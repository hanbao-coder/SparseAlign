"""困惑度评估模块

提供显存友好的逐层困惑度计算方法。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .layers.pipeline import HiddenStateDataset


@torch.no_grad()
def llama_evaluate_perplexity(model, test_tokens, args, logger, dataset: str,
                              log_wandb: bool = False, progress_callback=None):
    """逐层处理计算困惑度，同一时刻只有一层在 GPU 上。

    Args:
        model: 待评估的语言模型。
        test_tokens: 分词后的测试数据，需包含 ``input_ids`` 属性。
        args: 包含 ``device`` 的命名空间。
        logger: 日志记录器。
        dataset: 数据集名称。
        log_wandb: 是否记录到 wandb。
        progress_callback: 可选，每完成一层时调用 callback(cur, total)。

    Returns:
        困惑度浮点数。
    """
    logger.info("开始困惑度评估")
    device = args.device

    token_ids = test_tokens.input_ids
    n_samples = token_ids.numel() // model.seqlen

    original_use_cache = model.config.use_cache
    model.config.use_cache = False
    transformer_layers = model.model.layers

    # 将共享模块移到 GPU
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model.model.norm = model.model.norm.to(device)
    for layer in transformer_layers:
        layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(device)

    # 通过 Catcher 捕获第一层的嵌入
    transformer_layers[0] = transformer_layers[0].to(device)

    captured_states = torch.zeros(
        (n_samples, model.seqlen, model.config.hidden_size),
        dtype=torch.float16,
        device=torch.device("cpu"),
    )
    capture_info = {"index": 0, "attention_mask": None, "position_ids": None}

    class EmbeddingCatcher(nn.Module):
        """临时替换第一层来捕获输入嵌入。"""
        def __init__(self, wrapped_layer):
            super().__init__()
            self.wrapped = wrapped_layer

        def forward(self, hidden, **kwargs):
            captured_states[capture_info["index"]] = hidden.cpu()
            capture_info["index"] += 1
            capture_info["attention_mask"] = kwargs["attention_mask"]
            capture_info["position_ids"] = kwargs["position_ids"]
            raise ValueError

    transformer_layers[0] = EmbeddingCatcher(transformer_layers[0])

    for sample_idx in range(n_samples):
        batch = token_ids[:, (sample_idx * model.seqlen):((sample_idx + 1) * model.seqlen)].to(device)
        try:
            model(batch)
        except ValueError:
            pass

    # 恢复第一层，将共享模块移回 CPU
    transformer_layers[0] = transformer_layers[0].wrapped
    transformer_layers[0] = transformer_layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    for layer in transformer_layers:
        layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.cpu()

    torch.cuda.empty_cache()

    attn_mask = capture_info["attention_mask"]
    pos_ids = capture_info["position_ids"]

    # 逐层处理
    for layer_idx in range(len(transformer_layers)):
        logger.info(f"处理第 {layer_idx} 层")
        current_layer = transformer_layers[layer_idx].to(device)

        layer_data = HiddenStateDataset(captured_states, device=device)
        loader = DataLoader(layer_data, batch_size=1, shuffle=False, drop_last=False)

        for sample_pos, batch in loader:
            output = current_layer(batch, attention_mask=attn_mask, position_ids=pos_ids)[0]
            captured_states[sample_pos] = output.half().cpu()

        transformer_layers[layer_idx] = current_layer.cpu()
        del current_layer
        torch.cuda.empty_cache()
        if progress_callback:
            progress_callback(layer_idx + 1, len(transformer_layers))

    # 最终隐藏状态计算困惑度
    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(device)
    model.lm_head = model.lm_head.to(device)

    token_ids = token_ids.to(device)
    negative_log_likelihoods = []

    eval_data = HiddenStateDataset(captured_states, device=device)
    eval_loader = DataLoader(eval_data, batch_size=1, shuffle=False, drop_last=False)

    for sample_pos, hidden in eval_loader:
        if model.model.norm is not None:
            hidden = model.model.norm(hidden)
        logits = model.lm_head(hidden)

        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = token_ids[:, (sample_pos * model.seqlen):((sample_pos + 1) * model.seqlen)][:, 1:]

        loss_fn = nn.CrossEntropyLoss()
        sample_loss = loss_fn(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
        )
        negative_log_likelihoods.append(sample_loss.float() * model.seqlen)

    perplexity = torch.exp(torch.stack(negative_log_likelihoods).sum() / (n_samples * model.seqlen))

    logger.info("-" * 50)
    logger.info(f"WikiText2 困惑度: {perplexity.item():.3f}")
    logger.info("-" * 50)

    if log_wandb:
        import wandb
        wandb.log({f"{dataset}/perplexity": perplexity.item()})

    model.config.use_cache = original_use_cache
    return perplexity.item()
