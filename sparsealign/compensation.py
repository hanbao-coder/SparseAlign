"""逐层补偿流程: Wanda, L1/L2幅度剪枝, 和 SparseGPT 三种剪枝方案"""

import copy
import gc
import glob
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .training import CompensationConfig, layer_compensation_train
from .pruning import compress_wanda, compress_sparsegpt, compress_magnitude
from .layers.common import find_linear_layers
from .layers.pipeline import HiddenStateDataset


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------

def _capture_embeddings(model, dataloader, device):
    """通过 Catcher 包装第0层来捕获初始嵌入。

    Returns
    -------
    hidden_states : shape (nsamples, seqlen, hidden_size)
    attention_mask : attention mask
    position_ids : position ids
    """
    nsamples = len(dataloader)

    dtype = next(iter(model.parameters())).dtype
    hidden_states = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=device,
    )

    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            hidden_states[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            raise ValueError

    layers = model.model.layers
    layers[0] = layers[0].to(device)
    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        try:
            input_ids = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch["input_ids"].to(device)
            model(input_ids)
        except ValueError:
            pass

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    return hidden_states, cache["attention_mask"], cache["position_ids"]


# ---------------------------------------------------------------------------
# 检查点相关
# ---------------------------------------------------------------------------

def _save_checkpoint(model, idx, args, **kwargs):
    """保存层检查点到磁盘。"""
    ckpt_dir = os.path.join(args.checkpoint_dir, "layer_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, f"layer_{idx}.pt")
    tmp_path = ckpt_path + ".tmp"

    save_dict = {
        "finished_layer_idx": idx,
        "model_state_dict": model.state_dict(),
    }
    save_dict.update(kwargs)

    torch.save(save_dict, tmp_path)
    os.rename(tmp_path, ckpt_path)

    # 清理旧检查点
    prev_idx = idx - getattr(args, "save_intervals", 10)
    if prev_idx > 0:
        prev_ckpt = os.path.join(ckpt_dir, f"layer_{prev_idx}.pt")
        if os.path.exists(prev_ckpt):
            os.remove(prev_ckpt)


def _load_latest_checkpoint(ckpt_dir, device):
    """加载最新的层检查点。"""
    existing = glob.glob(os.path.join(ckpt_dir, "layer_*.pt"))
    if not existing:
        return None

    latest = max(existing, key=os.path.getctime)
    return torch.load(latest, map_location="cpu")


def _resume_from_checkpoint(model, args, device):
    """尝试从最新检查点恢复训练。"""
    ckpt_dir = os.path.join(args.checkpoint_dir, "layer_checkpoints")
    states = {}

    if not getattr(args, "resume", True):
        return 0, states

    ckpt = _load_latest_checkpoint(ckpt_dir, device)
    if ckpt is None:
        return 0, states

    try:
        model.load_state_dict(ckpt["model_state_dict"])

        start_idx = ckpt["finished_layer_idx"] + 1

        if "pruned_states" in ckpt:
            states["pruned_states"] = ckpt["pruned_states"].to(device)
        if "original_states" in ckpt:
            states["original_states"] = ckpt["original_states"].to(device)

        if "err_stream_state" in ckpt and "pruned_states" not in ckpt:
            states["pruned_states"] = ckpt["err_stream_state"].to(device)
        if "std_stream_state" in ckpt and "original_states" not in ckpt:
            states["original_states"] = ckpt["std_stream_state"].to(device)

        if "inps" in ckpt:
            states["inps"] = ckpt["inps"]
        if "outs" in ckpt:
            states["outs"] = ckpt["outs"]

        logging.info(f"从第 {start_idx} 层恢复")
        return start_idx, states

    except Exception as e:
        logging.error(f"检查点恢复失败: {e}")
        logging.warning("从第0层开始.")
        return 0, states


# ---------------------------------------------------------------------------
# Wanda 补偿流程
# ---------------------------------------------------------------------------

def compensate_wanda(model, dataloader, device, logger, args):
    """Wanda 剪枝 + 逐层补偿训练。

    Returns
    -------
    model : 补偿后的模型。
    """
    logger.info("=" * 60)
    logger.info("Wanda 剪枝 + 补偿训练开始")
    logger.info("=" * 60)

    config = CompensationConfig(
        epochs=getattr(args, "tune_epoch", 10),
        lr=getattr(args, "tune_lr", 1e-5),
        seed=getattr(args, "seed", 42),
        use_amp=getattr(args, "use_amp", True),
        checkpoint_dir=getattr(args, "checkpoint_dir", "./checkpoints"),
    )

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # 禁用 KV 缓存
    model.config.use_cache = False
    layers = model.model.layers

    # 关键组件移到 GPU
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    for layer in layers:
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "rotary_emb"):
            layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(device)

    # 捕获初始嵌入
    logger.info("捕获初始嵌入...")
    hidden_states, attention_mask, position_ids = _capture_embeddings(
        model, dataloader, device
    )
    logger.info("嵌入捕获完成.")

    # 初始化隐藏状态缓冲区
    pruned_states = hidden_states.clone().to(device)
    original_states = hidden_states.clone().to(device)
    inps = original_states
    outs = pruned_states

    # 尝试恢复
    start_idx, resume_states = _resume_from_checkpoint(model, args, device)
    if "pruned_states" in resume_states:
        pruned_states = resume_states["pruned_states"]
    if "original_states" in resume_states:
        original_states = resume_states["original_states"]
    if "inps" in resume_states:
        inps = resume_states["inps"]
    if "outs" in resume_states:
        outs = resume_states["outs"]

    if start_idx >= len(layers):
        logger.info("所有层已处理完毕，跳过补偿.")
        return model

    stage_losses = []

    for idx in range(start_idx, len(layers)):
        layer = layers[idx].to(device)

        # Step 1: 计算未剪枝输出
        logger.info(f"--- 第 {idx} 层: 计算原始输出 ---")
        layer.eval()

        if getattr(args, "is_train", True):
            with torch.no_grad():
                for b_idx, batch in enumerate(original_states):
                    batch_exp = batch.unsqueeze(0)
                    out = layer(
                        batch_exp,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        output_attentions=False,
                        use_cache=False,
                    )
                    original_states[b_idx] = out[0].half()

        # Step 2: 剪枝
        logger.info(f"--- 第 {idx} 层: {args.method.upper()} 剪枝 ---")
        if getattr(args, '_magnitude_norm', None) in ("l1_norm", "l2_norm"):
            layer, inps, outs, sparsity_constraint = compress_magnitude(
                layer, device, args, inps, outs, attention_mask, position_ids,
                norm=getattr(args, '_magnitude_norm', "l1_norm")
            )
        else:
            layer, inps, outs, sparsity_constraint = compress_wanda(
                layer, device, args, inps, outs, attention_mask, position_ids
            )
        inps = inps.cpu()
        outs = outs.cpu()
        torch.cuda.empty_cache()

        # Step 3: 补偿训练
        if getattr(args, "is_train", True) and config.epochs > 0:
            logger.info(f"--- 第 {idx} 层: 补偿训练 ---")
            best_loss, history = layer_compensation_train(
                layer_idx=idx,
                adapted_layer=layer,
                pruned_states=pruned_states.detach(),
                original_states=original_states.detach(),
                attention_mask=attention_mask,
                position_ids=position_ids,
                sparsity_constraint=sparsity_constraint,
                config=config,
                logger=logger,
                device=device,
            )
            stage_losses.append(best_loss)
        else:
            stage_losses.append(0.0)

        # Step 4: 计算补偿后输出
        logger.info(f"--- 第 {idx} 层: 计算补偿后输出 ---")
        if getattr(args, "is_train", True):
            layer.eval()
            with torch.no_grad():
                for b_idx, batch in enumerate(pruned_states):
                    batch_exp = batch.unsqueeze(0)
                    out = layer(
                        batch_exp,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        output_attentions=False,
                        use_cache=False,
                    )
                    pruned_states[b_idx] = out[0].half()

        # 层移回 CPU
        layer = layer.cpu()
        model.model.layers[idx] = layer
        del layer
        torch.cuda.empty_cache()
        logger.info(f"第 {idx} 层处理完成.\n")

        # 定期保存检查点
        save_interval = getattr(args, "save_intervals", 10)
        if getattr(args, "save_layer_ckpt", True) and (
            idx % save_interval == 0 or idx == len(layers) - 1
        ):
            logger.info(f"保存第 {idx} 层检查点...")
            _save_checkpoint(
                model, idx, args,
                pruned_states=pruned_states.cpu(),
                original_states=original_states.cpu(),
                inps=inps.cpu(),
                outs=outs.cpu(),
            )
            logger.info(f"检查点已保存: layer_{idx}.pt")

    del original_states, pruned_states, inps, outs
    return model


# ---------------------------------------------------------------------------
# SparseGPT 补偿流程
# ---------------------------------------------------------------------------

def compensate_sparsegpt(model, dataloader, device, logger, args):
    """SparseGPT 剪枝 + 逐层补偿训练。

    Returns
    -------
    model : 补偿后的模型。
    """
    logger.info("=" * 60)
    logger.info("SparseGPT 剪枝 + 补偿训练开始")
    logger.info("=" * 60)

    config = CompensationConfig(
        epochs=getattr(args, "tune_epoch", 0),
        lr=getattr(args, "tune_lr", 1e-5),
        seed=getattr(args, "seed", 42),
        use_amp=getattr(args, "use_amp", True),
        checkpoint_dir=getattr(args, "checkpoint_dir", "./checkpoints"),
    )

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(device)
    for layer in layers:
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "rotary_emb"):
            layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(device)

    # 捕获初始嵌入
    logger.info("捕获初始嵌入...")
    hidden_states, attention_mask, position_ids = _capture_embeddings(
        model, dataloader, device
    )
    logger.info("嵌入捕获完成.")

    pruned_states = hidden_states.clone().to(device)
    original_states = hidden_states.clone().to(device)
    inps = original_states
    outs = pruned_states

    start_idx, resume_states = _resume_from_checkpoint(model, args, device)
    if "pruned_states" in resume_states:
        pruned_states = resume_states["pruned_states"]
    if "original_states" in resume_states:
        original_states = resume_states["original_states"]
    if "inps" in resume_states:
        inps = resume_states["inps"]
    if "outs" in resume_states:
        outs = resume_states["outs"]

    if start_idx >= len(layers):
        logger.info("所有层已处理完毕，跳过补偿.")
        return model

    stage_losses = []

    for idx in range(start_idx, len(layers)):
        layer = layers[idx].to(device)

        # Step 1: 计算未剪枝输出
        logger.info(f"--- 第 {idx} 层: 计算原始输出 ---")
        layer.eval()

        if getattr(args, "is_train", True):
            with torch.no_grad():
                for b_idx, batch in enumerate(original_states):
                    batch_exp = batch.unsqueeze(0)
                    out = layer(
                        batch_exp,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        output_attentions=False,
                        use_cache=False,
                    )
                    original_states[b_idx] = out[0].half()

        # Step 2: SparseGPT 剪枝
        logger.info(f"--- 第 {idx} 层: SparseGPT 剪枝 ---")
        layer, inps, outs, sparsity_constraint = compress_sparsegpt(
            layer, device, args, inps, outs, attention_mask, position_ids
        )
        inps = inps.cpu()
        outs = outs.cpu()
        torch.cuda.empty_cache()

        # Step 3: 补偿训练
        if getattr(args, "is_train", True) and config.epochs > 0:
            logger.info(f"--- 第 {idx} 层: 补偿训练 ---")
            best_loss, history = layer_compensation_train(
                layer_idx=idx,
                adapted_layer=layer,
                pruned_states=pruned_states.detach(),
                original_states=original_states.detach(),
                attention_mask=attention_mask,
                position_ids=position_ids,
                sparsity_constraint=sparsity_constraint,
                config=config,
                logger=logger,
                device=device,
            )
            stage_losses.append(best_loss)
        else:
            stage_losses.append(0.0)

        # Step 4: 计算补偿后输出
        logger.info(f"--- 第 {idx} 层: 计算补偿后输出 ---")
        if getattr(args, "is_train", True):
            layer.eval()
            with torch.no_grad():
                for b_idx, batch in enumerate(pruned_states):
                    batch_exp = batch.unsqueeze(0)
                    out = layer(
                        batch_exp,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        output_attentions=False,
                        use_cache=False,
                    )
                    pruned_states[b_idx] = out[0].half()

        layer = layer.cpu()
        model.model.layers[idx] = layer
        del layer
        torch.cuda.empty_cache()
        logger.info(f"第 {idx} 层处理完成.\n")

        save_interval = getattr(args, "save_intervals", 10)
        if getattr(args, "save_layer_ckpt", True) and (
            idx % save_interval == 0 or idx == len(layers) - 1
        ):
            logger.info(f"保存第 {idx} 层检查点...")
            _save_checkpoint(
                model, idx, args,
                pruned_states=pruned_states.cpu(),
                original_states=original_states.cpu(),
                inps=inps.cpu(),
                outs=outs.cpu(),
            )
            logger.info(f"检查点已保存: layer_{idx}.pt")

    del original_states, pruned_states, inps, outs
    return model
