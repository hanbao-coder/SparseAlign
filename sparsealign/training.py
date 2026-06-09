"""训练配置与逐层补偿训练循环"""

import gc
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class CompensationConfig:
    """逐层补偿训练配置"""

    epochs: int = 10
    lr: float = 5e-5
    batch_size: int = 1
    weight_decay: float = 1e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    max_grad_norm: float = 1.0
    use_amp: bool = True
    seed: int = 42
    log_interval: int = 100
    checkpoint_dir: str = "./checkpoints"


# ---------------------------------------------------------------------------
# 损失函数
# ---------------------------------------------------------------------------

def compute_mse_loss(
    student_out: torch.Tensor,
    target_out: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """计算两个隐藏状态张量之间的 MSE 损失。"""
    if mask is not None:
        mask = mask.unsqueeze(-1).expand_as(student_out)
        student_out = student_out[mask]
        target_out = target_out[mask]

    return F.mse_loss(student_out, target_out)


def compute_cosine_loss(
    student_out: torch.Tensor,
    target_out: torch.Tensor,
) -> torch.Tensor:
    """计算 1 - 余弦相似度 损失。"""
    pred_flat = student_out.reshape(-1, student_out.shape[-1])
    target_flat = target_out.reshape(-1, target_out.shape[-1])
    cosine_sim = F.cosine_similarity(pred_flat, target_flat, dim=-1)
    return 1.0 - cosine_sim.mean()


# ---------------------------------------------------------------------------
# 逐层补偿训练
# ---------------------------------------------------------------------------

class _HiddenStateDataset(Dataset):
    """将剪枝/原始隐藏状态配对为数据集。"""

    def __init__(
        self,
        pruned_states: torch.Tensor,
        original_states: torch.Tensor,
    ):
        self.pruned = pruned_states
        self.original = original_states

    def __len__(self) -> int:
        return self.pruned.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pruned[idx], self.original[idx]


def layer_compensation_train(
    layer_idx: int,
    adapted_layer: nn.Module,
    pruned_states: torch.Tensor,
    original_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    sparsity_constraint,
    config: CompensationConfig,
    logger: logging.Logger,
    device: torch.device,
) -> Tuple[float, List[Dict]]:
    """对单个剪枝层进行补偿训练。"""
    adapted_layer = adapted_layer.float()
    params = [p for p in adapted_layer.parameters() if p.requires_grad]
    adapted_layer.train()

    optimizer = torch.optim.AdamW(
        params,
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-7,
    )

    scaler = GradScaler() if config.use_amp else None

    dataset = _HiddenStateDataset(pruned_states, original_states)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, drop_last=False,
    )

    total_params = sum(p.numel() for p in params)
    logger.info(f"=== 第 {layer_idx} 层补偿训练 ===")
    logger.info(f"可训练参数: {total_params:,}")
    logger.info(f"训练样本数: {len(dataset)}")

    best_loss = float("inf")
    training_history: List[Dict] = []

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_cosine = 0.0
        num_batches = 0

        for batch_idx, (pruned_batch, original_batch) in enumerate(dataloader):
            pruned_batch = pruned_batch.to(device)
            original_batch = original_batch.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=config.use_amp):
                outputs = adapted_layer(
                    pruned_batch,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_attentions=False,
                    use_cache=False,
                )
                compensated = outputs[0]

                loss_mse = compute_mse_loss(compensated, original_batch)
                loss_cosine = compute_cosine_loss(compensated, original_batch)
                loss = loss_mse

            if config.use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            if sparsity_constraint is not None:
                sparsity_constraint.apply_mask_to_gradients(adapted_layer)

            torch.nn.utils.clip_grad_norm_(params, max_norm=config.max_grad_norm)

            if config.use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += loss_mse.item()
            epoch_cosine += loss_cosine.item()
            num_batches += 1

            if batch_idx % config.log_interval == 0:
                logger.info(
                    f"第{layer_idx}层 Epoch{epoch} Batch{batch_idx}: "
                    f"Loss={loss.item():.6f} MSE={loss_mse.item():.6f} "
                    f"Cosine={loss_cosine.item():.6f}"
                )

        avg_loss = epoch_loss / num_batches
        avg_mse = epoch_mse / num_batches
        avg_cosine = epoch_cosine / num_batches
        current_lr = optimizer.param_groups[0]["lr"]

        training_history.append(
            {
                "epoch": epoch,
                "loss": avg_loss,
                "mse": avg_mse,
                "cosine": avg_cosine,
                "lr": current_lr,
            }
        )

        logger.info(
            f"第{layer_idx}层 Epoch{epoch} 总结: "
            f"Loss={avg_loss:.6f} MSE={avg_mse:.6f} "
            f"Cosine={avg_cosine:.6f} LR={current_lr:.2e}"
        )

        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss

    adapted_layer.eval()
    for p in params:
        p.requires_grad = False

    logger.info(
        f"=== 第 {layer_idx} 层补偿训练完成 ===\n"
        f"最终Loss: {best_loss:.6f} 训练Epoch: {config.epochs}"
    )

    return best_loss, training_history


# ---------------------------------------------------------------------------
# 全局知识蒸馏 (Stage 2)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _cache_teacher_logits(
    teacher_model: nn.Module,
    sample_inputs: list,
    cache_path: str,
    device: torch.device,
    logger: logging.Logger,
):
    """用教师模型推理所有样本，缓存 logits 到磁盘。"""
    logger.info("[蒸馏] 教师模型推理，缓存 logits...")
    os.makedirs(cache_path, exist_ok=True)

    teacher_model.eval()
    teacher_model = teacher_model.half().to(device)

    all_logits = []
    for i, (input_ids, _) in enumerate(sample_inputs):
        input_ids = input_ids.to(device)
        attention_mask = torch.ones(1, input_ids.shape[1], dtype=torch.long, device=device)
        position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)

        out = teacher_model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        all_logits.append(out.logits[:, :-1, :].cpu().half())

        if (i + 1) % 50 == 0:
            logger.info(f"  已推理 {i + 1}/{len(sample_inputs)} 条样本")

    torch.save(all_logits, os.path.join(cache_path, "teacher_logits.pt"))
    logger.info(f"[蒸馏] 共缓存 {len(all_logits)} 条样本 logits")

    del teacher_model
    torch.cuda.empty_cache()
    gc.collect()


def run_distillation(
    student_model: nn.Module,
    teacher_model: nn.Module,
    sample_inputs: list,
    distill_epochs: int,
    distill_lr: float,
    logger: logging.Logger,
    device: torch.device,
    cache_dir: str = "./cache",
):
    """全局知识蒸馏: 用教师模型的 logits 分布指导学生模型。"""
    if distill_epochs <= 0:
        return student_model

    logger.info("=" * 60)
    logger.info("全局知识蒸馏开始 (Stage 2)")
    logger.info(f"蒸馏轮数: {distill_epochs}  学习率: {distill_lr}")
    logger.info("=" * 60)

    _cache_teacher_logits(teacher_model, sample_inputs, cache_dir, device, logger)

    all_teacher_logits = torch.load(
        os.path.join(cache_dir, "teacher_logits.pt"), map_location="cpu"
    )

    student_model.train()
    student_model = student_model.to(device)

    trainable_params = [p for p in student_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=distill_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=distill_epochs, eta_min=1e-7
    )
    scaler = GradScaler()

    logger.info(f"可训练参数: {sum(p.numel() for p in trainable_params):,}")

    for epoch in range(distill_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for i, (input_ids, _) in enumerate(sample_inputs):
            input_ids = input_ids.to(device)
            teacher_logits = all_teacher_logits[i].to(device)

            attention_mask = torch.ones(1, input_ids.shape[1], dtype=torch.long, device=device)
            position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                out = student_model(
                    input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
                student_logits = out.logits[:, :-1, :].contiguous()

                loss = F.kl_div(
                    F.log_softmax(student_logits / 2.0, dim=-1),
                    F.softmax(teacher_logits / 2.0, dim=-1),
                    reduction="batchmean",
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1

            if i % 50 == 0:
                logger.info(f"[蒸馏] Epoch {epoch}  Sample {i}/{len(sample_inputs)}  Loss={loss.item():.6f}")

        avg_loss = epoch_loss / num_batches
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"[蒸馏] Epoch {epoch} 总结: Loss={avg_loss:.6f}  LR={current_lr:.2e}")
        scheduler.step()

    student_model.eval()
    logger.info("全局知识蒸馏完成.")
    return student_model
