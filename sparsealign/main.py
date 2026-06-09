"""SparseAlign: 大语言模型逐层补偿剪枝框架

主入口，支持:
  - 模型加载 (不剪枝，基准困惑度)
  - 纯剪枝 (只剪枝不补偿，对比效果)
  - Stage 1 补偿: 逐层隐藏状态对齐
  - Stage 2 蒸馏: 全局知识蒸馏
  - 评估: WikiText2 困惑度

用法:
  # 基准: 只加载模型
  python -m sparsealign.main models/Qwen2.5-1.5B-Instruct

  # 纯剪枝 (无反补)
  python -m sparsealign.main models/Qwen2.5-1.5B-Instruct --prune --method wanda --sparsity 0.5

  # 剪枝 + 补偿
  python -m sparsealign.main models/Qwen2.5-1.5B-Instruct --prune --method wanda --sparsity 0.5 --compensate --epochs 10 --lr 5e-5

  # 剪枝 + 补偿 + 蒸馏
  python -m sparsealign.main models/Qwen2.5-1.5B-Instruct --prune --method wanda --sparsity 0.5 --compensate --epochs 10 --lr 5e-5 --distill --distill_epochs 5
"""

import argparse
import logging
import os
import time

import torch
from transformers import AutoTokenizer

from .layers import common, loader
from .compensation import compensate_sparsegpt, compensate_wanda
from .data import get_loaders
from .evaluation import llama_evaluate_perplexity
from .training import run_distillation


def parse_args():
    parser = argparse.ArgumentParser(
        description="SparseAlign: 大语言模型逐层补偿剪枝框架"
    )

    parser.add_argument("model", type=str,
                        help="模型路径 (默认: models/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("dataset", type=str, nargs="?", default="wikitext2",
                        help="校准数据集 (固定为 wikitext2, 从 data/wikitext2/ 读取)")

    # 数据参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--nsamples", type=int, default=256, help="校准样本数")
    parser.add_argument("--seqlen", type=int, default=128, help="序列长度")

    # 剪枝参数
    parser.add_argument("--prune", action="store_true", help="启用剪枝")
    parser.add_argument("--method", type=str, default="wanda",
                        choices=["wanda", "sparsegpt", "l1_norm", "l2_norm"], help="剪枝算法")
    parser.add_argument("--sparsity", type=float, default=0.5, help="目标稀疏率")
    parser.add_argument("--prune_n", type=int, default=0, help="N:M稀疏的N (SparseGPT)")
    parser.add_argument("--prune_m", type=int, default=0, help="N:M稀疏的M (SparseGPT)")

    # Stage 1: 逐层补偿训练参数
    parser.add_argument("--compensate", action="store_true", help="启用逐层补偿训练")
    parser.add_argument("--epochs", type=int, default=10, help="每层补偿训练轮数")
    parser.add_argument("--lr", type=float, default=5e-5, help="补偿训练学习率")

    # Stage 2: 全局知识蒸馏参数
    parser.add_argument("--distill", action="store_true", help="启用全局知识蒸馏 (Stage 2)")
    parser.add_argument("--distill_epochs", type=int, default=5, help="蒸馏训练轮数")
    parser.add_argument("--distill_lr", type=float, default=1e-5, help="蒸馏学习率")

    # 输出参数
    parser.add_argument("--device", type=str, default="cuda:0", help="运行设备")
    parser.add_argument("--log_path", type=str, default="./logs", help="日志输出目录")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="检查点目录")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="缓存目录")
    parser.add_argument("--run_name", type=str, default="sparsealign", help="运行名称")
    parser.add_argument("--save", type=str, default="", help="保存补偿后模型的路径")
    parser.add_argument("--resume", action="store_true", default=True, help="从检查点恢复")
    parser.add_argument("--save_interval", type=int, default=4, help="每隔N层保存检查点")

    return parser.parse_args()


def _add_compat_args(args):
    """给 args 添加内部代码需要的别名属性。"""
    args.sparsity_ratio = args.sparsity
    args.tune_epoch = args.epochs
    args.tune_lr = args.lr
    args.is_train = args.compensate
    args.save_layer_ckpt = True
    args.save_intervals = args.save_interval
    return args


def setup_logging(args):
    os.makedirs(args.log_path, exist_ok=True)
    logger = logging.getLogger("SparseAlign")
    logger.setLevel(logging.INFO)

    ts = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.log_path, f"{args.run_name}_{ts}.log")

    fh = logging.FileHandler(save_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def main():
    args = parse_args()
    args = _add_compat_args(args)

    logger = setup_logging(args)
    logger.info(common.format_config(args))
    common.seed_everything(args.seed)

    # 加载模型
    logger.info(f"加载模型: {args.model}")
    model = loader.load_qwen_model(args.model, device=args.device)
    model.seqlen = args.seqlen
    model.eval()

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    dataloader, testloader, _ = get_loaders(
        n_samples=args.nsamples, seed=args.seed,
        seq_len=model.seqlen, model=args.model,
    )

    # Stage 1: 剪枝 + 逐层补偿
    if args.prune:
        if args.compensate:
            logger.info(f"=== Stage 1: {args.method} 剪枝 + 逐层补偿 ===")
        else:
            logger.info(f"=== Stage 1: {args.method} 剪枝 (无补偿) ===")

        if args.method == "wanda":
            model = compensate_wanda(model, dataloader, args.device, logger, args)
        elif args.method in ("l1_norm", "l2_norm"):
            setattr(args, '_magnitude_norm', args.method)
            model = compensate_wanda(model, dataloader, args.device, logger, args)
        elif args.method == "sparsegpt":
            model = compensate_sparsegpt(model, dataloader, args.device, logger, args)

    # Stage 2: 全局知识蒸馏
    if args.distill:
        logger.info("=== Stage 2: 全局知识蒸馏 ===")
        logger.info("加载教师模型 (原始未剪枝模型)...")
        teacher = loader.load_qwen_model(args.model, device="cpu")
        teacher.seqlen = args.seqlen
        teacher.eval()

        model = run_distillation(
            student_model=model,
            teacher_model=teacher,
            sample_inputs=dataloader,
            distill_epochs=args.distill_epochs,
            distill_lr=args.distill_lr,
            logger=logger,
            device=args.device,
            cache_dir=args.cache_dir,
        )
        del teacher
        torch.cuda.empty_cache()

    # WikiText2 困惑度评估（无论是否剪枝都会执行）
    logger.info("WikiText2 困惑度评估")
    _, testloader_eval, _ = get_loaders(
        n_samples=args.nsamples, seed=args.seed,
        seq_len=model.seqlen, model=args.model,
    )
    model.eval()
    llama_evaluate_perplexity(model, testloader_eval, args, logger=logger, dataset="wikitext2")

    # 保存模型
    if args.save:
        model.half()
        torch.save({"model": model, "tokenizer": tokenizer}, args.save)
        logger.info(f"模型已保存至: {args.save}")

    logger.info("完成.")


if __name__ == "__main__":
    main()
