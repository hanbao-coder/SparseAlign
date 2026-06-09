# SparseAlign · 稀疏对齐

> 大语言模型逐层补偿剪枝框架 —— 以 Qwen1.5-1.8B-Chat 为实验对象

---

## 项目简介

SparseAlign 是一个轻量级的大语言模型（LLM）压缩框架，针对 **高稀疏度下剪枝导致性能退化** 的问题，采用 **逐层隐藏状态对齐（Layer-wise Hidden State Alignment）** 的策略来拦截误差传播。

核心思路：每剪枝一层，立即用少量校准数据（约 128~256 条样本）将该层的输出分布微调对齐回原始模型的输出分布，从而避免误差随层数累积放大。

### 实验效果（Qwen1.5-1.8B-Chat，WikiText2，70% 稀疏度）

| 方案 | PPL | 相对仅剪枝的改善 |
|------|-----|------------------|
| 原始模型 | 33.62 | — |
| 仅剪枝（Wanda） | 298.11 | — |
| **剪枝 + 逐层补偿** | **82.48** | **↓ 72.4%** |

---

## 核心功能

- **多种剪枝算法**：Wanda（激活感知）、L1/L2 范数（幅值剪枝）、SparseGPT（二阶 Hessian）
- **逐层补偿机制**：MSE Loss 对齐剪枝层输出，冻结已剪枝权重位置
- **知识蒸馏（可选）**：全局 KL 散度对齐，进一步缩小剪枝模型与原始模型输出分布差距
- **PyQt5 可视化界面**：四种工作模式、参数配置、实时日志、实验结果对比
- **纯离线运行**：支持从本地目录加载模型与 WikiText2 数据集，无需联网

---

## 环境与依赖

| 项目 | 要求 |
|------|------|
| Python | ≥ 3.9 |
| PyTorch | ≥ 2.0.0 |
| CUDA GPU | 建议 8GB 显存以上（Qwen1.5-1.8B 约需 6GB） |

安装依赖：

```bash
pip install -r requirements.txt
```

依赖清单（见 `requirements.txt`）：

- `torch` — 深度学习框架
- `transformers` — 模型加载
- `datasets` — WikiText2 数据集加载
- `accelerate` — 训练加速工具
- `PyQt5` — 图形界面

---

## 离线数据准备

本项目设计为**纯离线运行**，请提前将以下文件放入对应目录：

### 1. 模型文件

将 Qwen1.5-1.8B-Chat 放入 `models/Qwen1.5-1.8B-Chat/` 目录下，结构如下：

```
models/Qwen1.5-1.8B-Chat/
├── config.json
├── model.safetensors   (或 pytorch_model.bin)
├── tokenizer.json
├── tokenizer_config.json
└── vocab.json
```

下载：<https://modelscope.cn/models/qwen/Qwen1.5-1.8B-Chat>

### 2. WikiText2 数据集

数据集已存放在 `data/wikitext2/` 目录下（arrow 格式），无需额外下载。如需重新生成，执行：

```python
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1")
ds.save_to_disk("data/wikitext2")
```

---

## 快速开始

提供两种使用方式：**图形界面（推荐新手）** 和 **命令行（适合脚本批量实验）**。

### 方式一：图形界面

```bash
python gui_main.py
```

界面提供四种工作模式：

1. **仅加载模型** — 评估原始模型 PPL，作为基准
2. **仅剪枝** — 对模型进行稀疏化，不做补偿
3. **剪枝 + 补偿** — 逐层剪枝并对齐输出分布
4. **剪枝 + 补偿 + 蒸馏** — 附加全局知识蒸馏

界面内可调整：剪枝算法、稀疏度、校准样本数、补偿轮次、学习率等参数，并实时显示日志与进度条。

### 方式二：命令行

```bash
# 1. 基准：仅加载模型，评估 PPL
python -m sparsealign.main models/Qwen1.5-1.8B-Chat

# 2. 仅剪枝（Wanda，50% 稀疏）
python -m sparsealign.main models/Qwen1.5-1.8B-Chat --prune --method wanda --sparsity 0.5

# 3. 剪枝 + 补偿（推荐配置）
python -m sparsealign.main models/Qwen1.5-1.8B-Chat \
    --prune --method wanda --sparsity 0.5 \
    --compensate --epochs 10 --lr 5e-5

# 4. 剪枝 + 补偿 + 蒸馏
python -m sparsealign.main models/Qwen1.5-1.8B-Chat \
    --prune --method wanda --sparsity 0.5 \
    --compensate --epochs 10 --lr 5e-5 \
    --distill --distill_epochs 5 --distill_lr 1e-5

# 使用 L1 范数剪枝
python -m sparsealign.main models/Qwen1.5-1.8B-Chat \
    --prune --method l1_norm --sparsity 0.5 --compensate --epochs 10 --lr 5e-5
```

> **高稀疏度提示**：稀疏度 ≥ 0.7 时，建议将 `--nsamples` 降至 128、`--epochs` 降至 3，避免过拟合导致 PPL 爆炸。

---

## 命令行参数一览

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `model`（位置参数） | 模型路径，如 `models/Qwen1.5-1.8B-Chat` | 必填 |
| `dataset`（位置参数） | 数据集名称，固定为 `wikitext2` | `wikitext2` |
| `--seed` | 随机种子 | 42 |
| `--nsamples` | 校准样本数 | 256 |
| `--seqlen` | 序列长度 | 128 |
| `--prune` | 启用剪枝 | False |
| `--method` | 剪枝算法：`wanda` / `sparsegpt` / `l1_norm` / `l2_norm` | `wanda` |
| `--sparsity` | 目标稀疏率（0~1） | 0.5 |
| `--prune_n` / `--prune_m` | N:M 稀疏模式（SparseGPT） | 0 / 0 |
| `--compensate` | 启用逐层补偿 | False |
| `--epochs` | 每层补偿训练轮数 | 10 |
| `--lr` | 补偿学习率 | 5e-5 |
| `--distill` | 启用全局知识蒸馏 | False |
| `--distill_epochs` | 蒸馏轮数 | 5 |
| `--distill_lr` | 蒸馏学习率 | 1e-5 |
| `--device` | 运行设备 | `cuda:0` |
| `--save` | 保存模型路径 | （空，不保存） |

---

## 算法流程

对于每个 Transformer 层，依次执行三个阶段：

```
输入 H_{l-1} → ① 剪枝 → ② 补偿微调 → ③ 传播输出 H_l → 下一层
```

1. **剪枝（Prune）**：根据所选算法（Wanda / L1 / L2 / SparseGPT）评估权重重要性，将低于阈值的权重置零
2. **补偿（Compensate）**：以 MSE Loss 为目标，在少量校准数据上微调剩余权重，使剪枝层的输出尽可能接近原始层输出
3. **传播（Propagate）**：将补偿后的输出作为下一层的输入，保证后续层接收到的隐藏状态与原始模型一致

代码实现上，每层的稀疏位置通过 `SparsityConstraint` 掩码锁定，补偿过程中不会影响已剪枝的位置。

---

## 项目结构

```
Laco-main/
├── gui_main.py              # GUI 启动器（python gui_main.py）
├── gui_config.json          # GUI 配置（自动保存/加载）
├── requirements.txt         # 依赖清单
├── models/                  # 本地模型目录（需自行放入）
├── data/
│   └── wikitext2/           # WikiText2 离线数据集
├── logs/                    # 运行日志（自动生成）
├── checkpoints/             # 逐层检查点（运行时自动生成）
└── sparsealign/             # 核心代码
    ├── main.py              # CLI 入口：加载/剪枝/补偿/蒸馏/评估
    ├── gui_app.py           # PyQt5 图形界面
    ├── pruning.py           # 剪枝算法：Wanda / SparseGPT / L1-L2
    ├── compensation.py      # 逐层补偿流水线
    ├── training.py          # 知识蒸馏训练循环
    ├── evaluation.py        # WikiText2 PPL 评估
    ├── data.py              # 数据加载器
    └── layers/              # 模型层相关工具
        ├── common.py        # 种子、配置打印等通用工具
        ├── loader.py        # 模型加载器
        └── pipeline.py      # 隐藏状态封装
```

---

## 常见问题

### Q1：补偿后 PPL 反而升高怎么办？
这通常是补偿数据与测试数据分布偏差、或在高稀疏度下过拟合导致。解决方法：
- 将 `--nsamples` 降至 128，`--epochs` 降至 3
- 降低学习率（如 `--lr 1e-5`）
- 先在 50% 稀疏度验证趋势，再逐步提升稀疏度
- 每次实验前清理旧检查点：`Remove-Item -Recurse -Force checkpoints`

### Q2：检查点的作用是什么？
检查点在每层补偿完成后保存（默认每 4 层保存一次），用于中断恢复。**不能直接用于补偿操作**，因为补偿是逐层进行的，必须从头开始执行流水线。

### Q3：仅加载模型模式下进度条不动？
`evaluation.py` 中的 `llama_evaluate_perplexity()` 已内置 `progress_callback` 回调。在 GUI 模式下会自动触发进度更新；CLI 模式则通过日志输出每层评估结果。

### Q4：可以使用其他数据集吗？
当前版本已简化为仅支持 WikiText2（离线加载）。如需更换数据集，修改 `sparsealign/data.py` 中的 `get_loaders()` 函数即可。

---

## License

基于原项目 LaCo（Liu et al., ACL 2025）开源协议。本项目在其基础上进行了精简、重构与功能扩展。

```bibtex
@inproceedings{liu2025laco,
    title     = {LaCo: Layer-wise Compensation for Pruned Large Language Models},
    author    = {Liu, Yingen and Wu, Fan and Pan, Xuyan and Li, Ruihui and Tang, Zhuo and Li, Kenli},
    booktitle = {ACL},
    year      = {2025}
}
```
