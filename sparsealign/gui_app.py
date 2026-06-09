"""SparseAlign GUI - PyQt5 图形界面

功能模式:
  1. 仅加载模型 (基准评估)
  2. 仅剪枝
  3. 剪枝 + 补偿
  4. 剪枝 + 补偿 + 蒸馏

剪枝算法: Wanda, L1_norm, L2_norm, SparseGPT
"""

import json
import os
import re
import sys
import time
import logging
import tempfile
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox,
    QDoubleSpinBox, QRadioButton, QButtonGroup, QTextEdit, QProgressBar,
    QFileDialog, QMessageBox, QTabWidget, QSplitter, QGridLayout,
    QFormLayout, QCheckBox, QFrame, QSizePolicy, QScrollArea,
    QTableWidget, QTableWidgetItem, QHeaderView, QStatusBar, QAction,
    QMenu, QMenuBar, QToolBar, QStyle
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSettings, QSize, QPropertyAnimation,
    QEasingCurve
)
from PyQt5.QtGui import QFont, QIcon, QColor, QPalette, QTextCursor


# ===========================================================================
# 常量与样式
# ===========================================================================

DEFAULT_PARAMS = {
    "model_path": "models/Qwen1.5-1.8B-Chat",
    "dataset_path": "data/wikitext2",
    "seed": 42,
    "nsamples": 512,
    "seqlen": 128,
    "method": "wanda",
    "sparsity": 0.7,
    "epochs": 10,
    "lr": 5e-5,
    "distill_epochs": 5,
    "distill_lr": 1e-5,
    "device": "cuda:0",
    "save_interval": 4,
}

MODE_LABELS = {
    0: "仅加载模型",
    1: "仅剪枝",
    2: "剪枝 + 补偿",
    3: "剪枝 + 补偿 + 蒸馏",
}

METHOD_LABELS = {
    "wanda": "Wanda (激活感知)",
    "l1_norm": "L1 范数",
    "l2_norm": "L2 范数",
    "sparsegpt": "SparseGPT (二阶)",
}

APP_STYLE = """
QMainWindow {
    background-color: #f5f6fa;
}
QGroupBox {
    font-weight: bold;
    border: 1px solid #dcdde1;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 18px;
    background-color: #ffffff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 8px;
    color: #2d3436;
}
QPushButton {
    border: 1px solid #dcdde1;
    border-radius: 4px;
    padding: 6px 16px;
    background-color: #ffffff;
    color: #2d3436;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #dfe6e9;
}
QPushButton#btnRun {
    background-color: #0984e3;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 10px 32px;
    font-size: 14px;
}
QPushButton#btnRun:hover {
    background-color: #0773c5;
}
QPushButton#btnRun:disabled {
    background-color: #b2bec3;
}
QPushButton#btnStop {
    background-color: #d63031;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 10px 32px;
    font-size: 14px;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    border: 1px solid #dcdde1;
    border-radius: 4px;
    padding: 4px 8px;
    background-color: #ffffff;
}
QTextEdit {
    border: 1px solid #dcdde1;
    border-radius: 4px;
    background-color: #2d3436;
    color: #b2bec3;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
}
QProgressBar {
    border: 1px solid #dcdde1;
    border-radius: 4px;
    text-align: center;
    height: 22px;
}
QProgressBar::chunk {
    background-color: #0984e3;
    border-radius: 3px;
}
QTableWidget {
    gridline-color: #dcdde1;
    border: 1px solid #dcdde1;
    border-radius: 4px;
}
QTableWidget::item {
    padding: 4px;
}
QHeaderView::section {
    background-color: #dfe6e9;
    padding: 4px;
    border: 1px solid #dcdde1;
    font-weight: bold;
}
QStatusBar {
    background-color: #dfe6e9;
    color: #636e72;
}
QMenuBar {
    background-color: #ffffff;
    border-bottom: 1px solid #dcdde1;
}
QMenuBar::item:selected {
    background-color: #dfe6e9;
}
QScrollArea {
    border: none;
}
QRadioButton {
    spacing: 6px;
}
"""


# ===========================================================================
# 后端运行线程
# ===========================================================================

class BackendWorker(QThread):
    """在独立线程中运行 sparsealign 后端"""

    log_signal = pyqtSignal(str)           # 日志行
    progress_signal = pyqtSignal(int, int)  # (current_layer, total_layers)
    finished_signal = pyqtSignal(dict)      # 结果字典
    error_signal = pyqtSignal(str)          # 错误信息
    status_signal = pyqtSignal(str)         # 状态文字

    def __init__(self, args_dict: dict):
        super().__init__()
        self.args_dict = args_dict
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            self._execute()
        except Exception as e:
            self.error_signal.emit(str(e))

    def _execute(self):
        a = self.args_dict

        # 检查路径
        model_path = a.get("model_path", "")
        if not os.path.isdir(model_path):
            self.error_signal.emit(f"模型路径无效: {model_path}")
            return

        self.status_signal.emit("正在导入模块...")
        from transformers import AutoTokenizer
        from .layers import common, loader
        from .compensation import compensate_wanda, compensate_sparsegpt
        from .data import get_loaders
        from .evaluation import llama_evaluate_perplexity
        from .training import run_distillation

        # 构造 args
        class Cfg:
            pass

        args = Cfg()
        args.model = model_path
        args.dataset = "wikitext2"
        args.seed = a.get("seed", 42)
        args.nsamples = a.get("nsamples", 512)
        args.seqlen = a.get("seqlen", 128)
        args.prune = a.get("prune", True)
        args.method = a.get("method", "wanda")
        args.sparsity = a.get("sparsity", 0.7)
        args.sparsity_ratio = args.sparsity
        args.prune_n = a.get("prune_n", 0)
        args.prune_m = a.get("prune_m", 0)
        args.compensate = a.get("compensate", False)
        args.epochs = a.get("epochs", 10)
        args.tune_epoch = args.epochs
        args.lr = a.get("lr", 5e-5)
        args.tune_lr = args.lr
        args.distill = a.get("distill", False)
        args.distill_epochs = a.get("distill_epochs", 5)
        args.distill_lr = a.get("distill_lr", 1e-5)
        args.device = a.get("device", "cuda:0")
        args.log_path = a.get("log_path", "./logs")
        args.checkpoint_dir = a.get("checkpoint_dir", "./checkpoints")
        args.cache_dir = a.get("cache_dir", "./cache")
        args.run_name = "sparsealign_gui"
        args.save = a.get("save", "")
        args.resume = a.get("resume", False)
        args.save_interval = a.get("save_interval", 4)
        args.save_layer_ckpt = True
        args.save_intervals = args.save_interval
        args.is_train = args.compensate

        # 日志
        os.makedirs(args.log_path, exist_ok=True)
        logger = logging.getLogger("SparseAlignGUI")
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        log_handler = _SignalLogHandler(self.log_signal)
        log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(log_handler)

        try:
            # 加载模型
            self.status_signal.emit("加载模型中...")
            logger.info(common.format_config(args))
            common.seed_everything(args.seed)

            self.log_signal.emit(f"加载模型: {args.model}")
            model = loader.load_qwen_model(args.model, device=args.device)
            model.seqlen = args.seqlen
            model.eval()

            tokenizer = AutoTokenizer.from_pretrained(
                args.model, trust_remote_code=True, local_files_only=True
            )
            dataloader, testloader, _ = get_loaders(
                n_samples=args.nsamples, seed=args.seed,
                seq_len=model.seqlen, model=args.model,
            )

            # 总层数
            total_layers = len(model.model.layers)
            self.progress_signal.emit(0, total_layers)

            # Stage 1: 剪枝 + 补偿
            if args.prune:
                if args.compensate:
                    self.status_signal.emit(f"Stage 1: {args.method} 剪枝 + 逐层补偿")
                else:
                    self.status_signal.emit(f"Stage 1: {args.method} 剪枝 (无补偿)")

                logger.info(f"=== Stage 1: {args.method} ===")

                if args.method in ("wanda", "l1_norm", "l2_norm"):
                    model = _compensate_with_progress(
                        compensate_wanda, model, dataloader, args.device, logger, args,
                        total_layers, self
                    )
                elif args.method == "sparsegpt":
                    model = _compensate_with_progress(
                        compensate_sparsegpt, model, dataloader, args.device, logger, args,
                        total_layers, self
                    )

            # Stage 2: 蒸馏
            if args.distill:
                self.status_signal.emit("Stage 2: 全局知识蒸馏")
                logger.info("=== Stage 2: 全局知识蒸馏 ===")
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

            # 评估
            self.status_signal.emit("评估 WikiText2 困惑度...")
            logger.info("WikiText2 困惑度评估")
            _, testloader_eval, _ = get_loaders(
                n_samples=args.nsamples, seed=args.seed,
                seq_len=model.seqlen, model=args.model,
            )
            model.eval()
            total_layers = len(model.model.layers)
            self.progress_signal.emit(0, total_layers)
            ppl = llama_evaluate_perplexity(
                model, testloader_eval, args, logger=logger, dataset="wikitext2",
                progress_callback=lambda cur, tot: self.progress_signal.emit(cur, tot)
            )

            # 保存
            if args.save:
                model.half()
                torch.save({"model": model, "tokenizer": tokenizer}, args.save)

            logger.info("完成.")

            # 收集结果
            results = {
                "success": True,
                "ppl": ppl,
                "sparsity": args.sparsity,
                "method": args.method,
                "mode": _get_mode_label(a),
                "total_layers": total_layers,
                "model_path": args.model,
            }
            self.finished_signal.emit(results)

        except Exception as e:
            import traceback
            logger.error(f"运行出错: {e}")
            logger.error(traceback.format_exc())
            self.error_signal.emit(str(e))


def _get_mode_label(a: dict) -> str:
    if a.get("distill"):
        return "剪枝 + 补偿 + 蒸馏"
    elif a.get("compensate"):
        return "剪枝 + 补偿"
    elif a.get("prune"):
        return "仅剪枝"
    return "仅加载模型"


def _compensate_with_progress(fn, model, dataloader, device, logger, args, total, worker):
    """包装 compensate 函数以发送进度信号。"""
    # 猴子补丁: 在每层处理后汇报进度
    import torch
    from .layers.pipeline import HiddenStateDataset
    from .training import CompensationConfig

    # 这里的补偿流程和 compensation.py 一样，但加入进度信号
    config = CompensationConfig(
        epochs=getattr(args, "tune_epoch", 10),
        lr=getattr(args, "tune_lr", 1e-5),
        seed=getattr(args, "seed", 42),
        use_amp=getattr(args, "use_amp", True),
        checkpoint_dir=getattr(args, "checkpoint_dir", "./checkpoints"),
    )

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(device)
    for layer in layers:
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "rotary_emb"):
            layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(device)

    # 捕获嵌入
    worker.status_signal.emit("捕获初始嵌入...")
    logger.info("捕获初始嵌入...")

    nsamples = len(dataloader)
    dtype = next(iter(model.parameters())).dtype
    hidden_states = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype, device=device,
    )
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class _Catcher(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.module = mod
        def forward(self, inp, **kw):
            hidden_states[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kw.get("attention_mask")
            cache["position_ids"] = kw.get("position_ids")
            raise ValueError

    layers[0] = layers[0].to(device)
    layers[0] = _Catcher(layers[0])
    for batch in dataloader:
        try:
            ids = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch["input_ids"].to(device)
            model(ids)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    logger.info("嵌入捕获完成.")

    pruned_states = hidden_states.clone().to(device)
    original_states = hidden_states.clone().to(device)
    inps = original_states
    outs = pruned_states

    worker.progress_signal.emit(0, total)

    for idx in range(total):
        if not worker._is_running:
            break

        layer = layers[idx].to(device)
        worker.status_signal.emit(f"处理第 {idx}/{total} 层...")
        worker.progress_signal.emit(idx, total)

        # 剪枝
        method_name = args.method
        if method_name == "wanda":
            from .pruning import compress_wanda
            layer, inps, outs, sparsity_constraint = compress_wanda(
                layer, device, args, inps, outs, attention_mask, position_ids
            )
        elif method_name in ("l1_norm", "l2_norm"):
            from .pruning import compress_magnitude
            layer, inps, outs, sparsity_constraint = compress_magnitude(
                layer, device, args, inps, outs, attention_mask, position_ids,
                norm=method_name
            )
        elif method_name == "sparsegpt":
            from .pruning import compress_sparsegpt
            layer, inps, outs, sparsity_constraint = compress_sparsegpt(
                layer, device, args, inps, outs, attention_mask, position_ids
            )

        inps = inps.cpu()
        outs = outs.cpu()

        # 补偿训练
        if args.is_train and config.epochs > 0:
            from .training import layer_compensation_train
            best_loss, _ = layer_compensation_train(
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

        # 计算补偿后输出
        if args.is_train:
            layer.eval()
            with torch.no_grad():
                for b_idx, batch in enumerate(pruned_states):
                    out = layer(
                        batch.unsqueeze(0),
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

        worker.progress_signal.emit(idx + 1, total)

    del original_states, pruned_states, inps, outs
    return model


class _SignalLogHandler(logging.Handler):
    """将 logging 消息转发到 Qt 信号"""

    def __init__(self, signal):
        super().__init__()
        self.signal = signal

    def emit(self, record):
        msg = self.format(record)
        self.signal.emit(msg)


# ===========================================================================
# UI 组件
# ===========================================================================

class PathSelector(QWidget):
    """文件/目录选择组件"""

    path_changed = pyqtSignal(str)

    def __init__(self, label: str, mode: str = "dir", parent=None):
        super().__init__(parent)
        self._mode = mode
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel(label)
        self._line = QLineEdit()
        self._btn = QPushButton("浏览...")
        self._btn.setFixedWidth(72)
        self._btn.clicked.connect(self._browse)

        layout.addWidget(self._label)
        layout.addWidget(self._line)
        layout.addWidget(self._btn)

        self._line.textChanged.connect(self.path_changed.emit)

    def _browse(self):
        if self._mode == "dir":
            path = QFileDialog.getExistingDirectory(self, "选择目录", self._line.text())
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择文件", self._line.text())
        if path:
            self._line.setText(path)

    @property
    def path(self) -> str:
        return self._line.text().strip()

    @path.setter
    def path(self, val: str):
        self._line.setText(val)


class ModeSelector(QWidget):
    """四模式选择器"""

    mode_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        for i, (key, label) in enumerate(MODE_LABELS.items()):
            rb = QRadioButton(label)
            rb.setChecked(i == 2)  # 默认剪枝+补偿
            self._group.addButton(rb, key)
            layout.addWidget(rb)

        self._group.buttonClicked.connect(self._on_clicked)

    def _on_clicked(self, btn):
        self.mode_changed.emit(self._group.id(btn))

    @property
    def current_mode(self) -> int:
        return self._group.checkedId()


# ===========================================================================
# 主窗口
# ===========================================================================

class SparseAlignGUI(QMainWindow):
    """SparseAlign 主窗口"""

    def __init__(self):
        super().__init__()
        self._worker: Optional[BackendWorker] = None
        self._results_history: List[Dict] = []
        self._config_path = "gui_config.json"
        self._init_ui()
        self._load_config()
        self._on_mode_changed(2)  # 默认剪枝+补偿

    # -------- UI 构建 --------

    def _init_ui(self):
        self.setWindowTitle("SparseAlign - 大语言模型逐层补偿剪枝框架")
        self.setMinimumSize(1100, 800)
        self.resize(1200, 900)

        # 中心区域
        central = QWidget()
        self.setCentralWidget(central)

        main_splitter = QSplitter(Qt.Vertical)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.addWidget(main_splitter)

        # 上半: 配置
        top_widget = QScrollArea()
        top_widget.setWidgetResizable(True)
        top_content = QWidget()
        top_layout = QVBoxLayout(top_content)
        top_layout.setSpacing(10)

        top_layout.addWidget(self._build_mode_group())
        top_layout.addWidget(self._build_path_group())
        top_layout.addWidget(self._build_param_group())
        top_layout.addWidget(self._build_control_bar())
        top_layout.addStretch()

        top_widget.setWidget(top_content)
        main_splitter.addWidget(top_widget)

        # 下半: 日志+结果
        bottom = QTabWidget()
        bottom.addTab(self._build_log_tab(), "运行日志")
        bottom.addTab(self._build_results_tab(), "结果展示")
        main_splitter.addWidget(bottom)
        main_splitter.setSizes([380, 420])

        # 状态栏
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status_lbl = QLabel("就绪")
        self._status.addWidget(self._status_lbl)
        self._mem_lbl = QLabel("")
        self._status.addPermanentWidget(self._mem_lbl)

        # 菜单
        self._init_menu()

    def _build_mode_group(self) -> QGroupBox:
        gb = QGroupBox("功能模式")
        layout = QHBoxLayout(gb)
        self._mode_selector = ModeSelector()
        self._mode_selector.mode_changed.connect(self._on_mode_changed)
        layout.addWidget(self._mode_selector)
        return gb

    def _build_path_group(self) -> QGroupBox:
        gb = QGroupBox("文件路径")
        layout = QVBoxLayout(gb)
        layout.setSpacing(6)

        self._model_path = PathSelector("模型路径:", "dir")
        self._dataset_path = PathSelector("数据集路径:", "dir")

        self._model_path.path = DEFAULT_PARAMS["model_path"]
        self._dataset_path.path = DEFAULT_PARAMS["dataset_path"]

        layout.addWidget(self._model_path)
        layout.addWidget(self._dataset_path)
        return gb

    def _build_param_group(self) -> QGroupBox:
        gb = QGroupBox("参数配置")
        grid = QGridLayout(gb)
        grid.setSpacing(8)

        # 剪枝算法
        row = 0
        grid.addWidget(QLabel("剪枝算法:"), row, 0)
        self._method_combo = QComboBox()
        for key, label in METHOD_LABELS.items():
            self._method_combo.addItem(label, key)
        self._method_combo.setCurrentIndex(0)
        grid.addWidget(self._method_combo, row, 1)

        # 稀疏率
        grid.addWidget(QLabel("稀疏率 (sparsity):"), row, 2)
        self._sparsity_spin = QDoubleSpinBox()
        self._sparsity_spin.setRange(0.1, 0.99)
        self._sparsity_spin.setSingleStep(0.05)
        self._sparsity_spin.setValue(DEFAULT_PARAMS["sparsity"])
        self._sparsity_spin.setSuffix(" (0~1)")
        grid.addWidget(self._sparsity_spin, row, 3)

        # 训练轮数
        row += 1
        grid.addWidget(QLabel("补偿训练轮数:"), row, 0)
        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(0, 100)
        self._epochs_spin.setValue(DEFAULT_PARAMS["epochs"])
        grid.addWidget(self._epochs_spin, row, 1)

        # 学习率
        grid.addWidget(QLabel("学习率:"), row, 2)
        self._lr_spin = QDoubleSpinBox()
        self._lr_spin.setRange(1e-6, 1e-2)
        self._lr_spin.setDecimals(6)
        self._lr_spin.setSingleStep(1e-5)
        self._lr_spin.setValue(DEFAULT_PARAMS["lr"])
        grid.addWidget(self._lr_spin, row, 3)

        # 样本数
        row += 1
        grid.addWidget(QLabel("校准样本数:"), row, 0)
        self._nsamples_spin = QSpinBox()
        self._nsamples_spin.setRange(8, 4096)
        self._nsamples_spin.setValue(DEFAULT_PARAMS["nsamples"])
        grid.addWidget(self._nsamples_spin, row, 1)

        # 随机种子
        grid.addWidget(QLabel("随机种子:"), row, 2)
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 99999)
        self._seed_spin.setValue(DEFAULT_PARAMS["seed"])
        grid.addWidget(self._seed_spin, row, 3)

        # 蒸馏参数 (初始隐藏)
        row += 1
        self._distill_epochs_label = QLabel("蒸馏轮数:")
        self._distill_epochs_spin = QSpinBox()
        self._distill_epochs_spin.setRange(1, 50)
        self._distill_epochs_spin.setValue(DEFAULT_PARAMS["distill_epochs"])
        self._distill_lr_label = QLabel("蒸馏学习率:")
        self._distill_lr_spin = QDoubleSpinBox()
        self._distill_lr_spin.setRange(1e-6, 1e-3)
        self._distill_lr_spin.setDecimals(7)
        self._distill_lr_spin.setSingleStep(1e-5)
        self._distill_lr_spin.setValue(DEFAULT_PARAMS["distill_lr"])
        grid.addWidget(self._distill_epochs_label, row, 0)
        grid.addWidget(self._distill_epochs_spin, row, 1)
        grid.addWidget(self._distill_lr_label, row, 2)
        grid.addWidget(self._distill_lr_spin, row, 3)

        # 保存这些蒸馏控件引用
        self._distill_widgets = [
            self._distill_epochs_label, self._distill_epochs_spin,
            self._distill_lr_label, self._distill_lr_spin,
        ]

        return gb

    def _build_control_bar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)

        self._btn_run = QPushButton("运行")
        self._btn_run.setObjectName("btnRun")
        self._btn_run.clicked.connect(self._on_run)
        self._btn_run.setMinimumWidth(140)

        self._btn_stop = QPushButton("停止")
        self._btn_stop.setObjectName("btnStop")
        self._btn_stop.clicked.connect(self._on_stop)
        self._btn_stop.setEnabled(False)
        self._btn_stop.setMinimumWidth(100)

        self._progress = QProgressBar()
        self._progress.setVisible(False)

        btn_save_cfg = QPushButton("保存配置")
        btn_save_cfg.clicked.connect(self._save_config)
        btn_load_cfg = QPushButton("加载配置")
        btn_load_cfg.clicked.connect(self._load_config)

        layout.addWidget(self._btn_run)
        layout.addWidget(self._btn_stop)
        layout.addWidget(self._progress)
        layout.addStretch()
        layout.addWidget(btn_save_cfg)
        layout.addWidget(btn_load_cfg)

        return bar

    def _build_log_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)

        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        layout.addWidget(self._log_view)

        clear_btn = QPushButton("清空日志")
        clear_btn.clicked.connect(self._log_view.clear)
        layout.addWidget(clear_btn)

        return w

    def _build_results_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        self._results_table = QTableWidget()
        self._results_table.setColumnCount(6)
        self._results_table.setHorizontalHeaderLabels([
            "模式", "剪枝算法", "稀疏率", "PPL", "PPL增长率(%)", "模型"
        ])
        self._results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self._results_table)

        btn_clear = QPushButton("清空结果")
        btn_clear.clicked.connect(self._on_clear_results)
        layout.addWidget(btn_clear)

        return w

    def _init_menu(self):
        menu = self.menuBar()

        file_menu = menu.addMenu("文件(&F)")
        file_menu.addAction("保存配置", self._save_config, Qt.CTRL + Qt.Key_S)
        file_menu.addAction("加载配置", self._load_config, Qt.CTRL + Qt.Key_O)
        file_menu.addSeparator()
        file_menu.addAction("退出", self.close, Qt.CTRL + Qt.Key_Q)

        help_menu = menu.addMenu("帮助(&H)")
        help_menu.addAction("使用说明", self._show_help)
        help_menu.addAction("关于", self._show_about)

    # -------- 模式切换 --------

    def _on_mode_changed(self, mode: int):
        # 模式 3 = 剪枝+补偿+蒸馏，显示蒸馏参数
        visible = (mode == 3)
        for w in self._distill_widgets:
            w.setVisible(visible)

        # 模式 0 = 仅加载模型，禁用剪枝/补偿参数
        self._method_combo.setEnabled(mode > 0)
        self._sparsity_spin.setEnabled(mode > 0)
        self._epochs_spin.setEnabled(mode >= 2)
        self._lr_spin.setEnabled(mode >= 2)
        self._distill_epochs_spin.setEnabled(mode == 3)
        self._distill_lr_spin.setEnabled(mode == 3)

    # -------- 运行控制 --------

    def _on_run(self):
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "提示", "已有任务在运行中.")
            return

        # 验证输入
        errors = self._validate_inputs()
        if errors:
            QMessageBox.warning(self, "输入错误", "\n".join(errors))
            return

        # 构建参数字典
        args = self._collect_args()

        # 日志
        self._log_view.clear()
        self._log(f"模式: {MODE_LABELS[self._mode_selector.current_mode]}", "info")
        self._log(f"模型: {args['model_path']}", "info")
        self._log(f"剪枝算法: {METHOD_LABELS.get(args['method'], args['method'])}", "info")
        self._log(f"稀疏率: {args['sparsity']}", "info")
        self._log("-" * 50)

        # 启动
        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._progress.setVisible(True)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)

        self._worker = BackendWorker(args)
        self._worker.log_signal.connect(self._log)
        self._worker.progress_signal.connect(self._on_progress)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.error_signal.connect(self._on_error)
        self._worker.status_signal.connect(self._on_status)
        self._worker.start()

    def _on_stop(self):
        if self._worker:
            self._log("正在停止...", "warn")
            self._worker.stop()
            self._worker.wait(3000)
        self._reset_ui()

    def _on_progress(self, current: int, total: int):
        if total > 0:
            pct = int(current / total * 100)
            self._progress.setValue(pct)
        self._progress.setFormat(f"层 {current}/{total}")

    def _on_finished(self, results: dict):
        self._log("-" * 50)
        ppl = results.get("ppl", -1)
        if ppl > 0:
            self._log(f"WikiText2 困惑度: {ppl:.4f}", "result")
        else:
            self._log("困惑度评估失败", "warn")

        results["timestamp"] = time.strftime("%H:%M:%S")
        self._results_history.append(results)
        self._update_results_table()
        self._reset_ui()

    def _on_error(self, msg: str):
        self._log(f"错误: {msg}", "error")
        QMessageBox.critical(self, "运行错误", f"任务执行失败:\n{msg}")
        self._reset_ui()

    def _on_status(self, text: str):
        self._status_lbl.setText(text)

    def _reset_ui(self):
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._progress.setVisible(False)
        self._status_lbl.setText("就绪")

    # -------- 日志 --------

    def _log(self, msg: str, level: str = "info"):
        import html
        colors = {"info": "#dfe6e9", "warn": "#fdcb6e", "error": "#ff7675",
                   "result": "#55efc4", "debug": "#b2bec3"}
        color = colors.get(level, "#dfe6e9")
        safe_msg = html.escape(msg)
        self._log_view.append(
            f'<span style="color:{color};">{safe_msg}</span>'
        )
        # 自动滚动
        cursor = self._log_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._log_view.setTextCursor(cursor)
        QApplication.processEvents()

    # -------- 结果表格 --------

    def _update_results_table(self):
        tbl = self._results_table
        tbl.setRowCount(len(self._results_history))

        for i, r in enumerate(self._results_history):
            ppl = r.get("ppl", -1)
            ppl_str = f"{ppl:.3f}" if ppl > 0 else "N/A"

            # PPL增长率 (相对于第一行)
            growth = "N/A"
            if i > 0 and len(self._results_history) > 0:
                base_ppl = self._results_history[0].get("ppl", -1)
                if base_ppl > 0 and ppl > 0:
                    growth = f"{(ppl - base_ppl) / base_ppl * 100:.1f}"

            items = [
                r.get("mode", ""),
                r.get("method", ""),
                f"{r.get('sparsity', 0):.0%}",
                ppl_str,
                growth,
                os.path.basename(r.get("model_path", "")),
            ]
            for j, txt in enumerate(items):
                item = QTableWidgetItem(txt)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if j == 4 and growth != "N/A":
                    try:
                        g = float(growth)
                        if g < 50:
                            item.setForeground(QColor("#00b894"))
                        elif g < 200:
                            item.setForeground(QColor("#fdcb6e"))
                        else:
                            item.setForeground(QColor("#d63031"))
                    except ValueError:
                        pass
                tbl.setItem(i, j, item)

    def _on_clear_results(self):
        self._results_history.clear()
        self._results_table.setRowCount(0)

    # -------- 配置保存/加载 --------

    def _collect_args(self) -> dict:
        mode = self._mode_selector.current_mode
        return {
            "model_path": self._model_path.path,
            "dataset_path": self._dataset_path.path,
            "seed": self._seed_spin.value(),
            "nsamples": self._nsamples_spin.value(),
            "seqlen": 128,
            "method": self._method_combo.currentData(),
            "sparsity": self._sparsity_spin.value(),
            "prune": (mode > 0),
            "compensate": (mode >= 2),
            "epochs": self._epochs_spin.value(),
            "lr": self._lr_spin.value(),
            "distill": (mode == 3),
            "distill_epochs": self._distill_epochs_spin.value(),
            "distill_lr": self._distill_lr_spin.value(),
            "device": "cuda:0",
            "log_path": "./logs",
            "checkpoint_dir": "./checkpoints",
            "cache_dir": "./cache",
            "save_interval": 4,
            "resume": False,
        }

    def _apply_args(self, args: dict):
        self._model_path.path = args.get("model_path", "")
        self._dataset_path.path = args.get("dataset_path", "")
        self._seed_spin.setValue(args.get("seed", 42))
        self._nsamples_spin.setValue(args.get("nsamples", 512))

        method = args.get("method", "wanda")
        for i in range(self._method_combo.count()):
            if self._method_combo.itemData(i) == method:
                self._method_combo.setCurrentIndex(i)
                break

        self._sparsity_spin.setValue(args.get("sparsity", 0.7))
        self._epochs_spin.setValue(args.get("epochs", 10))

        lr = args.get("lr", 5e-5)
        if 1e-6 <= lr <= 1e-2:
            self._lr_spin.setValue(lr)

        self._distill_epochs_spin.setValue(args.get("distill_epochs", 5))
        dlr = args.get("distill_lr", 1e-5)
        if 1e-6 <= dlr <= 1e-3:
            self._distill_lr_spin.setValue(dlr)

    def _save_config(self):
        args = self._collect_args()
        args["gui_mode"] = self._mode_selector.current_mode
        try:
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(args, f, indent=2, ensure_ascii=False)
            self._status_lbl.setText(f"配置已保存到 {self._config_path}")
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))

    def _load_config(self):
        if not os.path.exists(self._config_path):
            return
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                args = json.load(f)
            self._apply_args(args)
            self._status_lbl.setText(f"配置已从 {self._config_path} 加载")
        except Exception as e:
            QMessageBox.warning(self, "加载失败", str(e))

    # -------- 验证 --------

    def _validate_inputs(self) -> List[str]:
        errors = []
        model = self._model_path.path
        if not model:
            errors.append("请选择模型路径")
        elif not os.path.isdir(model):
            errors.append(f"模型路径不存在: {model}")

        dataset = self._dataset_path.path
        if not os.path.isdir(dataset):
            errors.append(f"数据集路径不存在: {dataset}")

        if self._mode_selector.current_mode > 0:
            sparsity = self._sparsity_spin.value()
            if sparsity <= 0 or sparsity >= 1:
                errors.append("稀疏率必须在 0 到 1 之间 (不含边界)")
            if sparsity >= 0.9:
                errors.append("稀疏率过大会导致模型性能严重退化")

        return errors

    # -------- 帮助 --------

    def _show_help(self):
        text = (
            "<h2>SparseAlign 使用说明</h2>"
            "<h3>功能模式</h3>"
            "<ul>"
            "<li><b>仅加载模型</b>: 加载模型并评估基准困惑度，作为后续对比的基线</li>"
            "<li><b>仅剪枝</b>: 对模型权重进行稀疏化，不进行补偿训练</li>"
            "<li><b>剪枝 + 补偿</b>: 剪枝后逐层补偿训练，恢复模型性能</li>"
            "<li><b>剪枝 + 补偿 + 蒸馏</b>: 三阶段完整流程</li>"
            "</ul>"
            "<h3>剪枝算法</h3>"
            "<ul>"
            "<li><b>Wanda</b>: 基于权重范数×激活范数的重要性排序</li>"
            "<li><b>L1 范数</b>: 按权重绝对值和排序，剪掉最小的</li>"
            "<li><b>L2 范数</b>: 按权重平方和排序</li>"
            "<li><b>SparseGPT</b>: 二阶 Hessian 感知剪枝</li>"
            "</ul>"
            "<h3>参数说明</h3>"
            "<ul>"
            "<li><b>稀疏率</b>: 0.5=剪掉50%, 0.7=剪掉70%。建议0.3~0.7</li>"
            "<li><b>补偿训练轮数</b>: 每层训练多少轮，建议5~10</li>"
            "<li><b>学习率</b>: 补偿训练学习率，默认5e-5</li>"
            "<li><b>校准样本数</b>: 越多补偿越准但显存越大，建议128~512</li>"
            "</ul>"
        )
        QMessageBox.information(self, "使用说明", text)

    def _show_about(self):
        QMessageBox.about(self, "关于 SparseAlign",
                          "<h3>SparseAlign v1.0</h3>"
                          "<p>大语言模型逐层补偿剪枝框架</p>"
                          "<p>剪枝算法: Wanda / L1 / L2 / SparseGPT</p>"
                          "<p>补偿策略: 逐层隐藏状态对齐 + 全局知识蒸馏</p>")

    def closeEvent(self, event):
        self._save_config()
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        event.accept()


# ===========================================================================
# 入口
# ===========================================================================

def launch_gui():
    """启动 GUI 应用"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)

    window = SparseAlignGUI()
    window.show()

    sys.exit(app.exec_())
