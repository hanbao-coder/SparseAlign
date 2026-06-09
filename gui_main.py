#!/usr/bin/env python
"""SparseAlign GUI 启动器
用法: python gui_main.py
"""

import sys
import os

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sparsealign.gui_app import launch_gui

if __name__ == "__main__":
    launch_gui()
