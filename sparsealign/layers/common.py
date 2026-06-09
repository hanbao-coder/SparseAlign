"""Common utilities: seed, layer finder, arg formatting."""

import os
import random

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def find_linear_layers(module, layers=None, prefix=''):
    """Recursively collect all nn.Linear layers in a module.
    
    Returns a dict mapping layer names to the actual nn.Linear objects.
    """
    if layers is None:
        layers = [nn.Linear]
    if type(module) in layers:
        return {prefix: module}
    result = {}
    for name, child in module.named_children():
        result.update(find_linear_layers(
            child, layers=layers,
            prefix=prefix + '.' + name if prefix else name
        ))
    return result


def format_config(args):
    """Format argparse config for logging."""
    items = [f"{k}={v!r}" for k, v in vars(args).items()]
    lines = [", ".join(items[i:i + 5]) for i in range(0, len(items), 5)]
    return "Configuration:\n    " + "\n    ".join(lines)
