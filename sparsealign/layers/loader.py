"""Model loading utilities.

Loads model and tokenizer from the local ``models/`` directory.
"""

import torch


def _skip_init(*args, **kwargs):
    """No-op weight initializer to speed up model loading."""
    pass


def load_qwen_model(model_path, device="cuda:0"):
    """Load a Qwen model from a local directory.

    Args:
        model_path: Path to the model directory under ``models/``.
        device: Target device for the model.

    Returns:
        The loaded causal language model.
    """
    torch.nn.init.kaiming_uniform_ = _skip_init
    torch.nn.init.uniform_ = _skip_init
    torch.nn.init.normal_ = _skip_init

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
        local_files_only=True,
    )
    return model
