"""Data loading utilities for SparseAlign.

Loads WikiText2 dataset from the local ``data/wikitext2/`` directory.
"""

import os

import torch
from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def get_tokenizer(model_name: str):
    """Load tokenizer from local path."""
    return AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=True, local_files_only=True
    )


class CalibrationDataset(Dataset):
    """Wraps tokenized text into fixed-length sequences for calibration."""

    def __init__(self, input_ids: torch.Tensor, seq_len: int):
        self.seq_len = seq_len
        n_sequences = input_ids.numel() // seq_len
        self.sequences = input_ids[:n_sequences * seq_len].view(n_sequences, seq_len)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        ids = self.sequences[idx]
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
        }


def load_wikitext2(n_samples: int, seed: int, seq_len: int, tokenizer):
    """Load WikiText2 from local disk.

    Expects the dataset to be saved under ``data/wikitext2/`` via
    ``datasets.Dataset.save_to_disk()``.

    Returns
    -------
    train_batches : list
        ``n_samples`` random (input, target) pairs from the training split.
    test_tokens : Tensor
        Full tokenized test split for perplexity evaluation.
    """
    data_dir = os.path.join("data", "wikitext2")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"WikiText2 dataset not found at '{data_dir}'.\n"
            f"Please download it first:\n"
            f"  python -c \"from datasets import load_dataset; "
            f"load_dataset('wikitext', 'wikitext-2-raw-v1').save_to_disk('{data_dir}')\""
        )

    ds = load_from_disk(data_dir)
    train_split = ds["train"]
    test_split = ds["test"]

    train_tokens = tokenizer(" ".join(train_split["text"]), return_tensors="pt")
    test_tokens = tokenizer("\n\n".join(test_split["text"]), return_tensors="pt")

    rng = torch.Generator()
    rng.manual_seed(seed)

    train_batches = []
    upper_bound = train_tokens.input_ids.shape[1] - seq_len - 1
    sample_indices = torch.randint(0, upper_bound, (n_samples,), generator=rng).tolist()

    for start in sample_indices:
        end = start + seq_len
        chunk = train_tokens.input_ids[:, start:end]
        target = chunk.clone()
        target[:, :-1] = -100
        train_batches.append((chunk, target))

    return train_batches, test_tokens


def get_loaders(n_samples: int = 128, seed: int = 0, seq_len: int = 2048, model: str = ""):
    """Obtain WikiText2 calibration and evaluation data from local disk.

    Args:
        n_samples: Number of calibration samples to draw.
        seed: Random seed for reproducibility.
        seq_len: Length of each sequence.
        model: Model path used to load the matching tokenizer.

    Returns:
        Tuple of (train_batches, test_tokens, tokenizer).
    """
    tokenizer = get_tokenizer(model)
    train_batches, test_tokens = load_wikitext2(n_samples, seed, seq_len, tokenizer)
    return train_batches, test_tokens, tokenizer
