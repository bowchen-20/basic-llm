"""
Dataset and DataLoader helpers.

Works with any tokenizer that exposes encode(text) -> list[int],
so both CharTokenizer and BPETokenizer plug in without changes.
"""

import glob as _glob
import os
import torch
from torch.utils.data import Dataset, DataLoader


class TokenDataset(Dataset):
    """
    Sliding-window dataset over a flat token sequence.

    Each item is an (x, y) pair of length block_size where y = x shifted
    right by one — the standard next-token-prediction target.
    """

    def __init__(self, tokens: torch.Tensor, block_size: int):
        assert len(tokens) > block_size, "Token sequence shorter than block_size"
        self.tokens     = tokens
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[idx : idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def build_datasets(
    text: str,
    tokenizer,
    block_size: int,
    val_frac: float = 0.1,
) -> tuple[TokenDataset, TokenDataset]:
    """
    Tokenize text and split chronologically into train / val datasets.

    Uses a chronological split (not random) so val always contains the
    tail of the corpus — more realistic than random shuffling.
    """
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_val  = max(block_size + 1, int(len(tokens) * val_frac))
    return (
        TokenDataset(tokens[:-n_val],  block_size),
        TokenDataset(tokens[-n_val:],  block_size),
    )


def make_loader(
    dataset: TokenDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True,   # keeps batch size constant throughout training
    )


def build_datasets_from_dir(
    dir_path: str,
    tokenizer,
    block_size: int,
    val_frac: float = 0.1,
    pattern: str = "*.txt",
) -> tuple[TokenDataset, TokenDataset]:
    """
    Load all files matching pattern from dir_path, concatenate them,
    and return train / val datasets.

    Files are sorted by name so the split is deterministic.
    Each file is separated by a newline to prevent context bleed across
    document boundaries.
    """
    paths = sorted(_glob.glob(os.path.join(dir_path, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matching '{pattern}' in {dir_path!r}")
    parts: list[str] = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            parts.append(f.read())
    print(f"Loaded {len(parts)} file(s) from {dir_path!r} ({sum(len(t) for t in parts):,} chars)")
    return build_datasets("\n".join(parts), tokenizer, block_size, val_frac)
