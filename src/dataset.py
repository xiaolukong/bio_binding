"""
Dataset utilities for DNA-Protein binding affinity training.

Data format
-----------
Each .npy file stores one batch as a Python dict with two keys:
    'dna':  float32 array of shape (10, 8, 60,  512)
    'prot': float32 array of shape (10, 8, 514, 960)

    10  = data points per file (= batch size)
    8   = samples per data point (index 0 = positive, 1-7 = negatives)
    60  = DNA token length
    512 = DNA embedding dim  (DNABERT-2)
    514 = Protein token length
    960 = Protein embedding dim (ESM-2)

One file = one batch. __getitem__ loads one file (~160 MB) and returns all
10 data points. RAM peak = num_workers × 160 MB, regardless of dataset size.

File layout
-----------
    data/
        samples_001.npy
        samples_002.npy
        ...
        train_files.txt   # one filename per line
        val_files.txt
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


_LABELS = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # index 0 = positive


class BindingDataset(Dataset):
    """
    One file = one batch. __getitem__ loads a single file (~160 MB) and
    returns all data points in it as flat tensors ready for the model.
    RAM usage = num_workers x file size at any point during training.
    """

    def __init__(self, file_paths: list[str]):
        self._files = file_paths

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> dict:
        data     = np.load(self._files[idx], allow_pickle=True).item()
        dna_emb  = torch.from_numpy(data["dna"].copy())   # (10, 8, 60,  512)
        prot_emb = torch.from_numpy(data["prot"].copy())  # (10, 8, 514, 960)
        B        = dna_emb.size(0)   # data points in this file
        group    = dna_emb.size(1)   # 8

        labels = _LABELS.unsqueeze(0).expand(B, -1)  # (10, 8)

        return {
            "dna_emb":  dna_emb.view(B * group, *dna_emb.shape[2:]),    # (80, 60,  512)
            "prot_emb": prot_emb.view(B * group, *prot_emb.shape[2:]),  # (80, 514, 960)
            "labels":   labels.reshape(B * group),                       # (80,)
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """
    Each item in batch is already a full batch from one file.
    With DataLoader batch_size=1, this simply unwraps the single item.
    """
    assert len(batch) == 1, "DataLoader batch_size must be 1 (one file = one batch)"
    return batch[0]


def _resolve_file_list(data_dir: str, list_file: str) -> list[str]:
    with open(list_file) as f:
        names = [line.strip() for line in f if line.strip()]
    return [os.path.join(data_dir, name) for name in names]


def build_dataloaders(
    data_dir: str,
    train_list: str,
    val_list: str,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Args:
        data_dir:    directory containing .npy files (10 data points each)
        train_list:  text file listing training .npy filenames
        val_list:    text file listing validation .npy filenames
        num_workers: DataLoader worker processes; RAM peak = num_workers x 160 MB
        pin_memory:  True for CUDA, False for MPS/CPU
    """
    train_files = _resolve_file_list(data_dir, train_list)
    val_files   = _resolve_file_list(data_dir, val_list)

    train_ds = BindingDataset(train_files)
    val_ds   = BindingDataset(val_files)

    loader_kwargs = dict(
        batch_size=1,          # one file = one batch, batch composition done in __getitem__
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    return train_loader, val_loader
