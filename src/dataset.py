"""
Dataset utilities for DNA-Protein binding affinity training.

Data format
-----------
Each .npy file stores a dict with two keys:
    'dna':  float32 array of shape (N, 8, 60,  512)
    'prot': float32 array of shape (N, 8, 514, 960)

    N   = number of data points in this file
    8   = samples per data point (index 0 = positive, 1-7 = negatives)
    60  = DNA token length
    512 = DNA embedding dim (DNABERT-2 output)
    514 = Protein token length
    960 = Protein embedding dim (ESM-2 output)

File layout
-----------
    data/
        samples_001.npy
        samples_002.npy
        ...
        train_files.txt   # one filename per line, e.g. "samples_001.npy"
        val_files.txt

Train/val split is done at the file level to prevent data leakage between
data points that may share the same protein or experimental batch.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BindingDataset(Dataset):
    """
    Loads one or more .npy sample files and exposes individual data points.

    Each __getitem__ returns a dict with tensors of shape:
        dna_emb:  (8, 60,  512)  -- all 8 samples for this data point
        prot_emb: (8, 514, 960)
        labels:   (8,)           -- 1 for positive (index 0), 0 for negatives
    """

    def __init__(self, file_paths: list[str]):
        """
        Args:
            file_paths: list of absolute paths to .npy sample files.
        """
        dna_chunks  = []
        prot_chunks = []

        for path in file_paths:
            data = np.load(path, allow_pickle=True).item()
            dna_chunks.append(data["dna"])    # (N, 8, 60,  512)
            prot_chunks.append(data["prot"])  # (N, 8, 514, 960)

        dna_all  = np.concatenate(dna_chunks,  axis=0)  # (total_N, 8, 60,  512)
        prot_all = np.concatenate(prot_chunks, axis=0)  # (total_N, 8, 514, 960)

        # Store as float32 tensors; labels are fixed: index 0 = positive
        self.dna  = torch.from_numpy(dna_all)   # (total_N, 8, 60,  512)
        self.prot = torch.from_numpy(prot_all)  # (total_N, 8, 514, 960)

        n_samples  = self.dna.size(1)
        self.labels = torch.zeros(n_samples, dtype=torch.float32)
        self.labels[0] = 1.0  # index 0 is always the positive

    def __len__(self) -> int:
        return self.dna.size(0)

    def __getitem__(self, idx: int) -> dict:
        return {
            "dna_emb":  self.dna[idx],    # (8, 60,  512)
            "prot_emb": self.prot[idx],   # (8, 514, 960)
            "labels":   self.labels,      # (8,)  shared across all data points
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """
    Collates a list of data point dicts into flat batch tensors.

    Input:  list of B dicts, each with tensors (8, L, D)
    Output: flat tensors of shape (B*8, L, D) ready for the model
    """
    B          = len(batch)
    group_size = batch[0]["dna_emb"].size(0)   # 8

    dna_emb  = torch.stack([s["dna_emb"]  for s in batch], dim=0)  # (B, 8, 60,  512)
    prot_emb = torch.stack([s["prot_emb"] for s in batch], dim=0)  # (B, 8, 514, 960)
    labels   = torch.stack([s["labels"]   for s in batch], dim=0)  # (B, 8)

    return {
        "dna_emb":  dna_emb.view(B * group_size, *dna_emb.shape[2:]),    # (B*8, 60,  512)
        "prot_emb": prot_emb.view(B * group_size, *prot_emb.shape[2:]),  # (B*8, 514, 960)
        "labels":   labels.view(B * group_size),                          # (B*8,)
    }


def _resolve_file_list(data_dir: str, list_file: str) -> list[str]:
    """Reads a text file of filenames and returns absolute paths."""
    with open(list_file) as f:
        names = [line.strip() for line in f if line.strip()]
    return [os.path.join(data_dir, name) for name in names]


def build_dataloaders(
    data_dir: str,
    train_list: str,
    val_list: str,
    batch_size: int = 10,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Args:
        data_dir:   directory containing .npy files
        train_list: path to text file listing training .npy filenames
        val_list:   path to text file listing validation .npy filenames
        batch_size: number of data points per batch
        num_workers: DataLoader worker count
    """
    train_files = _resolve_file_list(data_dir, train_list)
    val_files   = _resolve_file_list(data_dir, val_list)

    train_ds = BindingDataset(train_files)
    val_ds   = BindingDataset(val_files)

    loader_kwargs = dict(
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    return train_loader, val_loader
