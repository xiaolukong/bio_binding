"""
Dataset utilities for DNA-Protein binding affinity training.

Expected data directory layout:

    data/
        pairs.csv          # columns: dna_id, protein_id, label (1=positive, 0=negative)
        dna_embeddings/    # one .pt file per DNA sequence: {dna_id}.pt, shape (L, 768)
        prot_embeddings/   # one .pt file per protein:      {protein_id}.pt, shape (L, 320)

pairs.csv groups positives and negatives by data_point_id:
    data_point_id, dna_id, protein_id, label
    0, dna_001, prot_A, 1
    0, dna_002, prot_A, 0
    0, dna_003, prot_A, 0
    ...
    1, dna_011, prot_B, 1
    ...

Each data_point_id must have exactly 1 positive and N negatives (default N=7).
"""

import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class BindingDataset(Dataset):
    """
    Returns one data point per __getitem__: a list of (dna_emb, prot_emb, label) tuples,
    where the first entry is always the positive pair.
    """

    def __init__(
        self,
        pairs_csv: str,
        dna_emb_dir: str,
        prot_emb_dir: str,
        max_dna_len: int = 256,
        max_prot_len: int = 256,
        n_negatives: int = 7,
    ):
        self.dna_emb_dir  = dna_emb_dir
        self.prot_emb_dir = prot_emb_dir
        self.max_dna_len  = max_dna_len
        self.max_prot_len = max_prot_len
        self.n_negatives  = n_negatives

        df = pd.read_csv(pairs_csv)
        self._validate_columns(df)

        # Group by data_point_id; each group = 1 positive + N negatives
        self.groups = []
        for _, group in df.groupby("data_point_id", sort=True):
            pos = group[group["label"] == 1]
            neg = group[group["label"] == 0]
            assert len(pos) == 1, f"data_point_id {group['data_point_id'].iloc[0]}: expected 1 positive, got {len(pos)}"
            assert len(neg) >= n_negatives, (
                f"data_point_id {group['data_point_id'].iloc[0]}: "
                f"need {n_negatives} negatives, got {len(neg)}"
            )
            neg = neg.head(n_negatives)
            self.groups.append(pd.concat([pos, neg]).reset_index(drop=True))

    @staticmethod
    def _validate_columns(df: pd.DataFrame):
        required = {"data_point_id", "dna_id", "protein_id", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"pairs.csv missing columns: {missing}")

    def _load_emb(self, directory: str, seq_id: str, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = os.path.join(directory, f"{seq_id}.pt")
        emb = torch.load(path, map_location="cpu")  # (L, D)
        if emb.dim() == 3:
            emb = emb.squeeze(0)
        # Truncate if needed
        if emb.size(0) > max_len:
            emb = emb[:max_len]
        pad_mask = torch.zeros(emb.size(0), dtype=torch.bool)  # all valid, no padding here
        return emb, pad_mask

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> list[dict]:
        group = self.groups[idx]
        samples = []
        for _, row in group.iterrows():
            dna_emb,  dna_mask  = self._load_emb(self.dna_emb_dir,  str(row["dna_id"]),     self.max_dna_len)
            prot_emb, prot_mask = self._load_emb(self.prot_emb_dir, str(row["protein_id"]), self.max_prot_len)
            samples.append({
                "dna_emb":       dna_emb,
                "prot_emb":      prot_emb,
                "dna_pad_mask":  dna_mask,
                "prot_pad_mask": prot_mask,
                "label":         int(row["label"]),
            })
        return samples  # list of n_positives + n_negatives dicts


def collate_fn(batch: list[list[dict]]) -> dict[str, torch.Tensor]:
    """
    batch: list of data points, each a list of (1 pos + N neg) sample dicts.
    Returns flat tensors with shape (B * (1+N), ...) where B = len(batch).
    DNA and Protein sequences within the batch are padded to the longest in the batch.
    """
    flat_samples = [s for dp in batch for s in dp]

    dna_embs   = [s["dna_emb"]  for s in flat_samples]
    prot_embs  = [s["prot_emb"] for s in flat_samples]
    labels     = torch.tensor([s["label"] for s in flat_samples], dtype=torch.float)

    # Pad sequences to max length in batch; pad_sequence pads at the end
    dna_padded  = pad_sequence(dna_embs,  batch_first=True, padding_value=0.0)
    prot_padded = pad_sequence(prot_embs, batch_first=True, padding_value=0.0)

    # Build padding masks: True = padded position (should be ignored)
    def make_pad_mask(seqs: list[torch.Tensor], padded: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(padded.size(0), padded.size(1), dtype=torch.bool)
        for i, s in enumerate(seqs):
            mask[i, : s.size(0)] = False  # valid positions
        return mask

    dna_pad_mask  = make_pad_mask(dna_embs,  dna_padded)
    prot_pad_mask = make_pad_mask(prot_embs, prot_padded)

    return {
        "dna_emb":       dna_padded,    # (B*(1+N), L_dna,  768)
        "prot_emb":      prot_padded,   # (B*(1+N), L_prot, 320)
        "dna_pad_mask":  dna_pad_mask,  # (B*(1+N), L_dna)
        "prot_pad_mask": prot_pad_mask, # (B*(1+N), L_prot)
        "labels":        labels,        # (B*(1+N),)
    }


def build_dataloaders(
    data_dir: str,
    train_csv: str,
    val_csv: str,
    batch_size: int = 4,      # number of data points per batch
    n_negatives: int = 7,
    max_dna_len: int = 256,
    max_prot_len: int = 256,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    dna_emb_dir  = os.path.join(data_dir, "dna_embeddings")
    prot_emb_dir = os.path.join(data_dir, "prot_embeddings")

    common = dict(
        dna_emb_dir=dna_emb_dir,
        prot_emb_dir=prot_emb_dir,
        max_dna_len=max_dna_len,
        max_prot_len=max_prot_len,
        n_negatives=n_negatives,
    )
    train_ds = BindingDataset(train_csv, **common)
    val_ds   = BindingDataset(val_csv,   **common)

    loader_common = dict(
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_common)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_common)

    return train_loader, val_loader
