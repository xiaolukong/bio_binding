"""
Training script for BindingTransformer.

Usage:
    python src/train.py \
        --data_dir data/ \
        --train_list data/train_files.txt \
        --val_list   data/val_files.txt

train_files.txt / val_files.txt: one .npy filename per line, e.g.:
    samples_001.npy
    samples_002.npy

Key design decisions:
- Loss: ranking InfoNCE over (1 pos + 7 neg) per data point
- Train/val split is file-level to prevent data leakage
- Temperature tau is fixed at 0.07 (enable --learnable_tau to make it trainable)
"""

import argparse
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
import numpy as np

from model import build_model
from dataset import build_dataloaders


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class RankingInfoNCE(nn.Module):
    """
    InfoNCE-style ranking loss.

    Input scores: flat tensor (B * group_size,) where the first entry of each
    group of `group_size` is the positive.

    Loss = mean over groups of: -log softmax(scores / tau)[0]
    """

    def __init__(self, group_size: int, tau: float = 0.07, learnable: bool = False):
        super().__init__()
        self.group_size = group_size
        if learnable:
            self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))
        else:
            self.register_buffer("log_tau", torch.tensor(math.log(tau)))

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        n_groups = scores.size(0) // self.group_size
        logits   = scores.view(n_groups, self.group_size) / self.tau
        labels   = torch.zeros(n_groups, dtype=torch.long, device=scores.device)
        return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(scores: np.ndarray, labels: np.ndarray, group_size: int) -> dict:
    n_groups = len(scores) // group_size
    scores_g = scores.reshape(n_groups, group_size)

    top1 = float((scores_g.argmax(axis=1) == 0).mean())

    try:
        auroc = float(roc_auc_score(labels, scores))
    except ValueError:
        auroc = float("nan")

    return {"auroc": auroc, "top1_acc": top1}


# ---------------------------------------------------------------------------
# Training / Validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, group_size):
    model.train()
    amp_dtype = get_autocast_dtype(device)
    total_loss = 0.0
    for batch in loader:
        dna_emb  = batch["dna_emb"].to(device)
        prot_emb = batch["prot_emb"].to(device)

        optimizer.zero_grad()
        with autocast(device_type=device.type, dtype=amp_dtype):
            scores = model(dna_emb, prot_emb)
            loss   = loss_fn(scores)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, group_size):
    model.eval()
    amp_dtype  = get_autocast_dtype(device)
    total_loss = 0.0
    all_scores, all_labels = [], []

    for batch in loader:
        dna_emb  = batch["dna_emb"].to(device)
        prot_emb = batch["prot_emb"].to(device)
        labels   = batch["labels"]

        with autocast(device_type=device.type, dtype=amp_dtype):
            scores = model(dna_emb, prot_emb)
            loss   = loss_fn(scores)

        total_loss += loss.item()
        all_scores.append(scores.cpu().float().numpy())
        all_labels.append(labels.numpy())

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_scores, all_labels, group_size)
    metrics["loss"] = total_loss / len(loader)
    return metrics


def get_autocast_dtype(device: torch.device):
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float16




def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path: str, epoch: int, model, optimizer, scheduler, metrics: dict):
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics":   metrics,
    }, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt.get("metrics", {})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train BindingTransformer")
    p.add_argument("--data_dir",    required=True,  help="directory containing .npy files")
    p.add_argument("--train_list",  required=True,  help="text file listing training .npy filenames")
    p.add_argument("--val_list",    required=True,  help="text file listing validation .npy filenames")
    p.add_argument("--output_dir",  default="checkpoints")
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=0.01)
    p.add_argument("--tau",         type=float, default=0.07)
    p.add_argument("--learnable_tau", action="store_true")
    p.add_argument("--warmup_frac", type=float, default=0.05)
    p.add_argument("--patience",    type=int,   default=10,  help="early stopping patience")
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--resume",      default=None, help="checkpoint path to resume from")
    # Model config overrides
    p.add_argument("--d_model",  type=int,   default=768)
    p.add_argument("--n_heads",  type=int,   default=12)
    p.add_argument("--n_layers", type=int,   default=6)
    p.add_argument("--d_ffn",    type=int,   default=3072)
    p.add_argument("--dropout",  type=float, default=0.1)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # --- Data ---
    group_size = 8  # 1 positive + 7 negatives, fixed by data format
    train_loader, val_loader = build_dataloaders(
        data_dir=args.data_dir,
        train_list=args.train_list,
        val_list=args.val_list,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # --- Model ---
    model_config = {
        "d_model":     args.d_model,
        "d_dna":       512,
        "d_prot":      960,
        "n_heads":     args.n_heads,
        "n_layers":    args.n_layers,
        "d_ffn":       args.d_ffn,
        "dropout":     args.dropout,
        "max_seq_len": 577,   # CLS(1) + DNA(60) + SEP(1) + Prot(514) + END(1)
    }
    model = build_model(model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params / 1e6:.1f}M")

    # --- Loss ---
    loss_fn = RankingInfoNCE(
        group_size=group_size,
        tau=args.tau,
        learnable=args.learnable_tau,
    ).to(device)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # --- LR schedule ---
    total_steps  = args.epochs * len(train_loader)
    warmup_steps = max(500, int(args.warmup_frac * total_steps))
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = GradScaler(enabled=(device.type == "cuda"))
    # --- Resume ---
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch - 1}")

    # --- Training loop ---
    best_auroc    = -1.0
    patience_left = args.patience

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device, group_size
        )
        scheduler.step()

        val_metrics = evaluate(model, val_loader, loss_fn, device, group_size)
        val_auroc   = val_metrics["auroc"]
        val_top1    = val_metrics["top1_acc"]
        val_loss    = val_metrics["loss"]

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"auroc={val_auroc:.4f} | "
            f"top1={val_top1:.4f} | "
            f"tau={loss_fn.tau.item():.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if val_auroc > best_auroc:
            best_auroc    = val_auroc
            patience_left = args.patience
            save_checkpoint(
                os.path.join(args.output_dir, "best.pt"),
                epoch, model, optimizer, scheduler, val_metrics,
            )
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"Early stopping at epoch {epoch} (best AUROC={best_auroc:.4f})")
                break

        if epoch % 5 == 0:
            save_checkpoint(
                os.path.join(args.output_dir, f"epoch_{epoch:03d}.pt"),
                epoch, model, optimizer, scheduler, val_metrics,
            )

    print(f"Training complete. Best val AUROC: {best_auroc:.4f}")


if __name__ == "__main__":
    main()
