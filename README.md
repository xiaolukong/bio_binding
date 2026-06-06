# DNA-Protein Binding Affinity Transformer

Transformer-based model for predicting DNA-protein binding affinity. Takes DNABERT-2 and ESM-2 embeddings as input and outputs a binding affinity scalar, trained with a ranking InfoNCE loss.

---

## 1. Environment Setup (GPU)

Requirements: Python 3.10+, CUDA 12.4, GTX 5060 or equivalent.

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# Install dependencies with CUDA 12.4 support
pip install -r requirements-gpu.txt
```

> **Mac (development only):** Use `pip install -r requirements.txt` instead.
> The training script auto-detects CUDA / MPS / CPU — no code changes needed.

---

## 2. Data Preparation

### 2.1 Sample file format

Each sample file is a `.npy` file containing a Python dict with two keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `dna` | `(N, 8, 60, 512)` | DNA embeddings from DNABERT-2 |
| `prot` | `(N, 8, 514, 960)` | Protein embeddings from ESM-2 |

- `N` — number of data points in the file
- `8` — samples per data point: **index 0 is the positive**, indices 1–7 are negatives
- Embeddings must be pre-extracted offline (see `src/extract_embeddings.py`)

Save files to the `data/` directory:

```
data/
    samples_001.npy
    samples_002.npy
    samples_003.npy
    ...
```

To convert raw DNA/protein `.npy` arrays into the combined format:

```python
import numpy as np

dna  = np.load("100.dna.embeddings.npy")   # (N, 8, 60, 512)
prot = np.load("100.protein.embeddings.npy")  # (N, 8, 514, 960)
np.save("data/samples_001.npy", {"dna": dna, "prot": prot})
```

### 2.2 Train / val split

Split is done at the **file level** to prevent data leakage between batches from the same experimental source.

Create two plain text files listing which sample files go to train and which to val (one filename per line):

```bash
# data/train_files.txt
samples_001.npy
samples_002.npy
samples_003.npy

# data/val_files.txt
samples_004.npy
samples_005.npy
```

A rough 80/20 split by file count is a good starting point.

---

## 3. Training

### 3.1 Run training

```bash
python src/train.py \
    --data_dir   data/ \
    --train_list data/train_files.txt \
    --val_list   data/val_files.txt \
    --output_dir checkpoints/
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 100 | Total training epochs |
| `--batch_size` | 10 | Data points per batch (80 sequences) |
| `--lr` | 1e-4 | Learning rate |
| `--patience` | 10 | Early stopping patience (epochs) |
| `--tau` | 0.07 | InfoNCE temperature |
| `--learnable_tau` | off | Make temperature a learnable parameter |
| `--num_workers` | 4 | DataLoader worker processes |
| `--resume` | — | Path to checkpoint to resume from |

### 3.2 Resume from checkpoint

```bash
python src/train.py \
    --data_dir   data/ \
    --train_list data/train_files.txt \
    --val_list   data/val_files.txt \
    --resume     checkpoints/best.pt
```

### 3.3 Monitor training

Each epoch prints a one-line summary to stdout:

```
Epoch 001 | train_loss=2.55 | val_loss=1.69 | auroc=0.97 | top1=0.85 | tau=0.0700 | lr=2.00e-05
```

| Field | Description |
|-------|-------------|
| `train_loss` | InfoNCE loss on training set |
| `val_loss` | InfoNCE loss on validation set |
| `auroc` | Area under ROC curve (main metric, higher is better) |
| `top1` | Fraction of data points where positive scores highest |
| `tau` | Current temperature value |
| `lr` | Current learning rate |

To log to a file and watch live:

```bash
python src/train.py \
    --data_dir   data/ \
    --train_list data/train_files.txt \
    --val_list   data/val_files.txt \
    2>&1 | tee checkpoints/train.log

# In a second terminal:
tail -f checkpoints/train.log
```

Checkpoints are saved to `--output_dir` as:
- `best.pt` — best validation AUROC so far (overwritten each improvement)
- `epoch_000.pt`, `epoch_005.pt`, ... — snapshot every 5 epochs
