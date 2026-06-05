"""
Offline embedding extraction for DNABERT-2 and ESM-2.

Outputs one .pt file per sequence to avoid re-running encoders during training.

Usage:
    # Extract DNA embeddings
    python src/extract_embeddings.py dna \
        --fasta data/dna_sequences.fasta \
        --output_dir data/dna_embeddings \
        --batch_size 16

    # Extract Protein embeddings
    python src/extract_embeddings.py protein \
        --fasta data/protein_sequences.fasta \
        --output_dir data/prot_embeddings \
        --batch_size 16

Input FASTA format:
    >seq_id
    ATCGATCG...

Output:
    {output_dir}/{seq_id}.pt  -- float32 tensor of shape (L, D), L = token count
"""

import argparse
import os
import torch
from pathlib import Path


# ---------------------------------------------------------------------------
# FASTA reader
# ---------------------------------------------------------------------------

def read_fasta(path: str) -> list[tuple[str, str]]:
    """Returns list of (seq_id, sequence) pairs."""
    records = []
    current_id, current_seq = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(current_seq)))
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_id is not None:
        records.append((current_id, "".join(current_seq)))
    return records


# ---------------------------------------------------------------------------
# DNA: DNABERT-2
# ---------------------------------------------------------------------------

def extract_dna_embeddings(
    fasta: str,
    output_dir: str,
    batch_size: int = 16,
    max_len: int = 256,
    device: str = "cuda",
):
    from transformers import AutoTokenizer, AutoModel

    print("Loading DNABERT-2...")
    tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M", trust_remote_code=True
    )
    encoder = AutoModel.from_pretrained(
        "zhihan1996/DNABERT-2-117M", trust_remote_code=True
    ).to(device).eval()

    records = read_fasta(fasta)
    os.makedirs(output_dir, exist_ok=True)
    skipped = 0

    print(f"Extracting {len(records)} DNA sequences...")
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        ids, seqs = zip(*batch)

        inputs = tokenizer(
            list(seqs),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = encoder(**inputs)
        # DNABERT-2 returns last_hidden_state: (B, L, 768)
        hidden = outputs.last_hidden_state.cpu().float()

        for j, seq_id in enumerate(ids):
            out_path = os.path.join(output_dir, f"{seq_id}.pt")
            if os.path.exists(out_path):
                skipped += 1
                continue
            # Strip padding: use attention_mask to find real length
            real_len = inputs["attention_mask"][j].sum().item()
            # Skip [CLS] and [SEP] tokens (indices 0 and real_len-1)
            emb = hidden[j, 1 : real_len - 1, :]  # (L_real, 768)
            torch.save(emb, out_path)

        if (i // batch_size) % 10 == 0:
            print(f"  {i + len(batch)}/{len(records)} sequences processed")

    print(f"Done. Skipped {skipped} already-extracted sequences.")


# ---------------------------------------------------------------------------
# Protein: ESM-2 t6_8M
# ---------------------------------------------------------------------------

def extract_protein_embeddings(
    fasta: str,
    output_dir: str,
    batch_size: int = 16,
    max_len: int = 256,
    device: str = "cuda",
):
    import esm

    print("Loading ESM-2 t6_8M...")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    repr_layer = 6  # last layer of t6

    records = read_fasta(fasta)
    os.makedirs(output_dir, exist_ok=True)
    skipped = 0

    print(f"Extracting {len(records)} protein sequences...")
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]

        # Truncate sequences before converting (ESM is slow with long seqs)
        batch_trunc = [(sid, seq[:max_len]) for sid, seq in batch]
        _, _, batch_tokens = batch_converter(batch_trunc)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
        token_repr = results["representations"][repr_layer].cpu().float()
        # token_repr: (B, L+2, 320) — includes <cls> and <eos>

        for j, (seq_id, seq) in enumerate(batch_trunc):
            out_path = os.path.join(output_dir, f"{seq_id}.pt")
            if os.path.exists(out_path):
                skipped += 1
                continue
            L = len(seq)
            # Strip <cls> (idx 0) and <eos> (idx L+1)
            emb = token_repr[j, 1 : L + 1, :]  # (L, 320)
            torch.save(emb, out_path)

        if (i // batch_size) % 10 == 0:
            print(f"  {i + len(batch)}/{len(records)} sequences processed")

    print(f"Done. Skipped {skipped} already-extracted sequences.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Extract DNA or Protein embeddings offline")
    sub = p.add_subparsers(dest="modality", required=True)

    dna_p = sub.add_parser("dna", help="Extract DNABERT-2 embeddings")
    dna_p.add_argument("--fasta",      required=True)
    dna_p.add_argument("--output_dir", required=True)
    dna_p.add_argument("--batch_size", type=int, default=16)
    dna_p.add_argument("--max_len",    type=int, default=60,
                       help="Max tokens (after BPE). DNA sequences are 60bp.")
    dna_p.add_argument("--device",     default="cuda")

    prot_p = sub.add_parser("protein", help="Extract ESM-2 embeddings")
    prot_p.add_argument("--fasta",      required=True)
    prot_p.add_argument("--output_dir", required=True)
    prot_p.add_argument("--batch_size", type=int, default=16)
    prot_p.add_argument("--max_len",    type=int, default=512,
                        help="Max amino acids. Protein sequences are 512aa.")
    prot_p.add_argument("--device",     default="cuda")

    return p.parse_args()


def main():
    args = parse_args()
    if args.modality == "dna":
        extract_dna_embeddings(
            fasta=args.fasta,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_len=args.max_len,
            device=args.device,
        )
    else:
        extract_protein_embeddings(
            fasta=args.fasta,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_len=args.max_len,
            device=args.device,
        )


if __name__ == "__main__":
    main()
