import math
import torch
import torch.nn as nn


class BindingTransformer(nn.Module):
    """
    DNA-Protein binding affinity predictor.

    Inputs:
        dna_emb:  (B, L_dna,  d_dna)   -- DNABERT-2 embeddings, frozen upstream
        prot_emb: (B, L_prot, d_prot)  -- ESM-2 embeddings, frozen upstream
        dna_pad_mask:  (B, L_dna)      -- True where padded
        prot_pad_mask: (B, L_prot)     -- True where padded

    Output:
        scalar: (B,)  -- affinity score (higher = stronger binding)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        d_model   = config["d_model"]       # 768
        d_dna     = config["d_dna"]         # 768  (DNABERT-2)
        d_prot    = config["d_prot"]        # 320  (ESM-2 t6_8M)
        n_heads   = config["n_heads"]       # 12
        n_layers  = config["n_layers"]      # 6
        d_ffn     = config["d_ffn"]         # 3072
        dropout   = config["dropout"]       # 0.1
        max_len   = config["max_seq_len"]   # 515

        # --- Modality projections ---
        self.dna_proj  = nn.Linear(d_dna,  d_model)
        self.prot_proj = nn.Linear(d_prot, d_model)

        # --- Special token embeddings ---
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.sep_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.end_token = nn.Parameter(torch.empty(1, 1, d_model))

        # --- Segment embeddings (A=DNA side, B=Protein side) ---
        self.segment_emb = nn.Embedding(2, d_model)  # 0=A, 1=B

        # --- Positional embedding ---
        self.pos_emb = nn.Embedding(max_len, d_model)

        # --- Transformer encoder (Pre-LN) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # --- CLS projection head ---
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.sep_token, std=0.02)
        nn.init.trunc_normal_(self.end_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        dna_emb: torch.Tensor,
        prot_emb: torch.Tensor,
        dna_pad_mask: torch.Tensor | None = None,
        prot_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = dna_emb.size(0)
        device = dna_emb.device

        # Project both modalities to d_model
        dna_x  = self.dna_proj(dna_emb)    # (B, L_dna,  d_model)
        prot_x = self.prot_proj(prot_emb)  # (B, L_prot, d_model)

        L_dna  = dna_x.size(1)
        L_prot = prot_x.size(1)

        # Expand special tokens to batch
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, d_model)
        sep = self.sep_token.expand(B, -1, -1)
        end = self.end_token.expand(B, -1, -1)

        # Concatenate: [CLS] | DNA | [SEP] | Prot | [END]
        x = torch.cat([cls, dna_x, sep, prot_x, end], dim=1)
        # x: (B, 1 + L_dna + 1 + L_prot + 1, d_model)
        L_total = x.size(1)

        # --- Segment embeddings ---
        # Segment A: CLS(1) + DNA(L_dna) + SEP(1)
        # Segment B: Prot(L_prot) + END(1)
        seg_ids = torch.cat([
            torch.zeros(1 + L_dna + 1, dtype=torch.long, device=device),
            torch.ones(L_prot + 1,     dtype=torch.long, device=device),
        ]).unsqueeze(0).expand(B, -1)  # (B, L_total)
        x = x + self.segment_emb(seg_ids)

        # --- Positional embeddings ---
        pos_ids = torch.arange(L_total, device=device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(pos_ids)

        # --- Padding mask for transformer ---
        # TransformerEncoder expects src_key_padding_mask: True = ignore
        # Layout: [CLS(1)] [DNA(L_dna)] [SEP(1)] [Prot(L_prot)] [END(1)]
        if dna_pad_mask is not None or prot_pad_mask is not None:
            # Special tokens are never masked
            no_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
            if dna_pad_mask is None:
                dna_pad_mask = torch.zeros(B, L_dna, dtype=torch.bool, device=device)
            if prot_pad_mask is None:
                prot_pad_mask = torch.zeros(B, L_prot, dtype=torch.bool, device=device)
            # [CLS] | DNA | [SEP] | Prot | [END]
            pad_mask = torch.cat(
                [no_mask, dna_pad_mask, no_mask, prot_pad_mask, no_mask], dim=1
            )
        else:
            pad_mask = None

        # --- Transformer ---
        x = self.transformer(x, src_key_padding_mask=pad_mask)  # (B, L_total, d_model)

        # --- CLS head ---
        h_cls = x[:, 0, :]          # (B, d_model)
        score = self.head(h_cls)    # (B, 1)
        return score.squeeze(-1)    # (B,)


def build_model(config: dict | None = None) -> BindingTransformer:
    default_config = {
        "d_model":     768,
        "d_dna":       768,
        "d_prot":      320,
        "n_heads":     12,
        "n_layers":    6,
        "d_ffn":       3072,
        "dropout":     0.1,
        "max_seq_len": 575,   # CLS(1) + DNA(60) + SEP(1) + Prot(512) + END(1)
    }
    if config is not None:
        default_config.update(config)
    return BindingTransformer(default_config)
