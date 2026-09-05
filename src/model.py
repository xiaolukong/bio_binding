import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlashAttention(nn.Module):
    """
    Multi-head self-attention using F.scaled_dot_product_attention (Flash Attention).
    Requires PyTorch >= 2.0. On supported hardware (Ampere+/Blackwell) this
    dispatches to the fused Flash Attention kernel, which is ~2-3x faster and
    uses O(L) memory instead of O(L^2) for the attention matrix.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.dropout  = dropout

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L, d_model = x.shape

        # Project to Q, K, V in one fused matmul
        qkv = self.qkv_proj(x)                          # (B, L, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)                  # each (B, L, d_model)

        # Reshape to (B, n_heads, L, d_head)
        def split_heads(t):
            return t.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Convert padding mask to attention bias expected by SDPA
        # key_padding_mask: (B, L), True = ignore -> attn_mask: (B, 1, 1, L) with -inf
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = torch.zeros(B, 1, 1, L, dtype=q.dtype, device=x.device)
            attn_mask = attn_mask.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        # Flash Attention — O(L) memory, fused kernel on Ampere/Blackwell
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)

        # Merge heads: (B, n_heads, L, d_head) -> (B, L, d_model)
        out = out.transpose(1, 2).contiguous().view(B, L, d_model)
        return self.out_proj(out)


class AttentionPooling(nn.Module):
    """
    Attention pooling over the final layer.

    A single learnable query attends over all (non-padded) tokens and produces
    a weighted sum. Unlike a CLS token, the pooling happens once on the final
    representation with a short gradient path to every token, which learns more
    easily on smaller datasets while still focusing on sparse binding sites.
    """

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.query  = nn.Parameter(torch.empty(1, 1, d_model))
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.drop   = nn.Dropout(dropout)
        self.scale  = d_model ** -0.5

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, L, d_model)   pad_mask: (B, L)  True = padding (ignore)
        B = x.size(0)
        q = self.query.expand(B, -1, -1)                  # (B, 1, d_model)
        k = self.k_proj(x)                                # (B, L, d_model)
        v = self.v_proj(x)                                # (B, L, d_model)

        scores = (q @ k.transpose(-2, -1)) * self.scale   # (B, 1, L)
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask.unsqueeze(1), float("-inf"))

        weights = self.drop(scores.softmax(dim=-1))       # (B, 1, L)
        pooled  = weights @ v                             # (B, 1, d_model)
        return pooled.squeeze(1)                          # (B, d_model)


class TransformerEncoderLayer(nn.Module):
    """Pre-LN Transformer encoder layer with Flash Attention."""

    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float):
        super().__init__()
        self.norm1    = nn.LayerNorm(d_model)
        self.attn     = FlashAttention(d_model, n_heads, dropout)
        self.norm2    = nn.LayerNorm(d_model)
        self.ffn      = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )
        self.drop     = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-LN: norm before sublayer, residual after
        x = x + self.drop(self.attn(self.norm1(x), key_padding_mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


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

        d_model  = config["d_model"]       # 768
        d_dna    = config["d_dna"]         # 512  (DNABERT-2)
        d_prot   = config["d_prot"]        # 960  (ESM-2 t6_8M)
        n_heads  = config["n_heads"]       # 12
        n_layers = config["n_layers"]      # 6
        d_ffn    = config["d_ffn"]         # 3072
        dropout  = config["dropout"]       # 0.1
        max_len  = config["max_seq_len"]   # 576

        # --- Modality projections ---
        self.dna_proj  = nn.Linear(d_dna,  d_model)
        self.prot_proj = nn.Linear(d_prot, d_model)

        # --- Special token embeddings ---
        self.sep_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.end_token = nn.Parameter(torch.empty(1, 1, d_model))

        # --- Segment embeddings (0=DNA side, 1=Protein side) ---
        self.segment_emb = nn.Embedding(2, d_model)

        # --- Positional embedding ---
        self.pos_emb = nn.Embedding(max_len, d_model)

        # --- Transformer encoder (Pre-LN + Flash Attention) ---
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ffn, dropout)
            for _ in range(n_layers)
        ])

        # --- Attention pooling over final layer ---
        self.attn_pool = AttentionPooling(d_model, dropout)

        # --- Projection head ---
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.sep_token, std=0.02)
        nn.init.trunc_normal_(self.end_token, std=0.02)
        nn.init.trunc_normal_(self.attn_pool.query, std=0.02)
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
        B      = dna_emb.size(0)
        device = dna_emb.device

        # Project both modalities to d_model
        dna_x  = self.dna_proj(dna_emb)    # (B, L_dna,  d_model)
        prot_x = self.prot_proj(prot_emb)  # (B, L_prot, d_model)

        L_dna  = dna_x.size(1)
        L_prot = prot_x.size(1)

        # Expand special tokens to batch
        sep = self.sep_token.expand(B, -1, -1)
        end = self.end_token.expand(B, -1, -1)

        # Concatenate: DNA | [SEP] | Prot | [END]
        x       = torch.cat([dna_x, sep, prot_x, end], dim=1)
        L_total = x.size(1)

        # --- Segment embeddings (A=0: DNA+SEP, B=1: Prot+END) ---
        seg_ids = torch.cat([
            torch.zeros(L_dna + 1, dtype=torch.long, device=device),
            torch.ones( L_prot + 1, dtype=torch.long, device=device),
        ]).unsqueeze(0).expand(B, -1)
        x = x + self.segment_emb(seg_ids)

        # --- Positional embeddings ---
        pos_ids = torch.arange(L_total, device=device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(pos_ids)

        # --- Build full padding mask: (B, L_total), True = ignore ---
        no_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        if dna_pad_mask is None:
            dna_pad_mask = torch.zeros(B, L_dna, dtype=torch.bool, device=device)
        if prot_pad_mask is None:
            prot_pad_mask = torch.zeros(B, L_prot, dtype=torch.bool, device=device)
        pad_mask = torch.cat([dna_pad_mask, no_mask, prot_pad_mask, no_mask], dim=1)

        # --- Transformer layers ---
        for layer in self.layers:
            x = layer(x, key_padding_mask=pad_mask)

        # --- Attention pooling head ---
        pooled = self.attn_pool(x, pad_mask=pad_mask)
        return self.head(pooled).squeeze(-1)   # (B,)


def build_model(config: dict | None = None) -> BindingTransformer:
    default_config = {
        "d_model":     768,
        "d_dna":       512,
        "d_prot":      960,
        "n_heads":     12,
        "n_layers":    6,
        "d_ffn":       3072,
        "dropout":     0.1,
        "max_seq_len": 576,   # DNA(60) + SEP(1) + Prot(514) + END(1)
    }
    if config is not None:
        default_config.update(config)
    return BindingTransformer(default_config)
