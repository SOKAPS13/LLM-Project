"""
model.py — Shared model definition
Imported by both pre-train.py and fine-tuning.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    vocab_size  : int   = 256   # overwritten from data/vocab.pkl at runtime
    d_model     : int   = 256
    n_heads     : int   = 8
    n_layers    : int   = 6
    d_ff        : int   = 1024
    max_seq_len : int   = 800   # overwritten from tokenized sequence length at runtime
    dropout     : float = 0.2
    # training (used by pre-train.py)
    epochs      : int   = 50    # 50 is enough for 10% subset; raise to 100+ for larger subsets
    batch_size  : int   = 256  # safer default for longer GMM/multiresolution sequences
    lr          : float = 3e-4
    weight_decay: float = 0.1
    grad_clip   : float = 1.0
    warmup_steps: int   = 200
    train_k     : int   = 8
    val_k       : int   = 1
    test_k      : int   = 1
    log_every   : int   = 10
    save_every  : int   = 25
    device      : str   = ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    data_dir    : str   = "./data"
    output_dir  : str   = "./checkpoints"


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads    = n_heads
        self.d_head     = d_model // n_heads
        self.dropout_p  = dropout
        self.qkv        = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj       = nn.Linear(d_model, d_model,     bias=False)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        def rsh(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = rsh(q), rsh(k), rsh(v)
        # scaled_dot_product_attention never materialises the full T×T matrix —
        # uses Flash Attention on MPS, saving ~10× memory vs the manual version
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,          # applies the causal mask internally
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TrafficGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg       = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb   = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.token_emb.weight   # weight tying
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, input_ids):
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device)
        x    = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))             # (B, T, vocab_size)

    def get_hidden_states(self, input_ids):
        """Returns (B, T, d_model) — used by fine-tuning classification head."""
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device)
        x    = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)