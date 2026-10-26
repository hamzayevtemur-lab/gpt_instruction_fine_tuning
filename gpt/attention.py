import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCausalAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_heads=cfg.n_heads
        self.head_dim=cfg.emb_dim//cfg.n_heads

        self.c_attn = nn.Linear(cfg.emb_dim, 3 * cfg.emb_dim, bias=cfg.qkv_bias)
        self.c_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim)

        self.attn_drop = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)


    def forward(self, x):
        b, seq_len, d=x.shape

        qkv=self.c_attn(x)
        q, k, v=qkv.split(d, dim=2)

        q = q.view(b, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=True
        )

        out = out.transpose(1, 2).contiguous().view(b, seq_len, d)

        return self.resid_drop(self.c_proj(out))



        