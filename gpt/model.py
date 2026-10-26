import math
import torch
import torch.nn as nn
from gpt.attention import MultiHeadCausalAttention
from gpt.config import GPTConfig

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (
            1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3)))
        )


class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(emb_dim))
        self.bias=nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean=x.mean(dim=-1, keepdim=True)
        var=x.var(dim=-1, unbiased=False, keepdim=True)
        out = (x-mean)/torch.sqrt(var + self.eps)
        return self.weight*out+self.bias



class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.c_fc=nn.Linear(cfg.emb_dim, 4*cfg.emb_dim)
        self.gelu=GELU()

        self.c_proj=nn.Linear(4*cfg.emb_dim, cfg.emb_dim)
        self.drop=nn.Dropout(cfg.dropout)


    def forward(self, x):
        return self.drop(self.c_proj(self.gelu(self.c_fc(x))))


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.ln_1 = LayerNorm(cfg.emb_dim)
        self.attn = MultiHeadCausalAttention(cfg)
        self.ln_2 = LayerNorm(cfg.emb_dim)
        self.mlp  = MLP(cfg)

    def forward(self, x):
        x=x+self.attn(self.ln_1(x))
        x=x+self.mlp(self.ln_2(x))
        return x

    

class GPT2Model(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()

        self.cfg=cfg

        self.transformer=nn.ModuleDict({
            "wte":  nn.Embedding(cfg.vocab_size, cfg.emb_dim),
            "wpe":  nn.Embedding(cfg.context_length, cfg.emb_dim),
            "drop": nn.Dropout(cfg.dropout),
            "h":    nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)]),
            "ln_f": LayerNorm(cfg.emb_dim)
        })
        
        self.lm_head=nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.transformer["wte"].weight

    def forward(self, input_ids):
        b, seq_len=input_ids.shape

        tok_emb=self.transformer["wte"](input_ids)
        pos_emb = self.transformer["wpe"](
            torch.arange(seq_len, device=input_ids.device)
        )
        x = self.transformer["drop"](tok_emb + pos_emb)

        for block in self.transformer["h"]:
            x = block(x)

        x=self.transformer["ln_f"](x)

        return self.lm_head(x) # [batch, seq_len, vocab_size]

        


    


