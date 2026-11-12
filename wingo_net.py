#!/usr/bin/env python3
"""WingoNet — Tilelli-style dual-head predictor (Fixed & Checkpoint-Compatible).

Input : [B, L] digit tokens (128) + [B, 89] handcrafted features
Output: size logit (BIG/SMALL) + 3 colour logits (RED/GREEN/VIOLET multi-label)

Supports loading both original training checkpoints (reg, qn, kn) and standard
attribute names (registers, q_norm, k_norm).
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_hidden: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_hidden, bias=False)
        self.w2 = nn.Linear(d_model, d_hidden, bias=False)
        self.w3 = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


def get_alibi_slopes(n_heads: int) -> Tensor:
    def pow2(n: int) -> Tensor:
        start = 2 ** (-math.pow(2, -(math.log2(n) - 3)))
        return torch.tensor([start * start ** i for i in range(n)])

    if math.log2(n_heads).is_integer():
        return pow2(n_heads)
    closest = 2 ** math.floor(math.log2(n_heads))
    return torch.cat([pow2(closest), pow2(closest // 2)[: n_heads - closest]])


class GQAAttention(nn.Module):
    def __init__(self, d_model: int, n_q: int = 8, n_kv: int = 2) -> None:
        super().__init__()
        self.n_q, self.n_kv = n_q, n_kv
        self.groups = n_q // n_kv
        self.d_head = d_model // n_q
        self.Wq = nn.Linear(d_model, n_q * self.d_head, bias=False)
        self.Wk = nn.Linear(d_model, n_kv * self.d_head, bias=False)
        self.Wv = nn.Linear(d_model, n_kv * self.d_head, bias=False)
        self.Wo = nn.Linear(n_q * self.d_head, d_model, bias=False)
        self.qn = RMSNorm(self.d_head)
        self.kn = RMSNorm(self.d_head)
        self.register_buffer("alibi", get_alibi_slopes(n_q), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        B, L, _ = x.shape
        q = self.qn(self.Wq(x).view(B, L, self.n_q, self.d_head))
        k = self.kn(self.Wk(x).view(B, L, self.n_kv, self.d_head))
        v = self.Wv(x).view(B, L, self.n_kv, self.d_head)
        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=2)
            v = v.repeat_interleave(self.groups, dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        pos = torch.arange(L, device=x.device)
        dist = (pos[None, :] - pos[:, None]).abs().neg().float()
        scores = scores + self.alibi.view(-1, 1, 1) * dist
        scores = scores.masked_fill(
            torch.ones(L, L, dtype=torch.bool, device=x.device).triu(1),
            float("-inf"),
        )
        attn = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, -1)
        return self.Wo(out)


class Block(nn.Module):
    def __init__(self, d_model: int, n_q: int, n_kv: int, ffn: int) -> None:
        super().__init__()
        self.an = RMSNorm(d_model)
        self.attn = GQAAttention(d_model, n_q, n_kv)
        self.fn = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.an(x))
        return x + self.ffn(self.fn(x))


class WingoNet(nn.Module):
    def __init__(
        self,
        vocab: int = 12,
        d_model: int = 256,
        n_layers: int = 5,
        n_q: int = 8,
        n_kv: int = 2,
        ffn_mult: int = 4,
        n_features: int = 89,
        n_registers: int = 4,
    ) -> None:
        super().__init__()
        ffn = d_model * ffn_mult
        self.embed = nn.Embedding(vocab, d_model)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.reg = nn.Parameter(torch.randn(n_registers, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            Block(d_model, n_q, n_kv, ffn) for _ in range(n_layers)
        )
        self.final = RMSNorm(d_model)
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_features, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.size_head = nn.Linear(d_model * 2, 1)
        self.colour_head = nn.Linear(d_model * 2, 3)

    def forward(self, tok: Tensor, feat: Tensor) -> tuple[Tensor, Tensor]:
        B, L = tok.shape
        x = self.embed(tok)
        regs = self.reg.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([regs, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        # NOTE: training forward does NOT apply self.final before pooling (unused,
        # stays at init). Match it exactly or live predictions diverge.
        x = x[:, self.reg.size(0) :, :]
        pooled = x.mean(dim=1)
        f = self.feat_mlp(feat)
        h = torch.cat([pooled, f], dim=-1)
        return self.size_head(h).squeeze(-1), self.colour_head(h)


def load_checkpoint(ckpt_path: str | Path, device: str = "cpu") -> tuple[WingoNet, dict]:
    """Load best.pt or final.pt seamlessly regardless of key naming convention."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}

    model = WingoNet(
        vocab=cfg.get("vocab", 12),
        d_model=cfg.get("dm", cfg.get("d_model", 256)),
        n_layers=cfg.get("layers", cfg.get("n_layers", 5)),
        n_q=cfg.get("nq", cfg.get("n_q", 8)),
        n_kv=cfg.get("nkv", cfg.get("n_kv", 2)),
        ffn_mult=cfg.get("ffn_mult", 4),
        n_features=cfg.get("n_feat", cfg.get("n_features", 89)),
        n_registers=cfg.get("n_reg", cfg.get("n_registers", 4)),
    )

    fixed_sd = {}
    for k, v in sd.items():
        k_new = "reg" if k == "registers" else k
        k_new = k_new.replace(".attn.q_norm.", ".attn.qn.").replace(".attn.k_norm.", ".attn.kn.")
        fixed_sd[k_new] = v

    model.load_state_dict(fixed_sd, strict=True)
    model.to(device).eval()
    return model, ckpt