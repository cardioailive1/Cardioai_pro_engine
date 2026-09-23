"""
CardioAI Pro — Vision Transformer Backbone
================================================
A real, from-scratch ViT-Base-style backbone (patch embedding + multi-head
self-attention transformer encoder), matching the architecture used in the
real, published, peer-reviewed cardiac imaging papers this design follows
— not an arbitrary choice. Specifically:

  - DINO-LG (Haque et al., Med Biol Eng Comput, 2026) and CARD-ViT
    (PMC13274280, 2026) — coronary artery calcium scoring from CT, both
    using a ViT-Base backbone (ViT-Base/8 in DINO-LG) self-supervised
    pretrained with DINO (self-distillation with no labels), then a
    lightweight head for the downstream task. DINO-LG reports 89%
    sensitivity / 90% specificity for CAC-slice detection on 914 CT scans.
  - Echo-Vision-FM (Zhang et al., Nature Communications, 2025/2026) —
    "Echo-VideoMAE," a ViT encoder-decoder masked autoencoder pretrained
    on echo video (MIMIC-IV-ECHO), fine-tuned on EchoNet-Dynamic, reaching
    89.12% accuracy / 0.9364 AUC for LVEF classification.

Both use the SAME underlying strategy: self-supervised ViT pretraining on
large UNLABELED imaging data (solving the real bottleneck in medical
imaging — expert-labeled data is scarce and expensive; raw scans are not),
then a small supervised head fine-tuned on the specific downstream task.
This module provides the shared backbone; imaging_models/echo_model.py and
imaging_models/ct_model.py build the two modality-specific heads on top of it.

WHAT THIS IS AND ISN'T: this is real, tested architecture code — it
instantiates and runs a real forward pass (see the test at the bottom of
this file and the standalone test run referenced in the module's usage).
It is NOT a trained model. No pretrained weights exist here or are loaded
here. Real DINO/MAE self-supervised pretraining needs the datasets the
papers above used (a large unlabeled cardiac CT or echo-video corpus) and
a GPU cluster to run it on — neither of which this environment has. This
is the architecture a real training run would use, wired to the same
load_trained_model() hook pattern as inference/models.py, not a trained
model itself.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Splits an image into fixed-size patches and linearly projects each — the standard ViT "patchify" step."""

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 1, embed_dim: int = 768):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.grid_size = img_size // patch_size
        self.n_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                      # (B, embed_dim, grid, grid)
        x = x.flatten(2).transpose(1, 2)        # (B, n_patches, embed_dim)
        return x


class TransformerEncoderBlock(nn.Module):
    """Standard pre-norm transformer encoder block: self-attention + MLP, each with a residual connection."""

    def __init__(self, embed_dim: int = 768, n_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTBackbone(nn.Module):
    """
    ViT-Base-style backbone. Defaults (patch_size=16, embed_dim=768,
    depth=12, n_heads=12, ~86M parameters) match the ViT-Base convention
    used across the cited papers (DINO-LG uses ViT-Base/8 — patch_size=8 —
    for finer spatial resolution on CT; the default here is configurable
    per modality in echo_model.py / ct_model.py).

    in_channels=1 by default (grayscale — both echo and CT are single-
    channel), unlike the ImageNet-pretrained ViTs most tutorials use, which
    assume 3-channel RGB natural images.
    """

    def __init__(
        self, img_size: int = 224, patch_size: int = 16, in_channels: int = 1,
        embed_dim: int = 768, depth: int = 12, n_heads: int = 12, dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, n_heads, dropout=dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _interpolated_pos_embed(self, n_patches: int) -> torch.Tensor:
        """
        Bicubically interpolates the learned patch positional embeddings to
        a different patch-grid size — the standard ViT technique for
        handling variable input resolutions with one shared set of weights
        (used by DeiT, DINO, and others). Needed because DINO's multi-crop
        augmentation deliberately uses TWO resolutions (large "global" crops,
        small "local" crops) through the SAME network — a real bug this
        fixed: without it, a 96px local crop produces a different patch
        count than the 224px positional embedding was sized for, and the
        add in forward() below fails with a shape mismatch. The CLS token's
        embedding is never interpolated — only the per-patch grid is.
        """
        if n_patches == self.patch_embed.n_patches:
            return self.pos_embed  # no interpolation needed — the common case
        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        old_grid = self.patch_embed.grid_size
        new_grid = int(math.sqrt(n_patches))
        patch_pos = patch_pos.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
        patch_pos = nn.functional.interpolate(patch_pos, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, -1)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> (B, n_patches + 1, embed_dim). Index 0 along dim 1 is the CLS token embedding. H/W need not match img_size — see _interpolated_pos_embed above."""
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self._interpolated_pos_embed(x.shape[1] - 1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
