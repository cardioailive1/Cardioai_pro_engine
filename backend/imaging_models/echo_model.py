"""
CardioAI Pro — Echo Video Model
=====================================
Follows Echo-Vision-FM's architecture pattern (Zhang et al., Nature
Communications, 2025/2026): a ViT backbone over echo frames, with temporal
modeling to capture the cardiac cycle — echo is fundamentally a VIDEO
modality (wall motion across a heartbeat is the clinically meaningful
signal), unlike the single-frame pixel statistics
integrations/dicom.py's extract_pixel_features() currently computes.

SIMPLIFICATION FROM THE PUBLISHED ARCHITECTURE, STATED HONESTLY: the real
Echo-VideoMAE uses joint spatiotemporal masked-patch pretraining — the
self-supervised objective operates on 3D (space + time) patches together,
learned from ~large-scale unlabeled echo video (MIMIC-IV-ECHO). This class
instead applies the 2D ViT backbone per-frame (spatial encoding), then a
separate lightweight transformer over the resulting frame embeddings
(temporal encoding) — a "late fusion" video architecture, simpler than
joint spatiotemporal pretraining but built on the same ViT backbone, and
adequate for supervised fine-tuning on a labeled downstream task (LVEF,
wall-motion classification) once frame-level features are available. Full
joint spatiotemporal MAE pretraining, matching the published architecture
exactly, would need a dedicated pretraining run on real echo video data at
MIMIC-IV-ECHO scale, which this environment cannot perform.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from imaging_models.vit_backbone import ViTBackbone


class EchoVideoViT(nn.Module):
    """
    Input:  (B, T, C, H, W) — a cine loop of T frames (a full or partial
            cardiac cycle), single-channel (C=1) grayscale, e.g. T=16-32
            frames sampled evenly across one heartbeat, H=W=224.
    Output: (B, n_outputs) — e.g. n_outputs=1 for LVEF regression (a
            continuous %), or n_outputs=n_classes for a categorical task
            (e.g. normal / reduced / severely-reduced ejection fraction).
    """

    def __init__(
        self, n_outputs: int = 1, img_size: int = 224, patch_size: int = 16,
        embed_dim: int = 768, spatial_depth: int = 12, spatial_heads: int = 12,
        temporal_layers: int = 2, temporal_heads: int = 8, dropout: float = 0.1,
    ):
        super().__init__()
        self.spatial_backbone = ViTBackbone(
            img_size=img_size, patch_size=patch_size, in_channels=1,
            embed_dim=embed_dim, depth=spatial_depth, n_heads=spatial_heads,
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=temporal_heads, dim_feedforward=embed_dim * 2,
            dropout=dropout, batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=temporal_layers)
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, 64, embed_dim))  # supports up to 64 frames
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        self.head = nn.Linear(embed_dim, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        frames = x.view(B * T, C, H, W)
        frame_embeds = self.spatial_backbone(frames)[:, 0]      # CLS token per frame: (B*T, embed_dim)
        frame_embeds = frame_embeds.view(B, T, -1)
        frame_embeds = frame_embeds + self.temporal_pos_embed[:, :T]
        temporal_out = self.temporal_encoder(frame_embeds)       # (B, T, embed_dim)
        pooled = temporal_out.mean(dim=1)                          # mean pool over the cardiac cycle
        return self.head(pooled)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
