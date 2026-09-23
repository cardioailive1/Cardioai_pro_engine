"""
CardioAI Pro — Lightweight Echo Model (MobileNetV3)
=========================================================
The edge-deployment counterpart to echo_model.py's EchoVideoViT. Real
published precedent for choosing MobileNet specifically here, not for CT:

  - A comparative study evaluating five architectures for automated
    echocardiogram ejection-fraction analysis found MobileNet was the
    best model for web/portable deployment, and shipped it running on a
    Raspberry Pi 4 (quad-core ARM, no GPU) at ~1 minute per video — still
    ~10x faster than manual expert analysis — with a stated path to
    smartphone deployment.
  - This matters because point-of-care/handheld cardiac ultrasound is a
    real, large, and growing clinical use case (see the POCUS literature:
    handheld devices used by non-expert operators at the bedside, in the
    ED, in resource-limited settings) with a genuine compute constraint
    that a hospital's cloud GPU cluster doesn't have to deal with.

WHY THIS ISN'T THE RIGHT CALL FOR CT: cardiac CT scanners are stationary
hospital equipment with server-room-adjacent compute already available —
there's no "point-of-care CT" the way there's point-of-care ultrasound, so
CT doesn't have the same edge-deployment driver. ct_model.py's ViT-based
CardiacCTViT, following DINO-LG's specific published architecture for
exactly the CAC-scoring task, stays the CT path. This isn't "MobileNet
instead of ViT" — it's the two architectures matched to where their real
published precedent actually applies.

Uses torchvision's standard, well-tested MobileNetV3 implementation
(matching Howard et al. 2019's published parameter counts exactly: 5.48M
for Large, 2.54M for Small — confirmed by instantiating both) rather than
a hand-rolled version — MobileNetV3's architecture is NAS-derived with
many specific per-block configurations that are easy to get subtly wrong
by hand; there's no methodological benefit to reimplementing it from
scratch the way vit_backbone.py's ViT was, since no cardiac-specific
architectural modification is being made here beyond the input/output
layers.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small


class MobileEchoNet(nn.Module):
    """
    Input:  (B, T, C, H, W) — a cine loop, same shape convention as
            EchoVideoViT, but T is typically fewer frames here (real-time
            or near-real-time inference is the whole point of choosing
            this backbone — a handheld device guiding a non-expert
            operator can't wait for a 32-frame ViT pass).
    Output: (B, n_outputs) — same interface as EchoVideoViT, so
            inference/models.py's load_trained_model() can select
            between the two based on deployment context (edge vs. cloud)
            without any other code caring which one is loaded.
    """

    def __init__(self, n_outputs: int = 1, variant: str = "large", temporal_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        if variant == "large":
            backbone = mobilenet_v3_large(weights=None)
            feature_dim = backbone.classifier[0].in_features  # 960 for Large
        elif variant == "small":
            backbone = mobilenet_v3_small(weights=None)
            feature_dim = backbone.classifier[0].in_features  # 576 for Small
        else:
            raise ValueError(f"variant must be 'large' or 'small', got {variant!r}")

        # Swap the first conv for single-channel (grayscale ultrasound)
        # input instead of the 3-channel RGB torchvision assumes.
        first_conv = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(
            1, first_conv.out_channels, kernel_size=first_conv.kernel_size,
            stride=first_conv.stride, padding=first_conv.padding, bias=False,
        )
        backbone.classifier = nn.Identity()  # drop the ImageNet classification head — use raw pooled features
        backbone.avgpool = nn.AdaptiveAvgPool2d(1)

        self.backbone = backbone
        self.feature_dim = feature_dim
        self.variant = variant

        # Lightweight temporal aggregation — a single small transformer
        # layer (or fewer) is appropriate here; the whole point of this
        # path is staying cheap enough for edge hardware, so this
        # shouldn't be as heavy as EchoVideoViT's temporal encoder.
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=4, dim_feedforward=feature_dim, dropout=dropout, batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=temporal_layers)
        self.head = nn.Linear(feature_dim, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        frames = x.view(B * T, C, H, W)
        feats = self.backbone(frames)              # (B*T, feature_dim)
        feats = feats.view(B, T, -1)
        temporal_out = self.temporal_encoder(feats)  # (B, T, feature_dim)
        pooled = temporal_out.mean(dim=1)
        return self.head(pooled)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
