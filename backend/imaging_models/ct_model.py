"""
CardioAI Pro — Cardiac CT Model
=====================================
Follows the DINO-LG (Haque et al., Med Biol Eng Comput, 2026) and CARD-ViT
(PMC13274280, 2026) pattern: a ViT backbone self-supervised pretrained
with DINO (self-distillation with no labels) on unlabeled CT slices, then
a lightweight linear head fine-tuned on a labeled downstream task —
DINO-LG's task is coronary artery calcium (CAC) risk classification;
this class is architecturally identical, with the task (number of output
classes) left configurable since the exact label set (e.g. CAC risk
category, or a binary plaque-present/absent) depends on which real,
labeled CT cohort ends up training it.

WHY DINO, SPECIFICALLY: DINO's self-supervised pretraining needs NO
labels at all — only the raw CT slices — which matters because unlabeled
CT scans are comparatively easy to obtain at scale, while a slice-level
CAC/plaque label needs expert radiologist annotation. DINO-LG reports
89% sensitivity / 90% specificity for CAC-slice detection pretrained on
just 914 CT scans (700 gated + 214 non-gated) — a training-set size an
individual health system's imaging archive could plausibly reach, unlike
the millions of images most foundation models require.

WHAT THIS CLASS DOES AND DOESN'T IMPLEMENT: this defines the model
ARCHITECTURE — the ViT backbone + linear head. It does not implement the
DINO self-supervised pretraining procedure itself (the student/teacher
network, momentum encoder, and centering/sharpening objective DINO uses)
— that is a real, separate training loop this environment has neither
the unlabeled CT corpus nor the GPU budget to run. This is the
architecture that pretraining loop would produce weights for.
"""
from __future__ import annotations

import torch.nn as nn

from imaging_models.vit_backbone import ViTBackbone


class CardiacCTViT(nn.Module):
    """
    Input:  (B, C, H, W) — a single CT slice, single-channel (C=1),
            windowed to the relevant Hounsfield-unit range before this
            model ever sees it (e.g. a cardiac/soft-tissue window for
            calcium scoring — that windowing is a preprocessing step, not
            something this class does).
    Output: (B, n_classes) — e.g. n_classes=4 for a CAC risk category
            (none / mild / moderate / severe, mirroring Agatston-score
            risk bands), or n_classes=2 for binary plaque presence.

    patch_size defaults to 8 (not 16, unlike EchoVideoViT) specifically
    because DINO-LG uses ViT-Base/8 for CT — finer patches give higher
    spatial resolution, which matters more for detecting small, dense
    calcium deposits than for the coarser structural task echo images
    are used for.
    """

    def __init__(
        self, n_classes: int = 4, img_size: int = 224, patch_size: int = 8,
        embed_dim: int = 768, depth: int = 12, n_heads: int = 12,
    ):
        super().__init__()
        self.backbone = ViTBackbone(
            img_size=img_size, patch_size=patch_size, in_channels=1,
            embed_dim=embed_dim, depth=depth, n_heads=n_heads,
        )
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        cls_embed = self.backbone(x)[:, 0]
        return self.head(cls_embed)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
