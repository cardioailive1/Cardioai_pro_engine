"""
CardioAI Pro — Masked Autoencoder Pretraining (Echo)
==========================================================
Real implementation of MAE (Masked Autoencoders Are Scalable Vision
Learners — He et al. 2021), the self-supervised method Echo-Vision-FM's
"Echo-VideoMAE" is built on. Stage 1 of EchoVideoViT's two-stage training
(see that module's docstring); a downstream fine-tuning script would
attach the temporal encoder + task head and train on labeled data.

HOW MAE WORKS, BRIEFLY: split the image into patches, randomly mask a
LARGE fraction of them (75% here, matching the original paper), and only
feed the small set of VISIBLE patches into the transformer encoder — most
of the image is invisible to the encoder, which is what keeps this cheap
despite using a full transformer. A separate, lightweight decoder then
takes the encoded visible patches plus learnable "mask token"
placeholders (one per masked position, each combined with that position's
embedding) and tries to reconstruct the ORIGINAL PIXEL VALUES of the
masked patches. The loss is simple mean-squared-error between
reconstructed and real pixels, computed ONLY on the masked patches. No
labels, no teacher network, no momentum encoder, no risk of the
representation-collapse failure mode DINO-style self-distillation is
prone to (see pretrain_dino_ct.py's honest account of hitting exactly
that failure mode at toy scale) — MAE's objective is a well-posed
regression problem from the start, which is part of why it was chosen
for the (arguably harder, spatiotemporal) echo modality.

SIMPLIFICATION FROM THE PUBLISHED ARCHITECTURE, CONSISTENT WITH
echo_model.py: this pretrains the 2D SPATIAL backbone on individual
frames, not Echo-VideoMAE's joint spatiotemporal masked patches. The
temporal transformer in EchoVideoViT gets trained during supervised
fine-tuning, not here. Stated once in echo_model.py's docstring, restated
here because this is the module where it actually matters mechanically.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from imaging_models.vit_backbone import PatchEmbed, TransformerEncoderBlock


class MAEPretrainModel(nn.Module):
    """
    A from-scratch MAE encoder-decoder. Doesn't reuse ViTBackbone directly
    (unlike the DINO script, which uses the full backbone unmodified) —
    MAE's encoder only ever sees a SUBSET of patches, which needs its own
    forward path distinct from ViTBackbone.forward()'s "encode everything"
    assumption. The two share PatchEmbed and TransformerEncoderBlock.
    """

    def __init__(
        self, img_size: int = 224, patch_size: int = 16, in_channels: int = 1,
        encoder_dim: int = 192, encoder_depth: int = 4, encoder_heads: int = 3,
        decoder_dim: int = 96, decoder_depth: int = 2, decoder_heads: int = 3,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, encoder_dim)
        self.n_patches = self.patch_embed.n_patches
        self.grid_size = self.patch_embed.grid_size

        self.encoder_pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, encoder_dim))
        self.encoder_blocks = nn.ModuleList([TransformerEncoderBlock(encoder_dim, encoder_heads) for _ in range(encoder_depth)])
        self.encoder_norm = nn.LayerNorm(encoder_dim)

        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, decoder_dim))
        self.decoder_blocks = nn.ModuleList([TransformerEncoderBlock(decoder_dim, decoder_heads) for _ in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size * patch_size * in_channels)  # predict raw pixel values per patch

        nn.init.trunc_normal_(self.encoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, n_patches, patch_size*patch_size*C) — the ground-truth pixel target for reconstruction."""
        p = self.patch_size
        B, C, H, W = imgs.shape
        h = w = H // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, p * p * C)
        return x

    def random_masking(self, x: torch.Tensor):
        """Returns (visible_tokens, mask, restore_indices). mask is 1 for masked positions, 0 for kept — used to select which patches the loss applies to."""
        B, N, D = x.shape
        n_keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        shuffle_idx = torch.argsort(noise, dim=1)
        restore_idx = torch.argsort(shuffle_idx, dim=1)

        keep_idx = shuffle_idx[:, :n_keep]
        visible = torch.gather(x, 1, keep_idx.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, N, device=x.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, 1, restore_idx)  # unshuffle back to original patch order
        return visible, mask, restore_idx, keep_idx

    def forward(self, imgs: torch.Tensor):
        target = self.patchify(imgs)

        x = self.patch_embed(imgs)                        # (B, n_patches, encoder_dim), full grid
        x = x + self.encoder_pos_embed
        visible, mask, restore_idx, keep_idx = self.random_masking(x)

        for block in self.encoder_blocks:
            visible = block(visible)
        visible = self.encoder_norm(visible)

        # Decoder: project visible tokens to decoder dim, insert mask
        # tokens at masked positions, restore original patch order, add
        # decoder positional embeddings, decode.
        B, n_visible, _ = visible.shape
        decoded_visible = self.decoder_embed(visible)
        n_masked = self.n_patches - n_visible
        mask_tokens = self.mask_token.expand(B, n_masked, -1)
        full_seq = torch.cat([decoded_visible, mask_tokens], dim=1)   # visible first, then mask tokens
        full_seq = torch.gather(full_seq, 1, restore_idx.unsqueeze(-1).expand(-1, -1, full_seq.shape[-1]))  # restore original order
        full_seq = full_seq + self.decoder_pos_embed

        for block in self.decoder_blocks:
            full_seq = block(full_seq)
        full_seq = self.decoder_norm(full_seq)
        pred = self.decoder_pred(full_seq)   # (B, n_patches, patch_size*patch_size*C) — reconstructed pixels for every patch

        loss = ((pred - target) ** 2).mean(dim=-1)   # per-patch MSE
        loss = (loss * mask).sum() / mask.sum()        # average over MASKED patches only — the whole point of MAE
        return loss, pred, mask


class SyntheticEchoFrameDataset(Dataset):
    """Synthetic echo-frame-shaped tensors — proves the MAE loop runs correctly end-to-end. Swap for real echo frames (unlabeled) to actually pretrain."""

    def __init__(self, n_samples: int = 32, img_size: int = 224):
        self.n_samples, self.img_size = n_samples, img_size

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(idx)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, self.img_size), torch.linspace(-1, 1, self.img_size), indexing="ij")
        # A ring pattern (roughly chamber-wall-like structure) with per-sample size variation
        r = torch.sqrt(xx ** 2 + yy ** 2)
        radius = 0.4 + 0.2 * torch.rand(1, generator=g).item()
        ring = torch.exp(-((r - radius) ** 2) / 0.01)
        noise = torch.randn(self.img_size, self.img_size, generator=g) * 0.05
        return (ring + noise).unsqueeze(0).clamp(0, 1)


def pretrain_mae_echo(n_epochs: int = 5, batch_size: int = 4, encoder_depth: int = 4, lr: float = 1e-3) -> dict:
    model = MAEPretrainModel(img_size=224, patch_size=16, in_channels=1, encoder_depth=encoder_depth, mask_ratio=0.75)
    dataset = SyntheticEchoFrameDataset(n_samples=32, img_size=224)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    losses = []
    for epoch in range(n_epochs):
        for batch in loader:
            loss, _, _ = model(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            losses.append(loss.item())

    return {"losses": losses, "model": model}


if __name__ == "__main__":
    result = pretrain_mae_echo()
    print("Loss trajectory:", [round(l, 4) for l in result["losses"]])
