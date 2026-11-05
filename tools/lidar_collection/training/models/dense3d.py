"""Dense 3D U-Net baseline; this is NOT the paper's sparse 4D network."""

import torch
from torch import nn
from torch.nn import functional as F


def block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(4, out_channels),
        nn.SiLU(),
        nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(4, out_channels),
        nn.SiLU(),
    )


class TerrainUNet(nn.Module):
    def __init__(self, base_channels=8):
        super().__init__()
        if base_channels < 4 or base_channels % 4:
            raise ValueError("base_channels must be a positive multiple of 4")
        c = base_channels
        self.encoder = nn.ModuleList([block(12, c), block(c, 2 * c), block(2 * c, 4 * c)])
        self.bottleneck = block(4 * c, 8 * c)
        self.decoder = nn.ModuleList([block(12 * c, 4 * c), block(6 * c, 2 * c), block(3 * c, c)])
        self.head = nn.Conv3d(c, 4, 1)
        # Avoid a dense hallucinated map in the first untrained feedback step.
        nn.init.constant_(self.head.bias, 0)
        with torch.no_grad():
            self.head.bias[0] = -2.0

    def forward(self, inputs):
        skips = []
        x = inputs
        for encoder in self.encoder:
            x = encoder(x)
            skips.append(x)
            x = F.avg_pool3d(x, 2)
        x = self.bottleneck(x)
        for decoder, skip in zip(self.decoder, reversed(skips), strict=True):
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = decoder(torch.cat((x, skip), dim=1))
        output = self.head(x)
        return output[:, :1], output[:, 1:].sigmoid()


def reconstruction_loss(logits, offsets, target, offset_weight=1.0):
    occupied = target[:, :1]
    positives = occupied.sum()
    pos_weight = ((occupied.numel() - positives) / positives.clamp_min(1)).clamp(1, 50)
    occupancy_loss = F.binary_cross_entropy_with_logits(logits, occupied, pos_weight=pos_weight)
    distances = torch.linalg.vector_norm(offsets - target[:, 1:], dim=1, keepdim=True)
    offset_loss = (distances * occupied).sum() / positives.clamp_min(1)
    return occupancy_loss + offset_weight * offset_loss
