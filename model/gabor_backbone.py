"""Gabor filter bank feature extractor — no pretrained weights needed.

Gabor filters are optimal for texture analysis at multiple orientations and
scales, making them naturally suited to woven materials like carbon fiber.
For each image we produce two feature maps:
  - 'fine'   : high-frequency responses (captures fiber weave pattern)
  - 'coarse' : low-frequency responses (captures larger structural anomalies)
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gabor_kernel(
    size: int,
    theta: float,
    sigma: float,
    lambd: float,
    gamma: float = 0.5,
) -> torch.Tensor:
    """Return a 2-D real Gabor kernel as a float32 tensor."""
    half = size // 2
    y, x = np.mgrid[-half : half + 1, -half : half + 1]

    x_t = x * np.cos(theta) + y * np.sin(theta)
    y_t = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-0.5 * (x_t**2 + gamma**2 * y_t**2) / sigma**2) * np.cos(
        2 * np.pi * x_t / lambd
    )
    gb -= gb.mean()
    std = gb.std()
    if std > 0:
        gb /= std
    return torch.from_numpy(gb.astype(np.float32))


def _make_bank(
    kernel_size: int,
    orientations: int,
    scales: List[float],
    lambdas: List[float],
) -> torch.Tensor:
    """Stack Gabor kernels over orientations and scales → (C, 1, kH, kW)."""
    kernels = []
    for sigma, lambd in zip(scales, lambdas):
        for i in range(orientations):
            theta = i * np.pi / orientations
            kernels.append(_gabor_kernel(kernel_size, theta, sigma, lambd))
    return torch.stack(kernels, dim=0).unsqueeze(1)  # (C, 1, kH, kW)


class GaborFeatureExtractor(nn.Module):
    """Two-level Gabor feature extractor producing 'fine' and 'coarse' maps."""

    def __init__(self, orientations: int = 8):
        super().__init__()
        self.orientations = orientations

        # Fine scale: small kernel, high frequency (captures weave period)
        fine_bank = _make_bank(
            kernel_size=15,
            orientations=orientations,
            scales=[2.0, 3.0, 4.0],
            lambdas=[4.0, 6.0, 8.0],
        )
        self.register_buffer("fine_bank", fine_bank)

        # Coarse scale: larger kernel, low frequency (captures structural changes)
        coarse_bank = _make_bank(
            kernel_size=31,
            orientations=orientations,
            scales=[5.0, 7.0, 10.0],
            lambdas=[10.0, 14.0, 20.0],
        )
        self.register_buffer("coarse_bank", coarse_bank)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Grayscale conversion (C-weighted)
        gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)

        pad_fine = self.fine_bank.shape[-1] // 2
        fine = F.conv2d(gray, self.fine_bank, padding=pad_fine).abs()
        # Downsample to 28×28 (matches WideResNet-50 layer2 spatial output for 224×224 input)
        fine = F.avg_pool2d(fine, kernel_size=8, stride=8)

        pad_coarse = self.coarse_bank.shape[-1] // 2
        coarse = F.conv2d(gray, self.coarse_bank, padding=pad_coarse).abs()
        # Downsample to 14×14 (matches WideResNet-50 layer3 spatial output)
        coarse = F.avg_pool2d(coarse, kernel_size=16, stride=16)

        return {"fine": fine, "coarse": coarse}
