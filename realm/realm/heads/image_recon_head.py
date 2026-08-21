# Copyright (c) 2025 Space and Terrestrial Autonomous Robotic Systems (STARS) Lab,
# University of Toronto Institute for Aerospace Studies (UTIAS).
# All rights reserved.
#
# This software is provided for research and educational purposes only.
# Redistribution and use, with or without modification, are permitted provided
# that this copyright notice and attribution are retained.
#
# Maintainer: Vincenzo Polizzi <polivicio@gmail.com>

"""
image_recon_head.py — Convolutional image reconstruction head for REALM.

Decodes ViT backbone patch tokens back into a dense image via a stack of
3x3-conv/BatchNorm/GELU blocks interleaved with bilinear upsampling.

Public API
----------
    ImageReconHead — patch tokens → reconstructed image
"""

from __future__ import annotations

import math

import torch.nn as nn
from torch import Tensor

__all__ = ["ImageReconHead"]


class ImageReconHead(nn.Module):
    """
    Convolutional decoder that reconstructs a dense image from patch tokens.

    The flat patch-token sequence is reshaped into a square spatial grid and
    progressively upsampled back to ``target_size`` through four conv blocks,
    ending in a 1x1 convolution that projects to ``out_channels``.

    Args:
        num_tokens:   Number of patch tokens (must be a perfect square).
        token_dim:    Dimensionality of each input patch token.
        target_size:  Output spatial resolution (square).
        out_channels: Number of output image channels (3 for RGB, 1 for
                      grayscale/intensity reconstruction).

    Example
    -------
    ::

        head  = ImageReconHead(num_tokens=1024, token_dim=768, target_size=448)
        image = head(features)
        # image: (B, 3, 448, 448)
    """

    def __init__(
        self,
        num_tokens: int = 1024,
        token_dim: int = 768,
        target_size: int = 448,
        out_channels: int = 3,
    ) -> None:
        super().__init__()

        self.Ph = int(math.isqrt(num_tokens))
        if self.Ph ** 2 != num_tokens:
            raise ValueError(f"num_tokens must be a perfect square, got {num_tokens}.")

        self.num_tokens   = num_tokens
        self.token_dim    = token_dim
        self.target_size  = target_size
        self.out_channels = out_channels

        def block(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.GELU(),
            )

        up = lambda: nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.net = nn.Sequential(
            block(token_dim, 512),   up(),   # 32 -> 64
            block(512, 256),         up(),   # 64 -> 128
            block(256, 128),         up(),   # 128 -> 256
            block(128, 64),
            nn.Upsample(size=target_size, mode="bilinear", align_corners=False),  # 256 -> target_size
            nn.Conv2d(64, out_channels, kernel_size=1),
        )

    def forward(self, features: Tensor | dict[str, Tensor], options: dict | None = None) -> Tensor:
        """
        Decode patch tokens into a reconstructed image.

        Args:
            features: Either the raw ``(B, N, C)`` patch-token tensor, or the
                      backbone feature dict containing ``"x_norm_patchtokens"``.
            options:  Unused, present for interface parity with other heads.

        Returns:
            Reconstructed image of shape ``(B, out_channels, target_size, target_size)``.

        Raises:
            KeyError:   If a dict is passed without a ``"x_norm_patchtokens"`` key.
            ValueError: If the patch token count does not match ``num_tokens``.
        """
        if isinstance(features, dict):
            if "x_norm_patchtokens" not in features:
                raise KeyError(
                    f"Required key 'x_norm_patchtokens' not found in features dict. "
                    f"Available keys: {list(features.keys())}"
                )
            patch_tokens = features["x_norm_patchtokens"]
        else:
            patch_tokens = features

        B, N, C = patch_tokens.shape
        if N != self.num_tokens:
            raise ValueError(
                f"Expected {self.num_tokens} patch tokens, got {N}."
            )

        x = patch_tokens.reshape(B, self.Ph, self.Ph, C).permute(0, 3, 1, 2).contiguous()
        return self.net(x)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_tokens={self.num_tokens}, token_dim={self.token_dim}, "
            f"target_size={self.target_size}, out_channels={self.out_channels})"
        )
