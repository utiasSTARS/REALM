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
seg_head.py — Lightweight segmentation head for REALM.

Accepts patch tokens from a ViT backbone, reshapes them into a spatial grid,
applies a 1×1 classification convolution, and optionally upsamples the logit
map to the original image resolution.

Public API
----------
    SegHead — patch-token → per-pixel class logits
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["SegHead"]


class SegHead(nn.Module):
    """
    Lightweight segmentation head for ViT-style patch token sequences.

    The head performs three operations:

    1. Reshapes the flat patch sequence into a 2-D spatial feature map.
    2. Applies a 1×1 convolution to produce per-patch class logits.
    3. Optionally bilinearly upsamples the logits to ``(H, W)``.

    Args:
        in_channels:  Dimensionality of each input patch token (``embed_dim``).
        num_classes:  Number of output segmentation classes.

    Raises:
        ValueError: If ``in_channels`` or ``num_classes`` are not positive.

    Example
    -------
    ::

        head   = SegHead(in_channels=768, num_classes=19)
        logits = head(patch_tokens, options={"H": 480, "W": 640, "upsample": True})
        # logits: (B, 19, 480, 640)
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}.")

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=True)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Xavier-uniform init for the classifier, zero bias."""
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, patch_tokens: Tensor, options: dict | None = None) -> Tensor:
        """
        Decode patch tokens into dense segmentation logits.

        Args:
            patch_tokens: Patch token tensor of shape ``(B, N, C)`` where
                          ``N`` must be a perfect square (e.g. 32×32 = 1024).
            options:      Optional inference-time settings dict with keys:

                          * ``"upsample"`` *(bool, default False)* — bilinearly
                            upsample logits to ``(H, W)``.
                          * ``"H"`` *(int)* — target height (required when
                            ``upsample=True``).
                          * ``"W"`` *(int)* — target width  (required when
                            ``upsample=True``).

        Returns:
            Class logit tensor. Shape is:

            * ``(B, num_classes, Ph, Pw)``      when ``upsample=False``
            * ``(B, num_classes, H, W)``        when ``upsample=True``

        Raises:
            ValueError: If ``patch_tokens`` is not 3-D, if ``N`` is not a
                        perfect square, or if ``upsample=True`` but ``H``/``W``
                        are not provided.
        """
        opts = options or {}

        if patch_tokens.ndim != 3:
            raise ValueError(
                f"Expected patch_tokens of shape (B, N, C), "
                f"got {tuple(patch_tokens.shape)}."
            )

        B, N, C = patch_tokens.shape

        # Infer square patch grid
        Ph = int(math.isqrt(N))
        if Ph * Ph != N:
            raise ValueError(
                f"Number of patches N={N} is not a perfect square. "
                "Cannot infer a square patch grid."
            )
        Pw = Ph

        # (B, N, C) → (B, C, Ph, Pw)
        x = patch_tokens.reshape(B, Ph, Pw, C).permute(0, 3, 1, 2).contiguous()

        logits = self.classifier(x)  # (B, num_classes, Ph, Pw)

        if opts.get("upsample", False):
            H = opts.get("H")
            W = opts.get("W")
            if H is None or W is None:
                raise ValueError(
                    "'H' and 'W' must be provided in options when upsample=True."
                )
            logits = F.interpolate(
                logits,
                size=(int(H), int(W)),
                mode="bilinear",
                align_corners=False,
            )

        return logits

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.in_channels}, "
            f"num_classes={self.num_classes})"
        )