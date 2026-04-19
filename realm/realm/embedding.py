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
embedding.py — Event-voxel patch embedding for REALM.

Maps event voxel grids ``(B, num_bins, H, W)`` to patch token sequences
``(B, num_patches, embed_dim)`` that are directly consumable by a
ViT-style transformer backbone.

Architecture
------------
    stem  →  EncoderBlock × num_downsample  →  1×1 proj  →  AdaptiveAvgPool  →  flatten

Public API
----------
    Vox2PatchEmbed   — full voxel-to-patch encoder
    EncoderBlock     — single strided downsampling block with optional residual
"""

from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["Vox2PatchEmbed", "EncoderBlock"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_2tuple(value: Union[int, tuple[int, int]]) -> tuple[int, int]:
    """
    Coerce *value* to a ``(int, int)`` tuple.

    Args:
        value: An ``int`` (replicated for both dimensions) or a length-2 tuple.

    Returns:
        A ``(height, width)`` integer tuple.

    Raises:
        TypeError:  If *value* is not an int or a tuple.
        ValueError: If a tuple is provided but its length is not 2.
    """
    if isinstance(value, int):
        return value, value
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(
                f"Expected a tuple of length 2, got length {len(value)}."
            )
        return int(value[0]), int(value[1])
    raise TypeError(
        f"Expected int or tuple[int, int], got {type(value).__name__}."
    )


# Valid normalisation layer identifiers
_NORM_TYPES = frozenset({"batch", "instance", "group"})


def _make_norm(norm: str, channels: int) -> nn.Module:
    """
    Instantiate a normalisation layer by name.

    Args:
        norm:     One of ``"batch"``, ``"instance"``, or ``"group"``.
        channels: Number of feature channels (used for all norm types).

    Returns:
        An ``nn.Module`` normalisation layer.

    Raises:
        ValueError: If *norm* is not a recognised type.

    Note:
        ``GroupNorm`` uses 32 groups. If ``channels`` is not divisible by 32
        a ``ValueError`` will be raised by PyTorch at construction time.
    """
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if norm == "group":
        return nn.GroupNorm(32, channels)
    raise ValueError(
        f"Unknown normalisation type: {norm!r}. "
        f"Choose from {sorted(_NORM_TYPES)}."
    )


# ---------------------------------------------------------------------------
# EncoderBlock
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """
    Strided downsampling block with an optional residual shortcut.

    Spatial resolution is halved (stride=2) by the first convolution.
    The residual path uses a 1×1 strided convolution to match dimensions.

    Args:
        in_channels:  Number of input feature channels.
        out_channels: Number of output feature channels.
        norm:         Normalisation type — ``"batch"``, ``"instance"``,
                      or ``"group"``.
        use_residual: If ``True``, add a learned residual shortcut.

    Raises:
        ValueError: If *norm* is not recognised.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str = "batch",
        use_residual: bool = True,
    ) -> None:
        super().__init__()

        if norm not in _NORM_TYPES:
            raise ValueError(
                f"Unknown normalisation type: {norm!r}. "
                f"Choose from {sorted(_NORM_TYPES)}."
            )

        self.use_residual = use_residual

        # --- Main path ---
        self.conv1  = nn.Conv2d(in_channels,  out_channels, 3, stride=2, padding=1, bias=False)
        self.norm1  = _make_norm(norm, out_channels)
        self.relu1  = nn.ReLU(inplace=True)
        self.conv2  = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.norm2  = _make_norm(norm, out_channels)
        self.relu2  = nn.ReLU(inplace=True)

        # --- Residual projection ---
        self.residual: nn.Module | None = None
        if use_residual:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False),
                _make_norm(norm, out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.relu1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))

        if self.residual is not None:
            identity = self.residual(x)
            out = out + identity

        return self.relu2(out)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"use_residual={self.use_residual})"
        )


# ---------------------------------------------------------------------------
# Vox2PatchEmbed
# ---------------------------------------------------------------------------

class Vox2PatchEmbed(nn.Module):
    """
    Converts event voxel grids into patch tokens consumable by REALM.

    The encoder follows a ResNet-style stem → strided-block stack pattern,
    finishing with a 1×1 projection and adaptive average pooling to produce
    a fixed-size patch grid regardless of input resolution.

    Architecture
    ------------
    ::

        (B, num_bins, H, W)
            │
            ▼  7×7 stem conv
        (B, C₀, H, W)
            │
            ▼  EncoderBlock × num_downsample  (stride 2 each)
        (B, Cₙ, H/2ⁿ, W/2ⁿ)
            │
            ▼  1×1 conv → BN → GELU
        (B, embed_dim, H', W')
            │
            ▼  AdaptiveAvgPool2d(patch_grid)
        (B, embed_dim, Ph, Pw)
            │
            ▼  flatten + transpose
        (B, Ph×Pw, embed_dim)

    Args:
        num_bins:       Number of temporal bins in the event voxel grid —
                        equals the number of input channels.
        patch_grid:     Output patch grid size ``(Ph, Pw)`` or a single int.
                        Determines ``num_patches = Ph × Pw``.
        embed_dim:      Dimensionality of each output patch token.
        base_channels:  Channel count after the stem. Doubles each block.
        num_downsample: Number of :class:`EncoderBlock` stages. Each halves
                        the spatial resolution, so the total stride is
                        ``2 ** num_downsample``.
        norm:           Normalisation type: ``"batch"``, ``"instance"``,
                        or ``"group"``.
        use_residual:   Toggle residual shortcuts in encoder blocks.

    Raises:
        ValueError: If *norm* is not recognised or *patch_grid* is invalid.

    Examples
    --------
    ::

        embed = Vox2PatchEmbed(num_bins=15, patch_grid=32, embed_dim=768)
        tokens = embed(voxels)           # (B, 1024, 768)
        out    = embed(voxels, aux=True) # dict with patch_tokens, patch_map, bottleneck
    """

    def __init__(
        self,
        num_bins: int,
        patch_grid: Union[int, tuple[int, int]] = 32,
        embed_dim: int = 768,
        base_channels: int = 64,
        num_downsample: int = 3,
        norm: str = "batch",
        use_residual: bool = True,
    ) -> None:
        super().__init__()

        if norm not in _NORM_TYPES:
            raise ValueError(
                f"Unknown normalisation type: {norm!r}. "
                f"Choose from {sorted(_NORM_TYPES)}."
            )
        if num_bins <= 0:
            raise ValueError(f"num_bins must be positive, got {num_bins}.")
        if num_downsample <= 0:
            raise ValueError(f"num_downsample must be positive, got {num_downsample}.")

        self.patch_grid    = _make_2tuple(patch_grid)
        self.embed_dim     = embed_dim
        self.num_patches   = self.patch_grid[0] * self.patch_grid[1]
        self.feature_stride = 2 ** num_downsample

        # Channel schedule: [C₀, C₁, …, Cₙ]
        channels = [base_channels * (2 ** i) for i in range(num_downsample + 1)]

        # --- Stem (large receptive field, no downsampling) ---
        self.stem = nn.Sequential(
            nn.Conv2d(num_bins, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            _make_norm(norm, channels[0]),
            nn.ReLU(inplace=True),
        )

        # --- Downsampling encoder ---
        self.encoder = nn.ModuleList([
            EncoderBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                norm=norm,
                use_residual=use_residual,
            )
            for i in range(num_downsample)
        ])

        # --- Projection to embedding dimension ---
        self.to_patch_embed = nn.Sequential(
            nn.Conv2d(channels[-1], embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        # --- Adaptive pool → fixed patch grid ---
        self.pool = nn.AdaptiveAvgPool2d(self.patch_grid)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Kaiming-normal init for Conv2d; ones/zeros for norm layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        voxel_grid: Tensor,
        aux: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        """
        Encode an event voxel grid into patch tokens.

        Args:
            voxel_grid: Input tensor of shape ``(B, num_bins, H, W)``.
            aux:        If ``True``, return a dictionary containing
                        intermediate feature maps in addition to the tokens.

        Returns:
            * ``aux=False`` *(default)*: ``(B, num_patches, embed_dim)`` tensor.
            * ``aux=True``: dictionary with keys:

              * ``"patch_tokens"`` — ``(B, num_patches, embed_dim)``
              * ``"patch_map"``    — ``(B, embed_dim, Ph, Pw)``
              * ``"bottleneck"``   — ``(B, Cₙ, H/stride, W/stride)``

        Raises:
            ValueError: If *voxel_grid* is not a 4-D tensor.
        """
        if voxel_grid.ndim != 4:
            raise ValueError(
                f"Expected a 4-D tensor (B, C, H, W), got shape {tuple(voxel_grid.shape)}."
            )

        x = self.stem(voxel_grid)

        for block in self.encoder:
            x = block(x)

        bottleneck = x

        x          = self.to_patch_embed(x)
        patch_map  = self.pool(x)
        # (B, embed_dim, Ph, Pw) → (B, Ph×Pw, embed_dim)
        patch_tokens = patch_map.flatten(2).transpose(1, 2)

        if aux:
            return {
                "patch_tokens": patch_tokens,
                "patch_map":    patch_map,
                "bottleneck":   bottleneck,
            }

        return patch_tokens

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"patch_grid={self.patch_grid}, "
            f"num_patches={self.num_patches}, "
            f"embed_dim={self.embed_dim}, "
            f"feature_stride={self.feature_stride})"
        )