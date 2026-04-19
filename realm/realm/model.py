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
model.py — Top-level REALM inference model.

REALM supports two forward modes:

* **Siamese** (dict input ``{"view1": Tensor, "view2": Tensor}``) — for
  stereo and dense-matching heads (e.g. Mast3r).
* **Single-view** (plain ``Tensor`` of shape ``(B, C, H, W)``) — for
  per-pixel prediction heads such as depth and segmentation.

The active encoder is selected based on :class:`ModelType`:

* ``Events``  — event voxel encoder only.
* ``RGB``     — RGB image encoder only.
* ``Hybrid``  — both encoders available; single-view selection is made
                by inspecting the input channel count.

Public API
----------
    ModelType  — enum of supported modality configurations
    REALM      — top-level inference module
"""

from __future__ import annotations

from enum import Enum
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from realm.utils.log import get_logger

__all__ = ["ModelType", "REALM"]

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Expected number of channels in an event voxel grid.
# Used by the Hybrid model to dispatch to the correct encoder.
_EVENT_CHANNELS: int = 5


# ---------------------------------------------------------------------------
# ModelType
# ---------------------------------------------------------------------------

class ModelType(Enum):
    """Modality configuration of a REALM model instance."""

    Events = "events"
    RGB    = "rgb"
    Hybrid = "hybrid"


# ---------------------------------------------------------------------------
# REALM
# ---------------------------------------------------------------------------

class REALM(nn.Module):
    """
    Top-level REALM inference model.

    Supports two forward modes selected automatically from the input type:

    * **Siamese** — ``x`` is a ``dict`` with keys ``"view1"`` and ``"view2"``.
      Both views are encoded independently, then decoded by a paired head
      (e.g. :class:`~realm.heads.mast3r.mast3r_head.Mast3rDecoder`).
    * **Single-view** — ``x`` is a plain ``(B, C, H, W)`` tensor.
      A single encoder runs, followed by an optional projector and head.

    The :attr:`model_type` is inferred from which encoders are provided:

    * Both ``encoder_ev`` and ``encoder_rgb`` → :attr:`ModelType.Hybrid`
    * Only ``encoder_ev``                     → :attr:`ModelType.Events`
    * Only ``encoder_rgb``                    → :attr:`ModelType.RGB`

    Args:
        encoder_ev:  Event-voxel encoder (or ``None``).
        encoder_rgb: RGB image encoder   (or ``None``).
        head:        Task head module.   (or ``None`` — returns raw features).
        projector:   Optional feature projector applied before the head.

    Raises:
        ValueError: If both ``encoder_ev`` and ``encoder_rgb`` are ``None``.
    """

    def __init__(
        self,
        encoder_ev:  nn.Module | None,
        encoder_rgb: nn.Module | None,
        head:        nn.Module | None,
        projector:   nn.Module | None = None,
    ) -> None:
        super().__init__()

        if encoder_ev is None and encoder_rgb is None:
            raise ValueError(
                "At least one encoder (encoder_ev or encoder_rgb) must be provided."
            )

        self.encoder_ev  = encoder_ev
        self.encoder_rgb = encoder_rgb
        self.projector   = projector
        self.head        = head

        # Detect siamese mode from the head
        self.siamese: bool = (
            head is not None and getattr(head, "siamese", False)
        )

        # Infer model type from encoder availability
        if encoder_ev is not None and encoder_rgb is not None:
            self.model_type = ModelType.Hybrid
        elif encoder_ev is not None:
            self.model_type = ModelType.Events
        else:
            self.model_type = ModelType.RGB

        logger.info(
            "REALM initialised — modality: %s | siamese: %s",
            self.model_type.value,
            self.siamese,
        )

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Union[Tensor, dict[str, Tensor]],
        options: dict | None = None,
    ) -> Union[Tensor, tuple, dict]:
        """
        Run inference on one or two views.

        Args:
            x:       Input data. Either:

                     * A ``dict`` with keys ``"view1"`` and ``"view2"``
                       (required for siamese heads).
                     * A plain ``(B, C, H, W)`` tensor (single-view heads).

            options: Inference-time settings forwarded to the task head
                     (e.g. ``{"upsample": True, "H": 480, "W": 640}``).
                     Ignored for siamese forward passes.

        Returns:
            Head output. Type depends on the active head:

            * **Siamese**: ``(res1, res2)`` — a pair of output dicts.
            * **Single-view**: head-specific tensor or dict.
            * **No head**: raw feature dict from the encoder.

        Raises:
            ValueError: If a siamese head receives non-dict input, or a
                        single-view head receives dict input.
            ValueError: If a ``Hybrid`` model receives input with an
                        unrecognised channel count.
        """
        opts = options or {}

        if self.siamese:
            if not isinstance(x, dict) or "view1" not in x or "view2" not in x:
                raise ValueError(
                    "Siamese head requires a dict input with keys "
                    "'view1' and 'view2'."
                )
            return self._forward_siamese(x["view1"], x["view2"])

        if isinstance(x, dict):
            raise ValueError(
                "This REALM model uses a non-siamese head. "
                "Pass a plain (B, C, H, W) tensor instead of a dict."
            )

        return self._forward_single(x, opts)

    # ------------------------------------------------------------------
    # Internal forward paths
    # ------------------------------------------------------------------

    def _forward_siamese(
        self,
        x1: Tensor,
        x2: Tensor,
    ) -> tuple[dict, dict]:
        """Encode two views and decode with the paired siamese head."""
        feat1, pos1, _ = self._encode(x1)
        feat2, pos2, _ = self._encode(x2)

        B = x1.shape[0]
        # true_shape: (B, 2) — (W, H) per REALM/Mast3r convention
        true_shape = (
            torch.tensor(
                np.int32([[x1.shape[3], x1.shape[2]]]),
                device=x1.device,
            )
            .expand(B, -1)
        )

        dec1, dec2 = self.head._decoder(feat1["x_norm_patchtokens"], pos1, feat2["x_norm_patchtokens"], pos2)

        # Run downstream heads in full precision to avoid AMP precision issues
        with torch.amp.autocast(enabled=False, device_type="cuda"):
            res1 = self.head._downstream_head(
                1, [tok.float() for tok in dec1], true_shape
            )
            res2 = self.head._downstream_head(
                2, [tok.float() for tok in dec2], true_shape
            )

        # Rename pts3d in res2 to signal it is expressed in the other view's frame
        res2["pts3d_in_other_view"] = res2.pop("pts3d")

        return res1, res2

    def _forward_single(self, x: Tensor, options: dict) -> Tensor | dict:
        """Encode a single view, optionally project, then run the task head."""
        features = self._encode(x)[0]  # dict from encoder

        if self.projector is not None:
            features = self.projector(features["x_norm_patchtokens"])

        if self.head is None:
            return features

        return self.head(features, options)

    # ------------------------------------------------------------------
    # Encoder dispatch
    # ------------------------------------------------------------------

    def _encode(self, x: Tensor) -> tuple[dict, Tensor, None]:
        """
        Encode a single view with the appropriate encoder.

        For :attr:`ModelType.Hybrid` models the encoder is selected by
        channel count: :data:`_EVENT_CHANNELS` → event encoder, otherwise
        RGB encoder.

        Args:
            x: ``(B, C, H, W)`` input tensor.

        Returns:
            A 3-tuple ``(features, pos, None)`` where:

            * ``features`` — encoder output dict (includes
              ``"x_norm_patchtokens"`` and ``"x_norm_clstoken"``).
            * ``pos``      — positional encoding for the patch grid.
            * ``None``     — reserved for future mask/confidence output.

        Raises:
            ValueError: If ``model_type`` is :attr:`ModelType.Hybrid` and the
                        channel count does not match :data:`_EVENT_CHANNELS`
                        or 3.
        """
        B, C, H, W = x.shape

        if self.model_type == ModelType.Events:
            encoder = self.encoder_ev
        elif self.model_type == ModelType.RGB:
            encoder = self.encoder_rgb
        else:  # Hybrid — dispatch by channel count
            if C == _EVENT_CHANNELS:
                encoder = self.encoder_ev
            elif C == 3:
                encoder = self.encoder_rgb
            else:
                raise ValueError(
                    f"Hybrid model received input with {C} channels. "
                    f"Expected {_EVENT_CHANNELS} (events) or 3 (RGB)."
                )

        features = encoder(x)

        pos = None
        if self.siamese and self.head is not None:
            h = H // self.head.patch_embed_size
            w = W // self.head.patch_embed_size
            pos = self.head.position_getter(B, h, w, x.device)

        return features, pos, None

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_type={self.model_type.value}, "
            f"siamese={self.siamese}, "
            f"has_projector={self.projector is not None})"
        )