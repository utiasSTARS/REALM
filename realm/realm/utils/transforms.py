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
transforms.py — Spatiotemporal data augmentation pipeline for REALM.

All transforms operate on two distinct data formats:

* **Images** — ``np.ndarray`` of shape ``(T, H, W, C)`` or ``(T, H, W)``
* **Event voxels** — ``torch.Tensor`` of shape ``(T, C, H, W)``

Each transform subclasses :class:`UniTransform` and declares a *target type*
(``"image"``, ``"event"``, or ``"both"``) so the dataset pipeline can route
data correctly.

Public API
----------
    UniTransform        — abstract base class for all transforms
    WarperM3ED          — warp RGB images to the event-camera frame (M3ED calibration)
    Warper              — warp RGB images to the event-camera frame (YAML calibration)
    Flip                — horizontal / vertical flip
    Crop                — deterministic rectangular crop
    Resize              — stretch or scale-to-cover + centre-crop
    ResizeAndCropRandom — scale-to-cover + random crop (temporally consistent)
    build_transforms    — construct a transform list from a config list-of-dicts
    is_resize           — type predicate helper
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import yaml

from realm.utils.log import get_logger

__all__ = [
    "UniTransform",
    "WarperM3ED",
    "Warper",
    "Flip",
    "Crop",
    "Resize",
    "ResizeAndCropRandom",
    "build_transforms",
    "is_resize",
    "rescale_matches"
]

logger = get_logger(__name__)

# ---------------------------------------------------------------------------

def rescale_matches(
    m: np.ndarray,
    src_h: int,
    src_w: int,
    tar_h: int,
    tar_w: int,
) -> np.ndarray:
    """
    Invert the *scale-to-cover + centre-crop* transform on 2-D keypoints.

    Given keypoints expressed in the **cropped** coordinate frame (the output
    of :class:`Resize` with ``keep_aspect_ratio=True``), this function maps
    them back to the **original** image coordinate frame.

    Args:
        m:     ``(N, 2)`` float array of ``[x, y]`` keypoint coordinates in
               the cropped/resized frame.
        src_h: Height of the original (pre-resize) image.
        src_w: Width  of the original (pre-resize) image.
        tar_h: Height of the resize target (i.e. the crop height).
        tar_w: Width  of the resize target (i.e. the crop width).

    Returns:
        ``(N, 2)`` float32 array of keypoints in the original image frame.

    Raises:
        ValueError: If *m* is not a 2-D array with shape ``(N, 2)``.
    """
    if m.ndim != 2 or m.shape[1] != 2:
        raise ValueError(
            f"Expected m to have shape (N, 2), got {m.shape}."
        )

    scale = max(tar_w / src_w, tar_h / src_h)

    scaled_w = int(round(src_w * scale))
    scaled_h = int(round(src_h * scale))

    offset_x = (scaled_w - tar_w) // 2
    offset_y = (scaled_h - tar_h) // 2

    m_orig = m.astype(np.float32, copy=True)
    m_orig[:, 0] = (m[:, 0] + offset_x) / scale  # x
    m_orig[:, 1] = (m[:, 1] + offset_y) / scale  # y

    return m_orig

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

_SizeArg = Union[int, tuple[int, int]]
_Data    = Union[np.ndarray, torch.Tensor]

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class UniTransform(ABC):
    """
    Abstract base class for all REALM spatiotemporal transforms.

    Every concrete transform must declare:

    * :meth:`type` — which modality it operates on (image / event / both).
    * :meth:`__call__` — the transform logic.

    Target-type constants are exposed as class attributes so subclasses
    can refer to them without repeating the string literals.
    """

    #: Map from modality name to integer tag used internally.
    TYPES: dict[str, int] = {"event": 1, "image": 2, "both": 3}

    @abstractmethod
    def type(self) -> int:
        """Return the integer modality tag for this transform."""

    @abstractmethod
    def __call__(self, data: _Data) -> _Data:
        """Apply the transform to *data* and return the result."""

    def __repr__(self) -> str:
        modality = {v: k for k, v in self.TYPES.items()}.get(self.type(), "?")
        return f"{self.__class__.__name__}(target={modality!r})"


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------

def _cover_dims(src_h: int, src_w: int, tgt_h: int, tgt_w: int) -> tuple[int, int]:
    """
    Compute the scaled dimensions for a *scale-to-cover* resize.

    The scale factor is the *larger* of the two axis ratios so both target
    dimensions are fully covered before cropping.

    Returns:
        ``(new_h, new_w)`` — integer dimensions after scaling.
    """
    scale = max(tgt_h / src_h, tgt_w / src_w)
    return int(round(src_h * scale)), int(round(src_w * scale))


def _intrinsics_to_K(intrinsics: np.ndarray) -> np.ndarray:
    """Build a 3×3 camera matrix from a ``[fx, fy, cx, cy]`` array."""
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


# ---------------------------------------------------------------------------
# WarperM3ED
# ---------------------------------------------------------------------------

class WarperM3ED(UniTransform):
    """
    Warp RGB images to the event-camera frame using M3ED HDF5 calibration.

    Operates on image batches of shape ``(T, H, W, C)`` or ``(T, H, W)``.

    Args:
        h5_path: Path to an M3ED HDF5 file containing calibration groups
                 ``/prophesee/left/calib`` (event camera) and
                 ``/ovc/rgb/calib`` (RGB camera).

    Raises:
        FileNotFoundError: If *h5_path* does not exist.
        KeyError:          If required calibration groups are absent.
    """

    def __init__(self, h5_path: str) -> None:
        import h5py

        path = str(h5_path)
        try:
            with h5py.File(path, "r") as f:
                source_map, target_inv_map = self._load_remapping(
                    f["/prophesee/left/calib"],
                    f["/ovc/rgb/calib"],
                )
        except OSError as exc:
            raise FileNotFoundError(
                f"WarperM3ED: cannot open HDF5 file: {path}"
            ) from exc

        self.remap_grid, self.remap_mask = self._create_full_map_and_mask(
            source_map, target_inv_map
        )
        logger.debug("WarperM3ED: remap grid shape=%s", self.remap_grid.shape)

    # ------------------------------------------------------------------

    def _load_remapping(
        self,
        target_group: Any,
        source_group: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load calibration and compute the source→target remap."""
        target_T = target_group["T_to_prophesee_left"][...]
        source_T = source_group["T_to_prophesee_left"][...]

        source_T_target = source_T @ np.linalg.inv(target_T)
        target_T_source = np.linalg.inv(source_T_target)

        # Event-camera (target) parameters
        target_K    = _intrinsics_to_K(target_group["intrinsics"][...])
        target_dist = target_group["distortion_coeffs"][...]
        target_size = tuple(target_group["resolution"][...].astype(int))
        target_P    = np.zeros((3, 4))
        target_P[:3, :3] = target_K
        target_R    = target_T_source[:3, :3]

        # RGB-camera (source) parameters
        source_K    = _intrinsics_to_K(source_group["intrinsics"][...])
        source_dist = source_group["distortion_coeffs"][...]
        source_size = tuple(source_group["resolution"][...].astype(int))
        source_P    = np.zeros((3, 4))
        source_P[:3, :3] = target_K
        source_P[0, 3]   = target_K[0, 0] * target_T_source[0, 3]
        source_P[1, 3]   = target_K[1, 1] * target_T_source[1, 3]

        map_target = np.stack(
            cv2.initUndistortRectifyMap(
                target_K, target_dist, target_R, target_P,
                target_size, cv2.CV_32FC1,
            ),
            axis=-1,
        )
        map_source = np.stack(
            cv2.initUndistortRectifyMap(
                source_K, source_dist, np.eye(3), source_P,
                source_size, cv2.CV_32FC1,
            ),
            axis=-1,
        )
        return map_source, self._invert_map(map_target)

    @staticmethod
    def _invert_map(F: np.ndarray, iterations: int = 10) -> np.ndarray:
        """Invert a dense remap grid by iterative refinement."""
        h, w = F.shape[:2]
        I = np.zeros_like(F)
        I[:, :, 1], I[:, :, 0] = np.indices((h, w))
        P = I.copy()
        for _ in range(iterations):
            P += (I - cv2.remap(F, P, None, cv2.INTER_LINEAR)) * 0.5
        return P

    def _create_full_map_and_mask(
        self,
        source_map: np.ndarray,
        target_inv_map: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compose source and inverted-target maps; build an invalid-pixel mask."""
        composed = cv2.remap(source_map, target_inv_map, None, cv2.INTER_LINEAR)
        masked   = cv2.remap(
            source_map, target_inv_map, None,
            cv2.INTER_LINEAR, borderValue=-1,
        )
        invalid_mask = masked[:, :, 0] == -1
        return composed, invalid_mask

    def type(self) -> int:
        return self.TYPES["image"]

    def __call__(self, img_batch: np.ndarray) -> np.ndarray:
        """
        Remap a batch of RGB images to the event-camera frame.

        Args:
            img_batch: ``(T, H, W, C)`` or ``(T, H, W)`` uint8/float array.

        Returns:
            Remapped array of the same shape and dtype.
        """
        out = []
        mx, my = self.remap_grid[:, :, 0], self.remap_grid[:, :, 1]
        for frame in img_batch:
            remapped = cv2.remap(frame, mx, my, cv2.INTER_LINEAR)
            remapped[self.remap_mask] = 0
            if remapped.ndim == 2:
                remapped = remapped[:, :, np.newaxis]
            out.append(remapped)
        return np.stack(out)


# ---------------------------------------------------------------------------
# Warper
# ---------------------------------------------------------------------------

class Warper(UniTransform):
    """
    Warp RGB images to the event-camera frame using a YAML calibration file.

    Operates on image batches of shape ``(T, H, W, C)`` or ``(T, H, W)``.

    Args:
        calib_path: Path to a YAML calibration file with ``cam0`` (RGB) and
                    ``cam1`` (event) sections, each containing ``intrinsics``,
                    ``distortion_coeffs``, ``resolution``, and optionally
                    ``T_cn_cnm1``.

    Raises:
        FileNotFoundError: If *calib_path* does not exist.
        KeyError:          If required calibration keys are absent.
    """

    def __init__(self, calib_path: str) -> None:
        from pathlib import Path

        path = Path(calib_path)
        if not path.is_file():
            raise FileNotFoundError(f"Warper: calibration file not found: {path}")

        with path.open("r") as fh:
            calib = yaml.safe_load(fh)

        self._cam0 = self._parse_cam(calib["cam0"])
        self._cam1 = self._parse_cam(calib["cam1"])
        self._maps = self._build_maps()
        logger.debug("Warper: remap maps built for resolution %s.", self._cam1["res"])

    # ------------------------------------------------------------------

    @staticmethod
    def _parse_cam(c: dict) -> dict:
        K = _intrinsics_to_K(np.asarray(c["intrinsics"], dtype=np.float64))
        T = np.asarray(c.get("T_cn_cnm1", np.eye(4)), dtype=np.float64)
        return {
            "K":   K,
            "D":   np.asarray(c["distortion_coeffs"], dtype=np.float64),
            "R":   T[:3, :3],
            "res": tuple(int(v) for v in c["resolution"]),
        }

    def _build_maps(self) -> dict[str, np.ndarray]:
        """Build pixel-lookup maps to warp RGB → event frame."""
        ev_w, ev_h = self._cam1["res"]
        grid_x, grid_y = np.meshgrid(
            np.arange(ev_w, dtype=np.float32),
            np.arange(ev_h, dtype=np.float32),
        )
        pts_ev = cv2.undistortPoints(
            np.dstack((grid_x, grid_y)).reshape(-1, 1, 2),
            self._cam1["K"],
            self._cam1["D"],
            R=None,
            P=None,
        )
        R_rel = self._cam0["R"] @ self._cam1["R"].T
        pts_3d = (
            R_rel
            @ cv2.convertPointsToHomogeneous(pts_ev)
            .reshape(-1, 3)
            .T
        ).T.reshape(-1, 1, 3)
        pts_rgb, _ = cv2.projectPoints(
            pts_3d,
            np.zeros(3),
            np.zeros(3),
            self._cam0["K"],
            self._cam0["D"],
        )
        return {
            "mx": pts_rgb[:, 0, 0].reshape(ev_h, ev_w).astype(np.float32),
            "my": pts_rgb[:, 0, 1].reshape(ev_h, ev_w).astype(np.float32),
        }

    def type(self) -> int:
        return self.TYPES["image"]

    def __call__(self, img_batch: np.ndarray) -> np.ndarray:
        """
        Warp a batch of RGB images to the event-camera frame.

        Args:
            img_batch: ``(T, H, W, C)`` or ``(T, H, W)`` array.

        Returns:
            Warped array of the same shape.
        """
        mx, my = self._maps["mx"], self._maps["my"]
        return np.stack([
            cv2.remap(
                frame, mx, my,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            for frame in img_batch
        ])


# ---------------------------------------------------------------------------
# Flip
# ---------------------------------------------------------------------------

class Flip(UniTransform):
    """
    Horizontal and/or vertical flip.

    Supports both image batches ``(T, H, W, C)`` (NumPy) and event voxel
    batches ``(T, C, H, W)`` (PyTorch).

    Args:
        horizontal: If ``True``, flip left-right.
        vertical:   If ``True``, flip top-bottom.
        target:     Modality — ``"image"``, ``"event"``, or ``"both"``.

    Raises:
        ValueError: If *target* is not a recognised modality string.
    """

    def __init__(
        self,
        horizontal: bool = False,
        vertical:   bool = False,
        target:     str  = "both",
    ) -> None:
        if target not in self.TYPES:
            raise ValueError(
                f"Unknown target {target!r}. Choose from {list(self.TYPES)}."
            )
        self.horizontal  = horizontal
        self.vertical    = vertical
        self.target_type = target

    def type(self) -> int:
        return self.TYPES[self.target_type]

    def __call__(self, data: _Data) -> _Data:
        if isinstance(data, torch.Tensor):
            if self.horizontal:
                data = TF.hflip(data)
            if self.vertical:
                data = TF.vflip(data)
            return data

        if isinstance(data, np.ndarray):
            for i in range(data.shape[0]):
                if self.horizontal:
                    data[i] = cv2.flip(data[i], 1)
                if self.vertical:
                    data[i] = cv2.flip(data[i], 0)
            return data

        raise TypeError(
            f"Flip: unsupported input type {type(data).__name__}. "
            "Expected np.ndarray or torch.Tensor."
        )


# ---------------------------------------------------------------------------
# Crop
# ---------------------------------------------------------------------------

class Crop(UniTransform):
    """
    Deterministic rectangular crop.

    Args:
        tl_corner: ``(x1, y1)`` top-left pixel coordinate (inclusive).
        br_corner: ``(x2, y2)`` bottom-right pixel coordinate (exclusive).
        target:    Modality — ``"image"``, ``"event"``, or ``"both"``.

    Raises:
        ValueError: If the crop region is degenerate (zero area) or *target*
                    is unrecognised.
    """

    def __init__(
        self,
        tl_corner: tuple[int, int],
        br_corner: tuple[int, int],
        target: str = "both",
    ) -> None:
        if target not in self.TYPES:
            raise ValueError(
                f"Unknown target {target!r}. Choose from {list(self.TYPES)}."
            )
        self.x1, self.y1 = int(tl_corner[0]), int(tl_corner[1])
        self.x2, self.y2 = int(br_corner[0]), int(br_corner[1])
        self.h = self.y2 - self.y1
        self.w = self.x2 - self.x1
        if self.h <= 0 or self.w <= 0:
            raise ValueError(
                f"Crop region has non-positive area: "
                f"tl={tl_corner}, br={br_corner} → (h={self.h}, w={self.w})."
            )
        self.target_type = target

    def type(self) -> int:
        return self.TYPES[self.target_type]

    def __call__(self, data: _Data) -> _Data:
        if isinstance(data, torch.Tensor):
            return TF.crop(data, self.y1, self.x1, self.h, self.w)

        if isinstance(data, np.ndarray):
            # Works for (T, H, W) and (T, H, W, C): trailing dims are implicit
            return data[:, self.y1 : self.y2, self.x1 : self.x2]

        raise TypeError(
            f"Crop: unsupported input type {type(data).__name__}. "
            "Expected np.ndarray or torch.Tensor."
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"tl=({self.x1}, {self.y1}), "
            f"br=({self.x2}, {self.y2}), "
            f"target={list(self.TYPES.keys())[list(self.TYPES.values()).index(self.type())]})"
        )


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------

class Resize(UniTransform):
    """
    Vectorised spatial resize.

    Two modes are supported:

    * ``keep_aspect_ratio=False`` *(default)* — stretches to ``target_size``
      exactly, potentially distorting the image.
    * ``keep_aspect_ratio=True`` — *scale-to-cover*: scales so both dimensions
      are at least as large as the target, then centre-crops the excess.

    Args:
        target_size:       ``(H, W)`` or a single int for a square target.
        target:            Modality — ``"image"``, ``"event"``, or ``"both"``.
        keep_aspect_ratio: Select resize mode (see above).

    Raises:
        ValueError: If *target_size* contains non-positive values or *target*
                    is unrecognised.
    """

    def __init__(
        self,
        target_size:       _SizeArg,
        target:            str  = "both",
        keep_aspect_ratio: bool = False,
    ) -> None:
        if target not in self.TYPES:
            raise ValueError(
                f"Unknown target {target!r}. Choose from {list(self.TYPES)}."
            )
        h, w = (target_size, target_size) if isinstance(target_size, int) else target_size
        if h <= 0 or w <= 0:
            raise ValueError(
                f"target_size must be positive, got ({h}, {w})."
            )
        self.h = h
        self.w = w
        self.target_type      = target
        self.keep_aspect_ratio = keep_aspect_ratio

    def type(self) -> int:
        return self.TYPES[self.target_type]

    def __call__(self, data: _Data, interpolation: int = cv2.INTER_LINEAR) -> _Data:
        if isinstance(data, np.ndarray):
            return self._resize_numpy(data, interpolation)
        if isinstance(data, torch.Tensor):
            return self._resize_tensor(data)
        raise TypeError(
            f"Resize: unsupported input type {type(data).__name__}. "
            "Expected np.ndarray or torch.Tensor."
        )

    def _resize_numpy(self, data: np.ndarray, interpolation: int) -> np.ndarray:
        src_h, src_w = data.shape[1], data.shape[2]
        if not self.keep_aspect_ratio:
            return np.stack([
                cv2.resize(frame, (self.w, self.h), interpolation=interpolation)
                for frame in data
            ])
        new_h, new_w = _cover_dims(src_h, src_w, self.h, self.w)
        y0 = (new_h - self.h) // 2
        x0 = (new_w - self.w) // 2
        return np.stack([
            cv2.resize(frame, (new_w, new_h), interpolation=interpolation)[
                y0 : y0 + self.h, x0 : x0 + self.w
            ]
            for frame in data
        ])

    def _resize_tensor(self, data: torch.Tensor) -> torch.Tensor:
        src_h, src_w = data.shape[-2], data.shape[-1]
        if not self.keep_aspect_ratio:
            return TF.resize(
                data, [self.h, self.w],
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )
        new_h, new_w = _cover_dims(src_h, src_w, self.h, self.w)
        scaled = TF.resize(
            data, [new_h, new_w],
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True,
        )
        return TF.center_crop(scaled, [self.h, self.w])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"target=({self.h}, {self.w}), "
            f"keep_aspect_ratio={self.keep_aspect_ratio})"
        )


# ---------------------------------------------------------------------------
# ResizeAndCropRandom
# ---------------------------------------------------------------------------

class ResizeAndCropRandom(UniTransform):
    """
    Scale-to-cover resize followed by a temporally-consistent random crop.

    When ``target="both"``, the crop coordinates are generated once on the
    first call (expected to be the image/NumPy call) and reused on the second
    call (the event/Tensor call) to guarantee spatial alignment.  The state
    is cleared after every image call so a fresh crop is drawn each sample.

    Args:
        target_size: ``(H, W)`` or a single int for a square target.
        target:      Modality — ``"image"``, ``"event"``, or ``"both"``.
        seed:        Optional fixed random seed for reproducible crops.

    Raises:
        ValueError: If *target_size* contains non-positive values or *target*
                    is unrecognised.
    """

    def __init__(
        self,
        target_size: _SizeArg,
        target:      str           = "both",
        seed:        Optional[int] = None,
    ) -> None:
        if target not in self.TYPES:
            raise ValueError(
                f"Unknown target {target!r}. Choose from {list(self.TYPES)}."
            )
        h, w = (target_size, target_size) if isinstance(target_size, int) else target_size
        if h <= 0 or w <= 0:
            raise ValueError(
                f"target_size must be positive, got ({h}, {w})."
            )
        self.h = h
        self.w = w
        self.target_type = target
        self._rng = random.Random(seed)

        # State: crop params shared between the image and event calls
        self._params: Optional[tuple[int, int, int, int]] = None

    def type(self) -> int:
        return self.TYPES[self.target_type]

    def _compute_params(self, src_h: int, src_w: int) -> tuple[int, int, int, int]:
        """Compute and cache scale + crop parameters for the current sample."""
        new_h, new_w = _cover_dims(src_h, src_w, self.h, self.w)
        top  = self._rng.randint(0, max(new_h - self.h, 0))
        left = self._rng.randint(0, max(new_w - self.w, 0))
        return new_h, new_w, top, left

    def __call__(self, data: _Data) -> _Data:
        is_numpy = isinstance(data, np.ndarray)
        is_tensor = isinstance(data, torch.Tensor)

        if not is_numpy and not is_tensor:
            raise TypeError(
                f"ResizeAndCropRandom: unsupported input type {type(data).__name__}. "
                "Expected np.ndarray or torch.Tensor."
            )

        src_h = data.shape[1] if is_numpy else data.shape[-2]
        src_w = data.shape[2] if is_numpy else data.shape[-1]

        # Generate new params on the image pass (or in non-"both" mode)
        # and reuse them on the event pass.
        if is_numpy or self.target_type != "both":
            self._params = self._compute_params(src_h, src_w)
        elif self._params is None:
            # Safety fallback: event call received without a prior image call
            logger.warning(
                "ResizeAndCropRandom: event call received before image call "
                "— generating independent crop params."
            )
            self._params = self._compute_params(src_h, src_w)

        new_h, new_w, top, left = self._params

        # --- NumPy (images) ---
        if is_numpy:
            resized = np.stack([
                cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                for frame in data
            ])
            cropped = resized[:, top : top + self.h, left : left + self.w]
            # Clear state after the image pass so it is refreshed next sample
            self._params = None
            return cropped

        # --- Tensor (events) ---
        resized = TF.resize(
            data, [new_h, new_w],
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True,
        )
        return TF.crop(resized, top, left, self.h, self.w)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"target=({self.h}, {self.w}))"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

#: Registry mapping config type strings to transform classes.
_TRANSFORM_REGISTRY: dict[str, type[UniTransform]] = {
    "warpm3ed":              WarperM3ED,
    "warp":                  Warper,
    "flip":                  Flip,
    "crop":                  Crop,
    "custom_crop":           Crop,
    "resize":                Resize,
    "resize_and_crop_random": ResizeAndCropRandom,
}


def build_transforms(cfg: Optional[list[dict]]) -> list[UniTransform]:
    """
    Construct a list of :class:`UniTransform` instances from a config.

    Each entry in *cfg* must be a dict with at least a ``"type"`` key.
    The optional ``"params"`` key may be a dict (forwarded as ``**kwargs``)
    or any other value (forwarded as a single positional argument).

    Args:
        cfg: List of transform config dicts, or ``None`` / empty list.

    Returns:
        Ordered list of instantiated transforms.

    Raises:
        ValueError: If a ``"type"`` key is missing or unrecognised.
        TypeError:  If a transform constructor rejects its parameters.

    Example YAML equivalent::

        transforms:
          - type: resize
            params:
              target_size: 448
              keep_aspect_ratio: true
          - type: flip
            params:
              horizontal: true
    """
    if not cfg:
        return []

    transforms: list[UniTransform] = []
    for entry in cfg:
        raw_type = entry.get("type")
        if not raw_type:
            raise ValueError(
                f"Transform entry is missing a 'type' key: {entry}"
            )
        t_key = str(raw_type).strip().lower()
        if t_key not in _TRANSFORM_REGISTRY:
            raise ValueError(
                f"Unknown transform type: {raw_type!r}. "
                f"Available: {sorted(_TRANSFORM_REGISTRY)}."
            )
        cls    = _TRANSFORM_REGISTRY[t_key]
        params = entry.get("params")
        try:
            if isinstance(params, dict):
                instance = cls(**params)
            elif params is not None:
                instance = cls(params)
            else:
                instance = cls()
        except TypeError as exc:
            raise TypeError(
                f"Failed to instantiate {cls.__name__} with params={params!r}: {exc}"
            ) from exc

        transforms.append(instance)
        logger.debug("Registered transform: %s", instance)

    return transforms


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------

def is_resize(transform: UniTransform) -> bool:
    """Return ``True`` if *transform* is a :class:`Resize` instance."""
    return isinstance(transform, Resize)