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
unified_dataset.py — Single-file HDF5 sequence dataset for REALM.

Supports DSEC, MVSEC, and M3ED dataset layouts.  Handles event-based and
RGB frame loading, optional depth / segmentation ground truth, and
per-frame event-activity masks.

Public API
----------
    UnifiedDataset — PyTorch Dataset over a single .h5 sequence file.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import hdf5plugin  # noqa: F401 — registers HDF5 compression filters as a side effect
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from dataloaders.semantic_labels import Id2label_11_Cityscapes, fromIdToTrainId
from realm.utils.transforms import UniTransform, build_transforms, is_resize
from realm.utils import get_logger, representation_factory, image_to_normalized_tensor

__all__ = ["UnifiedDataset"]

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# DSEC segmentation maps are cropped 40 px at the bottom relative to the event
# sensor resolution. We pad with the ignore label (255) to re-align them.
_DSEC_HEIGHT_MARGIN_PX: int = 40

# Default fallback event window when neither n_ev nor ms_ev is specified.
_DEFAULT_MS_EV: float = 33.0

# M3ED: skip the first minute of data (30 Hz × 60 s = 1800 frames; use 1980
# for a small safety margin).
_M3ED_SKIP_INIT_FRAMES: int = 1980


# ---------------------------------------------------------------------------
# Per-worker HDF5 handle store
# ---------------------------------------------------------------------------

@dataclass
class _H5Handles:
    """Container for all HDF5 dataset handles opened by one worker."""
    f:            h5py.File                       = field(default=None)
    dset_img:     Optional[h5py.Dataset]          = field(default=None)
    dset_ev_t:    Optional[h5py.Dataset]          = field(default=None)
    dset_ev_x:    Optional[h5py.Dataset]          = field(default=None)
    dset_ev_y:    Optional[h5py.Dataset]          = field(default=None)
    dset_ev_p:    Optional[h5py.Dataset]          = field(default=None)
    dset_ts:      Optional[h5py.Dataset]          = field(default=None)
    dset_idx:     Optional[h5py.Dataset]          = field(default=None)
    ms_map_idx:   Optional[h5py.Dataset]          = field(default=None)
    dset_img_t:   Optional[h5py.Dataset]          = field(default=None)
    semantic:     Optional[h5py.Dataset]          = field(default=None)
    depth_data:   Optional[h5py.Dataset]          = field(default=None)


# Thread-local storage so each DataLoader worker owns its own file handle.
_tls: threading.local = threading.local()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UnifiedDataset(Dataset):
    """
    Single-sequence HDF5 dataset supporting DSEC, MVSEC, and M3ED layouts.

    Each sample is a tuple of:
    * **events** — ``(C, H, W)`` or ``(T, C, H, W)`` voxel grid tensor.
    * **images** — ``(3, H, W)`` or ``(T, 3, H, W)`` normalised RGB tensor.
    * **label**  — segmentation ``(H, W)`` / depth ``(H, W)`` / token mask
                   ``(N,)``, depending on the active mode.

    Args:
        h5_path:              Path to the ``.h5`` sequence file.
        sequence_length:      Number of consecutive frames per sample.
        events_representation: Event representation type (e.g. ``"voxel_grid"``).
        num_bins:             Temporal bins for the voxel grid.
        normalize_event:      Whether to normalise the event tensor.
        n_ev:                 Fixed number of events per frame (``-1`` = disabled).
        ms_ev:                Event window in milliseconds (``-1`` = disabled).
        target_size:          Output spatial size as ``int`` or ``(W, H)`` tuple.
        stride:               Frame sampling stride.
        skip_init_frames:     Frames to skip at the start of the sequence.
        mask:                 If ``True``, return per-token activity mask.
        transforms:           List of transform config dicts (see
                              :func:`~dataloaders.transforms.build_transforms`).
        segmentation:         If ``True``, load semantic segmentation labels.
        depth:                If ``True``, load metric depth maps.
        map_semantics:        If ``True``, remap raw IDs to train IDs.
        min_n_ev:             Minimum event count; frames below this threshold
                              return blank images with a zeroed mask.
        patch_size:           ViT patch size used for mask token resolution.

    Raises:
        ValueError: If both ``n_ev`` and ``ms_ev`` are positive, if
                    ``target_size`` is invalid, or if the image dataset key
                    is not found in the HDF5 file.
    """

    def __init__(
        self,
        h5_path: str | Path,
        sequence_length: int = 5,
        events_representation: str = "voxel_grid",
        num_bins: int = 5,
        normalize_event: bool = True,
        n_ev: int = -1,
        ms_ev: float = -1,
        target_size: int | tuple[int, int] = 448,
        stride: int = 1,
        skip_init_frames: int = 0,
        mask: bool = False,
        transforms: list | None = None,
        segmentation: bool = False,
        depth: bool = False,
        map_semantics: bool = False,
        min_n_ev: int = 1000,
        patch_size: int = 14,
    ) -> None:

        if n_ev > 0 and ms_ev > 0:
            raise ValueError(
                "Only one of 'n_ev' or 'ms_ev' can be positive at a time."
            )

        self.h5_path         = Path(h5_path)
        self.is_m3ed         = "M3ED" in self.h5_path.parts or "M3ED" in self.h5_path.name
        self.seq_len         = sequence_length
        self.num_bins        = num_bins
        self.normalize_event = normalize_event
        self.segmentation    = segmentation
        self.depth           = depth
        self.map_semantics   = map_semantics
        self.n_ev            = n_ev
        self.ms_ev           = ms_ev
        self.stride          = max(1, stride)
        self.mask            = mask
        self.patch_size      = patch_size

        # Default event window
        if ms_ev <= 0 and n_ev <= 0:
            self.ms_ev = _DEFAULT_MS_EV
            logger.warning(
                "Neither 'ms_ev' nor 'n_ev' specified — defaulting to %.0f ms.",
                _DEFAULT_MS_EV,
            )

        # ------------------------------------------------------------------
        # Transforms
        # ------------------------------------------------------------------
        if self.is_m3ed and transforms:
            # Inject the h5 path into the M3ED warp transform config in-place
            for t in transforms:
                if t.get("type") == "warpm3ed":
                    t.setdefault("params", {})["h5_path"] = str(self.h5_path)
                    break

        raw_transforms = build_transforms(transforms) if transforms else []
        self.img_transforms   = [t for t in raw_transforms if t.type() & UniTransform.TYPES["image"]]
        self.event_transforms = [t for t in raw_transforms if t.type() & UniTransform.TYPES["event"]]

        # ------------------------------------------------------------------
        # HDF5 metadata (read once at construction; handle is then closed)
        # ------------------------------------------------------------------
        with h5py.File(self.h5_path, "r") as f:
            if self.is_m3ed:
                self.skip_init_frames = _M3ED_SKIP_INIT_FRAMES
                self.key_img = "/ovc/rgb/data"
                res = f["/prophesee/left/calib/resolution"][:]
                self.ev_height = int(res[1])
                self.ev_width  = int(res[0])
            else:
                self.skip_init_frames = skip_init_frames
                self.key_img = "images/data"
                if "events" in f and "height" in f["events"].attrs:
                    self.ev_height = int(f["events"].attrs["height"])
                    self.ev_width  = int(f["events"].attrs["width"])
                elif "camera_info" in f:
                    self.ev_height = int(f["camera_info"].attrs["height"])
                    self.ev_width  = int(f["camera_info"].attrs["width"])
                else:
                    logger.warning(
                        "Could not determine sensor resolution from '%s' — "
                        "falling back to 480×640.",
                        self.h5_path,
                    )
                    self.ev_height, self.ev_width = 480, 640

            if self.key_img not in f:
                raise ValueError(
                    f"Image dataset '{self.key_img}' not found in {self.h5_path}."
                )
            self.len_imgs = f[self.key_img].shape[0]

        # ------------------------------------------------------------------
        # Length
        # ------------------------------------------------------------------
        start_idx   = self.skip_init_frames
        end_idx     = self.len_imgs - self.seq_len + 1
        self.length = len(range(start_idx, end_idx, self.stride))

        # ------------------------------------------------------------------
        # Event representation
        # ------------------------------------------------------------------
        self.repr = representation_factory(
            rep_type=events_representation,
            height=self.ev_height,
            width=self.ev_width,
            channels=self.num_bins,
            normalize=self.normalize_event,
        )

        # ------------------------------------------------------------------
        # Target size & pre-built tensors
        # ------------------------------------------------------------------
        if isinstance(target_size, int):
            target_w = target_h = target_size
        elif isinstance(target_size, (tuple, list)) and len(target_size) == 2:
            target_w, target_h = int(target_size[0]), int(target_size[1])
        else:
            raise ValueError(
                "target_size must be an int or a (width, height) tuple, "
                f"got {target_size!r}."
            )

        self.target_h = target_h
        self.target_w = target_w

        num_tokens = (target_h // self.patch_size) * (target_w // self.patch_size)
        self.fallback_mask = torch.ones(num_tokens,  dtype=torch.float32)
        self.zero_mask     = torch.zeros(num_tokens, dtype=torch.float32)

        self.blank_img = torch.zeros(
            (self.seq_len, 3, target_h, target_w), dtype=torch.float32
        )

        # Depth padding to reach 448×448 (only meaningful when depth=True)
        pad_h = (448 - target_h) // 2
        pad_w = (448 - target_w) // 2
        self.depth_padding = (pad_w, 448 - target_w - pad_w, pad_h, 448 - target_h - pad_h)

        self.min_n_ev = min(min_n_ev, n_ev) if n_ev > 0 else min_n_ev

    # ------------------------------------------------------------------
    # Worker-safe HDF5 access
    # ------------------------------------------------------------------

    def _get_handles(self) -> _H5Handles:
        """
        Return the HDF5 handles for the current worker thread.

        Files are opened once per worker and stored in thread-local storage,
        which is safe for ``num_workers > 0`` because each worker process gets
        its own TLS and therefore its own independent file descriptor.
        """
        handles: _H5Handles | None = getattr(_tls, "handles", {}).get(str(self.h5_path))
        if handles is not None:
            return handles

        h = _H5Handles()
        h.f = h5py.File(self.h5_path, "r", swmr=True)

        if self.is_m3ed:
            h.dset_img   = h.f["/ovc/rgb/data"]
            h.dset_ev_t  = h.f["/prophesee/left/t"]
            h.dset_ev_x  = h.f["/prophesee/left/x"]
            h.dset_ev_y  = h.f["/prophesee/left/y"]
            h.dset_ev_p  = h.f["/prophesee/left/p"]
            if self.n_ev > 0:
                h.dset_idx   = h.f["/ovc/ts_map_prophesee_left_t"]
            if self.ms_ev > 0:
                h.ms_map_idx = h.f["/prophesee/left/ms_map_idx"]
                h.dset_img_t = h.f["/ovc/ts"]
        else:
            h.dset_img  = h.f["images/data"]
            h.dset_ts   = h.f["images/timestamps_us"]
            h.dset_ev_t = h.f["events/t"]
            h.dset_ev_x = h.f["events/x"]
            h.dset_ev_y = h.f["events/y"]
            h.dset_ev_p = h.f["events/p"]
            h.dset_idx  = h.f["events/ms_to_idx"]
            if self.segmentation:
                key = "/semantics/data" if "/semantics/data" in h.f else "/semantic/data"
                h.semantic = h.f[key]
            if self.depth:
                h.depth_data = h.f["/depths/data"]

        if not hasattr(_tls, "handles"):
            _tls.handles = {}
        _tls.handles[str(self.h5_path)] = h
        return h

    # ------------------------------------------------------------------
    # Event loading helpers
    # ------------------------------------------------------------------

    def _empty_event_tensor(self) -> Tensor:
        """Return the voxel representation of an empty event window."""
        empty = torch.empty(0, dtype=torch.float32)
        return self.repr(empty, empty, empty, empty)

    def _events_to_tensor(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        p: np.ndarray,
    ) -> Tensor:
        """Normalise timestamps and convert a raw event slice to a voxel tensor."""
        denom  = float(t[-1] - t[0]) + 1e-6
        t_norm = (t.astype(np.float32) - t[0]) / denom
        return self.repr(
            x=torch.from_numpy(x.astype(np.float32)),
            y=torch.from_numpy(y.astype(np.float32)),
            pol=torch.from_numpy(p.astype(np.float32)),
            time=torch.from_numpy(t_norm),
        )

    def _load_events_m3ed(self, h: _H5Handles, index: int) -> tuple[list[Tensor], int]:
        """Load event tensors for one sample from an M3ED file."""
        tensors: list[Tensor] = []
        total_events = 0

        for i in range(self.seq_len):
            curr_idx = index + i
            ev_start = ev_end = 0

            if self.n_ev > 0:
                ev_end   = int(h.dset_idx[curr_idx])
                ev_start = max(0, ev_end - self.n_ev)
            elif self.ms_ev > 0:
                t_ms     = int(h.dset_img_t[curr_idx]) // 1000
                ms_start = max(0, t_ms - int(self.ms_ev))
                max_idx  = h.ms_map_idx.shape[0] - 1
                ev_start = int(h.ms_map_idx[min(ms_start, max_idx)])
                ev_end   = int(h.ms_map_idx[min(t_ms,     max_idx)])

            if ev_end > ev_start:
                x = h.dset_ev_x[ev_start:ev_end]
                y = h.dset_ev_y[ev_start:ev_end]
                t = h.dset_ev_t[ev_start:ev_end]
                p = h.dset_ev_p[ev_start:ev_end]
                total_events += len(t)
                tensors.append(self._events_to_tensor(x, y, t, p))
            else:
                tensors.append(self._empty_event_tensor())

        return tensors, total_events

    def _load_events_standard(self, h: _H5Handles, index: int) -> tuple[list[Tensor], int]:
        """Load event tensors for one sample from a DSEC/MVSEC file."""
        tensors: list[Tensor] = []
        total_events = 0
        ms_len = h.dset_idx.shape[0]

        # Build timestamp array for the window [index-1 … index+seq_len]
        if index == 0:
            raw_ts = h.dset_ts[0:self.seq_len]
            ts_us  = np.concatenate((np.array([0], dtype=raw_ts.dtype), raw_ts))
        else:
            ts_us = h.dset_ts[index - 1 : index + self.seq_len]

        # Extrapolate if the file ends before we have enough timestamps
        expected = self.seq_len + 1
        if len(ts_us) < expected:
            avg_delta = float(np.mean(np.diff(ts_us))) if len(ts_us) > 1 else 33_333.0
            missing   = expected - len(ts_us)
            extra     = ts_us[-1] + avg_delta * np.arange(1, missing + 1)
            ts_us     = np.concatenate([ts_us, extra.astype(ts_us.dtype)])

        for i in range(self.seq_len):
            if self.ms_ev > 0:
                t_end   = ts_us[i + 1]
                t_start = max(0.0, t_end - self.ms_ev * 1_000)
            else:
                t_start = float(ts_us[i])
                t_end   = float(ts_us[i + 1])

            idx_start = max(0, min(int(t_start // 1_000), ms_len - 1))
            idx_end   = max(0, min(int(t_end   // 1_000), ms_len - 1))
            ev_start  = int(h.dset_idx[idx_start])
            ev_end    = int(h.dset_idx[idx_end])

            if self.n_ev > 0 and (ev_end - ev_start) > self.n_ev:
                ev_start = max(0, ev_end - self.n_ev)

            if ev_end > ev_start:
                x = h.dset_ev_x[ev_start:ev_end]
                y = h.dset_ev_y[ev_start:ev_end]
                t = h.dset_ev_t[ev_start:ev_end]
                p = h.dset_ev_p[ev_start:ev_end]
                total_events += x.shape[0]
                tensors.append(self._events_to_tensor(x, y, t, p))
            else:
                tensors.append(self._empty_event_tensor())

        return tensors, total_events

    # ------------------------------------------------------------------
    # Mask generation
    # ------------------------------------------------------------------

    def _gen_mask(self, events_tensor: Tensor) -> Tensor:
        """
        Generate a binary per-token activity mask from an event voxel grid.

        Spatial activity is pooled down to the token resolution with
        ``max_pool2d``, then binarised.

        Args:
            events_tensor: ``(C, H, W)`` voxel grid.

        Returns:
            ``(N,)`` float32 binary mask, where ``N = (H/patch) * (W/patch)``.
        """
        with torch.no_grad():
            activity   = events_tensor.abs().sum(dim=0, keepdim=True)  # (1, H, W)
            token_mask = F.max_pool2d(
                activity,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
            return (token_mask > 0).float().flatten()

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        index = self.skip_init_frames + index * self.stride
        h     = self._get_handles()

        # ---- Images -------------------------------------------------------
        imgs_np: np.ndarray = h.dset_img[index : index + self.seq_len]
        for t in self.img_transforms:
            imgs_np = t(imgs_np)

        # ---- Events -------------------------------------------------------
        if self.is_m3ed:
            event_list, total_ev = self._load_events_m3ed(h, index)
        else:
            event_list, total_ev = self._load_events_standard(h, index)

        events_tensor: Tensor = torch.stack(event_list)  # (T, C, H, W)
        for t in self.event_transforms:
            events_tensor = t(events_tensor)

        # ---- Token mask ---------------------------------------------------
        if self.mask:
            token_mask = torch.stack([
                self._gen_mask(events_tensor[i]) for i in range(events_tensor.shape[0])
            ])
        else:
            token_mask = self.fallback_mask.unsqueeze(0).expand(events_tensor.shape[0], -1)

        # ---- Below-threshold frames → blank ------------------------------
        if total_ev < self.min_n_ev:
            imgs_torch = self.blank_img.clone()
            token_mask = torch.zeros_like(token_mask)
        else:
            imgs_torch = torch.from_numpy(imgs_np).float()  # (T, H, W, C) or (T, H, W)

        # ---- Image tensor normalisation -----------------------------------
        # Ensure shape (T, 3, H, W)
        if imgs_torch.ndim == 3:
            imgs_torch = imgs_torch.unsqueeze(1)            # (T, H, W) → (T, 1, H, W)
        if imgs_torch.shape[1] == 1:
            imgs_torch = imgs_torch.expand(-1, 3, -1, -1)  # grey → RGB

        if self.depth:
            imgs_torch = F.pad(imgs_torch, self.depth_padding, mode="constant", value=0)

        imgs_torch = image_to_normalized_tensor(imgs_torch)

        # ---- Squeeze temporal dim for seq_len == 1 -----------------------
        if self.seq_len == 1:
            events_tensor = events_tensor.squeeze(0)
            token_mask    = token_mask.squeeze(0)
            imgs_torch    = imgs_torch.squeeze(0)

        # ---- Ground-truth labels -----------------------------------------
        if self.segmentation:
            return self._get_segmentation_sample(
                h, index, events_tensor, imgs_torch
            )

        if self.depth:
            return self._get_depth_sample(
                h, index, events_tensor, imgs_torch
            )

        return (
            events_tensor.float().contiguous(),
            imgs_torch.float().contiguous(),
            token_mask.float().contiguous(),
        )

    # ------------------------------------------------------------------
    # Ground-truth helpers
    # ------------------------------------------------------------------

    def _get_segmentation_sample(
        self,
        h:             _H5Handles,
        index:         int,
        events_tensor: Tensor,
        imgs_torch:    Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Load, align, and optionally remap semantic segmentation labels."""
        seg_np: np.ndarray = h.semantic[index : index + self.seq_len]

        # DSEC: pad the bottom to re-align with the event sensor frame
        margin = np.full(
            (seg_np.shape[0], _DSEC_HEIGHT_MARGIN_PX, seg_np.shape[2]),
            fill_value=255,
            dtype=seg_np.dtype,
        )
        seg_np = np.concatenate((seg_np, margin), axis=1)

        for t in self.img_transforms:
            if is_resize(t):
                seg_np = t(seg_np, interpolation=0)  # cv2.INTER_NEAREST
            else:
                seg_np = t(seg_np)

        if self.map_semantics:
            seg_np = fromIdToTrainId(seg_np, Id2label_11_Cityscapes)

        seg = torch.from_numpy(seg_np)
        if seg.ndim == 3 and self.seq_len == 1:
            seg = seg.squeeze(0)

        # Sanity-check label range
        valid = seg[seg != 255]
        if valid.numel() > 0 and valid.max().item() > 10:
            logger.warning(
                "Unexpected segmentation label %d found in %s (index %d). "
                "Max valid label is 10.",
                int(valid.max().item()), self.h5_path.name, index,
            )

        return (
            events_tensor.float().contiguous(),
            imgs_torch.float().contiguous(),
            seg.long().contiguous(),
        )

    def _get_depth_sample(
        self,
        h:             _H5Handles,
        index:         int,
        events_tensor: Tensor,
        imgs_torch:    Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Load and clean metric depth ground truth."""
        depth_np: np.ndarray = h.depth_data[index : index + self.seq_len].copy()

        for i in range(self.seq_len):
            np.nan_to_num(depth_np[i], copy=False, nan=-1.0, posinf=-1.0, neginf=-1.0)

        for t in self.img_transforms:
            depth_np = t(depth_np, interpolation=0)  # cv2.INTER_NEAREST

        depth = torch.from_numpy(depth_np)
        if depth.ndim == 3 and self.seq_len == 1:
            depth = depth.squeeze(0)

        return (
            events_tensor.float().contiguous(),
            imgs_torch.float().contiguous(),
            depth.float().contiguous(),
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"path={self.h5_path.name!r}, "
            f"seq_len={self.seq_len}, "
            f"samples={self.length}, "
            f"is_m3ed={self.is_m3ed})"
        )