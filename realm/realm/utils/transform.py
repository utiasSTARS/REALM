"""
transform.py — Spatial transforms for REALM image and event-voxel data.

Transforms operate on two distinct data formats:

* **Images** — ``np.ndarray`` of shape ``(H, W, C)`` or ``(T, H, W, C)``
* **Event voxels** — ``torch.Tensor`` of shape ``(C, H, W)`` or ``(T, C, H, W)``

Public API
----------
    Resize          — stretch or scale-to-cover + centre-crop
    rescale_matches — invert the scale-to-cover transform on 2-D keypoints
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from typing import Union

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

_SizeArg = Union[int, tuple[int, int]]
_Data    = Union[np.ndarray, torch.Tensor]

# ---------------------------------------------------------------------------
# Coordinate helper
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
# Resize transform
# ---------------------------------------------------------------------------

class Resize:
    """
    Spatial resize for images (NumPy) and event voxels (PyTorch).

    Two modes are supported:

    * ``keep_aspect_ratio=False`` *(default)* — stretches the input to
      exactly ``target_size``. Aspect ratio is **not** preserved.
    * ``keep_aspect_ratio=True`` — *scale-to-cover*: the input is scaled
      so that both spatial dimensions are **at least** as large as the
      target, then the excess is removed with a centre crop. Aspect ratio
      **is** preserved and no padding is introduced.

    Args:
        target_size:       Output spatial size as ``(H, W)`` or a single int
                           (applied to both height and width).
        keep_aspect_ratio: Select the resize mode (see above).

    Raises:
        TypeError:  If the input passed to ``__call__`` is neither a NumPy
                    array nor a PyTorch tensor.
        ValueError: If ``target_size`` contains non-positive values.

    Examples
    --------
    Resize a batch of events (tensor) with aspect-ratio preservation::

        resize = Resize((448, 448), keep_aspect_ratio=True)
        events_out = resize(events)   # (T, C, H, W) → (T, C, 448, 448)

    Resize a batch of images (numpy) with stretching::

        resize = Resize(256)
        images_out = resize(images)   # (T, H, W, C) → (T, 256, 256, C)
    """

    def __init__(
        self,
        target_size: _SizeArg,
        keep_aspect_ratio: bool = False,
    ) -> None:
        if isinstance(target_size, int):
            h = w = target_size
        else:
            h, w = target_size

        if h <= 0 or w <= 0:
            raise ValueError(
                f"target_size must contain positive integers, got ({h}, {w})."
            )

        self.h = h
        self.w = w
        self.keep_aspect_ratio = keep_aspect_ratio

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _cover_dims(self, src_h: int, src_w: int) -> tuple[int, int]:
        """
        Compute the intermediate (scaled) dimensions for scale-to-cover.

        The scale factor is chosen as the *larger* of the two axis ratios so
        that both target dimensions are fully covered before cropping.

        Returns:
            ``(new_h, new_w)`` — dimensions after scaling, before cropping.
        """
        scale = max(self.h / src_h, self.w / src_w)
        return int(round(src_h * scale)), int(round(src_w * scale))

    # ------------------------------------------------------------------
    # NumPy path  (images: H×W×C  or  T×H×W×C)
    # ------------------------------------------------------------------

    def _resize_numpy(
        self,
        data: np.ndarray,
        interpolation: int,
    ) -> np.ndarray:
        # Normalise to (T, H, W, C) for uniform processing
        squeezed = data.ndim == 3
        if squeezed:
            data = data[np.newaxis]          # (H, W, C) → (1, H, W, C)

        _, src_h, src_w, *_ = data.shape

        if not self.keep_aspect_ratio:
            result = np.stack([
                cv2.resize(frame, (self.w, self.h), interpolation=interpolation)
                for frame in data
            ])
        else:
            new_h, new_w = self._cover_dims(src_h, src_w)
            y0 = (new_h - self.h) // 2
            x0 = (new_w - self.w) // 2

            result = np.stack([
                cv2.resize(frame, (new_w, new_h), interpolation=interpolation)[
                    y0 : y0 + self.h, x0 : x0 + self.w
                ]
                for frame in data
            ])

        return result.squeeze(0) if squeezed else result

    # ------------------------------------------------------------------
    # Tensor path  (voxels: C×H×W  or  T×C×H×W)
    # ------------------------------------------------------------------

    def _resize_tensor(self, data: torch.Tensor) -> torch.Tensor:
        # Normalise to (T, C, H, W) for uniform processing
        squeezed = data.ndim == 3
        if squeezed:
            data = data.unsqueeze(0)         # (C, H, W) → (1, C, H, W)

        src_h, src_w = data.shape[-2:]

        if not self.keep_aspect_ratio:
            result = TF.resize(
                data,
                [self.h, self.w],
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )
        else:
            new_h, new_w = self._cover_dims(src_h, src_w)
            scaled = TF.resize(
                data,
                [new_h, new_w],
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )
            result = TF.center_crop(scaled, [self.h, self.w])

        return result.squeeze(0) if squeezed else result

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def __call__(
        self,
        data: _Data,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> _Data:
        """
        Apply the resize transform to *data*.

        Args:
            data:          Input array or tensor. See class docstring for
                           accepted shapes.
            interpolation: OpenCV interpolation flag used for NumPy inputs
                           (e.g. ``cv2.INTER_LINEAR``, ``cv2.INTER_NEAREST``).
                           Ignored for tensor inputs, which always use
                           bilinear interpolation with antialiasing.

        Returns:
            Resized array or tensor with the same type and number of
            dimensions as the input.

        Raises:
            TypeError: If *data* is neither a NumPy array nor a PyTorch tensor.
        """
        if isinstance(data, np.ndarray):
            return self._resize_numpy(data, interpolation)
        if isinstance(data, torch.Tensor):
            return self._resize_tensor(data)
        raise TypeError(
            f"Unsupported input type: {type(data).__name__}. "
            "Expected np.ndarray or torch.Tensor."
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"target=({self.h}, {self.w}), "
            f"keep_aspect_ratio={self.keep_aspect_ratio})"
        )