"""
vector_dataset.py — Standalone VECtor dataset loader.

Supports event-camera (ee), RGB (ii), and mixed (ie) modes.
The event representation is injected via ``representation_factory``
from ``dataloaders.representations`` — the dataset does not own
any conversion logic itself.

Expected sequence directory layout
------------------------------------
<sequence>/
    ├── events.hdf5                                   # raw events
    ├── <name>_left_camera/                           # RGB frames
    │   └── <timestamp>.png
    ├── <name>.gt.txt                                 # ground-truth poses
    ├── left_event_camera_intrinsic_results.yaml      # event-cam calibration
    └── left_regular_camera_intrinsic_results.yaml    # RGB-cam calibration
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np
import torch
import yaml
from natsort import natsorted
from torch.utils.data import Dataset

from realm.utils.representations import EventRepresentation, representation_factory
from dataloaders.eventslicer import EventSlicer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

CALIB_EVENT = "left_event_camera_intrinsic_results.yaml"
CALIB_RGB   = "left_regular_camera_intrinsic_results.yaml"


def load_calibration(calib_path: Path) -> dict:
    """
    Load a VECtor YAML calibration file.

    Returns a dict with keys: K (3×3), D (array), H (int), W (int).
    """
    with calib_path.open() as fh:
        cam = yaml.load(fh, Loader=yaml.FullLoader)

    return {
        "K": np.array(cam["camera_matrix"]["data"],          dtype=np.float64).reshape(3, 3),
        "D": np.array(cam["distortion_coefficients"]["data"], dtype=np.float64),
        "H": int(cam["image_height"]),
        "W": int(cam["image_width"]),
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VECtorDataset(Dataset):
    """
    VECtor dataset loader for event and/or RGB data.

    Args:
        dataset_path: Path to the sequence directory.
        is_ev:        If True, yields event voxels; otherwise yields RGB frames.
        ev_us:        Event window duration in microseconds (event mode only).
        representation: An instantiated ``EventRepresentation`` used to convert
                        raw events to a tensor.  Required when ``is_ev=True``;
                        ignored otherwise.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        is_ev: bool,
        ev_us: float = 33_000.0,
        representation: Optional[EventRepresentation] = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.is_ev        = is_ev
        self.ev_us        = float(ev_us)

        if is_ev and representation is None:
            raise ValueError(
                "An EventRepresentation must be provided when is_ev=True. "
                "Use representation_factory() from dataloaders.representations."
            )
        self.representation = representation

        # Calibration ----------------------------------------------------------
        calib_name = CALIB_EVENT if is_ev else CALIB_RGB
        self.calib_file = self.dataset_path / calib_name
        if not self.calib_file.exists():
            raise FileNotFoundError(f"Calibration file not found: {self.calib_file}")
        self.calibration = load_calibration(self.calib_file)

        # RGB files + timestamps -----------------------------------------------
        img_dirs = [
            d for d in self.dataset_path.iterdir()
            if d.is_dir() and d.name.endswith("left_camera")
        ]
        if not img_dirs:
            raise FileNotFoundError(
                f"No '*left_camera' directory found in {self.dataset_path}"
            )
        self._rgb_files = natsorted(list(img_dirs[0].glob("*.png")))
        if not self._rgb_files:
            raise FileNotFoundError(
                f"No PNG files found in {img_dirs[0]}"
            )

        # Timestamps are encoded in the PNG filenames (seconds as float)
        timestamps_s = np.array(
            [float(f.stem) for f in self._rgb_files], dtype=np.float64
        )
        self._timestamps_us = timestamps_s * 1e6  # → microseconds

        # Event slicer ---------------------------------------------------------
        self._slicer: Optional[EventSlicer] = None
        self._first_ev_us: float = 0.0

        if is_ev:
            h5_files = list(self.dataset_path.glob("*.hdf5"))
            if not h5_files:
                raise FileNotFoundError(
                    f"No .hdf5 event file found in {self.dataset_path}"
                )
            h5 = h5py.File(h5_files[0], "r")
            self._slicer = EventSlicer(h5)
            self._first_ev_us = float(self._slicer.get_start_time_us())

            # Drop timestamps that fall before the first recorded event
            valid = self._timestamps_us >= (self._first_ev_us + self.ev_us)
            if not valid.any():
                raise ValueError(
                    "No RGB timestamps are greater than the first event timestamp "
                    f"({self._first_ev_us:.0f} µs) + ev_us ({self.ev_us:.0f} µs)."
                )
            self._timestamps_us = self._timestamps_us[valid]
            self._rgb_files     = [self._rgb_files[i] for i in np.where(valid)[0]]

        logger.info(
            "VECtorDataset | %s | is_ev=%s | %d frames | ev_us=%.0f",
            self.dataset_path.name, is_ev, len(self), ev_us,
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._timestamps_us)

    def __getitem__(self, idx: int) -> tuple[float, torch.Tensor]:
        """
        Returns:
            (timestamp_seconds, data_tensor)

            data_tensor shape:
                - Event mode:  (C, H, W)  — output of the EventRepresentation
                - Image mode:  (3, H, W)  — float32 RGB in [0, 1]
        """
        t1_us = self._timestamps_us[idx]

        if self.is_ev:
            return t1_us / 1e6, self._load_events(t1_us)
        return t1_us / 1e6, self._load_image(idx)

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _load_events(self, t1_us: float) -> torch.Tensor:
        assert self._slicer is not None
        t0_us = max(t1_us - self.ev_us, self._first_ev_us)
        evs   = self._slicer.get_events(int(t0_us), int(t1_us))

        x   = torch.from_numpy(evs["x"]).to(torch.float64)
        y   = torch.from_numpy(evs["y"]).to(torch.float64)
        pol = torch.from_numpy(evs["p"]).to(torch.float64)
        t   = torch.from_numpy(evs["t"].astype(np.int64)).to(torch.float64)

        return self.representation(x=x, y=y, pol=pol, time=t)

    def _load_image(self, idx: int) -> torch.Tensor:
        bgr = cv2.imread(str(self._rgb_files[idx]))
        if bgr is None:
            raise OSError(f"Failed to read image: {self._rgb_files[idx]}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(rgb).permute(2, 0, 1)  # (3, H, W)


# ---------------------------------------------------------------------------
# Builder (used by validate_sequential.py)
# ---------------------------------------------------------------------------

def build_vector_datasets(
    dataset_path: str | Path,
    mode: str,
    ev_us: float,
    cfg: dict,
) -> tuple[VECtorDataset, Optional[VECtorDataset]]:
    """
    Instantiate VECtorDataset(s) from config and CLI args.

    The event representation is read from ``cfg['data']['representation']``
    and instantiated via ``representation_factory``.

    Args:
        dataset_path: Path to the VECtor sequence directory.
        mode:         'ee' | 'ii' | 'ie'
        ev_us:        Event window in microseconds.
        cfg:          Loaded YAML config dict.

    Returns:
        ``(primary_dataset, anchor_dataset)``

        - ``ee`` / ``ii``: ``anchor_dataset`` is ``None``.
        - ``ie``: ``primary`` is event-mode (query),
                  ``anchor_dataset`` is image-mode (anchor).
    """
    if mode not in ("ee", "ii", "ie"):
        raise ValueError(f"mode must be 'ee', 'ii', or 'ie', got {mode!r}")

    dataset_path = Path(dataset_path)
    data_cfg: dict = cfg.get("data", {})
    rep_cfg:  dict = data_cfg.get("representation", {})

    rep_type  = rep_cfg.get("type",      "voxel_grid")
    channels  = rep_cfg.get("channels",  5)
    normalize = rep_cfg.get("normalize", True)

    # Sensor resolution is fixed by hardware for VECtor
    H, W = 480, 640
    ev_repr = representation_factory(
        rep_type=rep_type, height=H, width=W,
        channels=channels, normalize=normalize,
    )
    logger.info(
        "Event representation: %s (channels=%d, normalize=%s)",
        rep_type, channels, normalize,
    )

    if mode == "ee":
        return VECtorDataset(dataset_path, is_ev=True,  ev_us=ev_us, representation=ev_repr), None
    if mode == "ii":
        return VECtorDataset(dataset_path, is_ev=False, ev_us=ev_us), None

    # ie — event queries, image anchor
    ev_ds  = VECtorDataset(dataset_path, is_ev=True,  ev_us=ev_us, representation=ev_repr)
    img_ds = VECtorDataset(dataset_path, is_ev=False, ev_us=ev_us)
    return ev_ds, img_ds