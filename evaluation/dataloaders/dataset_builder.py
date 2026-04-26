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
dataloader.py — REALM dataset and DataLoader construction.

Public API
----------
    REALM_Dataset          — multi-dataset PyTorch Dataset backed by HDF5 files.
    create_REALM_dataloader — factory that wraps REALM_Dataset in a DataLoader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from torch.utils.data import ConcatDataset, DataLoader, Dataset

from dataloaders.unified_dataset import UnifiedDataset
from realm.utils.log import get_logger

__all__ = ["REALM_Dataset", "create_REALM_dataloader"]

logger = get_logger(__name__)

# Valid split identifiers accepted by REALM_Dataset.
_Mode = Literal["train", "val", "test"]

# Subdirectory names to probe when the primary split directory is empty.
_SPLIT_FALLBACKS: dict[str, str] = {"val": "test", "test": "val"}


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_REALM_dataloader(
    config: dict,
    mode: _Mode,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int = 4,
) -> DataLoader:
    """
    Build a :class:`torch.utils.data.DataLoader` over a :class:`REALM_Dataset`.

    Args:
        config:          Dataset configuration dictionary (see :class:`REALM_Dataset`).
        mode:            One of ``"train"``, ``"val"``, or ``"test"``.
        batch_size:      Number of samples per batch.
        num_workers:     Number of worker processes for data loading.
        prefetch_factor: Batches to prefetch per worker. Automatically reduced
                         to ``2`` for ``"val"`` and ``"test"`` splits.

    Returns:
        A configured :class:`~torch.utils.data.DataLoader`.
    """
    dataset = REALM_Dataset(config, mode=mode)

    # Reduce prefetch pressure for non-training splits
    if mode in ("val", "test"):
        prefetch_factor = min(prefetch_factor, 2)

    logger.info(
        "DataLoader [%s] — %d samples | batch %d | workers %d",
        mode, len(dataset), batch_size, num_workers,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=(num_workers > 0),
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class REALM_Dataset(Dataset):
    """
    Multi-source dataset loader for REALM, backed by HDF5 files.

    Discovers ``.h5`` sequences under each dataset's root directory,
    instantiates a :class:`~dataloaders.unified_dataset.UnifiedDataset` per
    file, and concatenates them into a single flat dataset.

    Args:
        cfg:  Dataset configuration dictionary. Must contain at minimum:

              * ``"datasets"`` — list of dataset names to load.
              * ``"common"``   — shared options applied to every dataset.
              * One sub-dict per dataset name with dataset-specific overrides.

              Example layout::

                  {
                      "datasets": ["MVSEC", "DSEC"],
                      "common": {"sequence_length": 1, "nr_bins_per_data": 5},
                      "MVSEC": {"root": "./datasets/MVSEC", "depth": True},
                      "DSEC": {"root": "./datasets/DSEC", "segmentation": True},
                  }

        mode: Dataset split — one of ``"train"``, ``"val"``, or ``"test"``.

    Raises:
        ValueError:  If *cfg* is empty or ``"datasets"`` key is missing.
        RuntimeError: If no ``.h5`` files are found across all configured datasets.
    """

    def __init__(self, cfg: dict | None = None, mode: _Mode = "train") -> None:
        if not cfg:
            raise ValueError(
                "Configuration dictionary cannot be empty. "
                "Provide a valid dataset config."
            )

        self._cfg = cfg
        self._mode = mode
        self._datasets: list[Dataset] = []

        datasets_list: list[str] = cfg.get("datasets", [])
        if not datasets_list:
            raise ValueError(
                "Config must contain a non-empty 'datasets' list."
            )

        logger.info("[REALM_Dataset] Loading datasets: %s", datasets_list)

        for ds_name in datasets_list:
            self._load_single_dataset(ds_name)

        if not self._datasets:
            raise RuntimeError(
                "No datasets were loaded. "
                "Check that all configured root paths exist and contain .h5 files."
            )

        self._concat: ConcatDataset = ConcatDataset(self._datasets)
        logger.info(
            "[REALM_Dataset] Ready — %d sequences, %d total samples.",
            len(self._datasets), len(self._concat),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_single_dataset(self, ds_name: str) -> None:
        """Discover HDF5 files for *ds_name* and append UnifiedDataset instances."""
        cfg = self._cfg
        if ds_name not in cfg:
            logger.warning(
                "[REALM_Dataset] '%s' listed in 'datasets' but has no config entry — skipping.",
                ds_name,
            )
            return

        # Merge common options with dataset-specific overrides
        ds_cfg: dict = {**cfg.get("common", {}), **cfg[ds_name]}

        root_path = Path(ds_cfg["root"])
        h5_files = self._find_h5_files(root_path, ds_name, ds_cfg)

        if not h5_files:
            logger.warning(
                "[REALM_Dataset] No .h5 files found for '%s' — skipping.", ds_name
            )
            return

        logger.info("[%s] Found %d sequence(s).", ds_name, len(h5_files))

        for h5_path in h5_files:
            ds = UnifiedDataset(
                h5_path=h5_path,
                sequence_length=ds_cfg.get("sequence_length", 5),
                events_representation=ds_cfg.get("events_representation", "voxel_grid"),
                num_bins=ds_cfg.get("nr_bins_per_data", 5),
                normalize_event=ds_cfg.get("normalize_event", True),
                transforms=ds_cfg.get("transforms", None),
                n_ev=ds_cfg.get("n_ev", -1),
                ms_ev=ds_cfg.get("ms_ev", 20),
                target_size=ds_cfg.get("target_size", 448),
                stride=ds_cfg.get("stride", 1),
                skip_init_frames=ds_cfg.get("skip_init_frames", 0),
                mask=ds_cfg.get("mask", False),
                segmentation=ds_cfg.get("segmentation", False),
                depth=ds_cfg.get("depth", False),
                map_semantics=ds_cfg.get("map_semantics", False),
                min_n_ev=ds_cfg.get("min_n_ev", 1000),
            )
            self._datasets.append(ds)

    def _find_h5_files(
        self,
        root_path: Path,
        ds_name: str,
        ds_cfg: dict,
    ) -> list[Path]:
        """
        Locate ``.h5`` files for the current split, with a val/test fallback.

        The glob pattern is controlled by ``ds_cfg["h5_pattern"]`` if present,
        defaulting to ``"**/*_data.h5"`` for M3ED datasets and ``"**/*.h5"``
        for all others. M3ED detection is opt-in via ``ds_cfg["m3ed"]: true``.

        Args:
            root_path: Dataset root directory.
            ds_name:   Dataset name (used for logging only).
            ds_cfg:    Merged dataset configuration dictionary.

        Returns:
            Sorted list of resolved ``.h5`` paths, or an empty list if none found.
        """
        is_m3ed: bool = ds_cfg.get("m3ed", False)
        pattern: str = ds_cfg.get("h5_pattern", "**/*_data.h5" if is_m3ed else "**/*.h5")

        search_path = self._split_path(root_path, self._mode)
        h5_files = sorted(search_path.glob(pattern))

        # Val / test fallback: some datasets only have one of the two splits
        if not h5_files and self._mode in _SPLIT_FALLBACKS:
            fallback = _SPLIT_FALLBACKS[self._mode]
            fallback_path = self._split_path(root_path, fallback)
            h5_files = sorted(fallback_path.glob(pattern))
            if h5_files:
                logger.warning(
                    "[%s] No files under '%s', falling back to '%s' split.",
                    ds_name, self._mode, fallback,
                )

        return h5_files

    @staticmethod
    def _split_path(root: Path, mode: str) -> Path:
        """Return the subdirectory for *mode*, or *root* if mode is unknown."""
        if mode in ("train", "val", "test"):
            return root / mode
        return root

    def _rebuild_concat(self) -> None:
        """Rebuild the internal ConcatDataset after the dataset list changes."""
        self._concat = ConcatDataset(self._datasets)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_dataset(self, dataset: Dataset) -> None:
        """
        Append an additional dataset and rebuild the concatenation.

        Args:
            dataset: Any :class:`~torch.utils.data.Dataset` instance.
        """
        self._datasets.append(dataset)
        self._rebuild_concat()

    def get_datasets(self) -> list[Dataset]:
        """Return the list of individual datasets currently loaded."""
        return self._datasets

    def get_config(self) -> dict:
        """Return the configuration dictionary used to construct this dataset."""
        return self._cfg

    def __len__(self) -> int:
        return len(self._concat)

    def __getitem__(self, idx: int):
        return self._concat[idx]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"mode={self._mode!r}, "
            f"sequences={len(self._datasets)}, "
            f"samples={len(self)})"
        )