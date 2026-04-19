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
head_factory.py — Task-head factory for REALM.

Maps the ``type`` key in a head config dictionary to the corresponding
``nn.Module`` subclass and instantiates it with the remaining config keys.

Supported head types
--------------------
* ``"depth"``        → :class:`~realm.heads.depth_head.LinearDepthHead`
* ``"segmentation"`` → :class:`~realm.heads.seg_head.SegHead`
* ``"mast3r"``       → :class:`~realm.heads.mast3r.mast3r_head.Mast3rDecoder`

Usage
-----
    from realm.heads.head_factory import head_factory

    head = head_factory(cfg["head"])
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch.nn as nn

from realm.heads.depth_head import LinearDepthHead
from realm.heads.mast3r.mast3r_head import Mast3rDecoder
from realm.heads.seg_head import SegHead
from realm.utils.log import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_HEAD_REGISTRY: dict[str, type[nn.Module]] = {
    "depth":        LinearDepthHead,
    "segmentation": SegHead,
    "mast3r":       Mast3rDecoder,
}


def available_heads() -> list[str]:
    """Return the sorted list of registered head type names."""
    return sorted(_HEAD_REGISTRY)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def head_factory(config: dict) -> nn.Module:
    """
    Instantiate a task head from a configuration dictionary.

    The dictionary must contain a ``"type"`` key whose value is one of the
    registered head names (case-insensitive). All remaining keys are forwarded
    as keyword arguments to the head constructor.

    The input dictionary is **not mutated** — a shallow copy is taken before
    ``"type"`` is removed.

    Args:
        config: Head configuration dict, e.g.::

                    {"type": "depth", "embed_dim": 768, "num_bins": 15}

    Returns:
        An instantiated :class:`torch.nn.Module` task head.

    Raises:
        TypeError:  If *config* is not a dictionary.
        ValueError: If the ``"type"`` key is missing, empty, or not registered.
        TypeError:  If the head constructor rejects the provided keyword args
                    (re-raised with context identifying the head class).
    """
    if not isinstance(config, dict):
        raise TypeError(
            f"head_factory expects a dict, got {type(config).__name__}."
        )

    # Work on a copy so the caller's dict is not mutated by pop()
    cfg = copy.copy(config)

    raw_type = cfg.pop("type", None)
    if not raw_type:
        raise ValueError(
            "Head config must contain a non-empty 'type' key. "
            f"Available heads: {available_heads()}"
        )

    head_type = raw_type.strip().lower()
    if head_type not in _HEAD_REGISTRY:
        raise ValueError(
            f"Unknown head type: {raw_type!r}. "
            f"Available heads: {available_heads()}"
        )

    head_class = _HEAD_REGISTRY[head_type]
    logger.info("Building head: %s (%s)", head_type, head_class.__name__)

    try:
        head = head_class(**cfg)
    except TypeError as exc:
        raise TypeError(
            f"Failed to instantiate {head_class.__name__} with config {cfg}: {exc}"
        ) from exc

    logger.success("Head built successfully: %s", head_class.__name__)  # type: ignore[attr-defined]
    return head