# Copyright (C) 2026-present STARS Lab. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

# Copyright (c) 2025 Space and Terrestrial Autonomous Robotic Systems (STARS) Lab,
# University of Toronto Institute for Aerospace Studies (UTIAS).
# All rights reserved.
#
# This software is provided for research and educational purposes only.
# Redistribution and use, with or without modification, are permitted provided
# that this copyright notice and attribution are retained.
#
# Maintainer: Vincenzo Polizzi <vincenzo.polizzi@mail.utoronto.ca>

"""
realm.utils — Shared utilities for the REALM framework.

Sub-modules
-----------
    log        — Coloured logger, convenience helpers, and ASCII logo.
    vis        — Visualisation: match overlays, voxel→RGB, image normalisation.
    transform  — Spatial transforms for images (NumPy) and voxels (PyTorch).

Typical imports
---------------
    from realm.utils import get_logger, log_info, log_success
    from realm.utils import voxel_to_rgb_image, image_to_normalized_tensor
    from realm.utils import Resize, rescale_matches
"""

from realm.utils.log import (
    get_logger,
    log_error,
    log_info,
    log_success,
    log_warn,
    logo,
)
from realm.utils.transforms import (
    UniTransform,
    WarperM3ED,
    Warper,
    Flip,
    Crop,
    Resize,
    ResizeAndCropRandom,
    build_transforms,
    is_resize,
    rescale_matches
)

from realm.utils.vis import (
    VisMast3r,
    image_to_normalized_tensor,
    matches,
    vis_matches,
    voxel_to_rgb_image,
)
from realm.utils.representations import (
    EventRepresentation,
    VoxelGrid,
    EventFrame,
    Tencode,
    ERGO,
    representation_factory,
)



    

__all__ = [
    # log
    "get_logger",
    "log_info",
    "log_warn",
    "log_error",
    "log_success",
    "logo",
    # transform
    "UniTransform",
    "WarperM3ED",
    "Warper",
    "Flip",
    "Crop",
    "Resize",
    "ResizeAndCropRandom",
    "build_transforms",
    "is_resize",
    "rescale_matches",
    # vis
    "matches",
    "vis_matches",
    "VisMast3r",
    "voxel_to_rgb_image",
    "image_to_normalized_tensor",
    # representations
    "EventRepresentation",
    "VoxelGrid",
    "EventFrame",
    "Tencode",
    "ERGO",
    "representation_factory",
]