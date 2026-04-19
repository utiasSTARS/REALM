import logging
from collections import defaultdict
from typing import Dict, Optional

import torch
import torch.nn as nn

from .config import TEACHER_CFG


logger = logging.getLogger()


def get_teacher_outputs(
    image: torch.Tensor,
    teachers: Dict[str, torch.nn.Module],
    student_patch_size: int,
    tnorms: Optional[nn.ModuleDict],
    tnorm_ema_mom: float = 0.0,
) -> Dict[str, Dict[str, torch.Tensor]]:

    all_tout_dict = defaultdict(dict)

    for tname, tmodel in teachers.items():

        with torch.inference_mode():
            _image = image
            if student_patch_size != tmodel.patch_size:
                # resize image for the teacher model
                # such that spatial size of its output matches that of student
                _image = torch.nn.functional.interpolate(
                    image,
                    scale_factor=tmodel.patch_size / student_patch_size,
                    mode="bicubic",
                    align_corners=True,
                )

            tout_dict = teachers[tname].forward_features(_image)

        with torch.no_grad():
            for ttype in TEACHER_CFG[tname]["token_types"]:
                key = "x_norm_{}{}".format(
                    ttype, "token" if ttype == "cls" else "tokens"
                )
                tout = tout_dict[key]
                if tnorms is not None:
                    tout = tnorms[tname](tout, ttype, tnorm_ema_mom)
                all_tout_dict[tname][ttype] = tout

    return all_tout_dict
