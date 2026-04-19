import os
import logging
from collections import OrderedDict
from typing import List, Dict

import torch

from .config import TEACHER_CFG


logger = logging.getLogger()


def build_teachers(teacher_names: List[str]) -> Dict[str, torch.nn.Module]:
    teachers = OrderedDict()

    for tname in teacher_names:
        logger.info("Loading teacher '{}'".format(tname))
        teachers[tname] = _build_teacher(tname)

    return teachers


def _build_teacher(name):
    if name not in TEACHER_CFG.keys():
        raise ValueError(
            "Unsupported teacher name: {} (supported ones: {})".format(
                name, TEACHER_CFG.keys()
            )
        )

    ckpt_path = TEACHER_CFG[name]["ckpt_path"]
    ckpt_key = TEACHER_CFG[name]["ckpt_key"]

    if not os.path.exists(ckpt_path):
        raise ValueError("Invalid teacher model path/directory: {}".format(ckpt_path))

    if name.startswith("mast3r"):
        code_dir = TEACHER_CFG[name]["code_dir"]
        model = TEACHER_CFG[name]["loader"](code_dir, ckpt_path)

    else:
        # Teacher models which are loaded from the checkpoint files
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ckpt_key != "" and ckpt_key in state_dict.keys():
            state_dict = state_dict[ckpt_key]

        # dinov2 models require some modifications to the state_dict
        state_dict = _update_state_dict_for_dinov2_models(name, state_dict)

        model_args = {
            "img_size": TEACHER_CFG[name]["image_size"],
            "patch_size": TEACHER_CFG[name]["patch_size"],
        }
        for key in ["init_values", "num_register_tokens"]:
            if key in TEACHER_CFG[name]:
                model_args[key] = TEACHER_CFG[name][key]
        model = TEACHER_CFG[name]["loader"](**model_args)
        model.load_state_dict(state_dict, strict=True)

    model = model.cuda()
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model


def _update_state_dict_for_dinov2_models(tname, state_dict):

    if tname.startswith("multihmr"):
        state_dict = {
            k.replace("backbone.encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("backbone.encoder.")
        }

    # Add the "blocks.0" prefix to the transformer block keys
    state_dict = {k.replace("blocks.", "blocks.0."): v for k, v in state_dict.items()}

    return state_dict


def _test_teachers():
    """
    Load all teachers and test if they can be loaded successfully.
    """

    logging.basicConfig(level=logging.INFO)

    for tname in TEACHER_CFG.keys():
        logger.info("Testing teacher '{}'".format(tname))
        _ = _build_teacher(tname)
        logger.info(" - Teacher '{}' loaded successfully".format(tname))


if __name__ == "__main__":
    _test_teachers()
