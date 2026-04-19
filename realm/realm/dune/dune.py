import os
import logging
from typing import Dict, List, Tuple
from dataclasses import asdict

import torch
import torch.nn as nn

from realm.dune.teachers.forward import get_teacher_outputs
from realm.dune.teachers import TEACHER_CFG

from .options import EncoderOptions, ProjectorOptions
from .encoder import vision_transformer
from .projector.tp import TransformerProjector
from .teacher_norm import TeacherNorm
from .teacher_dropping import TeacherDropping
from .losses import unic_loss
from .model_utils import extra_repr


logger = logging.getLogger(__name__)


class DUNE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        projectors: nn.ModuleDict,
        teacher_norms: nn.ModuleDict,
        apply_last_enc_norm: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.projectors = projectors
        self.teacher_norms = teacher_norms
        self.apply_last_enc_norm = apply_last_enc_norm

    def extra_repr(self) -> str:
        return extra_repr(self)

    @property
    def patch_size(self) -> int:
        return self.encoder.patch_size  # type: ignore

    def get_encoder_output(
        self, image: torch.Tensor, concat_cls_patch=True
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        out_dict = self.encoder(image, apply_norm=self.apply_last_enc_norm)

        enc_out = out_dict
        if concat_cls_patch:
            enc_out = torch.cat(
                [
                    out_dict["x_norm_clstoken"].unsqueeze(1),
                    out_dict["x_norm_patchtokens"],
                ],
                dim=1,
            )

        return enc_out

    def get_projector_output(
        self,
        image: torch.Tensor,
        teacher: str = "dino2reg_vitlarge_14",
        reshape_patch_tokens: bool = True,
        # for compatibility with dense evaluations
        return_cls_token: bool = True,
        return_as_list: bool = False,
    ):

        enc_out = self.get_encoder_output(image)

        proj_out = {
            "cls": (pout := self.projectors[teacher](enc_out))[:, 0],
            "patch": pout[:, 1:],
        }

        cls_token = proj_out["cls"]
        patch_tokens = proj_out["patch"]

        if reshape_patch_tokens:
            B, _, w, h = image.shape
            patch_tokens = (
                patch_tokens.reshape(B, w // self.patch_size, h // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

        if return_cls_token and return_as_list:
            return [cls_token, patch_tokens]
        elif return_cls_token:
            return cls_token, patch_tokens
        elif return_as_list:
            return [patch_tokens]
        else:
            return patch_tokens

    def forward(
        self,
        image: torch.Tensor,
        dset_name: List[str],
        teachers: Dict[str, nn.Module],
        tdrop: TeacherDropping,
        tnorms_ema_mom: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        enc_out = self.get_encoder_output(image)
        proj_out = {
            tname: {"cls": (pout := proj(enc_out))[:, 0], "patch": pout[:, 1:]}
            for tname, proj in self.projectors.items()
        }

        teacher_out = get_teacher_outputs(
            image,
            teachers,
            self.patch_size,
            self.teacher_norms,
            tnorms_ema_mom if self.training else 0.0,
        )

        loss, loss_dict = unic_loss(
            proj_out,
            teacher_out,
            dset_name,
            tdrop,
        )

        return loss, loss_dict


def build_encoder(image_size: int, extra_args: Dict) -> nn.Module:
    args = asdict(EncoderOptions())
    args.update(extra_args)
    args["image_size"] = image_size
    return vision_transformer.get_model(**args)


def build_projector(input_dim: int, output_dim: int, extra_args: Dict) -> nn.Module:
    args = asdict(ProjectorOptions(input_dim, output_dim))
    args.update(extra_args)
    return TransformerProjector(**args)


def load_student_encoder_from_checkpoint(ckpt_fname, ckpt_key="model"):
    assert os.path.isfile(ckpt_fname), "Student checkpoint ({}) not found!".format(
        ckpt_fname
    )
    ckpt = torch.load(ckpt_fname, "cpu", weights_only=False)

    encoder = build_encoder(ckpt["args"].image_size, ckpt["args"].enc_args)

    state_dict = ckpt.get(ckpt_key, ckpt)
    encoder.load_state_dict(
        {
            k.replace("module.", "")
            .replace("_orig_mod.", "")
            .replace("encoder.", ""): v
            for k, v in state_dict.items()
            if "encoder." in k
        }
    )

    iter = ckpt.get("iter", 0)
    logger.info(
        "Loaded student encoder from checkpoint {} trained for {} iterations".format(
            ckpt_fname, iter
        )
    )

    return encoder, iter


def build_student_from_args(args):
    encoder = build_encoder(args.image_size, args.enc_args)
    if not hasattr(args.enc_args, "num_heads"):
        args.proj_args["num_heads"] = encoder.num_heads

    projectors = {}
    teacher_norms = {}

    for tname in args.teachers:
        proj_indim: int = encoder.embed_dim
        proj_outdim: int = TEACHER_CFG[tname]["num_features"]
        proj = build_projector(proj_indim, proj_outdim, args.proj_args)
        projectors[tname] = proj

        teacher_norms[tname] = TeacherNorm(
            TEACHER_CFG[tname]["token_types"], proj_outdim
        )

    projectors = nn.ModuleDict(projectors)
    teacher_norms = nn.ModuleDict(teacher_norms)

    model = DUNE(encoder, projectors, teacher_norms)

    return model


def load_student_from_checkpoint(ckpt_fname, ckpt_key="model"):
    assert os.path.isfile(ckpt_fname), ckpt_fname
    ckpt = torch.load(ckpt_fname, "cpu", weights_only=False)

    model = build_student_from_args(ckpt["args"])

    state_dict = ckpt.get(ckpt_key, ckpt)
    state_dict = {
        k.replace("module.", "").replace("_orig_mod.", ""): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict)

    iter = ckpt.get("iter", 0)

    logger.info(
        "Loaded student from checkpoint {} trained for {} iterations".format(
            ckpt_fname, iter
        )
    )

    return model, iter


def load_dune_encoder_from_checkpoint(*args, **kwargs):
    """
    Loads only the encoder part of the DUNE model from a checkpoint.
    """
    return load_student_encoder_from_checkpoint(*args, **kwargs)


def load_dune_from_checkpoint(*args, **kwargs):
    """
    Loads the complete DUNE model from a checkpoint.
    """
    return load_student_from_checkpoint(*args, **kwargs)
