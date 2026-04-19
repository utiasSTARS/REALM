import sys

import torch


class Mast3rEncoder(torch.nn.Module):
    """
    PyTorch module to keep only the vision encoder of a Mast3r model.
    The original class definition is here:
    https://github.com/naver/dust3r/blob/9869e71f9165aa53c53ec0979cea1122a569ade4/dust3r/model.py#L46
    If renorm_images is True, it is assumed that the input images are already normalized
    by the ImageNet normalization, and they will be re-normalized by the Mast3r normalization.
    """

    # default values for Mast3r normalization follow Dust3r, here:
    # https://github.com/naver/dust3r/blob/9869e71f9165aa53c53ec0979cea1122a569ade4/dust3r/utils/image.py#L23
    _mast3r_image_mean = (0.5, 0.5, 0.5)
    _mast3r_image_std = (0.5, 0.5, 0.5)
    _imagenet_image_mean = (0.485, 0.456, 0.406)
    _imagenet_image_std = (0.229, 0.224, 0.225)

    def __init__(self, code_dir, ckpt_path, renorm_images=True):
        super().__init__()

        sys.path.insert(0, code_dir)
        from mast3r.model import AsymmetricMASt3R

        self.model = AsymmetricMASt3R.from_pretrained(ckpt_path)
        self.renorm_images = renorm_images
        self.num_features = self.model.enc_embed_dim
        self.patch_size = self.model.patch_embed.patch_size[0]

        self.register_buffer(
            "master_image_mean",
            torch.tensor(self._mast3r_image_mean).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "master_image_std",
            torch.tensor(self._mast3r_image_std).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_image_mean",
            torch.tensor(self._imagenet_image_mean).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_image_std",
            torch.tensor(self._imagenet_image_std).view(1, 3, 1, 1),
        )

    def extra_repr(self) -> str:
        return "renorm_images={}".format(self.renorm_images)

    def forward_features(self, x):
        if self.renorm_images:
            # revert already-applied ImageNet normalization
            x = x * self.imagenet_image_std + self.imagenet_image_mean
            # apply Mast3r normalization
            x = (x - self.master_image_mean) / self.master_image_std

        x_patchtokens, _, _ = self.model._encode_image(x, true_shape=None)

        x_clstoken = x_patchtokens.mean(dim=1)

        return {
            "x_norm_clstoken": x_clstoken,
            "x_norm_regtokens": None,
            "x_norm_patchtokens": x_patchtokens,
            "x_prenorm": None,
            "masks": None,
        }


def mast3r(code_dir, ckpt_path, **kwargs):
    return Mast3rEncoder(code_dir, ckpt_path)
