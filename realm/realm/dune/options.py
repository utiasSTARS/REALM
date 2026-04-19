from dataclasses import dataclass


@dataclass
class EncoderOptions:
    """This class encompasses the most important options for specifying an
    encoder module. Most likely, this should contain the details to create
    a ViT-based encoder.
    """

    arch: str = "vit_base"
    image_size: int = 336
    patch_size: int = 14
    num_register_tokens: int = 4
    layerscale_init: float = 0.0001
    qkv_bias: bool = True
    ln_affine: bool = True


@dataclass
class ProjectorOptions:
    """This class encompasses the most important options for specifying a
    projector module.
    """

    input_dim: int
    output_dim: int
    num_blocks: int = 1
    num_heads: int = 12
    layerscale_init: float = 0.0001
    qkv_bias: bool = True
    ln_affine: bool = True
    scale: float = 1.0
