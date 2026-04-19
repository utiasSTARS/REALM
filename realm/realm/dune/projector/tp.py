from functools import partial

import torch
import torch.nn as nn

from ..common.block import Block


class TransformerProjector(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_blocks: int = 1,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        layerscale_init: float = 1e-4,
        ln_affine: bool = True,
        scale: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        assert input_dim > 0, input_dim
        assert output_dim > 0, output_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.scale = nn.Parameter(torch.ones(1).float()) if scale == 0.0 else scale
        norm_layer = partial(nn.LayerNorm, eps=1e-6, elementwise_affine=ln_affine)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=input_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    layerscale_init=layerscale_init,
                    norm_layer=norm_layer,
                    **kwargs,
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear = nn.Linear(input_dim, output_dim)

    def extra_repr(self):
        repr = "num_heads={}, scale={}".format(self.num_heads, self.scale)
        return repr

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        x = self.linear(x)

        return self.scale * x
