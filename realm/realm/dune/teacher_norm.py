from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist



class TeacherNorm(nn.Module):
    def __init__(
        self, token_types: list[str], dim, ema_momentum: float = 0.1, eps: float = 1e-3
    ):
        super().__init__()
        normalizers = {}
        for ttype in token_types:
            assert ttype in ["cls", "patch"], f"Invalid token type: {ttype}"
            agg_dims = [0] if ttype == "cls" else [0, 1]
            normalizers[ttype] = StandardNormalizer(
                dim, agg_dims, ema_momentum=ema_momentum, eps=eps
            )
        self.normalizers = nn.ModuleDict(normalizers)

    def forward(self, x, ttype: str, ema_momentum: Optional[float] = None):
        return self.normalizers[ttype](x, ema_momentum)


class StandardNormalizer(nn.Module):
    def __init__(self, dim, agg_dims, ema_momentum: float = 0.1, eps: float = 1e-3):
        super().__init__()
        self.agg_dims = agg_dims  # which dimensions to aggregate over
        self.ema_momentum = ema_momentum
        self.ema_momentum_last = 0.0  # set automatically, just for logging
        self.eps = eps

        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    def extra_repr(self) -> str:
        repr_str = "eps={}, ema_momentum={:.3f}, ema_momentum_last={:0.3f},\n\tmean={},\n\tstd={}".format(
            self.eps,
            self.ema_momentum,
            self.ema_momentum_last,
            self.mean.data,
            self.std.data,
        )
        return repr_str

    def forward(self, x, ema_momentum: Optional[float] = None):
        assert (
            len(x.shape) == self.agg_dims[-1] + 2
        ), "Data is not compatible with aggregation dims"

        if ema_momentum is None:
            ema_momentum = self.ema_momentum

        self.ema_momentum_last = ema_momentum

        if not self.training or ema_momentum == 0:
            # At inference time, or when not updating the statistics

            with torch.autocast(device_type=x.device.type, dtype=torch.float32):
                x = (x - self.mean) / torch.clamp(self.std, min=self.eps)

        else:
            # During training, update the statistics using EMA.

            # Gather data across all GPUs.
            x_all = concat_all_gather(x.contiguous())

            with torch.autocast(device_type=x.device.type, dtype=torch.float32):
                mean = x_all.mean(dim=self.agg_dims, keepdim=False)
                std = x_all.std(dim=self.agg_dims, keepdim=False)
                x = (x - mean) / torch.clamp(std, min=self.eps)

                self.mean.copy_(self.mean * (1 - ema_momentum) + mean * ema_momentum)
                self.std.copy_(self.std * (1 - ema_momentum) + std * ema_momentum)

        return x


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if not distributed.is_enabled():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(distributed.get_global_size())
    ]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
