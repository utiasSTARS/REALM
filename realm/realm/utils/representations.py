"""
Event camera representation implementations.

Supported representations:
    - EventFrame   : Red/blue polarity frame
    - VoxelGrid    : Bilinear-interpolated voxel grid
    - Tencode      : 3-channel timestamp encoding
    - ERGO         : Mixed-density event stack (12 channels)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class EventRepresentation(ABC):
    """Base class for all event-camera representations."""

    @abstractmethod
    def convert(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        pol: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert raw events to a tensor representation.

        Args:
            x:    (N,) integer pixel x-coordinates.
            y:    (N,) integer pixel y-coordinates.
            pol:  (N,) polarities in {0, 1} or {-1, 1}.
            time: (N,) timestamps (arbitrary unit, monotonically increasing).

        Returns:
            Tensor of shape (C, H, W).
        """

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        pol: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        return self.convert(x, y, pol, time)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        x: torch.Tensor,
        y: torch.Tensor,
        pol: torch.Tensor,
        time: torch.Tensor,
    ) -> None:
        if not (x.shape == y.shape == pol.shape == time.shape):
            raise ValueError(
                f"All input tensors must have the same shape, got "
                f"x={x.shape}, y={y.shape}, pol={pol.shape}, time={time.shape}"
            )
        if x.ndim != 1:
            raise ValueError(f"Expected 1-D tensors, got ndim={x.ndim}")


# ---------------------------------------------------------------------------
# EventFrame
# ---------------------------------------------------------------------------

class EventFrame(EventRepresentation):
    """
    3-channel (C, H, W) uint8 frame.

    Channel mapping (RGB order after permute):
        - Channel 0 (R): positive polarity pixels → 255
        - Channel 1 (G): unused / zero
        - Channel 2 (B): negative polarity pixels → 255
    """

    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width
        self._blank = torch.zeros((3, height, width), dtype=torch.uint8)

    def convert(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        pol: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        if x.numel() == 0:
            return self._blank.clone()

        self._validate_inputs(x, y, pol, time)

        xs = torch.clamp(x.long(), 0, self.width - 1)
        ys = torch.clamp(y.long(), 0, self.height - 1)
        ps = pol.float() * 2.0 - 1.0  # {0,1} → {-1,+1}

        # Vectorised accumulation via scatter_add
        flat_idx = (ys * self.width + xs).long()
        sum_events = torch.zeros(
            self.height * self.width, dtype=torch.float32, device=x.device
        )
        sum_events.scatter_add_(0, flat_idx, ps)
        sum_events = sum_events.reshape(self.height, self.width)

        img = torch.zeros((3, self.height, self.width), dtype=torch.uint8, device=x.device)

        pos = torch.nonzero(sum_events > 0, as_tuple=False)
        neg = torch.nonzero(sum_events < 0, as_tuple=False)

        if pos.numel() > 0:
            img[0, pos[:, 0], pos[:, 1]] = 255  # R channel → positive
        if neg.numel() > 0:
            img[2, neg[:, 0], neg[:, 1]] = 255  # B channel → negative

        return img


# ---------------------------------------------------------------------------
# VoxelGrid
# ---------------------------------------------------------------------------

class VoxelGrid(EventRepresentation):
    """
    Voxel grid with bilinear interpolation in the time domain.

    Args:
        channels:   Number of temporal bins.
        height:     Sensor height.
        width:      Sensor width.
        normalize:  If True, apply per-voxel z-score normalisation.
    """

    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        normalize: bool = True,
    ) -> None:
        self.channels = channels
        self.height = height
        self.width = width
        self.normalize = normalize
        # kept only for device-agnostic clone; never mutated after init
        self._template = torch.zeros(
            (channels, height, width), dtype=torch.float32, requires_grad=False
        )

    def convert(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        pol: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        if x.numel() == 0:
            return self._template.clone()

        self._validate_inputs(x, y, pol, time)

        C, H, W = self.channels, self.height, self.width
        device = x.device

        # Normalise timestamps to [0, C-1]
        t_min, t_max = time[0], time[-1]
        delta = (t_max - t_min).clamp(min=1e-6)
        t_norm = (C - 1) * (time - t_min) / delta

        xs = x.int()
        ys = y.int()
        ts = t_norm.int()
        value = 2.0 * pol.float() - 1.0  # {0,1} → {-1,+1}

        voxel = torch.zeros(C * H * W, dtype=torch.float32, device=device)

        for dx in (0, 1):
            for dy in (0, 1):
                for dt in (0, 1):
                    xl = xs + dx
                    yl = ys + dy
                    tl = ts + dt

                    valid = (
                        (xl >= 0) & (xl < W) &
                        (yl >= 0) & (yl < H) &
                        (tl >= 0) & (tl < C)
                    )

                    w = (
                        value
                        * (1.0 - (xl.float() - x).abs())
                        * (1.0 - (yl.float() - y).abs())
                        * (1.0 - (tl.float() - t_norm).abs())
                    ).float()

                    idx = (tl.long() * H * W + yl.long() * W + xl.long())[valid]
                    voxel.scatter_add_(0, idx, w[valid])

        voxel = voxel.reshape(C, H, W)

        if self.normalize:
            mask = voxel.nonzero(as_tuple=True)
            if mask[0].numel() > 0:
                mean = voxel[mask].mean()
                std = voxel[mask].std()
                voxel[mask] = (voxel[mask] - mean) / std.clamp(min=1e-6)

        return voxel


# ---------------------------------------------------------------------------
# Tencode
# ---------------------------------------------------------------------------

class Tencode(EventRepresentation):
    """
    3-channel timestamp encoding (C, H, W) in [-1, 1].

    Channel mapping:
        - Channel 0 (R): 255 for positive polarity pixels
        - Channel 1 (G): temporal recency value for all pixels
        - Channel 2 (B): 255 for negative polarity pixels
    """

    def __init__(
        self,
        height: int,
        width: int,
        delta_t: float | None = None,
    ) -> None:
        self.height = height
        self.width = width
        self.delta_t = delta_t
        self._blank = torch.zeros((3, height, width), dtype=torch.float32)

    def convert(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        pol: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        if x.numel() == 0:
            return self._blank.clone()

        self._validate_inputs(x, y, pol, time)
        device = x.device

        xs = torch.clamp(x.long(), 0, self.width - 1)
        ys = torch.clamp(y.long(), 0, self.height - 1)

        t_max = time.max()
        delta = float(self.delta_t) if self.delta_t is not None else float((t_max - time.min()).clamp(min=1e-6))
        temporal_values = torch.clamp(255.0 * (t_max - time) / delta, 0.0, 255.0).float()

        frame = torch.zeros((3, self.height, self.width), dtype=torch.float32, device=device)

        pos_mask = pol > 0
        neg_mask = ~pos_mask

        frame[0].index_put_((ys[pos_mask], xs[pos_mask]), torch.tensor(255.0, device=device))
        frame[2].index_put_((ys[neg_mask], xs[neg_mask]), torch.tensor(255.0, device=device))
        frame[1].index_put_((ys, xs), temporal_values)

        return frame / 127.5 - 1.0  # normalise to [-1, 1]


# ---------------------------------------------------------------------------
# ERGO — Mixed-density event stack
# ---------------------------------------------------------------------------

class _Operations:
    """
    Compute a single event surface by scattering events onto a (H, W) grid.

    Args:
        func:        Feature to compute: 'timestamp' | 'polarity' | 'count' |
                     'timestamp_pos' | 'timestamp_neg' | 'count_pos' | 'count_neg'
        aggregation: Scatter reduction: 'mean' | 'sum' | 'max' | 'variance'
        height:      Grid height.
        width:       Grid width.
    """

    _VALID_FUNCS = frozenset({
        "timestamp", "polarity", "count",
        "timestamp_pos", "timestamp_neg",
        "count_pos", "count_neg",
    })
    _VALID_AGGS = frozenset({"mean", "sum", "max", "variance"})

    def __init__(self, func: str, aggregation: str, height: int, width: int) -> None:
        if func not in self._VALID_FUNCS:
            raise ValueError(f"Unknown func {func!r}. Valid: {self._VALID_FUNCS}")
        if aggregation not in self._VALID_AGGS:
            raise ValueError(f"Unknown aggregation {aggregation!r}. Valid: {self._VALID_AGGS}")
        self.func = func
        self.aggregation = aggregation
        self.height = height
        self.width = width

    def __call__(self, events: np.ndarray) -> np.ndarray:
        # events: (N, 4) columns = [x, y, t, p]
        index = torch.as_tensor(
            events[:, 0] + events[:, 1] * self.width, dtype=torch.int64
        )

        if self.func == "timestamp":
            return self._scatter(torch.as_tensor(events[:, 2], dtype=torch.float32), index)
        if self.func == "polarity":
            return self._scatter(torch.as_tensor(events[:, 3], dtype=torch.float32), index)
        if self.func == "count":
            return self._scatter(torch.ones(len(events), dtype=torch.float32), index)

        # Polarity-filtered variants
        if "pos" in self.func:
            mask = events[:, 3] == 1
        else:
            mask = events[:, 3] == -1
            if mask.sum() == 0:
                mask = events[:, 3] == 0  # fallback for {0,1} polarity encoding

        filtered = events[mask]
        if len(filtered) == 0:
            return np.zeros((self.height, self.width), dtype=np.float32)

        idx_f = torch.as_tensor(
            filtered[:, 0] + filtered[:, 1] * self.width, dtype=torch.int64
        )

        if "timestamp" in self.func:
            src = torch.as_tensor(filtered[:, 2], dtype=torch.float32)
        else:  # count
            src = torch.ones(len(filtered), dtype=torch.float32)

        return self._scatter(src, idx_f)

    # Mapping from our aggregation names to torch.scatter_reduce_ reduce strings
    _REDUCE_MAP = {"sum": "sum", "mean": "mean", "max": "amax"}

    def _scatter(self, src: torch.Tensor, index: torch.Tensor) -> np.ndarray:
        n = self.height * self.width
        if self.aggregation == "variance":
            mean    = self._scatter_reduce(src,      index, n, "mean")
            mean_sq = self._scatter_reduce(src ** 2, index, n, "mean")
            result  = (mean_sq - mean ** 2).reshape(self.height, self.width)
        else:
            reduce = self._REDUCE_MAP[self.aggregation]
            result = self._scatter_reduce(src, index, n, reduce).reshape(self.height, self.width)
        return result.numpy()

    @staticmethod
    def _scatter_reduce(
        src: torch.Tensor,
        index: torch.Tensor,
        dim_size: int,
        reduce: str,
    ) -> torch.Tensor:
        """
        Thin wrapper around scatter_reduce_ (requires PyTorch >= 2.0).

        include_self=False ensures empty buckets stay at 0
        for sum/mean and are not inflated for amax.
        """
        out = torch.zeros(dim_size, dtype=src.dtype)
        return out.scatter_reduce_(0, index, src, reduce=reduce, include_self=False)


class _MixedDensityEventStack:
    """
    Stack events into a multi-channel representation using mixed-density windows.

    Args:
        stack_size:                     Number of output channels.
        height:                         Sensor height.
        width:                          Sensor width.
        indexes_functions_aggregations: Tuple of (window_indices, funcs, aggregations).
        stacking_type:                  'SBN' (by number) or 'SBT' (by time).
    """

    def __init__(
        self,
        stack_size: int,
        height: int,
        width: int,
        indexes_functions_aggregations: tuple,
        stacking_type: str,
    ) -> None:
        if stacking_type not in ("SBN", "SBT"):
            raise ValueError(f"stacking_type must be 'SBN' or 'SBT', got {stacking_type!r}")
        self.stack_size = stack_size
        self.height = height
        self.width = width
        self.ifa = indexes_functions_aggregations
        self.stacking_type = stacking_type

    def stack(
        self,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:
        assert len(x) == len(y) == len(p) == len(t)

        t = t - t.min()
        stacked_events = self._make_stack(x, y, p, t)

        rep = np.zeros((self.height, self.width, self.stack_size), dtype=np.float32)
        for i, event_dict in enumerate(stacked_events):
            surface = next(iter(event_dict.values()))
            rep[:, :, i] = surface

        return rep

    def _create_windows(
        self,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        t: np.ndarray,
    ) -> list[tuple]:
        windows = [(x, y, p, t)]

        if self.stacking_type == "SBN":
            n = x.shape[0] // 3
            for i in range(3):
                sl = slice(i * n, (i + 1) * n)
                windows.append((x[sl], y[sl], p[sl], t[sl]))
            cur_n = len(t)
            xw, yw, pw, tw = x.copy(), y.copy(), p.copy(), t.copy()
            for _ in range(3):
                cur_n //= 2
                xw, yw, pw, tw = xw[cur_n:], yw[cur_n:], pw[cur_n:], tw[cur_n:]
                windows.append((xw, yw, pw, tw))

        elif self.stacking_type == "SBT":
            factor = 1 / 3
            for i in range(3):
                mask = (t >= i * factor) & (t <= (i + 1) * factor)
                windows.append((x[mask], y[mask], p[mask], t[mask]))
            xw, yw, pw, tw = x.copy(), y.copy(), p.copy(), t.copy()
            f = 1.0
            for _ in range(4):
                f /= 2
                mask = tw <= f
                xw, yw, pw, tw = xw[mask], yw[mask], pw[mask], tw[mask]
                windows.append((xw, yw, pw, tw))

        return windows

    def _make_stack(
        self,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        t: np.ndarray,
    ) -> list[dict]:
        t_norm = t - t.min()
        rng = t_norm.max() - t_norm.min()
        t_s = t_norm / max(rng, 1e-6)

        windows = self._create_windows(x, y, p, t_s)
        win_idxs, funcs, aggs = self.ifa
        stacked = []

        for i in range(self.stack_size):
            try:
                result = self._stack_data(
                    *windows[win_idxs[i]],
                    func=funcs[i],
                    aggregation=aggs[i],
                )
            except Exception as exc:
                logger.warning("ERGO channel %d failed (%s); filling with zeros.", i, exc)
                result = {"": np.zeros((self.height, self.width), dtype=np.float32)}
            stacked.append(result)

        return stacked

    def _stack_data(
        self,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        t_s: np.ndarray,
        func: str,
        aggregation: str,
    ) -> dict:
        assert len(x) == len(y) == len(p) == len(t_s)

        events = np.stack([x, y, t_s, p], axis=1)
        surface = _Operations(func, aggregation, self.height, self.width)(events)
        key = f"{func.capitalize()}_{aggregation.capitalize()}"
        return {key: surface}


class ERGO(EventRepresentation):
    """
    12-channel Mixed-Density Event Stack representation.

    Args:
        height:        Sensor height.
        width:         Sensor width.
        stacking_type: 'SBN' (by number of events) or 'SBT' (by time). Default 'SBN'.
    """

    _WINDOW_INDEXES = [0, 3, 2, 6, 5, 6, 2, 5, 1, 0, 4, 1]
    _FUNCTIONS = [
        "polarity", "timestamp_neg", "count_neg", "polarity",
        "count_pos", "count", "timestamp_pos", "count_neg",
        "timestamp_neg", "timestamp_pos", "timestamp", "count",
    ]
    _AGGREGATIONS = [
        "variance", "variance", "mean", "sum",
        "mean", "sum", "mean", "mean",
        "max", "max", "max", "mean",
    ]
    STACK_SIZE = 12

    def __init__(
        self,
        height: int,
        width: int,
        stacking_type: str = "SBN",
    ) -> None:
        self.height = height
        self.width = width
        self._stack = _MixedDensityEventStack(
            stack_size=self.STACK_SIZE,
            height=height,
            width=width,
            indexes_functions_aggregations=(
                self._WINDOW_INDEXES,
                self._FUNCTIONS,
                self._AGGREGATIONS,
            ),
            stacking_type=stacking_type,
        )
        self._blank = torch.zeros((self.STACK_SIZE, height, width), dtype=torch.float32)

    def convert(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        pol: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        if x.numel() == 0:
            return self._blank.clone()

        self._validate_inputs(x, y, pol, time)
        device = x.device

        res = self._stack.stack(
            x=x.cpu().numpy(),
            y=y.cpu().numpy(),
            p=pol.cpu().numpy(),
            t=time.cpu().numpy(),
        )
        # (H, W, C) → (C, H, W)
        return torch.from_numpy(res).permute(2, 0, 1).to(device)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def representation_factory(
    rep_type: str,
    height: int,
    width: int,
    channels: int = 5,
    normalize: bool = True,
    ergo_stacking_type: str = "SBN",
) -> EventRepresentation:
    """
    Instantiate an event representation by name.

    Args:
        rep_type:            One of 'event_frame' | 'voxel_grid' | 'tencode' | 'ergo'.
        height:              Sensor height in pixels.
        width:               Sensor width in pixels.
        channels:            Temporal bins — used only for 'voxel_grid'.
        normalize:           Z-score normalisation — used only for 'voxel_grid'.
        ergo_stacking_type:  'SBN' or 'SBT' — used only for 'ergo'.

    Returns:
        An EventRepresentation instance.
    """
    if rep_type == "event_frame":
        return EventFrame(height, width)
    if rep_type == "voxel_grid":
        return VoxelGrid(channels, height, width, normalize)
    if rep_type == "tencode":
        return Tencode(height, width)
    if rep_type == "ergo":
        return ERGO(height, width, stacking_type=ergo_stacking_type)
    raise ValueError(
        f"Unknown representation type {rep_type!r}. "
        f"Valid types: 'event_frame', 'voxel_grid', 'tencode', 'ergo'."
    )