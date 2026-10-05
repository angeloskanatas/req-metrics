"""Turn token layouts into the inputs the metrics expect.

Frame-sequence encoders (one token per time step) already yield (T, D)
trajectories and (N, D) pooled vectors. Spectrogram-patch encoders yield a
(F, T, D) grid of tokens per sample; the metrics do not know about grids, so the
caller chooses a layout here and the choice is recorded in the pooling label.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def grid_to_trajectory(grid: Tensor, *, time_axis: int = 1, mode: str = "freq_concat") -> Tensor:
    """Time trajectory of an (F, T, D) patch grid: one vector per time step.

    "freq_concat" concatenates the F frequency patches of each time step into (T, F * D), the
    readout of MSM-MAE (Niizumi et al., 2022, arXiv:2204.12260, Sec. 3.3); "freq_mean" averages
    them into (T, D). The result is a valid input for trajectory metrics at the sample level.

    Args:
        grid: One sample's tokens (F, T, D) with time on `time_axis`, class tokens removed.
        time_axis: Which of the first two axes is time.
        mode: "freq_concat" or "freq_mean".

    Returns:
        (T, F * D) or (T, D) tensor, time-ordered.
    """
    if grid.ndim != 3 or time_axis not in (0, 1):
        raise ValueError(f"expected (F, T, D) or (T, F, D) with time_axis in {{0, 1}}, got shape {tuple(grid.shape)}")
    g = grid if time_axis == 0 else grid.transpose(0, 1)  # (T, F, D)
    if mode == "freq_concat":
        return g.reshape(g.shape[0], -1)
    if mode == "freq_mean":
        return g.mean(dim=1)
    raise ValueError(f"unknown mode {mode!r}")


def grid_to_tokens(grid: Tensor) -> Tensor:
    """All patches of a (F, T, D) grid as one (F * T, D) token cloud, for the sample and population levels."""
    if grid.ndim != 3:
        raise ValueError(f"expected (F, T, D), got shape {tuple(grid.shape)}")
    return grid.reshape(-1, grid.shape[-1])


def grid_to_pooled(grid: Tensor, *, mode: str = "gap", freq_chunks: int = 1, time_chunks: int = 1) -> Tensor:
    """One vector per sample from a (F, T, D) patch grid, with the fixed readouts of the audio literature.

    "gap": mean over all patches, (D,). "freq_concat_mean": frequency patches concatenated,
    then the mean over time, (F * D,); the standard global readout of the
    M2D lineage (Niizumi et al., 2022; Riou et al., 2024). "partitioned": the
    grid is average-pooled into freq_chunks x time_chunks blocks that are
    concatenated, (freq_chunks * time_chunks * D,), the region pooling of Gu
    et al. (2026, arXiv:2606.25713); (1, 1) is "gap" and (F, 1) is "freq_concat_mean".
    All three are parameter-free, so metrics computed on them stay label-free.

    Args:
        grid: Tokens of one sample, shape (F, T, D); class tokens removed.
        mode: "gap", "freq_concat_mean" or "partitioned".
        freq_chunks, time_chunks: Block counts for "partitioned".

    Returns:
        Pooled vector.
    """
    if grid.ndim != 3:
        raise ValueError(f"expected (F, T, D), got shape {tuple(grid.shape)}")
    if mode == "gap":
        return grid.mean(dim=(0, 1))
    if mode == "freq_concat_mean":
        return grid.mean(dim=1).reshape(-1)  # mean over time, bands concatenated
    if mode == "partitioned":
        f, t, d = grid.shape
        if not (1 <= freq_chunks <= f and 1 <= time_chunks <= t):
            raise ValueError("chunk counts must lie between 1 and the grid size")
        blocks = torch.nn.functional.adaptive_avg_pool2d(grid.permute(2, 0, 1).unsqueeze(0), (freq_chunks, time_chunks))
        return blocks.squeeze(0).permute(1, 2, 0).reshape(-1)  # (fc, tc, D) -> concatenated
    raise ValueError(f"unknown mode {mode!r}")


def frame_tokens_to_pooled(frames: Tensor, *, mode: str = "mean") -> Tensor:
    """One (D,) vector from a (T, D) frame sequence: "mean" or "max" over time, or the "last" frame.

    Mean pooling is the usual readout of encoders; the final token is the
    readout for causal decoders, whose last position attends to the whole sequence.
    All are parameter-free. Learnable poolers trained with task labels
    (attention pooling, learned layer fusion) are probe-side readouts and are
    not inputs for label-free metrics.
    """
    if frames.ndim != 2:
        raise ValueError(f"expected (T, D), got shape {tuple(frames.shape)}")
    if mode == "mean":
        return frames.mean(dim=0)
    if mode == "max":
        return frames.max(dim=0).values
    if mode == "last":
        return frames[-1]
    raise ValueError(f"unknown mode {mode!r}")


def strip_prefix_tokens(tokens: Tensor, n_prefix: int = 1) -> Tensor:
    """Drop class or register tokens from the front of a (..., L, D) token sequence."""
    if n_prefix < 0 or n_prefix >= tokens.shape[-2]:
        raise ValueError("n_prefix must be smaller than the number of tokens")
    return tokens[..., n_prefix:, :]


def stack_samples(samples: Sequence[Tensor]) -> Tensor:
    """Stack equal-length per-sample tensors into one batch tensor along a new first axis."""
    shapes = {tuple(c.shape) for c in samples}
    if len(shapes) != 1:
        raise ValueError(f"samples have different shapes {sorted(shapes)}; keep them as a list for the sample level")
    return torch.stack(list(samples), dim=0)
