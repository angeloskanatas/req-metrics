"""Construction and description of augmented views and pitch-shifted copies.

LiDAR, InfoNCE and DiME consume a (q, N, D) stack of views of the same N
samples. What a view is decides what the score means: LiDAR measures
invariance to exactly the perturbations that produced the views (Thilak et
al., 2024, Sec. 3), so a run with the objective's own positive-pair
construction and a run with a shared augmentation chain answer different
questions. Views are built from user-supplied encoder and augmentation
callables, and a ViewSpec describing them is stored with every result.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ViewSpec:
    """How a (q, N, D) view stack was produced; recorded next to every view-based result.

    Attributes:
        source: "objective" when views are the training objective's own positives
            (fresh crop plus the method's augmentation chain, masks redrawn for
            masked-prediction methods); "shared" when one augmentation chain is
            applied to every model, as in the protocol of Kanatas et al. (2026); "crops" when
            views differ only by the random crop.
        augmentations: Names with parameters, e.g. "PitchShift(-4..4 semitones, p=0.5)".
        excluded: Augmentations deliberately left out, e.g. "PitchShift" for tonal tasks.
        q: Number of views per sample.
        seed: Seed of the augmentation draws, if any.
        notes: Anything else needed to reproduce the views.
    """

    source: str
    augmentations: tuple[str, ...] = ()
    excluded: tuple[str, ...] = ()
    q: int = 2
    seed: int | None = None
    notes: str = ""

    def describe(self) -> str:
        """One-line description stored in the records' views field."""
        aug = ", ".join(self.augmentations) or "none"
        exc = f"; excluded: {', '.join(self.excluded)}" if self.excluded else ""
        return f"{self.source}, q={self.q}, augmentations: {aug}{exc}"


def stack_views(*views: Tensor) -> Tensor:
    """Stack q per-view (N, D) tensors, e.g. loaded from disk, into one (q, N, D) tensor."""
    if len(views) < 2:
        raise ValueError("need at least two views")
    shape = tuple(views[0].shape)
    if any(v.ndim != 2 or tuple(v.shape) != shape for v in views):
        raise ValueError("all views must be (N, D) tensors with the same shape")
    return torch.stack(views, dim=0)


@torch.no_grad()
def make_views(
    encode: Callable[[object], Tensor],
    inputs: Sequence,
    augment: Callable[[object, torch.Generator], object],
    q: int,
    *,
    seed: int = 0,
    batch_size: int = 64,
    collate: Callable[[list], object] | None = None,
) -> Tensor:
    """Build a (q, N, D) view stack from encoder and augmentation callables.

    Each of q passes draws a seeded generator, augments every batch with it and encodes it.
    Keep inputs in a fixed order so row i is the same sample in every view. For masked objectives,
    put the mask draw inside `encode` (train mode, zero dropout). For stored embeddings, use
    stack_views.

    Args:
        encode: Maps an augmented batch to (B, D).
        inputs: Indexable samples in a fixed order (list, tensor, dataset).
        augment: Perturbs a batch given a torch.Generator.
        q: Number of views.
        seed: Base seed; pass p uses seed + p.
        batch_size: Samples per encode call.
        collate: Builds a batch from a list of items; default torch.stack for tensors.

    Returns:
        Views (q, N, D), on the encoder's output device.
    """
    if q < 2:
        raise ValueError("q must be >= 2")
    n = len(inputs)
    if collate is None:
        collate = lambda items: torch.stack(items) if torch.is_tensor(items[0]) else items  # noqa: E731
    passes = []
    for p in range(q):
        gen = torch.Generator().manual_seed(seed + p)
        chunks = []
        for start in range(0, n, batch_size):
            batch = collate([inputs[i] for i in range(start, min(start + batch_size, n))])
            z = encode(augment(batch, gen))
            if z.ndim != 2:
                raise ValueError(f"encode must return (B, D), got shape {tuple(z.shape)}")
            chunks.append(z.detach())
        passes.append(torch.cat(chunks, dim=0))
    return torch.stack(passes, dim=0)


@dataclass(frozen=True)
class ShiftSpec:
    """How the pitch-shifted copies for PTE were produced; recorded with every PTE result.

    Attributes:
        method: e.g. "waveform pitch shift (librosa)", "CQT frame crop".
        semitones: The nonzero shifts k, in semitones.
        up_is_positive: True when positive k raised the pitch of the audio (the convention of
            Kanatas et al., 2026, target phase -2 pi omega k / 12); False for the opposite action
            (STONE's CQT crop, positive sign). The PTE estimator assumes True.
        clip_seconds: Clip duration fed to the encoder, if fixed.
        notes: Anything else needed to reproduce the shifted inputs.
    """

    method: str
    semitones: tuple[int, ...]
    up_is_positive: bool = True
    clip_seconds: float | None = None
    notes: str = ""

    def describe(self) -> str:
        """One-line description stored in the records' shifts field."""
        sign = "up" if self.up_is_positive else "down"
        return f"{self.method}; k in {list(self.semitones)} ({sign} for positive k)"


@torch.no_grad()
def make_shifted(
    encode: Callable[[object], Tensor],
    inputs: Sequence,
    shift: Callable[[object, int], object],
    semitones: Sequence[int],
    *,
    batch_size: int = 64,
    collate: Callable[[list], object] | None = None,
) -> dict[int, Tensor]:
    """Encode k-semitone shifted copies of every sample: {k: (N, D)} aligned with the unshifted rows.

    `shift(batch, k) -> batch` is the user's audio pitch shift (deterministic given k);
    `encode(batch) -> (B, D)` the user's layer readout.

    Args:
        encode: Maps a batch to a (B, D) tensor.
        inputs: Indexable samples in a fixed order.
        shift: Applies a k-semitone pitch shift to a batch.
        semitones: Nonzero shifts to produce.
        batch_size: Samples per encode call.
        collate: Builds a batch from a list of items; default torch.stack for tensors, list otherwise.

    Returns:
        Dict from k to (N, D) representations.
    """
    if any(k == 0 for k in semitones) or not semitones:
        raise ValueError("semitones must be nonzero")
    n = len(inputs)
    if collate is None:
        collate = lambda items: torch.stack(items) if torch.is_tensor(items[0]) else items  # noqa: E731
    out: dict[int, Tensor] = {}
    for k in semitones:
        chunks = []
        for start in range(0, n, batch_size):
            batch = collate([inputs[i] for i in range(start, min(start + batch_size, n))])
            z = encode(shift(batch, int(k)))
            if z.ndim != 2:
                raise ValueError(f"encode must return (B, D), got shape {tuple(z.shape)}")
            chunks.append(z.detach())
        out[int(k)] = torch.cat(chunks, dim=0)
    return out
