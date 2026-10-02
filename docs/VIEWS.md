# Views for LiDAR, InfoNCE and DiME

The three view-based metrics take a tensor of shape (q, N, D): q views of the
same N clips at one layer. The library does not know how the views were made
and does not need to; but the score is only interpretable together with that
information, so every view-based result carries a `ViewSpec`.

## Two patterns

**Monitoring during training, the objective's own positives.** LiDAR is
defined with respect to the perturbations the objective treats as positives
(Thilak et al., 2024, Section 3). During training the views are produced by
re-running the data pipeline: each pass redraws the random crop and applies
the method's augmentation chain, and for masked-prediction objectives the
token mask is redrawn per view by running the encoder in train mode with
dropout at zero. Objectives without augmentations still yield views that
differ by the crop. In the library this is `make_views(encode, inputs,
augment, q)`, where `encode` is the user's layer readout and `augment` the
user's perturbation; the library owns only the loop and the seeding.
`ViewSpec(source="objective", ...)` records it.

**Post-hoc comparison across models, a shared chain.** When models trained
with different objectives are compared on equal terms, one augmentation
chain is applied to every model's inputs, the per-view pooled embeddings are
stored, and the stack is formed with `stack_views(view_0, view_1, ...)`. The
protocol of Kanatas et al. (2026, Section 3.2) used pitch shifting
by up to four semitones, time stretching by a factor between 0.85 and 1.15,
additive Gaussian noise, gain changes, time shifts and low-pass filtering,
with 10 views per clip for LiDAR and 2 for InfoNCE on 10,000 clips,
and it removed the augmentation that alters the task-defining attribute per
task family: pitch shifts for tonal tasks, time stretching for rhythm tasks.
`ViewSpec(source="shared", excluded=("PitchShift",), ...)` records this. It
departs from LiDAR's original use with each method's own augmentations:
cross-model comparison then measures invariance to a chosen chain rather than
to what each model was trained on.

## What each metric does with the views

- LiDAR: clips are classes, views are within-class samples; the other clips
  act as negatives only through the between-class scatter. Needs N above D.
- InfoNCE: exactly two views; every other clip in the batch is a negative, so
  the loss depends on N and is reported with the 1 - L / log N bound.
- DiME: exactly two views; the baseline is the joint entropy under random
  re-pairing of the second view, so no explicit negatives.

## Fields the record keeps

source, augmentations with parameters, excluded augmentations, q, seed, plus
the metric's own parameters (LiDAR delta and denominators, InfoNCE
temperature, DiME alpha, kernel and permutation count). These fields are
mandatory because a view-based score cannot be compared without them.
