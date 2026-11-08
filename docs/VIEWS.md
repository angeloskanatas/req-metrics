# Views for LiDAR, InfoNCE and DiME

LiDAR, InfoNCE and DiME take a tensor of shape (q, N, D): q views of the same N samples at one layer.
The views are augmented copies of each sample, and the clean input is not one of them (Thilak et al.,
2024, Section 3; Skean et al., 2025, Appendix D). What a score means depends on how the views were
made, so every result records it in a `ViewSpec`.

## Views during training

LiDAR is defined with respect to the perturbations that the training objective treats as positives
(Thilak et al., 2024, Section 3). During training the views come from running the data pipeline
again: each pass draws a new random crop and applies the method's augmentations. For masked-prediction
objectives, each view gets a new token mask by running the encoder in train mode with dropout set to
zero. Objectives without augmentations still give views that differ in their crop.

`make_views(encode, inputs, augment, q)` builds the views with seeded passes, where `encode` reads out
the layer and `augment` is your perturbation. `LayerMonitor` does the same during training and reads
the crops from a live loader. Both draw each view independently. Some objectives draw positives around
a shared anchor, for example crops near one center. For these, use a view loader whose dataset fixes
the anchor of each sample and draws a new offset at each pass, or build the stack yourself. No
published LiDAR protocol says how to combine views of different kinds, such as global and local crops,
in one sample's class. Record these views with `ViewSpec(source="objective", ...)`.

## Views for comparing models

To compare models trained with different objectives on equal terms, apply one augmentation chain to the
inputs of every model, store the pooled embedding of each view, and build the stack with
`stack_views(view_0, view_1, ...)`. Kanatas et al. (2026, Section 2.2) used pitch shifts of up to four
semitones, time stretching by a factor between 0.85 and 1.15, additive Gaussian noise, gain changes,
time shifts and low-pass filtering, with 10 views per clip for LiDAR and 2 for InfoNCE, on 10,000
clips. For each task family they left out the augmentation that changes what the task measures: pitch
shifts for tonal tasks and time stretching for rhythm tasks. `ViewSpec(source="shared",
excluded=("PitchShift",), ...)` records this. With a shared chain, the score measures invariance to
that chain rather than to the augmentations each model was trained with, which is not how LiDAR was
originally used.

## How each metric uses the views

- LiDAR treats each sample as a class and its views as points of that class. The other samples enter
  only through the between-class scatter. It needs N larger than D.
- InfoNCE uses two views, or the mean over view pairs when q > 2. Every other sample is a negative, so
  the loss depends on N and on the temperature. The extras hold log N - L and the contrastive accuracy.
- DiME needs exactly two views. Its baseline is the joint entropy after randomly re-pairing the second
  view, so it needs no explicit negatives.

## What the record keeps

The source, the augmentations and their parameters, the excluded augmentations, q, the seed, and the
metric's own parameters (LiDAR's delta and denominators, the InfoNCE temperature, DiME's alpha, kernel
and number of permutations). Two view-based scores cannot be compared without these fields.
