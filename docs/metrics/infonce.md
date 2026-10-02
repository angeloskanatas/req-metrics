# `infonce`

Full-batch InfoNCE loss between two views of the same clips.

- Input: `views`
- Canonical preprocessing: `center+l2`
- Tags: paper-canonical
- Shared cache: none
- Origin: https://arxiv.org/abs/1807.03748
- Cite: `DBLP:journals/corr/abs-1807-03748` (Representation Learning with Contrastive Predictive Coding (2018)); `DBLP:conf/icml/SkeanAZPNLS25` (Layer by Layer: Uncovering Hidden Representations in Language Models (2025))

## Definition, protocol and pitfalls

van den Oord, Li and Vinyals (2018, arXiv:1807.03748), Eq. 4: the
cross-entropy of identifying each clip's second view among all N second
views, with logits the scaled similarities. Rows are centered and
L2-normalized so logits are cosines over the temperature, the
preprocessing of the Skean et al. (2025, ICML) reference implementation and of every
stored result file of Kanatas et al. (2026). Lower loss means the layer is more invariant
to the augmentations relative to clip identity. The bound of van den Oord et al.,
I >= log N - L is reported in nats and as the fraction 1 - L / log N;
for unrelated views the loss exceeds log N by about half the variance of
the scaled similarities, so the bound can be negative. The temperature
is a protocol constant that must be recorded: the
reference implementation uses 0.1, the runs of Kanatas et al. (2026) 0.3. The shared
augmentation chain contained a pitch shift, which confounds this metric
on tonal tasks unless that augmentation is removed.

Args:
    views: Two views, shape (2, N, D).
    temperature: Softmax temperature.
    center: Mean-center each view over clips.
    l2: Scale each row to unit norm.

Returns:
    value: InfoNCE loss in nats.
    extras: mi_lower_bound (1 - L / log N), mi_bound_nats (log N - L).
