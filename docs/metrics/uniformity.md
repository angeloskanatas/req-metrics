# `uniformity`

Uniformity: log mean pairwise Gaussian potential on the unit sphere.

- Input: `points`
- Canonical preprocessing: `l2`
- Tags: relational
- Shared cache: none
- Origin: https://arxiv.org/abs/2005.10242
- Cite: `wang2020uniformity` (Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere (2020))

## Definition, protocol and pitfalls

Wang and Isola (2020, ICML, arXiv:2005.10242, Sec. 4.1.2): log mean_{i<j} exp(-t ||u_i -
u_j||^2) over L2-normalized rows, t = 2; lower is more uniform. Corollary 1 bounds it below
by -2t + log 0F1(; D/2; t^2). Computed from Gram blocks in float64, O(N^2 D).

Args:
    x: Points (N, D).
    t: Kernel scale.
    center: Mean-center before normalizing.
    chunk: Rows per Gram block.

Returns:
    value: uniformity in nats.
    extras: lower_bound for this D and t, its large-D limit -2t, gap = value - lower_bound.
