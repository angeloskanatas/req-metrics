# `uniformity`

Uniformity: log mean pairwise Gaussian potential on the unit sphere.

- Input: `points`
- Canonical preprocessing: `l2`
- Tags: relational
- Shared cache: none
- arXiv: https://arxiv.org/abs/2005.10242
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

The value depends on D through the bound, so compare gap or excess across embedding
dimensions, not the raw value; the follow-up to SPHERE-JEPA (2026) derives the same
uniform baseline as the expected kernel under the uniform law and reads the Gaussian
potential as a kernel MMD to it.

Returns:
    value: uniformity in nats.
    extras: lower_bound for this D and t, its large-D limit -2t, gap = value - lower_bound,
        excess = gap / -lower_bound in [0, 1]: 0 for a uniform cloud, 1 for a single point.
