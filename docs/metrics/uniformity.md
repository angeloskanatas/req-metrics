# `uniformity`

Uniformity: log average pairwise Gaussian potential on the unit sphere.

- Input: `points`
- Canonical preprocessing: `l2`
- Tags: relational
- Shared cache: none
- Origin: https://arxiv.org/abs/2005.10242
- Cite: `wang2020uniformity` (Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere (2020))

## Definition, protocol and pitfalls

Wang and Isola (2020, ICML, arXiv:2005.10242, Sec. 4.1.2):
L_uniform = log mean_{i<j} exp(-t ||u_i - u_j||^2) over L2-normalized
features, t = 2 in the paper and its reference code. The uniform
distribution on the sphere is its unique minimizer (Prop. 1); Corollary 1
gives the range [-2t + log 0F1(; D/2; t^2), 0], the lower end reached only
by a perfectly uniform encoder and 0 only by a constant one. Lower values
mean points are spread more evenly. Pairwise distances come from the Gram
matrix in row chunks, ||u - v||^2 = 2 - 2 u.v, in float64; O(N^2 D).

Args:
    x: Points (N, D).
    t: Kernel scale of the Gaussian potential.
    center: Mean-center before normalizing (off in the source paper).
    chunk: Rows per Gram block.

Returns:
    value: L_uniform in nats, in [lower_bound, 0].
    extras: lower_bound for this D and t, its large-D limit -2t, gap = value - lower_bound.
