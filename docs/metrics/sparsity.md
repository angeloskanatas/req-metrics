# `sparsity`

Fraction of active entries and a Hoyer-type l1/l2 density ratio.

- Input: `points`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- Origin: https://arxiv.org/abs/2602.01456
- Cite: `kuang2026lpjepa` (Rectified LpJEPA: Joint-Embedding Predictive Architectures with Sparse and Maximum-Entropy Representations (2026))

## Definition, protocol and pitfalls

Rectified LpJEPA (Kuang et al., 2026, arXiv:2602.01456, appendix):
m_l0 = E[||x||_0] / D, the fraction of nonzero entries, 0 for all-zero
vectors and 1 for fully dense ones; m_l1 = E[||x||_1^2 / ||x||_2^2] / D,
which is 1/D for a one-hot vector and 1 for a dense vector with equal
magnitudes. Pre-activation transformer states are dense, so m_l0 is
informative only after a rectifying nonlinearity; m_l1 varies
continuously and is the returned value.

Args:
    x: Points (N, D).

Returns:
    value: m_l1.
    extras: m_l0.
