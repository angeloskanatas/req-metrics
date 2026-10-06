# `sparsity`

Sparsity: Hoyer-type l1/l2 density and the fraction of active entries.

- Input: `points`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- arXiv: https://arxiv.org/abs/2602.01456
- Cite: `kuang2026lpjepa` (Rectified LpJEPA: Joint-Embedding Predictive Architectures with Sparse and Maximum-Entropy Representations (2026))

## Definition, protocol and pitfalls

Kuang et al. (2026, arXiv:2602.01456, appendix): m_l1 = E[||x||_1^2 / ||x||_2^2] / D, from
1/D for a one-hot vector to 1 for equal magnitudes; m_l0 = E[||x||_0] / D, informative only
after a rectifying nonlinearity.

Args:
    x: Points (N, D).

Returns:
    value: m_l1.
    extras: m_l0.
