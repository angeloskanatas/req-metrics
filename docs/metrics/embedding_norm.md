# `embedding_norm`

Mean Euclidean norm of the representations.

- Input: `points`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- Origin: https://arxiv.org/abs/2502.09252
- Cite: `draganov2025norms` (On the Importance of Embedding Norms in Self-Supervised Learning (2025))

## Definition, protocol and pitfalls

Draganov et al. (2025, arXiv:2502.09252): pre-normalization norms govern convergence and
shrink on unexpected samples. Take the representation before any L2 normalization.

Args:
    x: Representations (N, D).

Returns:
    value: mean norm.
    extras: std, median, min, max, cv (std over mean).
