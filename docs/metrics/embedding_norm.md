# `embedding_norm`

Mean Euclidean norm of the representations, with its spread.

- Input: `points`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- Origin: https://arxiv.org/abs/2502.09252
- Cite: `draganov2025norms` (On the Importance of Embedding Norms in Self-Supervised Learning (2025))

## Definition, protocol and pitfalls

Draganov et al. (2025, arXiv:2502.09252) show that although cosine-based
self-supervised objectives embed on a hypersphere, the norms of the
pre-normalization embeddings govern convergence rates and encode the
network's confidence, with smaller norms on unexpected samples. Tracked
per layer during training the mean norm is a convergence monitor; the
coefficient of variation separates a few outlier clips from a uniform
rescaling. Take the representation before any L2 normalization.

Args:
    x: Representations, shape (N, D).

Returns:
    value: mean norm.
    extras: std, median, min, max, cv (std over mean).
