# `token_gram_drift`

Squared Frobenius drift of a clip's token cosine Gram matrix against a reference field.

- Input: `pair`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- Origin: https://arxiv.org/abs/2508.10104
- Cite: `simeoni2025dinov3` (DINOv3 (2025))

## Definition, protocol and pitfalls

Drift of a clip's token cosine Gram matrix against a reference field.

DINOv3's Gram anchoring term (Simeoni et al., 2025, arXiv:2508.10104, Sec. 4): ||X_S X_S^T -
X_G X_G^T||_F^2 on L2-normalized tokens, divided by P^2 here so clips of different length
compare. As a monitor, the reference is an earlier sweep of the same model. Called directly;
compute() and compute_pairs() do not route it.

Args:
    tokens: Current token field (T, D).
    reference: Reference field of the same clip (T, D_ref).

Returns:
    value: mean squared difference of the two cosine Gram matrices.
    extras: none.
