# `token_gram_drift`

Squared Frobenius drift of a clip's token cosine Gram matrix against a reference field.

- Input: `pair`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- Origin: https://arxiv.org/abs/2508.10104
- Cite: `simeoni2025dinov3` (DINOv3 (2025))

## Definition, protocol and pitfalls

Squared Frobenius distance between the cosine Gram matrices of two token fields of the same clip.

The quantity DINOv3's Gram anchoring regularizes (Simeoni et al., 2025,
arXiv:2508.10104, Sec. 4): with X_S and X_G the (P, d) L2-normalized
local features of the current and of a reference model, the loss is
||X_S X_S^T - X_G X_G^T||_F^2, which pins the similarity structure while
letting the features move. Divided by P^2 here so clips of different
length are comparable. As a monitor, the reference is an earlier sweep of
the same model; rising drift late in training together with weakening
dense probes is the degradation signature that motivated the loss. A
two-field metric: call it directly with the two token fields; compute()
and compute_pairs() do not route it.

Args:
    tokens: Current token field, shape (T, D).
    reference: Reference token field of the same clip, shape (T, D_ref).

Returns:
    value: mean squared difference of the two cosine Gram matrices.
    extras: none.
