# `token_cosine`

Mean cosine similarity between distinct tokens of one clip.

- Input: `tokens`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- Origin: https://arxiv.org/abs/2602.08626
- Cite: `marouani2026clspatch` (Revisiting [CLS] and Patch Token Interaction in Vision Transformers (2026))

## Definition, protocol and pitfalls

The patch-to-patch similarity tracked by Marouani et al. (2026, ICLR,
arXiv:2602.08626) around each block of a ViT, here on the layer output.
Near 1 means the tokens are interchangeable (a collapsed field), near 0
means they spread out. Computed exactly on up to n_tokens tokens sampled
without replacement.

Args:
    tokens: One clip's tokens, shape (T, D), class tokens removed.
    n_tokens: Tokens sampled when T exceeds it.
    seed: Sampling seed.

Returns:
    value: mean off-diagonal cosine.
    extras: n_tokens used.
