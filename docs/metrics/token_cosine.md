# `token_cosine`

Mean cosine similarity between distinct tokens of one clip.

- Input: `tokens`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- Origin: https://arxiv.org/abs/2602.08626
- Cite: `marouani2026clspatch` (Revisiting [CLS] and Patch Token Interaction in Vision Transformers (2026))

## Definition, protocol and pitfalls

Marouani et al. (2026, ICLR, arXiv:2602.08626): near 1 for a collapsed token field. Exact on
up to n_tokens tokens sampled without replacement.

Args:
    tokens: One clip's tokens (T, D), class tokens removed.
    n_tokens: Tokens sampled when T exceeds it.
    seed: Sampling seed.

Returns:
    value: mean off-diagonal cosine.
    extras: n_tokens.
