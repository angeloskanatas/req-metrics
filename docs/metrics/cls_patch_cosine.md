# `cls_patch_cosine`

Mean cosine between the class token(s) and the patch tokens of one clip.

- Input: `tokens`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- Origin: https://arxiv.org/abs/2602.08626
- Cite: `marouani2026clspatch` (Revisiting [CLS] and Patch Token Interaction in Vision Transformers (2026))

## Definition, protocol and pitfalls

Marouani et al. (2026, ICLR, arXiv:2602.08626) show that [CLS] and patch
tokens diverge at specific layers inside each block although they share
the same operators, and that disentangling them improves dense features.
Pass the token sequence with its leading class token(s) as the model
emits it; the mean is over all (class, patch) pairs.

Args:
    tokens_with_cls: One clip's full token sequence, shape (n_prefix + T, D).
    n_prefix: Number of leading class or register tokens.

Returns:
    value: mean cosine between class and patch tokens.
    extras: std over patches.
