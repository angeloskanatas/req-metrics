# `cls_patch_cosine`

Mean cosine between the class token(s) and the patch tokens of one sample.

- Input: `tokens`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- Origin: https://arxiv.org/abs/2602.08626
- Cite: `marouani2026clspatch` (Revisiting [CLS] and Patch Token Interaction in Vision Transformers (2026))

## Definition, protocol and pitfalls

Marouani et al. (2026, ICLR, arXiv:2602.08626) show class and patch tokens diverging at
specific layers.

Args:
    tokens_with_cls: The sample's full token sequence (n_prefix + T, D).
    n_prefix: Leading class or register tokens.

Returns:
    value: mean cosine over (class, patch) pairs.
    extras: std over patches.
