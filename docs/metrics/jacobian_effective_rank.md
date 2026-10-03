# `jacobian_effective_rank`

Effective rank of a layer readout's input-output Jacobian (Chung and Kim, 2026).

- Input: `jacobian`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- Origin: https://arxiv.org/abs/2602.03282
- Cite: `chung2026globalgeometry` (Global Geometry Is Not Enough for Vision Representations (2026))

## Definition, protocol and pitfalls

Jacobian effective rank (JER), from Jacobian-vector products.

Chung and Kim (2026, arXiv:2602.03282, Eq. 1 and App. D): with s_i the singular
values of the products J(x) v_1, ..., J(x) v_k of the Jacobian at input x with k
random orthonormal input directions, JER(x) = (sum s_i)^2 / sum s_i^2, averaged over
inputs; at most k. It counts the input directions a readout responds to, a property
of the model's function rather than of the representation's geometry, and its
preferred direction depends on the task.

Args:
    jvps: (B, k, D) products for B inputs and k directions; D is the flattened readout size.

Returns:
    value: mean JER over inputs.
    extras: std over inputs, n_probes, n_inputs.
