# `jacobian_effective_rank`

Effective rank of a layer readout's input-output Jacobian (Chung and Kim, 2026).

- Input: `jacobian`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- Origin: https://arxiv.org/abs/2602.03282
- Cite: `chung2026globalgeometry` (Global Geometry Is Not Enough for Vision Representations (2026))

## Definition, protocol and pitfalls

Jacobian effective rank (JER) from a sketch of the input-output Jacobian.

Chung and Kim (2026, ICML, arXiv:2602.03282, Eq. 1): with s_i the k leading singular values
of the Jacobian J(x) of a readout at input x, JER(x) = (sum s_i)^2 / sum s_i^2, averaged over
inputs; at most k. The singular values are those of the sketch, e.g. B = Q^T J from
jacobian_products, which estimates them by randomized range finding as the paper does
(Sec. 4.1; k = 32, 5 power iterations, 100 natural images, Gaussian noise as a control in
App. E.1). It counts the input directions a readout responds to, a property of the model's
function rather than of the representation's geometry, and its preferred direction depends
on the task.

Args:
    sketch: (B, k, M) per input, k rows whose singular values estimate those of J(x).

Returns:
    value: mean JER over inputs.
    extras: std over inputs, n_probes, n_inputs.
