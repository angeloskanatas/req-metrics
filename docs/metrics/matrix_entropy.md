# `matrix_entropy`

Matrix-based Renyi entropy of the trace-normalized Gram matrix.

- Input: `points`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: spectrum
- arXiv: https://arxiv.org/abs/2502.02013
- Cite: `giraldo2015matrixentropy` (Measures of Entropy From Data Using Infinitely Divisible Kernels (2015)); `DBLP:conf/icml/SkeanAZPNLS25` (Layer by Layer: Uncovering Hidden Representations in Language Models (2025))

## Definition, protocol and pitfalls

Sanchez Giraldo, Rao and Principe (2015, IEEE Trans. Inf. Theory); Skean et al. (2025, ICML,
arXiv:2502.02013, Eq. 1). S_alpha = log(sum lambda_i^alpha) / (1 - alpha) over the
eigenvalues of K / tr(K) with K = Z Z^T; alpha = 1 is the von Neumann entropy. Computed from
the singular values of Z, without the N x N matrix. Uncentered by default, as in the papers;
with center=True and alpha = 1 it equals effective_rank's normalized entropy under
spectrum="variance".

Args:
    x: Points (N, D), or a Spectrum of preprocessed points.
    alpha: Renyi order.
    normalization: "max" divides by log min(N, D); "logN", "logD" or "raw".
    center: Mean-center before the Gram matrix.

Returns:
    value: normalized entropy.
    extras: raw entropy in nats.
