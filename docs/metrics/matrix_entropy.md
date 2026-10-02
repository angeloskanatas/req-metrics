# `matrix_entropy`

Matrix-based Renyi entropy of the trace-normalized Gram matrix.

- Input: `points`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: spectrum
- Origin: https://arxiv.org/abs/2502.02013
- Cite: `giraldo2015matrixentropy` (Measures of Entropy From Data Using Infinitely Divisible Kernels (2015)); `DBLP:conf/icml/SkeanAZPNLS25` (Layer by Layer: Uncovering Hidden Representations in Language Models (2025))

## Definition, protocol and pitfalls

Sanchez Giraldo, Rao and Principe (2015, IEEE Trans. Inf. Theory) define
S_alpha(K) = log(sum_i lambda_i^alpha) / (1 - alpha) on the eigenvalues
of K / tr(K); alpha = 1 is the von Neumann entropy. Skean et al. (2025,
ICML, Eq. 1) apply it to the Gram matrix K = Z Z^T of a prompt's token
states ("prompt entropy") and of a dataset's mean-pooled states
("dataset entropy"); with a population argument these are the frames and
pooled populations. The nonzero eigenvalues of Z Z^T are the squared
singular values of Z, so the entropy is computed from the spectrum
without forming an N x N matrix. The Gram matrix is not clamped: the
reference implementation zero-clamps negative entries, which is not a
numerical safeguard and inflated the entropy by 13 to 22 percent on audio
foundation-model states. The reference library's alpha = 2 shortcut
divides the squared Frobenius norm by N^2, which assumes a kernel matrix
with unit diagonal; on a trace-normalized Gram matrix it overstates the
entropy by exactly 2 log N. The entropy here is computed from the
eigenvalues for every alpha. Not centered by default, following the papers.

Args:
    x: Points (N, D), or a Spectrum of already preprocessed points.
    alpha: Renyi order; 1.0 gives von Neumann entropy.
    normalization: "max" divides by log min(N, D); "logN", "logD", "raw".
    center: Mean-center before the Gram matrix (off in the papers).

Returns:
    value: normalized entropy.
    extras: raw entropy in nats.
