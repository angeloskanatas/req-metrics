# `intrinsic_dimension/mle`

Levina-Bickel MLE intrinsic dimension, k = 10..20.

- Input: `points`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: neighbors
- Cite: `levina2004mle` (Maximum Likelihood Estimation of Intrinsic Dimension (2004))

## Definition, protocol and pitfalls

Levina-Bickel maximum-likelihood intrinsic dimension.

Levina and Bickel (2004, NeurIPS, Eqs. 8-9): m_k(x) = [(1/(k-1)) sum_{j<k} log(T_k(x) /
T_j(x))]^-1 from the neighbor distances T_j, averaged over points and over k in k_range
(10 to 20 in the paper). unbiased=True divides by k - 2, which they note is asymptotically
unbiased.

Args:
    x: Points (N, D), duplicates removed first, or a Neighbors table with k >= k_range[1].
    k_range: Inclusive range of neighbor counts.
    unbiased: Use k - 2 in the denominator.

Returns:
    value: estimated dimension.
    extras: id_k{k} for every k in the range.
