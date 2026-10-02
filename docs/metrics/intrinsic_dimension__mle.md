# `intrinsic_dimension/mle`

Levina-Bickel MLE intrinsic dimension, k = 10..20.

- Input: `points`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: neighbors
- Cite: `levina2004mle` (Maximum Likelihood Estimation of Intrinsic Dimension (2004))

## Definition, protocol and pitfalls

Levina-Bickel maximum-likelihood intrinsic dimension.

Levina and Bickel (2004, NeurIPS), Eq. 8: with T_j(x) the distance to
the j-th neighbor, m_k(x) = [ (1/(k-1)) sum_{j<k} log(T_k(x)/T_j(x)) ]^-1;
Eq. 9 averages m_k over all points and then over k from k1 to k2, which
they fix at 10 and 20. Dividing by k-2 instead of k-1 makes the per-point
estimate asymptotically unbiased (their remark after Eq. 8).

Args:
    x: Points (N, D), or a Neighbors table with k >= k_range[1].
    k_range: Inclusive range of neighbor counts averaged over.
    unbiased: Use k-2 in the denominator.

Returns:
    value: estimated dimension.
    extras: id_k{k} for every k in the range.
