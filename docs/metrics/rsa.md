# `rsa`

Spearman correlation of the pairwise-distance vectors of two representations (Kriegeskorte et al., 2008).

- Input: `pair`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- Cite: `kriegeskorte2008rsa` (Representational similarity analysis -- connecting the branches of systems neuroscience (2008)); `koepke2026cave` (Back into Plato's Cave: Examining Cross-modal Representational Convergence at Scale (2026))

## Definition, protocol and pitfalls

Representational similarity analysis: correlation of the pairwise-distance vectors of two representations.

Kriegeskorte, Mur and Bandettini (2008, Frontiers in Systems Neuroscience): each representation
gives the distances between all pairs of items, and the two vectors (i < j) are compared by
Spearman rank correlation; 1 for the same geometry, 0 in expectation for unrelated ones. The
distance is cosine by default, as in Koepke et al. (2026), "euclidean", or "correlation" (one
minus the Pearson correlation across features, the paper's choice). Reads the full ordering of
pairs, where CKA weights the leading directions. N(N - 1)/2 distances per representation, built
in row chunks.

Args:
    x_a, x_b: (N, D_a) and (N, D_b), rows of the same items.
    distance: "cosine", "euclidean" or "correlation".
    method: "spearman" (ranks) or "pearson" (raw distances).
    chunk: Rows per distance block.

Returns:
    value: the correlation.
    extras: n_pairs.
