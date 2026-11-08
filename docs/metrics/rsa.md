# `rsa`

Spearman correlation of the pairwise-distance vectors of two representations (Kriegeskorte et al., 2008).

- Input: `pair`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- Cite: `kriegeskorte2008rsa` (Representational similarity analysis - connecting the branches of systems neuroscience (2008)); `koepke2026cave` (Back into Plato's Cave: Examining Cross-modal Representational Convergence at Scale (2026)); `cui2022deconfounded` (Deconfounded Representation Similarity for Comparison of Neural Networks (2022))

## Definition, protocol and pitfalls

Representational similarity analysis: correlation of the pairwise-distance vectors of two representations.

Kriegeskorte, Mur and Bandettini (2008, Frontiers in Systems Neuroscience): each representation
gives the distances between all pairs of items, and the two vectors (i < j) are compared by
Spearman rank correlation; 1 for the same geometry, 0 in expectation for unrelated ones. The
default distance is "cosine", as in Koepke et al. (2026, arXiv:2604.18572); "euclidean" and
"correlation" (one minus the Pearson correlation across features, the choice of Kriegeskorte et
al.) are the alternatives. RSA reads the full ordering of pairs, whereas CKA weights the leading
directions. Each representation gives N(N - 1)/2 distances, built in row chunks.

With confound, the deconfounded RSA of Cui, Kumar, Marttinen and Kaski (2022, NeurIPS,
arXiv:2202.00095, Eq. 12): the squared Euclidean distances of a representation of the same items
are regressed out of each representation's squared Euclidean distances (least squares, no
intercept) and the residuals are correlated. As in the paper, this needs distance="euclidean".
See cka for when deconfounding applies.

Args:
    x_a, x_b: (N, D_a) and (N, D_b), rows of the same items.
    distance: "cosine", "euclidean" or "correlation".
    method: "spearman" (ranks) or "pearson" (raw distances).
    chunk: Rows per distance block.
    confound: (N, D_0) representation whose distance structure is removed from both.
    confound_b: The confound of x_b when it differs from x_a's (defaults to confound).

Returns:
    value: the correlation, or the deconfounded correlation when confound is given.
    extras: n_pairs; with confound also rsa (the plain correlation), alpha_a and alpha_b.
