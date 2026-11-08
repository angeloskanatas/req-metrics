# `dense_structure`

Dense representation structure estimator: k-means separability of each sample's tokens plus the effective rank of one token per sample (Dai et al., 2025).

- Input: `tokens`
- Canonical preprocessing: `l2`
- Tags: none
- Shared cache: none
- arXiv: https://arxiv.org/abs/2510.17299
- Cite: `dai2025structural` (Exploring Structural Degradation in Dense Representations for Self-supervised Learning (2025))

## Definition, protocol and pitfalls

Dense representation structure estimator, DSE (Dai et al., 2025, NeurIPS, arXiv:2510.17299, Sec. 4.1, Eq. 5).

A label-free estimate of how well a linear head can classify the tokens of a sample set, built from a
nearest-class-mean error bound (their Thm. 2 and Cor. 5). It adds class separability on k-means
pseudo-labels to the effective dimensionality of the tokens,

    DSE = M_inter - M_intra + lambda * M_dim.

- M_intra, the radius of a cluster: the sum of the singular values of its centered tokens over
  sqrt(N_j - 1), averaged over clusters; computed on each sample's tokens with k_sample = 3 clusters and on
  the tokens of group_size = 8 samples with k_group = 24 clusters, and the two averaged (their App. D.4).
- M_inter: each token's distance to the nearest center of another cluster, from the k_group clustering.
- M_dim: the effective rank (Roy and Vetterli) of one token per sample, the samples' tokens drawn at
  random, averaged over draws; the samples should outnumber the dimensions.

variant="released" (the default) follows the authors' code (github.com/EldercatSAM/SSL-Degradation,
estimator.compute_DSE, commit b75fd51), which produced their published values and differs from Eq. 5 as
printed: M_inter and both radii are divided by the mean distance of all tokens to their mean, the radius
denominator is sqrt(max(N_j - 1, d)), M_inter is the mean over all tokens rather than over clusters, and
M_dim is divided by min(samples, d). These make the value free of the scale and, for M_dim, of the width,
so layers and models can be compared. variant="paper" follows Eq. 5 as printed. lambda is 1 here; the
paper sets it across candidates to Std(M_inter - M_intra) / Std(M_dim), which recombine_dense_structure
applies to the components of a set of checkpoints or layers. The authors L2-normalize the tokens (their
README's --normalize, l2=True) and measure the last layer of each checkpoint on 2,048 ImageNet images in
batches of 1,024. Other differences from the released code: its k-means++ seeding weights candidates by
distance rather than squared distance, it scores a sample with no cluster of two or more tokens as 1.0
where this function skips it, and it has no cap on the draws.

Args:
    x: Tokens (N, D) with groups, or token fields (B, T, D) of B samples.
    groups: The sample of every row of a 2D x, (N,); compute() passes it at level="population".
    k_sample: Clusters per sample.
    k_group: Clusters per group of samples.
    group_size: Samples per group, taken in order.
    l2: Scale every token to unit length first.
    variant: "released" or "paper".
    draws: Number of one-token-per-sample draws for M_dim; None uses as many as the shortest sample
        has tokens, as the released code does.
    iters: Maximum Lloyd iterations (20 in the released code).
    seed: Seed of the k-means seeding and of the draws.

Returns:
    value: DSE with lambda = 1.
    extras: m_inter, m_intra (the mean of m_intra_sample and m_intra_group), m_intra_sample,
        m_intra_group, m_dim, separability (m_inter - m_intra), scale (the mean distance to the mean),
        n_samples, n_groups, draws, n_samples_skipped.
