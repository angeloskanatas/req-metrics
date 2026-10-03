# `cluster_quality`

k-means Davies-Bouldin index, with the inertia in the extras (Whetten et al., 2025).

- Input: `points`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- Cite: `whetten2025early` (Towards Early Prediction of Self-Supervised Speech Model Performance (2025))

## Definition, protocol and pitfalls

Clustering quality of a point cloud under k-means: Davies-Bouldin index and inertia.

Whetten et al. (2025, Interspeech, Sec. 3.1.1) fit k-means with k = 1024 and
k-means++ seeding to the frame embeddings of one layer and report the inertia
(within-cluster sum of squares, their Eq. 2) and the Davies-Bouldin index
(their Eq. 3) as label-free indicators of downstream speech performance,
computed early in pretraining. In their study lower inertia goes with better
recognition while the Davies-Bouldin index correlates in the opposite direction,
so read the two together and compare runs at the same layer, N and k. The paper
fits scikit-learn's mini-batch k-means; full-batch Lloyd iterations are used
here, with a seed, so values are reproducible. Inertia scales with N and the
norm of the embeddings; the extras also give it per point. Use the frames
population to cluster all frames of a corpus, as in the paper.

Args:
    x: Points (N, D), N well above k.
    k: Number of clusters.
    score: "davies_bouldin" or "inertia", the value returned.
    iters: Maximum Lloyd iterations.
    seed: Seed of the k-means++ initialization.

Returns:
    value: the chosen score.
    extras: davies_bouldin, inertia, inertia_per_point, k, n_empty_clusters.
