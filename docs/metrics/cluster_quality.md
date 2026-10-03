# `cluster_quality`

k-means Davies-Bouldin index, with the inertia in the extras (Whetten et al., 2025).

- Input: `points`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- Cite: `whetten2025early` (Towards Early Prediction of Self-Supervised Speech Model Performance (2025))

## Definition, protocol and pitfalls

Clustering quality under k-means: Davies-Bouldin index and inertia.

Whetten et al. (2025, Interspeech, Sec. 3.1.1, Eqs. 2-3): k-means with k = 1024 and k-means++
seeding on the frames of one layer, early in pretraining. In their study lower inertia goes
with better recognition and the Davies-Bouldin index with worse, so read both, at equal
layer, N and k. Full-batch Lloyd iterations with a seed replace the paper's mini-batch
k-means.

Args:
    x: Points (N, D), N well above k.
    k: Number of clusters.
    score: "davies_bouldin" or "inertia".
    iters: Maximum Lloyd iterations.
    seed: Seed of the k-means++ initialization.

Returns:
    value: the chosen score.
    extras: davies_bouldin, inertia, inertia_per_point, k, n_empty_clusters.
