# Metric reference

This document collects the definitions, sources and verification that apply across metrics, and
the protocol of Kanatas et al. (2026). The text for each metric is in its estimator docstring,
which `scripts/build_metric_cards.py` renders into `docs/metrics/`.

## 1. Conventions

Estimators are pure functions on tensors, and each returns `MetricResult(value, extras)`. `x` is a
point cloud `(N, D)`, `z` a single trajectory `(T, D)`, and `views` a stack `(q, N, D)` of `q`
views of the same `N` samples. A token field is `(T, D)` per sample.

Metrics never see token grids. `layouts.py` turns an `(F, T, D)` spectrogram-patch grid into a
`(T, F * D)` trajectory by concatenating the frequency patches of each time step (MSM-MAE, Niizumi
et al., 2022, Sec. 3.3), into an `(F * T, D)` token cloud, or into one pooled vector. Frame
sequences are already trajectories. The choice is recorded in `pooling`, and `make_pooler` offers
the same readouts by name during training. The table lists the input each family expects and the
readouts its sources used.

| Family | Input the estimator sees | Readouts and precedent |
|---|---|---|
| spectral, relational, manifold (intrinsic dimension) | a point cloud `(N, D)` | one vector per sample (class token for ViTs and global average pooling for CNNs in Arputharaj et al. 2026; backbone output in RankMe and LiDAR; time mean in Kanatas et al. 2026; frequency-concatenated or partitioned means for 2D audio encoders), the tokens of one sample (Skean et al. 2025 prompt entropy) or of all samples (Razzhigaev et al. 2024), or the time steps of a trajectory layout |
| trajectory curvature | a time-ordered `(T, D)` sequence per sample | frame encoders directly; 2D encoders through the frequency-concatenated or frequency-mean trajectory |
| token fields | the patch tokens of one sample, class token kept for `cls_patch_cosine` | Darcet et al. 2024, Marouani et al. 2026, DINOv3 |
| views and equivariance | one vector per sample and per view or shift | pooled embeddings in Thilak et al. 2024 and Wang and Isola 2020; for pitch, a frequency-preserving readout keeps the quantity the probe must find |

Label-free metrics take parameter-free readouts only: the class token, the mean of the patch
tokens, their concatenation, the max, the last token of a causal decoder, the
frequency-concatenated time mean (MSM-MAE, Riou et al. 2024), and block-partitioned means (Gu et
al. 2026). Learnable probe-side readouts, such as attention pooling or learned layer fusion, are
trained with task labels and would make the metric measure the probe. A pooling head trained by
the self-supervised objective itself is part of the frozen model and is labeled as its own pooling.
The vision proxies (RankMe, LiDAR, alpha-ReQ, IdEst) are all computed on fixed features.

- Only metrics defined in a published paper are registered, each with its input contract,
  canonical preprocessing and citation keys into `references.bib`.
- Preprocessing is never implicit. The `center`, `standardize` and `l2` arguments of each
  estimator define what it applies. The registry holds their defaults, the pipeline applies them
  before building the shared spectrum, and every record stores the preprocessing actually applied.
  Estimators whose definition normalizes rows (the cosine measures) register `l2` as their
  preprocessing.
- The level (`sequence`, `sample`, `population`) is an argument of the pipeline, not a property of
  the estimator (Section 2).
- Citations name author, year and venue, and arXiv ids are given once per metric. "Published
  protocol" means the configuration behind the reported numbers of Kanatas et al. (2026).
- Tags are carried into every record. `paper-canonical` marks the metrics and parameters of that
  paper's protocol, and `relational` and `collapse-indicator` mark families.

## 2. Families and levels

| Family | Modules and metrics | Input |
|---|---|---|
| Spectral | spectral: effective_rank (RankMe, with normalized_rank = RankMe*, the variance convention and NerVE's spectral entropy in the extras), matrix_entropy, alpha_req, anisotropy/spectral (NESum in extras), participation_ratio (bias corrections in extras), eigenvalue_early_enrichment | points |
| Relational | relational: self_clustering, uniformity, normalized_std, anisotropy/cosine | points |
| Manifold | dimension: intrinsic_dimension/twonn, /gride, /mle, /mlid, /mst | points |
| Not in that taxonomy | clustering: cluster_quality (k-means), dense_structure (k-means separability of each sample's tokens plus the effective rank of one token per sample; population level); distribution: gaussianity, sparsity, embedding_norm; trajectory: trajectory_curvature (sample level); views: lidar, infonce, dime, alignment (augmented views); equivariance: pte (pitch shifts); compare: information_imbalance, neighborhood_overlap, cycle_knn, cka, svcca, rsa, procrustes, ridge_predictivity (representation pairs); tokens: token_gram_drift (token fields); functional: jacobian_effective_rank (the model and its inputs) | points, trajectories, views, shifts, pairs, tokens, Jacobian sketches |

The spectral, relational and manifold families follow the taxonomy of Arputharaj, Jönsson and
Eilertsen (2026, TMLR). In every family, a metric is a functional of one of three kinds of input,
and that input is what the caller has to provide.

- The embedded cloud is the set of vectors of a sample set alone, one per sample or the tokens of
  each sample, with or without time order (spectral, relational, manifold, clustering,
  distribution, token fields, trajectory curvature). These metrics depend on the model only
  through its image of the data.
- The map under a change of the input compares the same items, re-embedded after a
  transformation, with the originals. The transformation comes at three scales: the augmentations
  of the objective (views: lidar, infonce, dime, alignment), a group action with a prescribed
  response (equivariance: pte under pitch shifts), and an infinitesimal random perturbation, which
  needs the model itself (functional: jacobian_effective_rank).
- Two representations of the same items can be two layers, checkpoints, models or modalities
  (compare: cka, svcca, rsa, procrustes, ridge_predictivity, neighborhood_overlap, cycle_knn,
  information_imbalance; tokens: token_gram_drift for two token fields of one sample).

No kind reduces to another. An isotropic cloud can have a rank-deficient Jacobian (Chung and Kim,
2026), two representations can be similar while a readout fails, and an equivariant layer can have
a low rank. Trajectory curvature is a boundary case. It follows the map along the data's own
temporal path but needs no re-embedding, so it is grouped with the cloud.

### Levels

The level decides which vectors form the point cloud, and the estimator stays the same. The caller
extracts the vectors: one per sample, or the tokens of each sample (the frames of a 1D encoder,
the patches of a 2D one).

- `sequence`: one vector per sample, with `layers[l]` of shape `(N, D)` (token or time mean, class
  token, final token, or a grid readout). This is the setting of RankMe, LiDAR, Arputharaj et al.
  (2026) and Kanatas et al. (2026), who call these sequence-level metrics, and of the layer-wise
  intrinsic-dimension studies of Valeriani et al. (2023, token mean) and Cheng et al. (2025, ICLR,
  last token). Sample sizes are discussed in `docs/DESIGN.md`, section 6. With several clips per
  track, the nearest neighbors of a clip can be clips of its own track (the regime effect described
  under `population`). `group_ids` keeps one clip per track, as Kanatas et al. (2026) did.
- `sample`: the tokens of each sample form one cloud, and `layers[l]` is a sequence of `(T_i, D)`
  tensors. The estimator runs per sample, and the value is the mean of the finite per-sample
  values, with the std, median, min, max and the count of failures in the extras. Viswanathan et
  al. (2025) estimate the intrinsic dimension of the 1,024 tokens of each prompt and average over
  prompts, and Pedashenko et al. (2026, EACL) do the same per text. Skean et al. (2025) compute
  prompt entropy on the tokens of one prompt, and Sadok and Alameda-Pineda (2026) compute their
  speech metrics per sample and then average. Trajectory curvature (Hosseini and Fedorenko, 2023)
  and DINOv3's Gram anchoring are per sample by definition. The value describes the sample's own
  manifold, so a neighbor estimator at this level sees temporally adjacent frames. The value also
  depends on the token count T_i: a spectrum has at most min(T_i, D) nonzero values, and
  Pedashenko et al. exclude texts shorter than 150 tokens, where the variance of their estimates is
  high. Compare values at equal T_i, that is, at equal duration and token rate.
- `population`: the tokens of all samples form one cloud. `layers[l]` is a sequence of `(T_i, D)`
  tensors, from which `n_tokens` tokens are drawn at random. This is the setting of the anisotropy
  literature, where pairs of token vectors are drawn across a corpus (Ethayarajh, 2019; Godey et
  al., 2024; Timkey and van Schijndel, 2021). Razzhigaev et al. (2024) compute anisotropy and TwoNN
  on batches of at least 4,096 vectors from different contexts, shuffled before batching for
  TwoNN, and average over batches. Ruppik et al. (2025, NeurIPS) pool the tokens of 7,000 to 10,000
  sequences, deduplicate them and draw 60,000. Whetten et al. (2025) compute a global effective
  rank over all frames of about an hour of audio. Spectral metrics need n above D, as at the
  sequence level.

  For neighbor estimators, the number of samples sets the regime (Osman, Baroni and Macocco,
  2026). Take n points from c samples, about m = n / c per sample. While k < m, the k neighbors of
  a point can all be tokens of its own sample (the local regime, close to the sample level). Once
  k >= m, they must reach other samples (the global regime). The transition at c = n / k adds a
  spurious peak to layer profiles and reverses the dependence on c. The argument assumes, as they
  verify for words, that the tokens of a sample lie closer to each other than to the tokens of
  other samples. TwoNN reads two neighbors, so its value is in the global regime only up to two
  tokens per sample (n <= 2c), and mLID at k = 64 only up to 64. Compare such values at equal n, k
  and sample count. Records at this level carry `n_samples`. Records of the neighbor estimators
  also carry `same_sample_fraction`, the mean share of the neighbors behind the value that are
  tokens of the point's own sample (the diagnostic of their Sec. 5). It is near 1 in the local
  regime and near 1 / c without sample structure.

  Repeated draws with different seeds give the batch spread of Razzhigaev et al. Class, register
  and first-position tokens can distort such a cloud (Timkey and van Schijndel, 2021, report
  cosine above 0.99 between position-0 tokens), so strip them first with
  `layouts.strip_prefix_tokens`, or with `n_prefix` and the `tokens` readout. Token-field metrics
  whose definition pairs tokens within one sample (`cls_patch_cosine`, `token_cosine`; `per_sample`
  in the registry) still run per sample at this level. `token_norm_outliers` reads the population
  tokens, as Darcet et al. (2024) set their cutoff from a pooled norm histogram.

Rosina Fernandez, Guillaume and Wisniewski (2025) compare the cosine similarity of frame pairs
from the same recording and from different recordings, and find the two distributions similar. No
other study cited here compares the sample and population levels of one estimator on the same
data. Viswanathan et al. (2025, App. D) find that token-level and prompt-level intrinsic dimension
follow different trends, and Osman et al. (2026) find that pooled neighbor estimates change regime
with the number of items. The level is therefore part of the protocol and is recorded with every
value. The papers name the levels differently. Pasad et al. (2023) call pooled frames frame-level,
Kanatas et al. (2026) call per-clip token sequences frame-level, and Hosseini et al. (2026) call a
per-sequence average sequence-level. The levels in this library are named by how the vectors are
grouped.

## 3. Differences from other implementations

Where a reference implementation deviates from the published definition, the docstring says so.

- Spectral group. Matrix-based entropy is the Renyi entropy of the squared singular values, so it
  is computed from the shared spectrum without an N x N matrix. The alpha = 2 shortcut in repitl's
  `matrixAlphaEntropy` divides the Frobenius norm by N^2 and overstates the entropy of a
  trace-normalized Gram matrix by exactly 2 log N (17.30 versus 1.29 at N = 3000). The defaults
  use the constants of the original code: LeJEPA's Epps-Pulley grid t in [-5, 5] with 17 points,
  target and weight exp(-t^2/2), and 256 directions; VISReg's sliced Wasserstein quantiles
  i/(N+1); RankMe's 25,600 samples; and Chung and Kim's isotropy score 1 - lambda_1 / sum(lambda).

  reptrix computes RankMe from the eigenvalues of the covariance (PCA), which centers and squares
  the singular values (p_k = s_k^2 / sum s^2). The RankMe paper takes the singular values of the
  uncentered embeddings (p_k = s_k / sum s). Both conventions are exposed (`center`, and
  `spectrum="variance"`), and the card states the default. RankMe's epsilon inside the logarithm
  is omitted, so zero singular values contribute nothing. The reference matrix-entropy code clamps
  negative Gram entries to zero, which raised the entropy by 13 to 22 percent on audio
  foundation-model states. The Gram matrix is not clamped here. RankMe-t (Aldeneh et al., 2024) is
  the effective rank of time-summed sample vectors. For samples of equal length this equals the
  effective rank of the mean-pooled vectors, since the two differ by one global scale.
- Neighbor group. TwoNN and GRIDE follow DADApy step for step. DADApy labels each GRIDE scale by
  its outer rank, so `gride_k8` is the estimate from the ratio of the 8th to the 4th neighbor
  distance. The Levina-Bickel MLE follows their Eq. 8 and 9 with k = 10..20, not
  scikit-dimension's single k = 20. The estimators are ported rather than wrapped, so that one kNN
  table serves every estimator on torch tensors and GPUs without a scikit-learn and Cython
  dependency. The port is credited in NOTICE.
- Trajectory group. Reference values for the curvature: 0 for a straight line, 120 degrees
  (signed) and 60 degrees (absolute) for independent frames, and 90 degrees for a random walk.
  Zero-length steps are excluded and counted rather than scored as 90 degrees.
- View group. The default LiDAR delta of 1e-4 is the one in the analysis code of Skean et al.
  (2025); Kanatas et al. (2026) used 1e-6. Thilak et al. state unbiased estimates without giving
  the denominators. The choice rescales S_b and S_w by constants, which leaves the value unchanged
  at delta 0 and otherwise changes only the relative size of delta. With more than two views,
  InfoNCE averages the two-view loss over view pairs. Unlike repitl, DiME never replaces the
  N x N Gram Hadamard product with D x D covariances when N > D, because the two differ:
  single-matrix entropies agree, Hadamard products do not.
- Equivariance. PTE trains its probe as in Kanatas et al. (2026): mixed-shift batches, Adam,
  early stopping, and the best state restored.
- Comparison group. Ranks for the information imbalance are counted per chunk, so no N x N table
  is stored. On tied data (binary rolls, labels, repeated items), tied distances share their
  average rank, as in the code of Del Tatto et al. (2024, PNAS, arXiv:2305.10817), and a tie at
  the k-th neighbor is averaged over the tied items. The value then equals the mean over random
  tie-breaking and stays at 1 in expectation for independent spaces. DADApy (arXiv:2205.03373)
  keeps the order of its neighbor search. The code of Acevedo et al. gives tied items the lowest
  rank of the tie and counts every item tied at the k-th distance, which puts a random space at
  about 0.5 against two balanced classes (1.0 here). `calibrate`, `calibrate_layers` and
  `jackknife` reuse the tables of each space for N up to 3,000 (0.5 s instead of 31 s for five
  permutations on a 13 x 3 layer grid at N = 1,276).
- Relational group and corrections: see Section 6.

## 4. Reproducing Kanatas et al. (2026)

`protocols.get("kanatas2026")` pins these variants and parameters for each input kind:

| metric | estimator / variant | preprocessing | level | views |
|---|---|---|---|---|
| intrinsic_dimension/twonn, /gride | TwoNN; GRIDE at the 8th-neighbor scale, reported as consistent | none | sequence | 1 |
| effective_rank | singular spectrum, largest 2048 values | center | sequence | 1 |
| anisotropy | spectral | center + L2 | sequence | 1 |
| trajectory_curvature | k = 1, signed; the folded convention is in the extras | none | sample | 1 |
| lidar | delta 1e-6, n and nq denominators, largest 2048 eigenvalues | none (LDA centers) | sequence | 10 |
| infonce | temperature 0.3 | center + L2 | sequence | 2 |
| pte | linear probe, omega 7, phase distance | training-split mean of the originals | sequence | 1 + 11 shifts |

The protocol uses 10,000 clips of 15 seconds, one per track. Pooled vectors are time means for
encoders and the final-token state for autoregressive decoders.

Every record carries the metric and variant, the estimator parameters, the preprocessing, the
level and pooling label, the number of views and of items, the seed, the representation width and
the library version. PTE is trained with a linear probe on 10,000 clips, centered with the
training-split mean. The extras of every run hold the cross-power distance, which includes the
magnitude, next to the phase distance.

## 5. Intrinsic dimension: caveats

MLE, TwoNN and GRIDE target the pointwise dimension, which cannot increase under Lipschitz maps
(Schulte and Rügamer, 2026, AISTATS). Every standard layer type is Lipschitz, so a layer-wise ID
profile that rises is an artifact of the estimator, driven by growing nearest-neighbor distances
and representation norms, and not a rise in the true dimension. The estimates are biased lower
bounds whose bias is not consistent across layers, and their layer-wise pattern moves together
with the von Neumann entropy of the centered Gram matrix (the variance convention of
`effective_rank`). Report ID profiles as geometric descriptors rather than manifold dimensions,
and read them next to the spectral entropies. The plug-in participation ratio has a separate
sample-size bias of about PR/N (Chun et al., 2026). See the `correction` argument of
`participation_ratio`.

Duplicates matter for the two-neighbor estimators. An exact duplicate has r_1 = 0 and an infinite
ratio, and its neighbors have a ratio of exactly 1. DADApy's 2NN keeps such rows unless
`remove_identical_points` is called, whereas the neighbor estimators here (TwoNN, GRIDE, MLE,
mLID) remove exact duplicate rows first. On MERT-v1-95M embeddings of 990 GTZAN clips, which
contain 13 exact duplicates, the two conventions differ by about 20 percent at every layer (for
example 12.6 against 15.6 at layer 6), while GRIDE at the scale of the 8th neighbor is unaffected.
Under the same convention, both match DADApy to all printed digits. Deduplicate the corpus before
estimating, as the published protocol does. When duplicates were removed, `compute()` records the
number of distinct rows in `extras["n_distinct"]`, and `twonn` reports the count it used in
`extras["n_used"]`.

## 6. Relational metrics, collapse indicators and corrections

Sources for this section: Tsitsulin, Munkhoeva and Perozzi (2023, TAG-ML at ICML,
arXiv:2305.16562); He and Ozay (2022, ICML); Wang and Isola (2020, ICML, arXiv:2005.10242) with
their reference code; Chen and He (2021, CVPR, arXiv:2011.10566); Chun, Canatar, Chung and Lee
(2026, ICLR, arXiv:2509.26560) with their reference code; Arputharaj, Jönsson and Eilertsen (2026,
TMLR, arXiv:2608.23182); Schulte and Rügamer (2026, AISTATS, arXiv:2604.20276); Liao et al. (2024,
CISS, arXiv:2312.04823) with their code.

### self_clustering (Tsitsulin et al., 2023, Def. 3.5)

The input is the L2-normalized rows W, without centering. Q is the sum of the squared cosine over
all pairs. It is compared with its expectation on a uniform sphere, N + N(N-1)/D, and with its
maximum under collapse, N^2: (Q - N - N(N-1)/D) / (N^2 - N - N(N-1)/D). The paper writes Q as the
Frobenius norm of W W^T, but the expectation and the maximum it states are those of the squared
norm. This implementation uses the squared norm, so that a single point gives exactly 1 and a
uniform cloud gives 0. It is computed from the D x D second moment in O(N D^2). The TMLR study
reports it as a reliable negative predictor for self-supervised vision models. Under its linear
probe it is uninformative for supervised models, and for ViTs it sits near 0 when they are
self-supervised and near 1 when they are supervised. The study also finds rho = -0.999 with
diffusion spectral entropy.

### uniformity and alignment (Wang and Isola, 2020)

`uniformity` is the log of the mean pairwise Gaussian potential exp(-t ||u - v||^2) over
L2-normalized points, with t = 2, computed from Gram blocks in float64. By Corollary 1 its range is
[-2t + log 0F1(; D/2; t^2), 0], and only a perfectly uniform encoder reaches the lower end. The
extras hold lower_bound, gap and excess = gap / -lower_bound, which runs from 0 for a uniform
encoder to 1 for a collapsed one. The bound moves with D, so compare gap or excess across
embedding dimensions, not the raw value.

`alignment` is the mean over view pairs and samples of ||u_a - u_b||^alpha, with alpha = 2, on
L2-normalized views (input kind views).

### normalized_std (Chen and He, 2021, Sec. 4.1)

`normalized_std` is the mean over channels of the sample std of z / ||z||_2. It is 0 under
complete collapse and about 1/sqrt(D) for an isotropic cloud (extras: reference and ratio). It
monitors complete collapse only and does not see dimensional collapse.

### participation_ratio corrections (Chun et al., 2026)

1/PR_naive is about 1/N + 1/D + 1/PR (their Sec. 3), so the plug-in estimate is biased low by
roughly PR/N. `correction="row"` removes the sample-size term, for the case of network activations
where all units are observed (their Sec. 4.5). `"both"` also removes the
unit-subsampling term. The corrected estimators center algebraically and take the raw matrix. All
estimates come from one pass and are in the extras of every record. Quartic index sums are
evaluated in closed form (Gram matrix, column moments); the port of the MIT reference code is
credited in NOTICE. The plug-in estimate remains the default, as in published numbers.

### Extras added to existing metrics

`anisotropy.ne_sum` is sum(lambda)/lambda_1 of the anisotropy's spectrum, that is, under its
default preprocessing of centered, row-normalized points. NESum (He and Ozay, 2022, Def. 4.1, on
the eigenvalues of the feature covariance) and the stable rank of Tsitsulin et al. (2023, on
centered data) leave the rows unnormalized, so they agree with ne_sum only without the row
normalization. `effective_rank.normalized_rank` is RankMe/D (RankMe* of Tsitsulin et al., 2023).
For `alpha_req`, the fit range changes the sign of the correlation with accuracy (TMLR study,
Appendix B.1), so record it.

## 7. neighborhood_overlap and cycle_knn

The neighborhood overlap is defined by Doimo, Glielmo, Ansuini and Laio (2020, NeurIPS,
arXiv:2007.03506, Eq. 1), and Valeriani et al. (2023, NeurIPS) use it for transformers. It takes a
pair input and is the mean over items of the fraction of the k nearest neighbors (self excluded)
that two representations share. It is 1 when neighborhoods are preserved, and k/(N - 1) in
expectation for unrelated spaces (Gröger, Wen and Brbić, 2026, Prop. 4.2). The mutual
k-nearest-neighbor alignment of Huh et al. (2024, ICML, App. A, Eq. 11) is the same quantity. They
compute it with k = 10 on 1,024 image-caption pairs, after clamping every activation at one
threshold per model (the mean over items of each item's 0.95 quantile of absolute activations) and
L2-normalizing, so that inner-product neighbors are cosine neighbors. Here `l2=True` ranks cosine
neighbors and is recorded as preprocessing. Doimo et al. and Valeriani et al. use k = 30 at
ImageNet scale.

At fixed k, the value falls as the gallery grows and when an item has several valid partners,
which RSA and CKA do not register (Koepke et al., 2026). Compare values at equal n and k on
one-to-one pairs, and read them as a local statistic. `compute_pairs(layers,
metrics=["neighborhood_overlap"])` builds one k-NN table per layer and intersects them for every
pair. The overlap is symmetric, so it answers a different question from the information imbalance
(directional predictability of neighbor ranks) and from `cycle_knn` below.

`jaccard=True` divides the shared count of each item by the size of the union of its two neighbor
sets instead of by k. This is the k-NN Jaccard similarity of the ReSi benchmark (Klabunde et al.,
2025, ICLR, App. A.1, Eq. 24; Section 9), which attributes it to Wang et al. (2022), ranks it
first in vision, and computes it with k = 10 on cosine neighbors (`k=10, l2=True`). The two
normalizations read the same neighbor sets and differ by a monotone transform per item. The record
carries the choice in `params`, and the chance level becomes the expectation of m/(2k - m) over
the hypergeometric shared count m (Gröger et al., Theorem C.10). The retention score of Jiang et
al. (2026) is this Jaccard variant.

On discrete data, the k-th distance of an item can tie with further items, and its neighbor set
is then not unique. In that case the shared count is taken as its expectation with the tied items
drawn at random, independently in the two spaces (the sum over items of the product of their
inclusion probabilities). This is exact for the overlap and a plug-in for the Jaccard ratio. A
fixed order of tied items, shared by both spaces, inflates the overlap of unrelated spaces, while
under this rule two independent four-class label spaces stay at the chance level (tested). Without
ties the value is the plain count.

`cycle_knn` is the cycle k-nearest-neighbor consistency that Huh et al. (2024, App. B, Fig. 11,
called Table 11 in their text) list beside the mutual k-NN and compute in their code as
`knn_A[knn_B]`. It is the fraction of items that are among the k nearest neighbors in A of one of
their k nearest neighbors in B: a first hop in B, then a return hop in A. Zhang et al. (2026) write
it cycle-kNN(A -> B), and Gröger et al. (2026, Eq. 35) write cycle-kNN_k(A, B). It reads the same
k-NN tables as the overlap but is not symmetric for k >= 2. Zhang et al. prove symmetry for k = 1,
give a six-point example with 5/6 in one ordering and 1/2 in the other (App. A), and read the gap
between the orderings as a difference in how compact the neighborhoods of the two spaces are.
Their paper is a preprint. Gröger et al. find that the measure keeps its trend with
language-model capability after permutation calibration, as the overlap does (App. E.8).

Identical representations score 1 only when every item is a nearest neighbor of one of its own
nearest neighbors, so the diagonal of a layer map falls below 1 where items have no reciprocal
neighbor. For independent representations, each return hop succeeds with probability k/(N - 1)
(Gröger et al., Prop. C.9), which bounds the chance level by k^2/(N - 1). The bound is stored as
`chance_bound`. `compute_pairs(layers, metrics=["cycle_knn"])` records the value A -> B for every
ordered pair and the other ordering in `extras["reverse"]`. A tie at the k-th distance is resolved
by the order of the neighbor search, as in the three sources. Its expectation under random
tie-breaking depends on how ties in different rows are broken together, so it is not computed. On
discrete data, prefer the information imbalance or the overlap, which average over tied items.

## 8. cka, svcca and rsa

These are similarity indices for the same items in two representations. `cka` is linear centered
kernel alignment (Kornblith et al., 2019, ICML, arXiv:1905.00414, Table 1), invariant to
orthogonal maps and isotropic scaling. `debiased=True` uses the unbiased HSIC estimator, which
matters when N is not large relative to the widths. `svcca` (Raghu et al., 2017, NeurIPS,
arXiv:1706.05806) keeps the SVD directions that carry 99 percent of the summed singular values
(App. A) and averages the canonical correlations between them (Eq. 1). It is invariant to
invertible linear maps of the kept subspaces, and therefore needs N well above the kept widths.
Both are closed-form, and per-layer summaries are computed once per call.

Post hoc, `compute_pairs` gives the layer-by-layer similarity map of a model. During training, the
same fixed items at two checkpoints give the drift of each layer. This is how Raghu et al. (Sec.
4.1) use it: they compare every layer during training with its final state.
`LayerMonitor(drift_metrics=[...])` computes the drift at every sweep, each layer against its
first or its previous sweep on the monitoring items.

Under independence, the biased linear CKA has a baseline of order D/N (Gröger et al., 2026, Prop.
4.1; Murphy, Zylberberg and Fyshe, 2024), which `debiased=True` removes. Gröger et al. find that
their permutation calibration agrees with that correction. The maximum over the L_A x L_B pairs of
a map is inflated by the number of pairs (their Sec. 4.2), so report the map, and calibrate a
reported maximum against pairings permuted over items.

`rsa` is representational similarity analysis (Kriegeskorte, Mur and Bandettini, 2008). Each
representation gives the vector of distances between all pairs of items, and the two vectors are
compared by Spearman rank correlation: 1 for the same geometry, 0 in expectation for unrelated
representations. The default distance is cosine, the choice of Koepke et al. (2026). Euclidean
distance and the correlation distance of Kriegeskorte et al. are options, and `method="pearson"`
correlates the raw distances. RSA grades the full ordering of pairs, where CKA weights the leading
directions, so the two can disagree on the same pairs (Koepke et al., 2026, Sec. 4). The cost is
N(N - 1)/2 distances per representation, built in row chunks. `compute_pairs` holds one ranked
vector per layer, so the registry caps it at 4,000 items.

`confound=` gives the deconfounded CKA and RSA of Cui, Kumar, Marttinen and Kaski (2022, NeurIPS,
arXiv:2202.00095). The similarity structure of a representation of the same items (the input, or
the previous layer in their recursive variant, App. D.2) is regressed out of each side by least
squares over the N^2 entries, without intercept (Eq. 3-5). The CKA residuals are projected onto
the PSD cone (Eq. 7-8) and compared with the biased HSIC (Eq. 11). The RSA residuals of squared
Euclidean distances are rank-correlated (Eq. 12). Both keep the invariances of CKA and RSA (their
Props. 3.1-3.2). Deconfounding applies when both sides inherit a similarity structure from the
same input: layers of one model, two models or two checkpoints on the same items (their
pretrained-against-finetuned case, Sec. 4.3), or a layer against a target computed from the same
input. `confound_b` takes a second input when the two sides have different ones. Cui et al. expect
the same conclusions as CKA within one domain (App. G), find the linear fit best in shallow layers
(App. A), and advise against removing an input structure that is itself of interest (Sec. 5). The
deconfounded CKA needs one N x N kernel and one eigendecomposition per representation, so
`compute_pairs` caps it at 4,000 items. The authors' code breaks rank ties by sort order and uses a general eigensolver on the symmetric
residual, where this implementation uses average ranks and a symmetric eigensolver.

## 9. Pairs across models and modalities

The pair metrics take any two representations of the same items with aligned rows: the layers of
one model, two checkpoints, two models, or two modalities with paired items. Cheng et al. (2025,
ICLR, Fig. 4, App. G and H) compare the layers of two language models with the information
imbalance and linear CKA. Acevedo et al. (2026, TMLR, arXiv:2505.17101 v7) compare languages,
image classes and image-caption pairs with the imbalance, with unit-length or plain Euclidean
distances and no binarization. They compute it on the average of the last T tokens of each item,
which predicts better than the last token alone or the concatenation (their Sec. 3.1.1; a `pool`
callable here). A shuffled pairing, which their code provides, sits near 1. Huh et al. (2024)
measure cross-modal alignment as the neighborhood overlap of paired images and captions
(Section 7).

Such comparisons have limits:

- At fixed k, the overlap decays with gallery size and with many-to-many pairing while RSA and CKA
  do not, and the agreement that survives sits in the leading eigenmodes (Koepke et al., 2026).
- After permutation calibration, the convergence reported by CKA, SVCCA and Procrustes distance
  disappears, while the neighborhood overlap and `cycle_knn` keep it (Gröger et al., 2026).
- Across 24 similarity measures on graph, language and vision models, no measure wins everywhere,
  and preprocessing is part of the measure (Klabunde et al., 2025, ICLR, the ReSi benchmark).

In this library, the sample size, k, preprocessing and pairing are part of a pair record, and
records are compared only at equal values. The imbalance, the overlap and the cycle consistency
read neighborhoods, CKA and SVCCA read the global geometry, and a cross-model claim rests on both.
The full L_A x L_B map is reported rather than its maximum.

The permutation-null calibration of Gröger et al. (2026) is in `req_metrics.calibration`:

- `calibrate` calibrates one score.
- `calibrate_layers` calibrates an aggregate over layer pairs. One permutation is applied to every
  layer, and for the frames of one clip whole groups are permuted.
- `calibrate_grid` gives the calibrated map itself. Every layer pair is calibrated from the same
  draws, so a cell equals `calibrate` on that pair. The maximum over the calibrated cells is not
  the calibrated maximum.
- `jackknife` gives the half-sample spread that Acevedo et al. report.

The imbalance, the overlap and `cycle_knn` reuse their distance or neighbor tables across the
draws, since a row permutation only relabels a table. Calibrating a 25 x 25 grid therefore costs
the tables once plus O(N k) per cell and draw.

`ridge_predictivity` is the encoding-model measure of the brain-alignment literature (Tuckute,
Feather, Boebinger and McDermott, 2023, PLOS Biology, Methods; Pepino, Riera, Kamienkowski and
Ferrer, 2025, arXiv:2511.16849, Sec. 4.3). Ridge regression predicts the columns of the second
representation from the first. Each column is scored on held-out items by the squared Pearson
correlation, with negative correlations set to 0, and the scores are averaged over folds and then
over columns. The median is in the extras. Tuckute et al. take the median over ten random 83/82
splits, and their shipped code fits one `RidgeCV` per target with leave-one-out, as here. As in
both papers, the predictor's columns are demeaned on the training rows and not scaled (Tuckute et
al. found this non-isotropic prior to predict better than normalized columns). The penalty is
chosen per target column by leave-one-out cross-validation inside the training rows. Here the
grid is relative to the largest squared singular value of the training predictor (25 values from
1e-8 to 1e4), where theirs is an absolute 1e-50 to 1e49 that "rarely fell on the boundaries".

Regression finds the subspace of the predictor that explains the target and ignores the rest. This
is why both papers pair it with RSA, and why it is the measure for a layer against a target
computed from the same input (a piano roll, a feature set, a measurement). Their reliability
correction (the attenuation formula with repeated measurements) is left to the caller. The measure
is asymmetric. `compute_pairs` keeps one SVD per layer and fold, so the registry caps it at 4,000
items.

`procrustes` is the squared orthogonal Procrustes distance in the normalization of Ding, Denain
and Steinhardt (2021, NeurIPS, arXiv:2108.01661, Eqs. 6-7). The columns are centered, each matrix
is divided by its Frobenius norm, and the distance is 2 - 2 ||x_a^T x_b||_* in [0, 2], with the
nuclear norm. Unequal widths need no padding (ReSi zero-pads instead, with the same value). The
extras hold the alignment ||x_a^T x_b||_*. Lower is closer. `compute_pairs` keeps one thin SVD per
layer, as for `svcca`, and reads each pair from the singular values of the product of the factors.

Ding et al. score 30,480 representation pairs of BERT and ResNet models by rank correlation with a
functional difference (Table 1):

- Across layers and pretraining seeds, the distance tracks probing accuracy about as well as CKA
  (Spearman 0.86-0.89 against 0.88-0.91).
- It detects the deletion of principal components that CKA misses (Sec. 3.2). CKA's distance to a
  layer stripped of 97% of its components stays below its distance between seeds, while probing
  accuracy falls from 80% to 63%. Procrustes detects the deletion at 85% and PWCCA at 55%
  (benchmark 2, Spearman 0.86 against CKA's 0.75).
- Across fine-tuning seeds, it tracks out-of-distribution accuracy better than CKA (Sec. 4.1, 0.55
  against 0.46, bootstrap intervals in App. E).

With shared singular vectors, the distance is the squared distance between the singular values,
whereas CKA is a quartic dominated by the largest ones (their Discussion). The width effect of
Gröger et al. (2026) applies across models, so a cross-model comparison goes through `calibrate`.

## 10. cluster_quality and dense_structure

`cluster_quality` follows Whetten et al. (2025, Interspeech), who run k-means with k = 1024 and
k-means++ seeding on the frame embeddings of a layer and report the inertia and the
Davies-Bouldin index as label-free indicators computed early in pretraining. In their study the
two correlate with recognition in opposite directions, so read them together and compare them at
the same layer, N and k. Seeded full-batch Lloyd iterations replace the paper's mini-batch
k-means, so values are reproducible.

### dense_structure (Dai, Xu, Wen, Liu and Huang, 2025, NeurIPS, arXiv:2510.17299)

The dense representation structure estimator (DSE) predicts how well a linear head reads the
tokens of a sample set (in the paper, the patches of images). It follows from a nearest-class-mean
error bound (their Thm. 2 and Cor. 5): the error is small when clusters are tight relative to the
distance between them and when the tokens use many dimensions. DSE = M_inter - M_intra +
lambda M_dim, where:

- M_intra is the radius of a k-means cluster, the sum of the singular values of its centered
  tokens over sqrt(N_j - 1), averaged over clusters. It is computed on the tokens of each sample
  with 3 clusters and on the tokens of 8 samples with 24 clusters, and the two are averaged (their
  App. D.4).
- M_inter is the distance of each token to the nearest center of another cluster, from the
  24-cluster grouping.
- M_dim is the effective rank of one token per sample, drawn at random and averaged over draws.
  The samples should outnumber the dimensions.

The estimator needs the sample of every token, so it runs at the population level
(`compute(..., level="population")`, all tokens with `n_tokens=None`), where the pipeline passes
the sample of each token. Called directly, it takes a `(B, T, D)` tensor or `groups`. The default
`variant="released"` computes what the authors' code does. Their numbers come from that code,
which differs from Eq. 5 as printed:

- Tokens are scaled to unit length (`--normalize` in their README).
- M_inter and both radii are divided by the mean distance of the tokens to their mean.
- The radius denominator is sqrt(max(N_j - 1, d)).
- M_inter averages over tokens rather than clusters.
- M_dim is divided by min(samples, d).

These changes make the value free of the scale and, for M_dim, of the width, which comparing
layers and models needs. `variant="paper"` follows Eq. 5 as printed. lambda is 1 in each value.
The paper sets it over the candidates to Std(M_inter - M_intra) / Std(M_dim), which the authors
apply afterwards to the saved components of all checkpoints (`recombine_dense_structure`). Their
code has no license, so this implementation is written from these definitions. Its k-means++
seeding weights by squared distance, where theirs weights by distance.

Across the training checkpoints of sixteen self-supervised vision models, at the last layer and on
2,048 ImageNet images, Kendall's tau between DSE and linear segmentation mIoU averages 0.57. RankMe
and LiDAR reach -0.08 and -0.30 on image-level embeddings, and 0.24 and 0.33 on patch tokens (their
Table 1). Separability alone gives 0.39, and M_dim alone, which is RankMe on tokens, gives 0.24
(Table 10). DSE is weaker for depth estimation, a regression task (tau 0.07 to 0.75 by model,
Table 9). Their selection rule keeps the three best local maxima within two checkpoints (Alg. 1,
`top_local_maxima`) and gains 3.0 mIoU over the last checkpoint at about 1% of the cost of
supervised selection. Whether DSE orders the layers of one model, rather than checkpoints, is
untested.

## 11. jacobian_effective_rank

Chung and Kim (2026, ICML, arXiv:2602.03282, Eq. 1) define it as the participation ratio (sum
s_i)^2 / sum s_i^2 of the k leading singular values of the input-output Jacobian J(x) of a
readout, averaged over inputs. It is at most k. The singular values are estimated by randomized
range finding (Halko et al., 2011) from k random orthonormal input directions with subspace
iteration. Their protocol uses 32 directions, 5 power iterations and 100 ImageNet validation
images on the final embedding (Sec. 4.1, App. I.4), with Gaussian-noise inputs as a control (App.
E.1).

`jacobian_products` implements the estimator, and `LayerMonitor` applies it to the first
`jacobian_items` inputs of the monitoring set, for the readout of every hooked layer. The first
products serve all layers in one forward-mode pass per direction, and each power iteration costs
2k passes per layer. `jacobian_input` sets the tensor the Jacobian is taken with respect to, for
instance a spectrogram rather than the waveform. Values are comparable at equal k, power
iterations, inputs and readout. Chung and Kim find the measure predictive of compositional binding
and state that it is not a universal quality measure.
