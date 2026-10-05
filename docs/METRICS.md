# Metric reference

Definitions, sources and verification of the metrics, and the protocol of Kanatas et
al. (2026). The per-metric text is in the estimator docstrings and is rendered into
`docs/metrics/` by `scripts/build_metric_cards.py`; this document covers what
applies across metrics.

## 1. Conventions

- Estimators are pure functions on tensors. `x` is a point cloud `(N, D)`, `z`
  a single trajectory `(T, D)`, `views` a stack `(q, N, D)` of `q` views of
  the same `N` samples, a token field is `(T, D)` per sample. Each returns
  `MetricResult(value, extras)`.
- Layouts: metrics never see token grids. `layouts.py` turns an `(F, T, D)`
  spectrogram-patch grid into a `(T, F * D)` trajectory by concatenating the
  frequency patches of each time step (MSM-MAE, Niizumi et al., 2022, Sec.
  3.3), into an `(F * T, D)` token cloud, or into one pooled vector; frame
  sequences are already trajectories. The choice is recorded in `pooling`,
  and `make_pooler` offers the same readouts by name during training. What
  each family expects, with the readout its sources used:

  | Family | Input the estimator sees | Readouts and precedent |
  |---|---|---|
  | spectral, relational, intrinsic dimension, local geometry | a point cloud `(N, D)` | one vector per sample (class token for ViTs and global average pooling for CNNs in Arputharaj et al. 2026; backbone output in RankMe and LiDAR; time mean in Kanatas et al. 2026; frequency-concatenated or partitioned means for 2D audio encoders), the tokens of one sample (Skean et al. 2025 prompt entropy) or of all samples (Razzhigaev et al. 2024), or the time steps of a trajectory layout |
  | trajectory curvature | a time-ordered `(T, D)` sequence per sample | frame encoders directly; 2D encoders through the frequency-concatenated or frequency-mean trajectory |
  | token fields | the patch tokens of one sample, class token kept for `cls_patch_cosine` | Darcet et al. 2024, Marouani et al. 2026, DINOv3 |
  | views and equivariance | one vector per sample and per view or shift | pooled embeddings in Thilak et al. 2024 and Wang and Isola 2020; for pitch, a frequency-preserving readout keeps the quantity the probe must find |
- Readouts: label-free metrics take parameter-free readouts only: class
  token, mean of patch tokens, their concatenation, max, the last token of a
  causal decoder, frequency-concatenated time mean (MSM-MAE, Riou et al.
  2024), and block-partitioned means (Gu et al. 2026). Learnable probe-side
  readouts (attention pooling, learned layer fusion) are trained with task
  labels and would make the metric measure the probe; a pooling head trained
  by the self-supervised objective itself is part of the frozen model and is
  labelled as its own pooling. The vision proxies (RankMe, LiDAR, alpha-ReQ,
  IdEst) are all computed on fixed features.
- Inclusion rule: only metrics defined in a published paper are registered,
  each with its input contract, canonical preprocessing and citation keys
  into `references.bib`.
- Preprocessing is never implicit. Each estimator's `center`, `standardize` and
  `l2` arguments define what it applies; the registry holds their defaults, the
  pipeline applies them before building the shared spectrum, and every record
  stores the preprocessing actually applied.
- The level (`sequence`, `sample`, `population`) is a pipeline argument, not a
  property of the estimator; see Section 2.
- Citations name author, year and venue; arXiv ids are given once per metric.
  "Published protocol" means the configuration behind the reported numbers of
  Kanatas et al. (2026).
- Tags are carried into every record: `paper-canonical` marks the metrics and
  parameters of that paper's protocol; `relational` and `collapse-indicator` are
  family markers.

## 2. Families, in the vocabulary of Arputharaj et al. (2026)

| Family (TMLR 2026) | req-metrics modules and metrics | Input |
|---|---|---|
| Spectral | spectral: effective_rank (RankMe, with normalized_rank = RankMe*, the variance convention and NerVE's spectral entropy in the extras), matrix_entropy, alpha_req, anisotropy (NESum in extras), participation_ratio (bias corrections in extras), eigenvalue_early_enrichment | points |
| Relational | relational: self_clustering, uniformity, normalized_std, cosine_anisotropy | points |
| Manifold | dimension: intrinsic_dimension/twonn, /gride, /mle, /mlid, /mst; local_geometry: neighborhood_curvature, local_rectifiability | points |
| Not in that taxonomy | clustering: cluster_quality (k-means); distribution: gaussianity, sparsity, embedding_norm; trajectory: trajectory_curvature (sample level); views: lidar, infonce, dime, alignment (augmented views); equivariance: pte (pitch shifts); compare: information_imbalance, neighborhood_overlap, cycle_knn, cka, svcca, rsa (representation pairs); tokens (token fields); functional: jacobian_effective_rank (the model and its inputs) | points, trajectories, views, shifts, pairs, tokens, Jacobian sketches |

The study's own set is alpha-ReQ, RankMe, NE Sum, condition number, Self-Cluster,
DSE and TwoNN ID, all on the final backbone output of 260 vision models. This
registry covers all of them except condition number and DSE (rejected, with reasons,
in Section 6) and adds the trajectory, view, shift, pair and token families that
single-vector studies cannot express.

### Levels

The level decides which vectors form the point cloud; the estimator does not change.
The inputs are vectors the caller extracted: one per sample, or the tokens of each
sample (the frames of a 1D encoder, the patches of a 2D one).

- `sequence`: one vector per sample, `layers[l]` of shape `(N, D)` (token or time
  mean, class token, final token, or a grid readout). The setting of RankMe, LiDAR,
  Arputharaj et al. (2026) and Kanatas et al. (2026), who call these sequence-level
  metrics, and of the layer-wise intrinsic-dimension studies of Valeriani et al.
  (2023, token mean) and Cheng et al. (2025, ICLR, last token). Sample sizes:
  `docs/DESIGN.md`, section 6. With several clips per track, a clip's nearest
  neighbors can be clips of its own track (the regime effect described under
  `population`); `group_ids` keeps one clip per track, as Kanatas et al. (2026) did.
- `sample`: each sample's tokens form one cloud, `layers[l]` a sequence of
  `(T_i, D)` tensors; the estimator runs per sample and the values are aggregated
  (mean of the finite values; std, median, min, max and the count of failures in the
  extras). Viswanathan et al. (2025) estimate the intrinsic dimension of each
  prompt's 1,024 tokens and average over prompts; Pedashenko et al. (2026, EACL) do
  the same per text; Skean et al. (2025) compute prompt entropy on the tokens of one
  prompt; Sadok and Alameda-Pineda (2026) compute their speech metrics per sample
  and then average; trajectory curvature (Hosseini and Fedorenko, 2023) and DINOv3's
  Gram anchoring are per sample by definition. The value describes the sample's own
  manifold, so a neighbor estimator here sees temporally adjacent frames, and it
  depends on the token count T_i: a spectrum has at most min(T_i, D) nonzero values,
  and Pedashenko et al. exclude texts shorter than 150 tokens, where the variance of
  their estimates is high. Compare at equal T_i, that is, equal duration and token
  rate.
- `population`: the tokens of all samples form one cloud, `layers[l]` a sequence of
  `(T_i, D)` tensors, from which n tokens are drawn at random. This is the setting
  of the anisotropy literature, where pairs of token vectors are drawn across a
  corpus (Ethayarajh, 2019; Godey et al., 2024; Timkey and van Schijndel, 2021); of
  Razzhigaev et al. (2024), who compute anisotropy and TwoNN on batches of at least
  4,096 vectors from different contexts, shuffled before batching for TwoNN, and
  average over batches; of Ruppik et al. (2025, NeurIPS), who pool the tokens of
  7,000 to 10,000 sequences, deduplicate them and draw 60,000; and of the global
  effective rank of Whetten et al. (2025), over all frames of about an hour of
  audio. Spectral metrics need n above D, as at the sequence level. For neighbor
  estimators the number of samples sets the regime (Osman, Baroni and Macocco,
  2026): with n points from c samples, about m = n / c per sample, the k neighbors
  of a point can all be tokens of its own sample while k < m (the local regime,
  close to the sample level) and must reach other samples once k >= m (the global
  regime). The transition at c = n / k adds a spurious peak to layer profiles and
  reverses the dependence on c; the argument assumes, as they verify for words, that
  a sample's tokens lie closer to each other than to other samples' tokens. TwoNN
  reads two neighbors, so its value is in the global regime only up to two tokens
  per sample (n <= 2c), and mLID at k = 64 up to 64. Compare such values at equal n,
  k and sample count. Records of this level carry `n_samples`, and those of the
  neighbor estimators `same_sample_fraction`, the mean share of a point's neighbors
  behind the value that are tokens of its own sample (their Sec. 5 diagnostic): near
  1 in the local regime, near 1 / c without sample structure. Repeated draws with
  different seeds give the batch spread of Razzhigaev et al.; class, register and
  first-position tokens can distort such a cloud (Timkey and van Schijndel, 2021,
  report cosine above 0.99 between position-0 tokens), so strip them first
  (`layouts.strip_prefix_tokens`, or `n_prefix` with the `tokens` readout).
  Token-field metrics whose definition pairs tokens within one sample
  (`cls_patch_cosine`, `token_cosine`; registry `per_sample`) run per sample here
  too, while `token_norm_outliers` reads the population tokens, as Darcet et al.
  (2024) set their cutoff from a pooled norm histogram.

Rosina Fernandez, Guillaume and Wisniewski (2025) compare the cosine similarity of
frame pairs from the same recording and from different recordings and find the two
distributions similar; no other study cited compares the sample and population
levels of one estimator on the same data. Viswanathan et al. (2025, App. D) find
that token-level and prompt-level intrinsic dimension follow different trends, and
Osman et al. (2026) that pooled neighbor estimates change regime with the number of
items, so the level is part of the protocol and is recorded with every value. The
papers' own names differ: Pasad et al. (2023) call pooled frames frame-level,
Kanatas et al. (2026) call per-clip token sequences frame-level, and Hosseini et al.
(2026) call a per-sequence average sequence-level; the levels here are named by how
the vectors are grouped.

## 3. Verification against reference implementations

Every port was checked against the code it was adopted from; where a reference
implementation deviates from the published definition, the docstring says so.

- Spectral group. Matrix-based entropy is the Renyi entropy of the squared
  singular values, so it is computed from the shared spectrum without an
  N x N matrix. The repitl `matrixAlphaEntropy` alpha = 2 shortcut divides
  the Frobenius norm by N^2 and overstates the entropy of a trace-normalized
  Gram matrix by exactly 2 log N (17.30 versus 1.29 at N = 3000); alpha = 1
  and non-integer alpha agree to 1e-13. Constants verified in the reference
  code: LeJEPA's Epps-Pulley grid t in [-5, 5] with 17 points, target and
  weight exp(-t^2/2), 256 directions; VISReg's sliced Wasserstein quantiles
  i/(N+1); RankMe's 25,600 samples; Chung and Kim's isotropy score
  1 - lambda_1 / sum(lambda). reptrix computes RankMe through PCA, which
  centers, while the RankMe paper does not center; both conventions are
  exposed and the default is stated on the card. RankMe's epsilon inside the
  logarithm is omitted, so zero singular values contribute nothing. The
  reference matrix-entropy code clamps negative Gram entries to zero, which
  raised the entropy by 13 to 22 percent on audio foundation-model states; the
  Gram matrix is not clamped here. RankMe-t (Aldeneh et al., 2024) is the
  effective rank of time-summed sample vectors; for samples of equal length this is the
  effective rank of the mean-pooled vectors, since the two differ by one global
  scale.
- Neighbor group. TwoNN and GRIDE follow DADApy step for step and reproduce
  its solver to machine precision (estimates and Fisher errors at every scale,
  checked by loading DADApy's own likelihood functions on the same ratios).
  DADApy labels each GRIDE scale by its outer rank, so `gride_k8` is the
  estimate from the ratio of the 8th to the 4th neighbor distance. The
  Levina-Bickel MLE follows their Eq. 8 and 9 with k = 10..20, not
  scikit-dimension's single k = 20. The estimators are ported rather than
  wrapped so that one kNN table serves every estimator on torch tensors and
  GPUs without a scikit-learn and Cython dependency; the port is attributed in
  NOTICE and tested for parity.
- Trajectory group. Signed and absolute conventions and k in {1, 4} agree
  with the analysis code of Kanatas et al. (2026) to 1e-7; analytic nulls
  hold (straight line 0, independent frames 120 and 60 degrees, random walk
  90 degrees, regular polygon exact). Zero-length steps are excluded and
  counted rather than scored as 90 degrees.
- View group. LiDAR reproduces the analysis code of Kanatas et al. (2026) to
  2e-14 with delta 1e-6; the default delta 1e-4 is that of the Skean et al.
  (2025) analysis code. Thilak et al. state unbiased estimates without giving
  denominators; the choice rescales S_b and S_w by constants, which leaves the
  value unchanged at delta 0 and otherwise changes only delta's relative size.
  InfoNCE matches to 1e-9 on two views; with more views it averages the
  two-view loss over view pairs. DiME's joint entropy matches repitl to 1e-15;
  unlike that analysis code, it never swaps the N x N Gram Hadamard product for
  D x D covariances when N > D, since the two differ (single-matrix entropies
  agree, Hadamard products do not).
- Equivariance. PTE reproduces the published training loop (mixed-shift
  batches, Adam, early stopping, best state restored) and agrees with the
  analysis code on identical data and seeds to six decimals.
- Comparison group. The information imbalance agrees with DADApy's
  `_return_imbalance` on full index tables to 1e-10 for k = 1 and k = 3;
  ranks are counted per chunk, so no N x N table is stored. Linear CKA
  reproduces the recorded outputs of the authors' reference notebook, biased
  and debiased, to 1e-11, and its Gram form when the width exceeds N. SVCCA
  equals `cca_core.get_cca_similarity` of the reference code at epsilon 0 on
  the SVD-reduced representations to 1e-12.
- Clustering. The Davies-Bouldin index and the inertia equal scikit-learn's
  `davies_bouldin_score` and k-means inertia on the same labels and seeding.
- Relational group and corrections: see Section 6.

## 4. Reproducing Kanatas et al. (2026)

`protocols.get("kanatas2026")` pins the variants and parameters per input kind. The
canonical set:

| metric | estimator / variant | preprocessing | level | views |
|---|---|---|---|---|
| intrinsic_dimension/twonn, /gride | TwoNN; GRIDE at the 8th-neighbor scale, reported as consistent | none | sequence | 1 |
| effective_rank | singular spectrum, largest 2048 values | center | sequence | 1 |
| anisotropy | spectral | center + L2 | sequence | 1 |
| trajectory_curvature | k = 1, signed; the folded convention is in the extras | none | sample | 1 |
| lidar | delta 1e-6, n and nq denominators, largest 2048 eigenvalues | none (LDA centers) | sequence | 10 |
| infonce | temperature 0.3 | center + L2 | sequence | 2 |
| pte | linear probe, omega 7, phase distance | none | sequence | 1 + 11 shifts |

10,000 clips of 15 seconds, one per track; pooled vectors are time means for
encoders and the final-token state for autoregressive decoders.

Every record written here carries metric and variant, estimator parameters,
preprocessing, level and pooling label, number of views, number of items, seed,
representation width and the library version. PTE is trained with a linear probe on
10,000 clips; the cross-power distance that includes the magnitude is in the extras
of every run next to the phase distance.

## 5. Intrinsic dimension: caveats

MLE, TwoNN and GRIDE target the pointwise dimension, which cannot increase under
Lipschitz maps (Schulte and Rügamer, 2026, AISTATS); every standard layer type is
Lipschitz, so a layer-wise ID profile that rises is an estimator artifact driven by
growing nearest-neighbor distances and representation norms, not a rising true
dimension. Estimates are biased lower bounds whose bias is not consistent across
layers, and the layer-wise pattern co-moves with the von Neumann entropy of the
centered Gram matrix (the variance convention of `effective_rank`). Report ID
profiles as geometric descriptors, not as manifold dimensions, and read them next to
the spectral entropies. The plug-in participation ratio has a separate, sample-size
bias of about PR/N (Chun et al., 2026); see the `correction` argument of
`participation_ratio`.

Duplicates matter for the two-neighbor estimators. An exact duplicate has r_1 = 0
and an infinite ratio, and its neighbors a ratio of exactly 1; DADApy's 2NN keeps
such rows unless `remove_identical_points` is called, whereas the neighbor
estimators here (TwoNN, GRIDE, MLE, mLID, neighborhood curvature) remove exact
duplicate rows first. On MERT-v1-95M embeddings of 990 GTZAN clips, which contain 13
exact duplicates, the two conventions differ by about 20 percent at every layer (for
example 12.6 against 15.6 at layer 6), while GRIDE at the scale of the 8th neighbor
is unaffected; both match DADApy to all printed digits under the same convention.
Deduplicate the corpus before estimating, as the published protocol does;
`compute()` records the number of distinct rows in `extras["n_distinct"]` when
duplicates were removed, and `twonn` reports the count it used in
`extras["n_used"]`.

## 6. Relational metrics, collapse indicators and corrections

Sources for this section: Tsitsulin, Munkhoeva and Perozzi (2023, TAG-ML at ICML,
arXiv:2305.16562); He and Ozay (2022, ICML); Wang and Isola (2020, ICML,
arXiv:2005.10242) with their reference code; Chen and He (2021, CVPR,
arXiv:2011.10566); Chun, Canatar, Chung and Lee (2026, ICLR, arXiv:2509.26560) with
their reference code; Arputharaj, Jönsson and Eilertsen (2026, TMLR,
arXiv:2608.23182); Schulte and Rügamer (2026, AISTATS, arXiv:2604.20276); Liao et
al. (2024, CISS, arXiv:2312.04823) with their code.

### self_clustering (Tsitsulin et al., 2023, Def. 3.5)
Points: L2-normalized rows W, no centering. Q = sum over all pairs of the squared
cosine, compared with its uniform-sphere expectation N + N(N-1)/D and its collapse
maximum N^2: (Q - N - N(N-1)/D) / (N^2 - N - N(N-1)/D). The paper writes Q as the
Frobenius norm of W W^T; the expectation and the maximum it states are those of the
squared norm, which this implementation uses so that a single point gives exactly 1
and a uniform cloud 0. Computed from the D x D second moment in O(N D^2). The TMLR
study reports it as a reliable negative predictor for self-supervised vision models
and uninformative for supervised ones, with values near 0 or 1 for supervised ViTs,
and rho = -0.999 with diffusion spectral entropy.

### uniformity and alignment (Wang and Isola, 2020)
uniformity: log of the mean pairwise Gaussian potential exp(-t ||u - v||^2) over
L2-normalized points, t = 2, computed from Gram blocks in float64. Corollary 1 range
[-2t + log 0F1(; D/2; t^2), 0], the lower end only for a perfectly uniform encoder
(extras: lower_bound, gap). alignment: mean over view pairs and samples of ||u_a -
u_b||^alpha, alpha = 2, on L2-normalized views (input kind views). Both equal the
two-line reference implementation.

### normalized_std (Chen and He, 2021, Sec. 4.1)
Mean over channels of the sample std of z / ||z||_2: 0 under complete collapse,
about 1/sqrt(D) for an isotropic cloud (extras: reference and ratio). A
complete-collapse monitor only; it does not see dimensional collapse.

### participation_ratio corrections (Chun et al., 2026)
1/PR_naive is about 1/N + 1/D + 1/PR (their Sec. 3), so the plug-in estimate is
biased low by roughly PR/N. correction="row" removes the sample-size term, the case
of network activations where all units are observed (their Sec. 4.5); "both" also
removes the unit-subsampling term. The corrected estimators center algebraically and
take the raw matrix; all estimates come from one pass and are in the extras of every
record. Quartic index sums are evaluated in closed form (Gram matrix, column
moments) and agree with the MIT reference code to 1e-12 (NOTICE). The plug-in
estimate remains the default for parity with published numbers.

### Extras added to existing metrics
anisotropy.ne_sum = sum(lambda)/lambda_1 (NESum of He and Ozay, 2022; stable rank of
Tsitsulin et al., 2023, on centered data). effective_rank.normalized_rank = RankMe/D
(RankMe* of Tsitsulin et al., 2023). alpha_req: the fit range changes the sign of
the correlation with accuracy (TMLR study, Appendix B.1); record it.

### Considered and not adopted (relational and spectral candidates)
Condition number (Tsitsulin et al.): sign reversals across datasets in their Table
3, and the TMLR study shows its correlation with accuracy is an artifact of OLS
conditioning that disappears under a k-NN probe. Coherence (Tsitsulin et al.): their
Table 1 marks it data-dependent and least stable; not in the TMLR study. Diffusion
spectral entropy (Liao et al.): bandwidth sigma in absolute embedding units with no
scale rule (paper exp(-d^2/sigma) versus code exp(-d^2/(2 sigma^2)), default 10),
which confounds layer-wise comparison as norms grow with depth; reference code
non-commercial; rho = -0.999 with self_clustering. Layer-wise representation
dynamics (Jiang et al., 2026): no code and no peer review yet; CKA and SVCCA, the
published measures their subspace distances build on, are in the layer-pair family.
Dense representation structure estimator (Dai et al., 2025, NeurIPS,
arXiv:2510.17299): the released code computes a different quantity from the paper's
Eq. 5 (scale normalizations and an intra-cluster denominator that the paper does not
state, and no lambda, which Eq. 5 defines over the checkpoints of a run), so
published values cannot be reproduced from the definition; the code carries no
license. Parameter- and representation-prediction probes (Plachouras et al., 2025,
IJCNN): they train a probe for every layer, transformation and evaluation, which is
too costly for monitoring during training, and the paper evaluates them on
final-layer features rather than as layer-selection measures; the authors' toolkit
provides them. Persistence (Shestov et al., 2025): persistent homology on
recommender embeddings, which would add a persistent-homology dependency. Task
Priors (Patel and Balestriero, 2025, NeurIPS UniReps workshop): the expected value
and variance of a linear objective under a Gibbs prior over label graphs, in closed
form; the prior needs a kernel, the model's own or a reference model's, and a
temperature, neither with a selection rule, the paper and its code use different
kernel normalizations, and in the high-temperature limit the mean is the kernel
alignment of the model with the prior, which `cka` normalizes. Q-Score (Kalibhat et
al., 2024, AAAI): a per-sample score from the strongly active features whose
activation rate lies in a percentile band tuned per model and dataset; it is
validated as a predictor of which samples a linear probe misclassifies, not for
ranking layers, checkpoints or models, and its authors state that their observations
do not directly extend to ViT encoders, whose representations are signed and not
sparse.

## 7. neighborhood_overlap and cycle_knn

Doimo, Glielmo, Ansuini and Laio (2020, NeurIPS, arXiv:2007.03506, Eq. 1); Valeriani
et al. (2023, NeurIPS) for the transformer use. Pair input: the mean over items of
the fraction of k nearest neighbors (self excluded) shared by two representations; 1
when neighborhoods are preserved, k/(N - 1) in expectation for unrelated spaces
(Gröger, Wen and Brbić, 2026, Prop. 4.2). The mutual k-nearest-neighbor alignment of
Huh et al. (2024, ICML, App. A, Eq. 11) is the same quantity: they compute it with k
= 10 on 1,024 image-caption pairs after clamping each dimension at its 0.95 quantile
and L2-normalizing, so that inner-product neighbors are cosine neighbors; `l2=True`
ranks cosine neighbors here and is recorded as preprocessing. k = 30 in Doimo et al.
and Valeriani et al. at ImageNet scale. Koepke et al. (2026) show that at fixed k
the value falls as the gallery grows (0.135 at n = 1,024 to 0.008 at 15 million for
k = 10, DINOv2 against OpenLlama), while k = n/100 is stable, and that it also falls
when an item has several valid partners (many captions per image), which RSA and CKA
do not register: compare values at equal n and k on one-to-one pairs, and read them
as a local statistic. `compute_pairs(layers, metric="neighborhood_overlap")` builds
one k-NN table per layer and intersects them for every pair. Symmetric, so it
answers a different question from the information imbalance (directional
predictability of neighbor ranks) and from `cycle_knn` below. The retention score
of Jiang et al. (2026) is the Jaccard variant of this quantity.

`cycle_knn` is the cycle k-nearest-neighbor consistency that Huh et al. (2024, App. A,
Table 11) list beside the mutual k-NN and compute in their code as `knn_A[knn_B]`: the
fraction of items that are among the k nearest neighbors in A of one of their k nearest
neighbors in B, a first hop in B and a return hop in A. Zhang et al. (2026) write it
cycle-kNN(A -> B) and Gröger et al. (2026, Eq. 35) cycle-kNN_k(A, B). It reads the same
k-NN tables as the overlap but is not symmetric for k >= 2: Zhang et al. prove symmetry
for k = 1 and give a six-point example with 5/6 in one ordering and 1/2 in the other
(App. A), and read the gap between the orderings as a difference in how compact the two
spaces' neighborhoods are. Across 58 point-cloud, vision and language models
they find the gap positive toward language in 530 of 638 vision-language pairs (mean
0.010 at k = 10 on 1,024 WIT pairs), with the sign unchanged for k from 3 to 50 (App.
E.4), where CKA and the overlap are symmetric by construction; the paper is a preprint.
Gröger et al. find that it keeps its trend with language-model capability after
permutation calibration, as the overlap does (App. E.8). Identical representations score
1 only when every item is a nearest neighbor of one of its own nearest neighbors, so the
diagonal of a layer map falls below 1 where items have no reciprocal neighbor. For
independent representations each return hop succeeds with probability k/(N - 1) (Gröger
et al., Prop. C.9), which bounds the chance level by k^2/(N - 1), stored as
`chance_bound`. `compute_pairs(layers, metric="cycle_knn")` records the value A -> B for
every ordered pair and the other ordering in `extras["reverse"]`.

## 8. cka, svcca and rsa

Pair similarity indices for the same items in two representations. `cka` is linear
centered kernel alignment (Kornblith et al., 2019, ICML, arXiv:1905.00414, Table 1),
invariant to orthogonal maps and isotropic scaling; `debiased=True` uses the
unbiased HSIC estimator, which matters when N is not large relative to the widths.
`svcca` (Raghu et al., 2017, NeurIPS, arXiv:1706.05806) keeps the SVD directions
that carry 99 percent of the summed singular values (App. A) and averages the
canonical correlations between them (Eq. 1); it is invariant to invertible linear
maps of the kept subspaces and therefore needs N well above the kept widths. Post
hoc, `compute_pairs` gives the layer-by-layer similarity map of a model. During
training, the same fixed items at two checkpoints give the drift of each layer, the
use of Raghu et al., Sec. 4.1, who compare every layer during training with its
final state; `LayerMonitor(drift_metrics=[...])` computes this at every sweep, each
layer against its first or its previous sweep on the monitoring items. Both are
closed-form; per-layer summaries are computed once per call.
Under independence the biased linear CKA has a baseline of order D/N (Gröger et al.,
2026, Prop. 4.1; Murphy, Zylberberg and Fyshe, 2024), which `debiased=True` removes;
Gröger et al. find that their permutation calibration agrees with that correction.
The maximum over the L_A x L_B pairs of a map is inflated by the number of pairs
(their Sec. 4.2): report the map, and calibrate a reported maximum against pairings
permuted over items.

`rsa` is representational similarity analysis (Kriegeskorte, Mur and Bandettini,
2008): each representation gives the vector of distances between all pairs of items,
and the two vectors are compared by Spearman rank correlation, 1 for the same
geometry and 0 in expectation for unrelated representations. The distance is cosine
by default, the choice of Koepke et al. (2026), with Euclidean and the paper's
correlation distance as options, and `method="pearson"` correlates the raw
distances. It grades the full ordering of pairs where CKA weights the leading
directions, which is why Koepke et al. find cross-modal RSA at 51 to 58 percent of
the vision-vision ceiling against 83 to 92 percent for CKA on the same pairs (their
Sec. 4); Gröger et al. (2026) calibrate it against permuted pairings like the other
measures. Cost is N(N - 1)/2 distances per representation, built in row chunks;
`compute_pairs` holds one ranked vector per layer, so the registry caps it at 4,000
items.

## 9. Pairs across models and modalities

The pair metrics take any two representations of the same items with aligned rows:
the layers of one model, two checkpoints, two models, or two modalities with paired
items. Cheng et al. (2025, ICLR, Fig. 4, App. G and H) compare the layers of two
language models with the information imbalance and linear CKA on the last-token
states of 10,000 sequences of 20 tokens, averaged over corpora and partitions, and
find the lowest imbalance where the intrinsic-dimension peaks of the two models
intersect. Acevedo et al. (2025) compare translations of one sentence across
languages, images of one class, and image-caption pairs (DeepSeek-V3 against DINOv2
and image-GPT on Flickr30k): the imbalance reaches its minimum in each model's
semantic layers and is asymmetric between modalities. Their protocol concatenates
the last 20 text tokens or the last 200 image tokens (a `pool` callable here),
binarizes activations with the sign function and ranks Hamming distances, which are
the Euclidean ranks of the signed vectors (`torch.sign(x)`), on 5,000 pairs; a
shuffled pairing gives 1 at every layer. Huh et al. (2024) measure cross-modal
alignment as the neighborhood overlap of paired images and captions (Section 7) and
report that it rises with language-model performance. Koepke et al. (2026) find that
the rise saturates for recent models, that fixed-k overlap decays with gallery size
and with many-to-many pairing while RSA (Section 8) and CKA stay stable, and that
the agreement which survives is coarse: cross-modal RSA peaks on the top three
eigenmodes and erodes as finer modes are added. Gröger et al. (2026) find that after
permutation calibration the convergence reported by CKA, SVCCA and Procrustes
distance disappears while the neighborhood overlap keeps it. Zhang et al. (2026,
preprint) use the asymmetry of `cycle_knn` (Section 7) to ask in which direction the
convergence runs, and find vision and point-cloud models closer to the neighborhood
structure of language than the reverse. Klabunde et al. (2025, ICLR; the ReSi
benchmark) ground 24 similarity measures in six tests over graph, language and vision
models: no measure wins everywhere; the k-NN Jaccard overlap ranks first in vision,
linear CKA and distance correlation in language, neighborhood measures in graphs, and
RSA and SVCCA do not stand out. In their layer-monotonicity test on ResNet-18 (their
Table 3) the Spearman correlation between similarity and layer distance is 0.87 for
CKA and 0.97 for RSA and distance correlation, against 0.55 for the k-NN Jaccard, and
two Procrustes measures that differ only in a unit-norm step rank differently, so the
preprocessing is part of the measure. For this library: the sample size, k,
preprocessing and pairing are part of a pair record and are compared only at equal
values; the imbalance, the overlap and the cycle consistency read neighborhoods, CKA
and SVCCA the global geometry, and a cross-model claim rests on both; the full L_A x
L_B map is reported rather than its maximum.

Considered and not adopted from this literature: the centered kernel
nearest-neighbor alignment of Huh et al. (2024, App. A), a k-NN-masked CKA; the
variable-k overlap of Koepke et al. (2026), defined for a query set inside a growing
gallery; the k-NN Jaccard overlap of ReSi, |intersection| / |union| per item where
`neighborhood_overlap` is |intersection| / k, the same neighbor sets under a per-item
monotone transform; the permutation null calibration of Gröger et al. (2026): their
App. E.4 shows the calibrated CKA agreeing with the debiased estimator that
`debiased=True` provides, their Theorem C.10 gives the overlap's null k/(N - 1) in
closed form, the imbalance's is 1 by construction and the cycle consistency's is
bounded by k^2/(N - 1), their App. E.7 finds no width drift for RSA, and the remaining
use, a p-value for a maximum over layer pairs, needs 200 or more permutations of the
whole map (App. E.6); the Procrustes and angular shape distances of Williams et al.
(2021, NeurIPS), proper metrics that admit clustering and nearest-neighbor analyses
over collections of networks, which need equal widths or a PCA step, lose their
convergence trend after calibration in Gröger et al. (App. E.8) and rank in the upper
half of ReSi without leading any domain; the displacement cosine of Shang et al. (2026),
which fits an orthogonal map on held-out items; the barycentric consistency of Saha,
He and Khosla (2026) and the Procrustes dispersion of Hosseini et al. (2026), which
score items across a pool of models; and the ridge predictivity of He, Trott and
Khosla (2025), a fitted mapping.

## 10. cluster_quality

Whetten et al. (2025, Interspeech): k-means with k = 1024 and k-means++ seeding on
the frame embeddings of a layer, reported as the inertia and the Davies-Bouldin
index, as label-free indicators computed early in pretraining. In their study the
two correlate with recognition in opposite directions, so they are read together and
compared at the same layer, N and k. Full-batch Lloyd iterations with a seed replace
the paper's mini-batch k-means, so values are reproducible.

## 11. jacobian_effective_rank

Chung and Kim (2026, ICML, arXiv:2602.03282, Eq. 1): the participation ratio (sum
s_i)^2 / sum s_i^2 of the k leading singular values of a readout's input-output
Jacobian J(x), averaged over inputs; at most k. The singular values are estimated by
randomized range finding (Halko et al., 2011) from k random orthonormal input
directions with subspace iteration; their protocol uses 32 directions, 5 power
iterations and 100 ImageNet validation images on the final embedding (Sec. 4.1, App.
I.4), with Gaussian-noise inputs as a control (App. E.1). `jacobian_products`
implements the estimator and `LayerMonitor` applies it to the first `jacobian_items`
inputs of the monitoring set, for the readout of every hooked layer: the first
products serve all layers in one forward-mode pass per direction, and each power
iteration costs 2k passes per layer. `jacobian_input` sets the tensor the Jacobian
is taken with respect to (a spectrogram rather than the waveform, for instance).
Values are comparable at equal k, power iterations, inputs and readout. Chung and
Kim find the measure predictive of compositional binding and state that it is not a
universal quality measure.

