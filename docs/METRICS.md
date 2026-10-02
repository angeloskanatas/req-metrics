# Metric reference

What every metric computes, where it comes from, how it was verified, and
what the published analysis of Kanatas et al. (2026) recorded. The full
per-metric text lives in the estimator docstrings and is rendered unchanged
into `docs/metrics/` (one card per metric, built by
`scripts/build_metric_cards.py`); this document holds what cuts across
metrics.

## 1. Conventions

- Estimators are pure functions on tensors. `x` is a point cloud `(N, D)`, `z`
  a single trajectory `(T, D)`, `views` a stack `(q, N, D)` of `q` views of
  the same `N` clips, a token field is `(T, D)` per clip. Each returns
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
  | spectral, relational, intrinsic dimension, local geometry | a point cloud `(N, D)` | one vector per clip (class token for ViTs and global average pooling for CNNs in Arputharaj et al. 2026; backbone output in RankMe and LiDAR; time mean in Kanatas et al. 2026; frequency-concatenated or partitioned means for 2D audio encoders), or the token cloud of a clip (Skean et al. 2025 prompt entropy), or frames after a trajectory layout |
  | trajectory curvature | a time-ordered `(T, D)` sequence per clip | frame encoders directly; 2D encoders through the frequency-concatenated or frequency-mean trajectory |
  | token fields | the patch tokens of one clip, class token kept for `cls_patch_cosine` | Darcet et al. 2024, Marouani et al. 2026, DINOv3 |
  | views and equivariance | one vector per clip and per view or shift | pooled embeddings in Thilak et al. 2024 and Wang and Isola 2020; for pitch, a frequency-preserving readout keeps the quantity the probe must find |
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
- Preprocessing is never implicit. A metric whose definition is on the
  covariance centers inside the estimator and says so; anything else is
  applied by the pipeline from the registry entry and written into the record.
- Population (`pooled`, `frames`, `tokens`) is a pipeline argument, not a
  property of the estimator.
- Citations name author, year and venue; arXiv ids are given once per metric.
  "Published protocol" means the configuration behind the reported numbers of
  Kanatas et al. (2026).
- Tags mark provenance with respect to that paper and are carried into every
  record: `paper-canonical` (the variant and parameters behind its headline
  correlation analysis), `heldout-canonical` (the variant used in its held-out
  validation), `paper-figure-config` (the configuration of one of its figures),
  `methods-stated` (the variant its methods text names), `computed-not-in-paper`
  (computed in that analysis, not reported), `unpublished-variant`; `relational`
  and `collapse-indicator` are family markers.

## 2. Families, in the vocabulary of Arputharaj et al. (2026)

| Family (TMLR 2026) | req-metrics modules and metrics | Input |
|---|---|---|
| Spectral | spectral: effective_rank (RankMe, with normalized_rank = RankMe*), effective_rank/variance, spectral_entropy, matrix_entropy, alpha_req, anisotropy (NESum in extras), participation_ratio (+ corrected), eigenvalue_early_enrichment, gaussianity, sparsity | points |
| Relational | relational: self_clustering, uniformity, normalized_std; anisotropy/cosine | points |
| Manifold | dimension: intrinsic_dimension (TwoNN), gride, mle, mlid, mst_dimension; local_geometry: neighborhood_curvature, local_rectifiability | points |
| Not in that taxonomy | trajectory: trajectory_curvature (frames); views: lidar, infonce, dime, alignment (augmented views); equivariance: pte (pitch shifts); compare: information_imbalance (layer pairs); tokens and norms (token fields) | frames, views, shifts, pairs, tokens |

The study's own set is alpha-ReQ, RankMe, NE Sum, condition number, Self-Cluster,
DSE and TwoNN ID, all on the final backbone output of 260 vision models; this
registry covers all of them except condition number and DSE (rejected, with
reasons, in Section 6) and adds the frame, view, shift, pair and token
families that single-vector studies cannot express.



## 3. Verification against reference implementations

Every port was checked against the code it was adopted from; where a
reference implementation deviates from the published definition, the
docstring says so.

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
  exposed and the default is stated on the card.
- Neighbor group. TwoNN and GRIDE follow DADApy step for step and reproduce
  its solver to machine precision (estimates and Fisher errors at every scale,
  checked by loading DADApy's own likelihood functions on the same ratios).
  DADApy labels each GRIDE scale by its outer rank, so `gride_k8` is the
  estimate from the ratio of the 8th to the 4th neighbor distance. The
  Levina-Bickel MLE follows their Eq. 8 and 9 with k = 10..20, not
  scikit-dimension's single k = 5. The estimators are ported rather than
  wrapped so that one kNN table serves every estimator on torch tensors and
  GPUs without a scikit-learn and Cython dependency; the port is attributed in
  NOTICE and tested for parity.
- Trajectory group. Signed and absolute conventions and k in {1, 4} agree
  with the analysis code of Kanatas et al. (2026) to 1e-7; analytic nulls
  hold (straight line 0, independent frames 120 and 60 degrees, random walk
  90 degrees, regular polygon exact). Zero-length steps are excluded and
  counted rather than scored as 90 degrees.
- View group. LiDAR reproduces the analysis code of Kanatas et al. (2026) to
  2e-14 with their biased denominators and delta 1e-6; the default follows
  Thilak et al.'s unbiased estimates and the delta 1e-4 of the Skean et al.
  (2025) analysis code. InfoNCE matches to 1e-9. DiME's joint entropy matches
  repitl to 1e-15; unlike that analysis code, it never swaps the N x N Gram
  Hadamard product for D x D covariances when N > D, since the two differ
  (single-matrix entropies agree, Hadamard products do not).
- Equivariance. PTE reproduces the published training loop (mixed-shift
  batches, Adam, early stopping, best state restored) and agrees with the
  analysis code on identical data and seeds to six decimals.
- Comparison group. The information imbalance agrees with DADApy's
  `_return_imbalance` on full index tables to 1e-10 for k = 1 and k = 3;
  ranks are counted per chunk, so no N x N table is stored.
- Relational group and corrections: see Section 6.

## 4. Reproducing Kanatas et al. (2026)

`protocols.get("kanatas2026")` pins the variants and parameters per input
kind. The canonical set:

| metric | estimator / variant | preprocessing | population | views |
|---|---|---|---|---|
| intrinsic_dimension | GRIDE at the 8th-neighbor scale (correlation analysis); TwoNN (methods text) | none | pooled | 1 |
| effective_rank | singular spectrum | center | pooled | 1 |
| anisotropy | spectral | center + L2 | pooled | 1 |
| trajectory_curvature | k = 1, signed (text); absolute, negated (correlation analysis) | none | frames, 1,000 to 5,000 clips | 1 |
| lidar | effective-rank readout, delta 1e-6, biased denominators | none (LDA centers) | pooled | 10 |
| infonce | temperature 0.3 | center + L2 | pooled | 2 |
| pte | shared probe, linear, omega 7, phase distance | none | pooled | 1 + 11 shifts |

10,000 clips of 15 seconds, one per track; pooled vectors are time means for
encoders and the final-token state for autoregressive decoders.

What the published result files recorded and what this toolkit records: the
files carry centering, L2 normalization, the estimator method, the subset
size and the seed, but not the number of views for LiDAR and InfoNCE, the
InfoNCE temperature, the LiDAR ridge, the curvature gap, the ID estimator's
neighbor rank, or the pooling rule. Every record written here therefore
carries metric and variant, estimator parameters, preprocessing, population
and pooling label, number of views, number of items, seed, representation
width and the library version. Anisotropy was stored in two spectral variants
(with and without row normalization); the published one is centered + L2.
Frame-level curvature was computed on 1,000 to 5,000 clips per model, not
10,000. The PTE row of the published correlation analysis selects, per
model, the configuration among {linear, mlp} x {phase, cpsd} with the largest
absolute correlation; `pte` reproduces the canonical linear and phase score of
the text, `pte/mlp` and `pte/cpsd` the other configurations.

## 5. Intrinsic dimension: caveats

MLE, TwoNN and GRIDE target the pointwise dimension, which cannot increase
under Lipschitz maps (Schulte and Rügamer, 2026, AISTATS); every standard
layer type is Lipschitz, so a layer-wise ID profile that rises is an
estimator artefact driven by growing nearest-neighbour distances and
representation norms, not a rising true dimension. Estimates are biased lower
bounds whose bias is not consistent across layers, and the layer-wise
pattern co-moves with the von Neumann entropy of the centered Gram matrix
(`spectral_entropy`, `effective_rank/variance`). Report ID profiles as
geometric descriptors, not as manifold dimensions, and read them next to
the spectral entropies. The plug-in participation ratio has a separate,
sample-size bias of about PR/N (Chun et al., 2026); see
`participation_ratio/corrected`.

Duplicates matter for the two-neighbour estimators. An exact duplicate has
r_1 = 0 and an infinite ratio, and its neighbours a ratio of exactly 1; DADApy's
2NN keeps such rows unless `remove_identical_points` is called, whereas
`twonn` here removes exact duplicate rows first and records the count used.
On MERT-v1-95M embeddings of 990 GTZAN clips, which contain 13 exact
duplicates, the two conventions differ by about 20 percent at every layer
(for example 12.6 against 15.6 at layer 6), while GRIDE at the scale of the
8th neighbour is unaffected; both match DADApy to all printed digits under
the same convention. Deduplicate the corpus before estimating, as the
published protocol does, or read `extras["n_used"]`.

## 6. Relational metrics, collapse indicators and corrections

Sources for this section: Tsitsulin, Munkhoeva and Perozzi (2023,
TAG-ML at ICML, arXiv:2305.16562); He and Ozay (2022, ICML); Wang and Isola
(2020, ICML, arXiv:2005.10242) with their reference code; Chen and He (2021,
CVPR, arXiv:2011.10566); Chun, Canatar, Chung and Lee (2026, ICLR,
arXiv:2509.26560) with their reference code; Arputharaj, Jönsson and Eilertsen
(2026, TMLR, arXiv:2608.23182); Schulte and Rügamer (2026, AISTATS,
arXiv:2604.20276); Liao et al. (2024, CISS, arXiv:2312.04823) with their code.

### self_clustering (Tsitsulin et al., 2023, Def. 3.5)
Points: L2-normalized rows W, no centering. Q = sum over all pairs of the squared
cosine, compared with its uniform-sphere expectation N + N(N-1)/D and its
collapse maximum N^2: (Q - N - N(N-1)/D) / (N^2 - N - N(N-1)/D). The paper
writes Q as the Frobenius norm of W W^T; the expectation and the maximum it
states are those of the squared norm, which this implementation uses so that a
single point gives exactly 1 and a uniform cloud 0. Computed from the D x D
second moment in O(N D^2). The TMLR study reports it as a reliable negative
predictor for self-supervised vision models and uninformative for supervised
ones, with values near 0 or 1 for supervised ViTs, and rho = -0.999 with
diffusion spectral entropy.

### uniformity and alignment (Wang and Isola, 2020)
uniformity: log of the mean pairwise Gaussian potential exp(-t ||u - v||^2)
over L2-normalized points, t = 2, computed from Gram blocks in float64.
Corollary 1 range [-2t + log 0F1(; D/2; t^2), 0], the lower end only for a
perfectly uniform encoder (extras: lower_bound, gap). alignment: mean over
view pairs and clips of ||u_a - u_b||^alpha, alpha = 2, on L2-normalized views
(input kind views). Both equal the two-line reference implementation.

### normalized_std (Chen and He, 2021, Sec. 4.1)
Mean over channels of the sample std of z / ||z||_2: 0 under complete collapse,
about 1/sqrt(D) for an isotropic cloud (extras: reference and ratio). A
complete-collapse monitor only; it does not see dimensional collapse.

### participation_ratio corrections (Chun et al., 2026)
1/PR_naive is about 1/N + 1/D + 1/PR (their Sec. 3), so the plug-in estimate is
biased low by roughly PR/N. correction="row" removes the sample-size term, the
case of network activations where all units are observed (their Sec. 4.5);
"both" also removes the unit-subsampling term. The corrected estimators center
algebraically and take the raw matrix; `participation_ratio/corrected` registers
the row-corrected variant without pre-centering. Quartic index sums are evaluated
in closed form (Gram matrix, column moments) and agree with the MIT reference
code to 1e-12 (NOTICE). The plug-in estimate remains the default for parity
with published numbers.

### Extras added to existing metrics
anisotropy.ne_sum = sum(lambda)/lambda_1 (NESum of He and Ozay, 2022; stable rank
of Tsitsulin et al., 2023, on centered data). effective_rank.normalized_rank =
RankMe/D (RankMe* of Tsitsulin et al., 2023). alpha_req: the fit range changes the
sign of the correlation with accuracy (TMLR study, Appendix B.1); record it.

### Considered and not adopted (relational and spectral candidates)
Condition number (Tsitsulin et al.): sign reversals across datasets in their
Table 3, and the TMLR study shows its correlation with accuracy is an artefact of
OLS conditioning that disappears under a k-NN probe. Coherence (Tsitsulin et
al.): their Table 1 marks it data-dependent and least stable; not in the TMLR
study. Diffusion spectral entropy (Liao et al.): bandwidth sigma in absolute
embedding units with no scale rule (paper exp(-d^2/sigma) versus code
exp(-d^2/(2 sigma^2)), default 10), which confounds layer-wise comparison as
norms grow with depth; reference code non-commercial; rho = -0.999 with
self_clustering. Layer-wise representation dynamics (Jiang et al., 2026): no
code and no peer review yet; candidates for the layer-pair family. Persistence
(Shestov et al., 2025): persistent homology on recommender embeddings, which
would add a persistent-homology dependency.


## 7. neighborhood_overlap

Doimo, Glielmo, Ansuini and Laio (2020, NeurIPS, arXiv:2007.03506, Eq. 1);
Valeriani et al. (2023, NeurIPS) for the transformer use. Layer-pair
input: the mean over items of the fraction of k nearest neighbors (Euclidean,
self excluded) shared between two representations; 1 when neighborhoods are
preserved, k/(N-1) at chance. k = 30 in both papers at ImageNet scale, trend
robust to k. `compute_pairs(layers, metric="neighborhood_overlap")` builds one
k-NN table per layer and intersects them for every pair. Symmetric, so it
answers a different question from the information imbalance (directional
predictability of neighbor ranks); the two are the layer-pair family. The
retention score of Jiang et al. (2026) is the Jaccard variant of this quantity.
