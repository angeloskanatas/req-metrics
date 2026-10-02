# `anisotropy`

Spectral anisotropy of the centered, row-normalized matrix.

- Input: `points`
- Canonical preprocessing: `center+l2`
- Tags: paper-canonical
- Shared cache: spectrum
- Cite: `DBLP:conf/eacl/RazzhigaevMGODK24` (The Shape of Learning: Anisotropy and Intrinsic Dimensions in Transformer-Based
                  Models (2024)); `chung2026globalgeometry` (Global Geometry Is Not Enough for Vision Representations (2026)); `he2022whitened` (Exploring the Gap between Collapsed \& Whitened Features in Self-Supervised Learning (2022)); `tsitsulin2023unsupervised` (Unsupervised Embedding Quality Evaluation (2023))

## Definition, protocol and pitfalls

Fraction of total variance on the leading singular direction.

Razzhigaev et al. (2024, EACL Findings): s_1^2 / sum_k s_k^2 of the
centered embedding matrix; 1/D for an isotropic cloud, 1 when all
variance lies on one axis. The canonical protocol centers and then
L2-normalizes each row, so the score measures directional
concentration independent of norm. Chung and Kim (2026,
arXiv:2602.03282) report the complement 1 - lambda_1 / sum(lambda) as
the global isotropy score on the centered, non-normalized spectrum;
with l2=False the extras field equals it. The reciprocal sum(lambda) /
lambda_1 is the normalized eigenvalue sum NESum of He and Ozay (2022,
ICML, Def. 4.1), the whitening measure that equals the stable rank of
Tsitsulin et al. (2023) on centered data; it is returned in the extras
rather than as a separate metric. He and Ozay find its relation to
accuracy non-monotonic (too whitened is also worse), and Arputharaj et
al. (2026, TMLR) find it predictive for self-supervised but not for
supervised vision models.

Args:
    x: Points (N, D), or a Spectrum of already preprocessed points.
    center: Mean-center before the SVD.
    l2: Scale each row to unit norm after centering.

Returns:
    value: anisotropy in (0, 1].
    extras: isotropy_score = 1 - value; ne_sum = 1 / value (NESum, stable rank) of the analyzed matrix.
