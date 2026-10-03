# `anisotropy`

Spectral anisotropy of the centered, row-normalized matrix.

- Input: `points`
- Canonical preprocessing: `center+l2`
- Tags: paper-canonical
- Shared cache: spectrum
- Cite: `DBLP:conf/eacl/RazzhigaevMGODK24` (The Shape of Learning: Anisotropy and Intrinsic Dimensions in Transformer-Based
                  Models (2024)); `chung2026globalgeometry` (Global Geometry Is Not Enough for Vision Representations (2026)); `he2022whitened` (Exploring the Gap between Collapsed \& Whitened Features in Self-Supervised Learning (2022)); `tsitsulin2023unsupervised` (Unsupervised Embedding Quality Evaluation (2023))

## Definition, protocol and pitfalls

Spectral anisotropy: the share of variance on the leading direction.

Razzhigaev et al. (2024, EACL Findings): s_1^2 / sum s_k^2 of the centered matrix, 1/D for an
isotropic cloud and 1 for a single axis. Rows are L2-normalized after centering by default,
so the score ignores norms. With l2=False, 1 - value is the isotropy score of Chung and Kim
(2026) and 1 / value is NESum (He and Ozay, 2022, Def. 4.1), the stable rank of the centered
matrix.

Args:
    x: Points (N, D), or a Spectrum of preprocessed points.
    center: Mean-center before the SVD.
    l2: Scale rows to unit norm after centering.

Returns:
    value: anisotropy in (0, 1].
    extras: isotropy_score (1 - value), ne_sum (1 / value).
