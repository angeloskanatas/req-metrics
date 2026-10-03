"""Spectral metrics: functions of the singular spectrum of a point cloud.

All are read off one Spectrum, so the pipeline computes the SVD once per layer and
preprocessing. Given a tensor, each estimator applies its canonical preprocessing.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from req_metrics._types import InputKind, MetricResult, Preprocess
from req_metrics.preprocess import apply_preprocess
from req_metrics.registry import register_metric
from req_metrics.spectrum import Spectrum


def _spectrum(x: Tensor | Spectrum, pre: Preprocess) -> Spectrum:
    if isinstance(x, Spectrum):
        return x
    return Spectrum.from_points(apply_preprocess(x, pre), center=False)


def _renyi_entropy(p: Tensor, alpha: float) -> float:
    """Renyi entropy of order alpha of a probability vector; alpha = 1 is Shannon."""
    p = p[p > 0]
    if alpha == 1.0:
        return float(-(p * p.log()).sum())
    return float(torch.log((p**alpha).sum()) / (1.0 - alpha))


def _normalize_entropy(h: float, normalization: str, n: int, d: int) -> float:
    if normalization == "raw":
        return h
    denom = {"max": math.log(min(n, d)), "logN": math.log(n), "logD": math.log(d)}.get(normalization)
    if denom is None:
        raise ValueError(f"unknown normalization {normalization!r}")
    return h / denom if denom > 0 else float("nan")


def effective_rank(
    x: Tensor | Spectrum, *, spectrum: str = "singular", center: bool = True, max_eigenvalues: int | None = None
) -> MetricResult:
    """Effective rank: exponential of the Shannon entropy of the normalized spectrum.

    Roy and Vetterli (2007, EUSIPCO); RankMe (Garrido et al., 2023, ICML, arXiv:2210.02885).
    p_k = s_k / sum(s) over the singular values, or s_k^2 / sum(s^2) with spectrum="variance"
    (Skean et al., 2025). RankMe takes the uncentered embeddings and adds a small epsilon to
    p_k; here the default centers, because on autoregressive decoders the uncentered spectrum is
    dominated by the mean direction, and there is no epsilon. RankMe uses 25,600 samples; N
    should be well above D.

    Args:
        x: Points (N, D), or a Spectrum of preprocessed points.
        spectrum: "singular" or "variance".
        center: Mean-center before the SVD.
        max_eigenvalues: Keep only the largest singular values, as a truncated PCA does.

    Returns:
        value: effective rank in [1, min(N, D)]; 0 for a zero matrix (every row equal after
            centering), which has rank 0.
        extras: entropy; normalized_entropy, over log min(N, D) (with spectrum="variance", the
            spectral entropy of Jha et al., 2026, Eq. 1); normalized_rank, value / D; and the
            effective rank under the other convention.
    """
    s = _spectrum(x, Preprocess(center=center))
    if not float(s.singular_values.sum()) > 0.0:  # zero matrix: complete collapse
        return MetricResult(0.0, {})
    if max_eigenvalues is not None and s.singular_values.numel() > max_eigenvalues:
        s = Spectrum(singular_values=s.singular_values[:max_eigenvalues], n=s.n, d=s.d)
    h = _renyi_entropy(s.normalized(spectrum), 1.0)
    h_other = _renyi_entropy(s.normalized("variance" if spectrum == "singular" else "singular"), 1.0)
    h_max = math.log(min(s.n, s.d))
    return MetricResult(
        math.exp(h),
        {
            "entropy": h,
            "normalized_entropy": h / h_max if h_max > 0 else float("nan"),
            "normalized_rank": math.exp(h) / s.d,
            ("variance_convention" if spectrum == "singular" else "singular_convention"): math.exp(h_other),
        },
    )


def matrix_entropy(
    x: Tensor | Spectrum, *, alpha: float = 1.0, normalization: str = "max", center: bool = False
) -> MetricResult:
    """Matrix-based Renyi entropy of the trace-normalized Gram matrix.

    Sanchez Giraldo, Rao and Principe (2015, IEEE Trans. Inf. Theory); Skean et al. (2025, ICML,
    arXiv:2502.02013, Eq. 1). S_alpha = log(sum lambda_i^alpha) / (1 - alpha) over the
    eigenvalues of K / tr(K) with K = Z Z^T; alpha = 1 is the von Neumann entropy. Computed from
    the singular values of Z, without the N x N matrix. Uncentered by default, as in the papers;
    with center=True and alpha = 1 it equals effective_rank's normalized entropy under
    spectrum="variance".

    Args:
        x: Points (N, D), or a Spectrum of preprocessed points.
        alpha: Renyi order.
        normalization: "max" divides by log min(N, D); "logN", "logD" or "raw".
        center: Mean-center before the Gram matrix.

    Returns:
        value: normalized entropy.
        extras: raw entropy in nats.
    """
    s = _spectrum(x, Preprocess(center=center))
    h = _renyi_entropy(s.normalized("variance"), alpha)
    return MetricResult(_normalize_entropy(h, normalization, s.n, s.d), {"raw": h})


def alpha_req(x: Tensor | Spectrum, *, fit_range: tuple[int, int] = (10, 100), center: bool = True) -> MetricResult:
    """Power-law decay exponent of the covariance eigenspectrum (alpha-ReQ).

    Agrawal et al. (2022, NeurIPS): lambda_j ~ j^(-alpha), fitted by weighted least squares in
    log-log space with weights 1/j over fit_range (Stringer et al., 2019). Values near 1 go with
    good representations. The fit range can flip the sign of the correlation with accuracy
    (Arputharaj et al., 2026, App. B.1), so compare alphas only at equal fit_range.

    Args:
        x: Points (N, D), or a Spectrum of preprocessed points.
        fit_range: Half-open range of 0-based eigenvalue indices.
        center: Mean-center before the SVD.

    Returns:
        value: alpha.
        extras: r2 of the log-log fit.
    """
    s = _spectrum(x, Preprocess(center=center))
    eig = s.normalized("variance")
    lo, hi = fit_range
    hi = min(hi, eig.numel())
    if hi - lo < 5:
        raise ValueError(f"fit range {fit_range} leaves fewer than five eigenvalues of {eig.numel()}")
    window = eig[lo:hi]
    if (window <= 1e-10 * eig[0]).any():  # float32 roundoff leaves ~1e-14 relative residues on rank-deficient data
        raise ValueError("eigenvalues inside the fit range are numerically zero; the spectrum is rank deficient there")
    rank = torch.arange(lo + 1, hi + 1, dtype=eig.dtype, device=eig.device)
    y = torch.log(window)
    design = torch.stack([-torch.log(rank), torch.ones_like(rank)], dim=1)
    w = (1.0 / rank).unsqueeze(1)
    coef = torch.linalg.solve(design.T @ (design * w), (design * w).T @ y)
    pred = design @ coef
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = float(1.0 - ((y - pred) ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")
    return MetricResult(float(coef[0]), {"r2": r2})


def anisotropy_spectral(x: Tensor | Spectrum, *, center: bool = True, l2: bool = True) -> MetricResult:
    """Spectral anisotropy: the share of variance on the leading direction.

    Razzhigaev et al. (2024, EACL Findings): s_1^2 / sum s_k^2 of the centered matrix, 1/D for an
    isotropic cloud and 1 for a single axis. Rows are L2-normalized after centering by default,
    so the score ignores norms. With l2=False, 1 - value is the isotropy score of Chung and Kim
    (2026); 1 / value is NESum (He and Ozay, 2022, Def. 4.1), the stable rank on centered data.

    Args:
        x: Points (N, D), or a Spectrum of preprocessed points.
        center: Mean-center before the SVD.
        l2: Scale rows to unit norm after centering.

    Returns:
        value: anisotropy in (0, 1].
        extras: isotropy_score (1 - value), ne_sum (1 / value).
    """
    s = _spectrum(x, Preprocess(center=center, l2=l2))
    lam = s.eigenvalues
    value = float(lam[0] / lam.sum())
    return MetricResult(value, {"isotropy_score": 1.0 - value, "ne_sum": 1.0 / value})


def participation_ratio(
    x: Tensor | Spectrum, *, normalized: bool = True, center: bool = True, correction: str = "none"
) -> MetricResult:
    """Participation ratio (sum lambda)^2 / sum lambda^2 of the covariance eigenvalues.

    Jha et al. (2026, Eq. 2); divided by D, the G.PR of Chung and Kim (2026). The plug-in
    estimate is biased low by about PR/N. Chun, Canatar, Chung and Lee (2026, ICLR,
    arXiv:2509.26560, Sec. 4) give unbiased estimators: "row" corrects the sample size (all
    units observed), "col" the unit subsampling, "both" the two. All four come from one
    closed-form pass over the (N, D) matrix; correction chooses the value. A Spectrum, or
    center=False, gives the plug-in estimate only.

    Args:
        x: Points (N, D), or a Spectrum of preprocessed points (plug-in only).
        normalized: Divide by D.
        center: Mean-center; False gives the uncentered plug-in estimate only.
        correction: "none", "row", "col" or "both".

    Returns:
        value: the chosen estimate, divided by D if normalized.
        extras: participation_ratio (the chosen estimate) and the naive, row, col and both
            estimates, none divided by D.
    """
    if correction not in ("none", "row", "col", "both"):
        raise ValueError(f"unknown correction {correction!r}")
    if isinstance(x, Spectrum) or not center or x.shape[0] < 4:
        if correction != "none":
            raise ValueError("the corrections need the centered (N, D) matrix with N >= 4")
        s = _spectrum(x, Preprocess(center=center))
        lam = s.eigenvalues
        pr = float(lam.sum() ** 2 / (lam**2).sum())
        return MetricResult(pr / s.d if normalized else pr, {"participation_ratio": pr})
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError(f"expected (N, D) with D >= 2, got shape {tuple(x.shape)}")
    est = _chun_participation_ratios(x.double())
    pr = est["naive" if correction == "none" else correction]
    return MetricResult(pr / x.shape[1] if normalized else pr, {"participation_ratio": pr, **est})


def _chun_participation_ratios(x: Tensor) -> dict[str, float]:
    """Plug-in and bias-corrected participation ratios of Chun et al. (2026), from closed-form quartic sums.

    Rows are samples (P), columns units (Q). Each sum of v^{ab}_{ijkl} = X_ia X_ja X_kb X_lb
    over a row-index pattern is evaluated through the Q x Q second-moment matrix C,
    the row norms g, the column sums and the column moments, for all column pairs
    (a, b) and for a = b; distinct-column sums are their difference. The naive,
    row-, column- and both-corrected estimators follow the reference code.
    """
    p, q = x.shape
    xn = x / math.sqrt(p * q)
    c = xn.T @ xn  # (Q, Q)
    g = xn.square().sum(dim=1)  # row norms squared
    cs = xn.sum(dim=0)  # column sums
    c2 = torch.diagonal(c)  # column second moments
    c3 = xn.pow(3).sum(dim=0)
    g1 = xn @ cs  # G 1 = X (X^T 1)
    tr = float(g.sum())
    cs2 = float(cs.square().sum())
    t1, t1s = float(c.square().sum()), float(c2.square().sum())  # ijji
    t2, t2s = float(g.square().sum()), float(xn.pow(4).sum())  # iiii
    t3, t3s = float((g * g1).sum()), float((cs * c3).sum())  # ijjj
    t5, t5s = float(g1.square().sum()), float((cs.square() * c2).sum())  # ijjl
    t6, t6s = tr * tr, t1s  # iijj
    t7, t7s = tr * cs2, t5s  # iijl
    t9, t9s = cs2 * cs2, float(cs.pow(4).sum())  # ijlm
    t1d, t2d, t3d, t5d, t6d, t7d, t9d = t1 - t1s, t2 - t2s, t3 - t3s, t5 - t5s, t6 - t6s, t7 - t7s, t9 - t9s
    f1, f2, f3 = p / (p - 2), 2.0 / (p - 2), 1.0 / ((p - 1) * (p - 2))
    row_factor, col_factor = p / (p - 3), q / (q - 1)
    a_naive = t6 - (2.0 / p) * t7 + (1.0 / p) ** 2 * t9
    b_naive = t1 - (2.0 / p) * t5 + (1.0 / p) ** 2 * t9
    a_row = row_factor * (
        t6 - (2.0 / (p - 1)) * t7 + (1.0 / (p - 2)) * (4.0 * t3 - p * t2) + f3 * (t9 - 4.0 * t5 + 2.0 * t1 - t6)
    )
    b_row = row_factor * (t1 - f1 * t2 + f2 * (2.0 * t3 - t5) + f3 * (t6 - 2.0 * t7 + t9))
    a_col = t6d - (2.0 / p) * t7d + (1.0 / p) ** 2 * t9d
    b_col = t1d - (2.0 / p) * t5d + (1.0 / p) ** 2 * t9d
    a_both = (
        row_factor
        * col_factor
        * (
            t6d
            - (2.0 / (p - 1)) * t7d
            + (1.0 / (p - 2)) * (4.0 * t3d - p * t2d)
            + f3 * (t9d - 4.0 * t5d + 2.0 * t1d - t6d)
        )
    )
    b_both = row_factor * col_factor * (t1d - f1 * t2d + f2 * (2.0 * t3d - t5d) + f3 * (t6d - 2.0 * t7d + t9d))
    return {"naive": a_naive / b_naive, "row": a_row / b_row, "col": a_col / b_col, "both": a_both / b_both}


def eigenvalue_early_enrichment(x: Tensor | Spectrum, *, center: bool = True) -> MetricResult:
    """Top-heaviness of the covariance spectrum over the ambient dimension (EEE).

    Jha et al. (2026, arXiv:2603.06922, Eq. 3): EEE = (2/D) sum_k (S_k - k/D), with S_k the
    cumulative variance fraction of the k largest eigenvalues over all D directions. 0 for a flat
    spectrum, approaching 1 when one direction carries all the variance; scale-invariant.

    Args:
        x: Points (N, D), or a Spectrum of preprocessed points.
        center: Mean-center before the SVD.

    Returns:
        value: EEE in [0, 1).
        extras: none.
    """
    s = _spectrum(x, Preprocess(center=center))
    lam = torch.sort(s.padded_eigenvalues(), descending=True).values
    cum = torch.cumsum(lam, 0) / lam.sum()
    k = torch.arange(1, s.d + 1, dtype=lam.dtype, device=lam.device)
    return MetricResult(float((2.0 / s.d) * (cum - k / s.d).sum()), {})


_P = InputKind.POINTS
register_metric(
    "effective_rank",
    cache="spectrum",
    inputs=_P,
    preprocess=Preprocess(center=True),
    citation=("DBLP:conf/eusipco/RoyV07", "DBLP:conf/icml/GarridoBNL23", "jha2026nerve"),
    arxiv="2210.02885",
    tags=("paper-canonical",),
)(effective_rank)
register_metric(
    "matrix_entropy",
    cache="spectrum",
    inputs=_P,
    preprocess=Preprocess(),
    citation=("giraldo2015matrixentropy", "DBLP:conf/icml/SkeanAZPNLS25"),
    arxiv="2502.02013",
)(matrix_entropy)
register_metric(
    "alpha_req",
    cache="spectrum",
    inputs=_P,
    preprocess=Preprocess(center=True),
    citation=("DBLP:conf/nips/AgrawalMGR22", "stringer2019highdim"),
)(alpha_req)
register_metric(
    "anisotropy",
    cache="spectrum",
    inputs=_P,
    preprocess=Preprocess(center=True, l2=True),
    citation=(
        "DBLP:conf/eacl/RazzhigaevMGODK24",
        "chung2026globalgeometry",
        "he2022whitened",
        "tsitsulin2023unsupervised",
    ),
    tags=("paper-canonical",),
    description="Spectral anisotropy of the centered, row-normalized matrix.",
)(anisotropy_spectral)
register_metric(
    "participation_ratio",
    inputs=_P,
    preprocess=Preprocess(center=True),
    citation=("chung2026globalgeometry", "jha2026nerve", "chun2026dimensionality"),
    arxiv="2602.03282",
)(participation_ratio)
register_metric(
    "eigenvalue_early_enrichment",
    cache="spectrum",
    inputs=_P,
    preprocess=Preprocess(center=True),
    citation=("jha2026nerve",),
    arxiv="2603.06922",
)(eigenvalue_early_enrichment)
