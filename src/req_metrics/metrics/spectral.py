"""Spectral metrics: functions of the singular spectrum of a point cloud.

Effective rank, matrix-based entropy, alpha-ReQ, spectral anisotropy and
eigenvalue early enrichment are read off one Spectrum, so a pipeline computes the SVD once per layer and
preprocessing choice and passes the Spectrum instead of the tensor.

Every estimator applies its canonical preprocessing itself when given a
tensor, so a direct call reproduces the published definition; when given a
Spectrum it trusts that the caller preprocessed the data the same way.
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


def effective_rank(x: Tensor | Spectrum, *, spectrum: str = "singular", center: bool = True) -> MetricResult:
    """Effective rank: exponential of the Shannon entropy of the normalized spectrum.

    Roy and Vetterli (2007, EUSIPCO) define it on the singular values,
    p_k = s_k / sum(s), erank = exp(-sum p_k log p_k), bounded by 1 and the
    rank. RankMe (Garrido et al., 2023, ICML) uses the same quantity as a
    label-free predictor of linear-probe accuracy and for hyperparameter
    selection, computed on 25,600 samples; their appendix shows convergence
    in the number of samples for 2048-dimensional outputs, so N should be an
    order of magnitude above D. RankMe-t (Aldeneh et al., 2024) is the same
    quantity on frame sequences summed over time, one vector per utterance,
    which is the pooled population with mean pooling up to a per-clip scale
    that the effective rank ignores. Skean et al. (2025, ICML) and the reptrix
    library normalize the eigenvalues of the covariance instead (s_k^2),
    which weights the leading directions more heavily; pass
    spectrum="variance" for that convention. Both conventions come from the
    same spectrum, so the other one is always in the extras. With
    spectrum="variance", normalized_entropy is the spectral entropy of NerVE
    (Jha et al., 2026, arXiv:2603.06922, Eq. 1).

    The matrix is mean-centered before the SVD. RankMe as published does
    not center; the reptrix reference implementation does, through PCA.
    Centering changes conclusions: on autoregressive decoders the
    uncentered spectrum is dominated by the mean direction and the
    layer-wise correlation with downstream accuracy changes sign. RankMe's
    epsilon inside the logarithm is omitted; zero singular values contribute
    zero entropy exactly.

    Args:
        x: Points (N, D), or a Spectrum of already preprocessed points.
        spectrum: "singular" (Roy-Vetterli, RankMe) or "variance" (Skean, reptrix).
        center: Mean-center before the SVD.

    Returns:
        value: effective rank in [1, min(N, D)].
        extras: entropy, normalized_entropy (over log min(N, D)), normalized_rank = value / D
            (RankMe* of Tsitsulin et al., 2023, the fraction of the width in use), and the
            effective rank under the other spectrum convention.
    """
    s = _spectrum(x, Preprocess(center=center))
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

    Sanchez Giraldo, Rao and Principe (2015, IEEE Trans. Inf. Theory) define
    S_alpha(K) = log(sum_i lambda_i^alpha) / (1 - alpha) on the eigenvalues
    of K / tr(K); alpha = 1 is the von Neumann entropy. Skean et al. (2025,
    ICML, Eq. 1) apply it to the Gram matrix K = Z Z^T of a prompt's token
    states ("prompt entropy") and of a dataset's mean-pooled states
    ("dataset entropy"); with a population argument these are the frames and
    pooled populations. The nonzero eigenvalues of Z Z^T are the squared
    singular values of Z, so the entropy is computed from the spectrum
    without forming an N x N matrix. The Gram matrix is not clamped: the
    reference implementation zero-clamps negative entries, which is not a
    numerical safeguard and inflated the entropy by 13 to 22 percent on audio
    foundation-model states. The reference library's alpha = 2 shortcut
    divides the squared Frobenius norm by N^2, which assumes a kernel matrix
    with unit diagonal; on a trace-normalized Gram matrix it overstates the
    entropy by exactly 2 log N. The entropy here is computed from the
    eigenvalues for every alpha. Not centered by default, following the papers;
    with center=True and alpha = 1 it equals the normalized entropy of
    effective_rank(spectrum="variance").

    Args:
        x: Points (N, D), or a Spectrum of already preprocessed points.
        alpha: Renyi order; 1.0 gives von Neumann entropy.
        normalization: "max" divides by log min(N, D); "logN", "logD", "raw".
        center: Mean-center before the Gram matrix (off in the papers).

    Returns:
        value: normalized entropy.
        extras: raw entropy in nats.
    """
    s = _spectrum(x, Preprocess(center=center))
    h = _renyi_entropy(s.normalized("variance"), alpha)
    return MetricResult(_normalize_entropy(h, normalization, s.n, s.d), {"raw": h})


def alpha_req(x: Tensor | Spectrum, *, fit_range: tuple[int, int] = (10, 100), center: bool = True) -> MetricResult:
    """Power-law decay exponent of the covariance eigenspectrum (alpha-ReQ).

    Agrawal et al. (2022, NeurIPS) fit lambda_j ~ j^(-alpha) to the sorted
    eigenvalues of the empirical covariance. Small alpha (at most about 1)
    indicates a dense encoding, large alpha a rapidly decaying, sparse one;
    both too high and too low a value imply poor generalization, with good
    representations in a range close to 1, matching the infinite-width
    linear-regression result that min-norm solutions generalize iff alpha = 1.
    The fit is the Stringer et al. (2019, Nature) recipe used by the
    reference implementation: weighted least squares in log-log space over
    the eigenvalue indices in fit_range, with weights 1/j to down-weight the
    tail. The exponent is independent of how the spectrum is normalized.
    The fit range changes conclusions: Arputharaj et al. (2026, TMLR,
    Appendix B.1) find the correlation of alpha with accuracy flipping sign
    between this range and indices [10, 0.9 D], so record fit_range with
    every value and do not compare alphas fitted over different ranges.

    Args:
        x: Points (N, D), or a Spectrum of already preprocessed points.
        fit_range: Half-open range of 0-based eigenvalue indices used in the fit.
        center: Mean-center before the SVD (the covariance is centered by definition).

    Returns:
        value: alpha.
        extras: r2 of the log-log fit over fit_range.
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
    """Fraction of total variance on the leading singular direction.

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
    """
    s = _spectrum(x, Preprocess(center=center, l2=l2))
    lam = s.eigenvalues
    value = float(lam[0] / lam.sum())
    return MetricResult(value, {"isotropy_score": 1.0 - value, "ne_sum": 1.0 / value})


def participation_ratio(
    x: Tensor | Spectrum, *, normalized: bool = True, center: bool = True, correction: str = "none"
) -> MetricResult:
    """Participation ratio of the covariance eigenvalues, (sum lambda)^2 / sum lambda^2.

    A classical count of directions that carry variance, between 1 and D
    (NerVE, Jha et al., 2026, Eq. 2). Divided by D it is the global
    participation ratio G.PR of Chung and Kim (2026, arXiv:2602.03282),
    in (0, 1] with 1 for an isotropic cloud; that paper finds it, like other
    global geometry statistics, uncorrelated with compositional binding,
    which is the caveat to carry.

    The plug-in estimate is biased downward at finite N: Chun, Canatar, Chung
    and Lee (2026, ICLR, arXiv:2509.26560, Sec. 3) show 1/PR_naive is about
    1/N + 1/D + 1/PR, so the relative bias is about PR/N. Their unbiased
    estimators average the quartic index sums over distinct indices (Sec. 4):
    "row" removes the sample-size bias, the case of network activations where
    all D units are observed (their Sec. 4.5), "col" removes the
    unit-subsampling term and "both" removes the two. The plug-in and the three
    corrected estimates come from one pass over the (N, D) matrix, in closed
    form from the D x D second-moment matrix and column moments, checked
    against the reference implementation; all four are in the extras and
    correction chooses the value. Use "row" when N is below about 100 times the
    expected ratio, which includes small monitoring buffers. A Spectrum, or
    center=False, gives the plug-in estimate only.

    Args:
        x: Points (N, D), or a Spectrum of already preprocessed points (plug-in only).
        normalized: Divide by D.
        center: Mean-center; False gives the uncentered plug-in estimate only.
        correction: "none" (plug-in), "row", "col" or "both" (Chun et al., 2026).

    Returns:
        value: the chosen estimate, divided by D if normalized.
        extras: participation_ratio (the chosen estimate, not divided by D) and the
            naive, row, col and both estimates.
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

    NerVE (Jha et al., 2026, arXiv:2603.06922, Eq. 3): the mean gap between
    the cumulative variance fraction of the k largest eigenvalues and the
    uniform reference k/D, normalized to [0, 1): EEE = (2/D) sum_k (S_k - k/D)
    over all D ambient directions, unused ones counted as zero variance.
    0 for a flat spectrum, approaching 1 when one direction carries all the
    variance. Scale-invariant.

    Args:
        x: Points (N, D), or a Spectrum of already preprocessed points.
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
    preprocess=Preprocess(),
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
