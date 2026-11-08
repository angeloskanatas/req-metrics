# `ridge_predictivity`

Held-out variance of one representation's columns predicted by ridge regression from the other (Tuckute et al., 2023); the ordering matters.

- Input: `pair`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- Cite: `tuckute2023many` (Many but not all deep neural network audio models capture brain responses and exhibit correspondence between model stages and brain regions (2023)); `pepino2025brainlike` (Better audio representations are more brain-like: linking model-brain alignment with performance in downstream auditory tasks (2025))

## Definition, protocol and pitfalls

Held-out variance of x_b's columns predicted by ridge regression from x_a.

The encoding-model protocol of Tuckute, Feather, Boebinger and McDermott (2023, PLOS Biology,
doi:10.1371/journal.pbio.3002366, Methods) and Pepino, Riera, Kamienkowski and Ferrer (2025,
arXiv:2511.16849, Sec. 4.3). The items are split into folds. On each training fold the
predictor's columns are demeaned but not scaled (their non-isotropic prior), and the ridge
penalty of each target column is chosen by leave-one-out cross-validation within the training
rows. The held-out predictions are scored by the squared Pearson correlation with the target
column, with negative correlations set to 0, and the folds are averaged; Tuckute et al. take the
median over ten random 83/82 splits instead. Regression finds the subspace of x_a that predicts
x_b and ignores the rest, where RSA and CKA weigh every direction (Tuckute et al., Results;
Pepino et al., Sec. 2.1.2). The reliability (noise-ceiling) correction of Tuckute et al. is not
applied. The penalty grid is relative to the data: lambda_grid gives (low, high, count)
multiples of the largest squared singular value of the training predictor, log-spaced. Each fold
needs one SVD of the training predictor, in float64; the reverse direction is computed the same
way.

Args:
    x_a: (N, D_a), the predictor.
    x_b: (N, D_b), the representation predicted, scored column by column.
    n_folds: Seeded K-fold split of the items.
    seed: Seed of the split.
    lambda_grid: (low, high, count) of the relative penalty grid.

Returns:
    value: mean over x_b's columns of the held-out r^2.
    extras: median (over columns), reverse and reverse_median (x_a predicted from x_b), n_folds,
        log10_lambda_median and reverse_log10_lambda_median (median log10 of the chosen penalties, per
        direction).
