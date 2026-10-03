# `token_norm_outliers`

Fraction of high-norm tokens, and the channel concentration of the largest one.

- Input: `tokens`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- Origin: https://arxiv.org/abs/2309.16588
- Cite: `darcet2024registers` (Vision Transformers Need Registers (2024)); `jiang2025noregisters` (Vision Transformers Don't Need Trained Registers (2025)); `sun2024massive` (Massive Activations in Large Language Models (2024))

## Definition, protocol and pitfalls

Darcet et al. (2024, ICLR, arXiv:2309.16588) set an absolute cutoff (150 for DINOv2) from the
norm histogram of the tokens of many images and note that it varies across models, so a token
here is an outlier when its norm exceeds factor times the median of the tokens given: one
clip under the "frames" population, the pooled tokens under "tokens". The extras describe the
largest token's channel energies, after Jiang et al. (2025) and Sun et al. (2024).

Args:
    tokens: One clip's tokens (T, D), class tokens removed.
    factor: Cutoff relative to the median norm.
    cutoff: Absolute cutoff overriding factor.

Returns:
    value: fraction of outlier tokens.
    extras: max_over_median, median_norm, max_norm, channel_participation_ratio,
        top1_channel_mass.
