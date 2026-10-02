# `token_norm_outliers`

Fraction of high-norm tokens, and how concentrated the largest token is in a few channels.

- Input: `tokens`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- Origin: https://arxiv.org/abs/2309.16588
- Cite: `darcet2024registers` (Vision Transformers Need Registers (2024)); `jiang2025noregisters` (Vision Transformers Don't Need Trained Registers (2025)); `sun2024massive` (Massive Activations in Large Language Models (2024))

## Definition, protocol and pitfalls

Darcet et al. (2024, ICLR, arXiv:2309.16588) find that large ViTs
repurpose a few low-information patches as high-norm tokens; they use an
absolute cutoff on the token norm (150 for DINOv2) and observe that the value
varies across models, so the default here is relative: a token is an
outlier when its norm exceeds factor times the median token norm. Jiang
et al. (2025, arXiv:2506.08010) trace the high norms to a sparse set of
"register neurons", and Sun et al. (2024, COLM, arXiv:2402.17762)
describe the analogous massive activations in language models as a few
channels orders of magnitude above the median. The extras therefore
report the participation ratio of the squared channel energies of the
highest-norm token (near 1 when one channel carries it, near D when the
energy is spread) and the share of its largest channel.

Args:
    tokens: One clip's tokens, shape (T, D), class tokens removed.
    factor: Relative cutoff on the median norm.
    cutoff: Absolute norm cutoff overriding factor, as in Darcet et al.

Returns:
    value: fraction of outlier tokens.
    extras: max_over_median, median_norm, max_norm, channel_participation_ratio, top1_channel_mass.
