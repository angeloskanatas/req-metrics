# `infonce`

InfoNCE loss between augmented views of the same samples.

- Input: `views`
- Canonical preprocessing: `center+l2`
- Tags: paper-canonical
- Shared cache: none
- arXiv: https://arxiv.org/abs/1807.03748
- Cite: `DBLP:journals/corr/abs-1807-03748` (Representation Learning with Contrastive Predictive Coding (2018)); `DBLP:conf/icml/SkeanAZPNLS25` (Layer by Layer: Uncovering Hidden Representations in Language Models (2025))

## Definition, protocol and pitfalls

van den Oord, Li and Vinyals (2018, arXiv:1807.03748, Eq. 4): the cross-entropy of
identifying each sample's view b among all N samples' views b from its view a, with cosine
logits over the temperature. By default rows are centered and L2-normalized, as in the code of
Skean et al. (2025), whose paper does not state it. Lower is more invariant to the
augmentations. With q > 2 views the loss is averaged over the pairs a < b, the full graph of
Tian et al. (2020, Eq. 8), or over the pairs (anchor, b), their core view (Eq. 7), for a
non-exchangeable view such as a clean or global one. symmetric=True adds the reverse direction
of each pair (Tian et al., Eq. 4). The bound of van den Oord et al., log N - L, cannot exceed
log N; for unit vectors with nearly orthogonal negatives and identical views it approaches the
smaller of log N and 1 / temperature. Compare values at equal N and temperature, and read
contrastive_accuracy, which has no such ceiling.

Args:
    views: Views (q, N, D), q >= 2.
    temperature: Softmax temperature.
    center: Mean-center each view over samples.
    l2: Scale rows to unit norm.
    symmetric: Average both directions of each pair.
    anchor: View paired with every other view; None pairs all views.

Returns:
    value: mean loss in nats.
    extras: log_n_minus_loss, contrastive_accuracy (top-1 of the positive), n_pairs.
