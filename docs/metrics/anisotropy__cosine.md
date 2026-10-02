# `anisotropy/cosine`

Mean pairwise cosine similarity between distinct samples.

- Input: `points`
- Canonical preprocessing: `none`
- Tags: unpublished-variant
- Shared cache: none
- Cite: `ethayarajh2019contextual` (How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings (2019)); `DBLP:conf/eacl/GodeyCS24` (Anisotropy Is Inherent to Self-Attention in Transformers (2024)); `timkey2021rogue` (All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality (2021))

## Definition, protocol and pitfalls

Ethayarajh (2019, EMNLP) measures anisotropy as the expected cosine
between representations of random inputs; Godey et al. (2024, EACL) track
the same average across layers and attribute it to self-attention.
Computed exactly in O(N D) from the sum of the unit vectors. Timkey and
van Schijndel (2021, EMNLP) show this measure is often dominated by one
to five rogue dimensions and recommend standardizing before computing
it; centering alone removes the shared mean direction.

Args:
    x: Points (N, D).
    center: Mean-center before normalizing (off in the papers).

Returns:
    value: mean off-diagonal cosine in [-1/(N-1), 1].
    extras: none.
