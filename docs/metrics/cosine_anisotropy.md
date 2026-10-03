# `cosine_anisotropy`

Cosine anisotropy: mean cosine similarity between distinct samples.

- Input: `points`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- Cite: `ethayarajh2019contextual` (How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings (2019)); `DBLP:conf/eacl/GodeyCS24` (Anisotropy Is Inherent to Self-Attention in Transformers (2024)); `timkey2021rogue` (All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality (2021))

## Definition, protocol and pitfalls

Ethayarajh (2019, EMNLP); Godey et al. (2024, EACL). Uncentered, it reflects the shared mean
direction, which the spectral anisotropy of the centered matrix removes; Timkey and van
Schijndel (2021, EMNLP) show it is often driven by a few rogue dimensions. Exact in O(N D)
from the sum of the unit vectors.

Args:
    x: Points (N, D).
    center: Mean-center before normalizing.

Returns:
    value: mean off-diagonal cosine in [-1/(N-1), 1].
    extras: none.
