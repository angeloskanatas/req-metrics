# `information_imbalance`

Information imbalance from representation A to representation B.

- Input: `pair`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- arXiv: https://arxiv.org/abs/2104.15079
- Cite: `glielmo2022imbalance` (Ranking the information content of distance measures (2022)); `cheng2025emergence` (Emergence of a High-Dimensional Abstraction Phase in Language Transformers (2025)); `acevedo2025semantic` (A quantitative analysis of semantic information in deep representations of text and images (2025)); `glielmo2022dadapy` (DADApy: Distance-based Analysis of DAta-manifolds in Python (2022))

## Definition, protocol and pitfalls

Glielmo et al. (2022, PNAS Nexus, arXiv:2104.15079, Eq. 2): Delta(A -> B) = 2 <r_B | r_A = 1>
/ N, the mean rank in B of each point's nearest neighbor in A; about 2/N for identical spaces
and 1 for independent ones. A small Delta(A -> B) with a large Delta(B -> A) means A contains
the information in B. k > 1 averages the ranks of the k nearest A-neighbors, as in DADApy.
Ranks are counted exactly, without an N x N table. Used between the layers of one model and
between models by Cheng et al. (2025, ICLR), and between languages, images and image-caption
pairs by Acevedo et al. (2025), who binarize activations and rank Hamming distances: pass
torch.sign(x), whose Euclidean ranks are the Hamming ranks.

Args:
    x_a: Representation A, (N, D_a).
    x_b: Representation B, (N, D_b), rows of the same items as x_a.
    k: Nearest A-neighbors whose B-ranks are averaged.
    l2: Scale rows to unit norm first, so neighbors and ranks are cosine ones.
    neighbors_a, neighbors_b: Precomputed Neighbors tables with at least k neighbors.

Returns:
    value: Delta(A -> B).
    extras: reverse, Delta(B -> A).
