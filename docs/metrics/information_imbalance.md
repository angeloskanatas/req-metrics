# `information_imbalance`

Information imbalance from representation A to representation B.

- Input: `pair`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- arXiv: https://arxiv.org/abs/2104.15079
- Cite: `glielmo2022imbalance` (Ranking the information content of distance measures (2022)); `cheng2025emergence` (Emergence of a High-Dimensional Abstraction Phase in Language Transformers (2025)); `acevedo2025semantic` (A quantitative analysis of semantic information in deep representations of text and images (2025)); `glielmo2022dadapy` (DADApy: Distance-based Analysis of DAta-manifolds in Python (2022)); `deltatto2024robust` (Robust inference of causality in high-dimensional dynamical processes from the Information Imbalance of distance ranks (2024))

## Definition, protocol and pitfalls

Glielmo et al. (2022, PNAS Nexus, arXiv:2104.15079, Eq. 2): Delta(A -> B) = 2 <r_B | r_A = 1>
/ N, the mean rank in B of each point's nearest neighbor in A; 2/N for identical spaces without
ties and 1 in expectation for independent ones. A small Delta(A -> B) with a large
Delta(B -> A) means A contains the information in B. k > 1 averages the ranks of the k nearest
A-neighbors, as in DADApy (Glielmo et al., 2022, Patterns, arXiv:2205.03373). Ranks are counted
exactly, without storing an N x N table. Cheng et al. (2025, ICLR, arXiv:2405.15471) compare the
layers of one model and different models with it. Acevedo et al. (2026, TMLR, arXiv:2505.17101
v7) compare translations, same-class images and image-caption pairs, on token-averaged
representations with unit-length (l2=True) or plain Euclidean distances. Their definition
divides by N - 1 instead of N (N / (N - 1) in expectation for independent spaces), and their
code ranks each query against a disjoint half of the items; both change the value by O(1/N).
They report the standard deviation over five half-samples (req_metrics.calibration.jackknife)
and a shuffled-pair null near 1 (req_metrics.calibration.calibrate with lower_is_better=True
and best=0.0).

Ties (discrete data such as binary rolls, labels or repeated items) are handled without an
arbitrary order: tied B-distances share their average rank, as in the code of Del Tatto et al.
(2024, PNAS, arXiv:2305.10817), which keeps 1 as the expectation for independent spaces, and
items tied at A's k-th distance share the remaining weight equally, the expectation under a
random choice. Without ties the value is Glielmo et al.'s count exactly.

Args:
    x_a: Representation A, (N, D_a).
    x_b: Representation B, (N, D_b), rows of the same items as x_a.
    k: Nearest A-neighbors whose B-ranks are averaged.
    l2: Scale rows to unit norm first, so neighbors and ranks are cosine ones.
    neighbors_a, neighbors_b: Precomputed Neighbors tables of the (preprocessed) inputs; reused when they hold
        at least k + 1 neighbors (the extra one detects a tie at the k-th), rebuilt otherwise.

Returns:
    value: Delta(A -> B).
    extras: reverse, Delta(B -> A); asymmetry, Delta(A -> B) - Delta(B -> A) (Acevedo et al.'s A).
