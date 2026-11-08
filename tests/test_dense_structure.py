"""Dense representation structure estimator: parity with the authors' released code, invariances and the pipeline route."""

import math
import unittest

import torch

import req_metrics as rq
from req_metrics.metrics.clustering import _cluster_radius, _erank, dense_structure, recombine_dense_structure

# Values printed by estimator.py of github.com/EldercatSAM/SSL-Degradation (commit b75fd51), run on the inputs below.
REFERENCE_COMPONENTS = [  # (n, d, centered_singular_sum, effective_rank / min(n, d))
    (7, 12, 6.385323778259514, 0.8850277478485805),
    (40, 12, 11.927058010418701, 0.9535869463809242),
    (12, 40, 10.277416567547295, 0.9492489704606665),
]
REFERENCE_DSE = {  # normalize -> (M_inter, M_intra for one image, M_intra for 8 images, M_dim, DSE), compute_DSE defaults
    False: (0.013640605361717779, 3.370295310074754e-08, 3.370295091249872e-08, 0.8792371153831482, 0.892877687041914),
    True: (
        0.013820974149672878,
        3.3420182915082955e-08,
        3.3420182915082955e-08,
        0.8989875316619873,
        0.9128084723914772,
    ),
}


def clips(b=16, n=60, d=12, delta=1e-2, noise=1e-8, seed=0):
    """b samples of n tokens: one centre per sample, three sub-clusters delta apart, tiny noise."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(b, 1, d, generator=g, dtype=torch.float64)
    offs = delta * torch.randn(b, 3, d, generator=g, dtype=torch.float64)
    lab = torch.arange(n) % 3
    return base + offs[:, lab] + noise * torch.randn(b, n, d, generator=g, dtype=torch.float64)


class ReleasedCodeParityTests(unittest.TestCase):
    def test_radius_and_normalized_effective_rank(self):
        g = torch.Generator().manual_seed(1)
        for n, d, radius, erank in REFERENCE_COMPONENTS:
            r = torch.randn(n, d, generator=g, dtype=torch.float64)
            self.assertAlmostEqual(_cluster_radius(r, "released"), radius, places=12)
            self.assertAlmostEqual(_erank(r, True), erank, places=9)  # their 1e-12 inside the log: below 1e-10

    def test_components_match_compute_dse(self):
        # Well-separated sub-clusters, so both k-means implementations find the same partition; M_dim averages
        # random draws in both, here of nearly equal matrices.
        for normalize, (inter, intra_one, intra_eight, dim, dse) in REFERENCE_DSE.items():
            e = dense_structure(clips(), l2=normalize).extras
            self.assertAlmostEqual(e["m_inter"] / inter, 1.0, places=6)
            self.assertAlmostEqual(e["m_intra_sample"] / intra_one, 1.0, places=6)
            self.assertAlmostEqual(e["m_intra_group"] / intra_eight, 1.0, places=6)
            self.assertAlmostEqual(e["m_dim"] / dim, 1.0, places=4)
            self.assertAlmostEqual(e["m_inter"] - e["m_intra"] + e["m_dim"], dse, places=4)


class PropertyTests(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(3)
        centers = 4.0 * torch.randn(12, 5, 16, generator=g, dtype=torch.float64)  # 12 samples, 5 regions each
        self.x = centers[:, torch.arange(40) % 5] + torch.randn(12, 40, 16, generator=g, dtype=torch.float64)

    def test_released_value_is_scale_free(self):
        a = dense_structure(self.x, l2=False)
        b = dense_structure(7.3 * self.x, l2=False)
        self.assertAlmostEqual(a.value, b.value, places=8)  # float rounding in the distances, about 1e-9
        self.assertAlmostEqual(b.extras["scale"] / a.extras["scale"], 7.3, places=10)

    def test_paper_variant(self):
        rel, pap = dense_structure(self.x, l2=False), dense_structure(self.x, l2=False, variant="paper")
        self.assertAlmostEqual(pap.extras["m_dim"], rel.extras["m_dim"] * min(12, 16), places=9)  # same draws
        self.assertGreater(pap.extras["m_inter"], rel.extras["m_inter"])  # raw distances, here above the scale

    def test_separated_regions_beat_their_mixture(self):
        mixed = self.x.reshape(-1, 16)[torch.randperm(480, generator=torch.Generator().manual_seed(0))].reshape(
            12, 40, 16
        )
        self.assertGreater(
            dense_structure(self.x).extras["separability"], dense_structure(mixed).extras["separability"]
        )

    def test_grouped_and_batched_inputs_agree(self):
        groups = torch.arange(12).repeat_interleave(40)
        a = dense_structure(self.x)
        b = dense_structure(self.x.reshape(-1, 16), groups=groups)
        self.assertEqual(a.value, b.value)

    def test_input_errors(self):
        with self.assertRaises(ValueError):
            dense_structure(self.x.reshape(-1, 16))
        with self.assertRaises(ValueError):
            dense_structure(self.x[:1])
        with self.assertRaises(ValueError):
            dense_structure(self.x, variant="eq5")

    def test_population_level_route(self):
        samples = list(self.x)
        rec = rq.compute({0: samples, 1: [2 * s for s in samples]}, ["dense_structure"], level="population")
        direct = dense_structure(self.x)
        self.assertAlmostEqual(rec.rows[0].value, direct.value, places=12)
        self.assertEqual(rec.rows[0].extras["n_samples"], 12)
        with self.assertWarns(RuntimeWarning):  # the pipeline reports the failed record
            bad = rq.compute({0: samples}, ["dense_structure"], level="sample")
        self.assertTrue(math.isnan(bad.rows[0].value))
        self.assertIn("groups", bad.rows[0].extras.get("error", "") + str(bad.rows[0].extras))

    def test_recombination_lambda(self):
        comps = [
            {"separability": 0.1, "m_dim": 0.5},
            {"separability": 0.3, "m_dim": 0.6},
            {"separability": 0.2, "m_dim": 0.9},
        ]
        values, lam = recombine_dense_structure(comps)
        sep, dim = torch.tensor([0.1, 0.3, 0.2]), torch.tensor([0.5, 0.6, 0.9])
        self.assertAlmostEqual(lam, float(sep.std() / dim.std()), places=6)
        self.assertAlmostEqual(values[1], 0.3 + lam * 0.6, places=6)
        with self.assertRaises(ValueError):
            recombine_dense_structure(comps[:1])


if __name__ == "__main__":
    unittest.main()
