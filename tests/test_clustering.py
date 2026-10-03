"""k-means clustering quality: parity with scikit-learn and a separated-clusters case."""

import unittest

import torch

import req_metrics as rq
from req_metrics.metrics.clustering import _kmeans_pp, davies_bouldin, kmeans

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import davies_bouldin_score

    HAVE_SKLEARN = True
except ImportError:  # pragma: no cover
    HAVE_SKLEARN = False


def blobs(k=6, n=300, d=8, spread=6.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = spread * torch.randn(k, d, generator=g)
    return torch.cat([c + torch.randn(n, d, generator=g) for c in centers]).double()


class ClusterQualityTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_SKLEARN, "scikit-learn not installed")
    def test_davies_bouldin_matches_sklearn(self):
        for k, n, d in [(6, 400, 8), (64, 3000, 16)]:  # 64 centroids exercises the exact-distance path
            torch.manual_seed(k)
            x = torch.randn(n, d).double()
            _, labels, _ = kmeans(x, k, seed=0)
            self.assertAlmostEqual(davies_bouldin(x, labels), davies_bouldin_score(x.numpy(), labels.numpy()), places=9)

    @unittest.skipUnless(HAVE_SKLEARN, "scikit-learn not installed")
    def test_inertia_matches_sklearn_from_the_same_seeding(self):
        x = blobs()
        init = _kmeans_pp(x, 6, torch.Generator().manual_seed(0), 8192).numpy()
        ref = KMeans(6, init=init, n_init=1, max_iter=100, tol=0, algorithm="lloyd").fit(x.numpy())
        _, _, inertia = kmeans(x, 6, seed=0)
        self.assertAlmostEqual(inertia, ref.inertia_, places=6)

    def test_separated_clusters_score_better_than_noise(self):
        sep = rq.cluster_quality(blobs(k=8, n=200), k=8)
        noise = rq.cluster_quality(torch.randn(1600, 8), k=8)
        self.assertLess(sep.value, noise.value)
        self.assertEqual(sep.extras["n_empty_clusters"], 0.0)
        self.assertAlmostEqual(sep.extras["inertia_per_point"] * 1600, sep.extras["inertia"], places=6)
        r = rq.cluster_quality(blobs(k=8, n=200), k=8, score="inertia")
        self.assertEqual(r.value, r.extras["inertia"])

    def test_arguments_are_checked(self):
        with self.assertRaises(ValueError):
            rq.cluster_quality(torch.randn(50, 4), k=64)
        with self.assertRaises(ValueError):
            rq.cluster_quality(torch.randn(500, 4), k=8, score="silhouette")
        self.assertEqual(rq.get_metric("cluster_quality").inputs, rq.InputKind.POINTS)


if __name__ == "__main__":
    unittest.main()
