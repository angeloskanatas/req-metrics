"""Parity of the ported TwoNN and GRIDE routines with DADApy's own functions.

Runs only when DADApy's utility module can be loaded. It is loaded from its
file so that a broken compiled extension in the package init does not hide
the pure-Python likelihood code the port is checked against.
"""

import importlib.util
import sys
import types
import unittest

import numpy as np
import torch

import req_metrics as rq


def _load_dadapy_utils():
    try:
        spec = importlib.util.find_spec("dadapy")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    root = list(spec.submodule_search_locations)[0]
    pkg = types.ModuleType("dadapy")
    pkg.__path__ = [root]
    sub = types.ModuleType("dadapy._utils")
    sub.__path__ = [root + "/_utils"]
    saved = {k: sys.modules.get(k) for k in ("dadapy", "dadapy._utils")}
    sys.modules["dadapy"], sys.modules["dadapy._utils"] = pkg, sub
    try:
        s = importlib.util.spec_from_file_location("dadapy._utils.utils", root + "/_utils/utils.py")
        mod = importlib.util.module_from_spec(s)
        s.loader.exec_module(mod)
        return mod
    except Exception:
        return None
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


UT = _load_dadapy_utils()


@unittest.skipUnless(UT is not None, "dadapy utilities not importable")
class DadapyParityTests(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        v = torch.randn(1500, 4, generator=g)
        v = v / v.norm(dim=1, keepdim=True) * torch.rand(1500, 1, generator=g) ** 0.25
        self.x = (v @ torch.linalg.qr(torch.randn(12, 4, generator=g)).Q.T).float()
        self.nb = rq.Neighbors.from_points(self.x, 64)

    def test_gride_matches_dadapy_at_every_scale(self):
        ours = rq.gride(self.nb, scale=8, range_max=64)
        for n1 in (1, 2, 4, 8, 16, 32):
            mus = self.nb.ratios(2 * n1, n1).numpy().copy()
            d_ref = UT._argmax_loglik(np.float64, 0.001, 1000, mus.copy(), n1, 2 * n1, eps=1e-7)
            err_ref = (1 / UT._fisher_info_scaling(d_ref, mus, n1, 2 * n1, eps=5 * np.finfo(np.float64).eps)) ** 0.5
            self.assertAlmostEqual(ours.extras[f"id_rank{2 * n1}"], d_ref, places=10)
            self.assertAlmostEqual(ours.extras[f"err_rank{2 * n1}"], err_ref, places=10)

    def test_twonn_matches_dadapy_fit(self):
        from scipy.optimize import curve_fit

        mus = rq.Neighbors.from_points(torch.unique(self.x, dim=0), 2).ratios(2, 1).numpy()
        n = len(mus)
        n_eff = int(n * 0.9)
        xs = np.sort(np.log(mus))[:n_eff]
        ys = -np.log(1 - np.arange(1, n_eff + 1) / n)
        d_ref = curve_fit(lambda x, m: m * x, xs, ys)[0][0]
        self.assertAlmostEqual(rq.twonn(self.x).value, d_ref, places=7)
        self.assertAlmostEqual(rq.twonn(self.x, algorithm="ml").value, (n - 1) / np.log(mus).sum(), places=10)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(UT is not None, "dadapy utilities not importable")
class DadapyImbalanceParityTests(unittest.TestCase):
    """DADApy's _return_imbalance on full index tables equals our counted ranks (no ties in random data)."""

    def test_information_imbalance_matches_dadapy(self):
        import importlib.util as iu

        spec = iu.find_spec("dadapy")
        root = list(spec.submodule_search_locations)[0]
        pkg = types.ModuleType("dadapy")
        pkg.__path__ = [root]
        sub = types.ModuleType("dadapy._utils")
        sub.__path__ = [root + "/_utils"]
        saved = {k: sys.modules.get(k) for k in ("dadapy", "dadapy._utils")}
        sys.modules["dadapy"], sys.modules["dadapy._utils"] = pkg, sub
        try:
            s = iu.spec_from_file_location("dadapy._utils.metric_comparisons", root + "/_utils/metric_comparisons.py")
            mc = iu.module_from_spec(s)
            s.loader.exec_module(mc)
        except Exception as e:  # pragma: no cover
            self.skipTest(f"metric_comparisons not loadable: {e}")
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
        torch.manual_seed(1)
        a = torch.randn(400, 5).double()
        b = a[:, :3] + 0.3 * torch.randn(400, 3).double()
        idx_a = torch.cdist(a, a).argsort(dim=1).numpy()
        idx_b = torch.cdist(b, b).argsort(dim=1).numpy()
        rng = np.random.default_rng(0)
        for k in (1, 3):
            ref_ab = mc._return_imbalance(idx_a, idx_b, rng, k=k)
            ref_ba = mc._return_imbalance(idx_b, idx_a, rng, k=k)
            ours = rq.information_imbalance(a, b, k=k)
            self.assertAlmostEqual(ours.value, ref_ab, places=10)
            self.assertAlmostEqual(ours.extras["reverse"], ref_ba, places=10)
