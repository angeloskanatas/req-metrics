"""The version string in the package matches the installed distribution metadata (pyproject.toml)."""

import importlib.metadata
import unittest

import req_metrics as rq


class VersionTests(unittest.TestCase):
    def test_module_version_matches_distribution(self):
        self.assertEqual(rq.__version__, importlib.metadata.version("req-metrics"))


if __name__ == "__main__":
    unittest.main()
