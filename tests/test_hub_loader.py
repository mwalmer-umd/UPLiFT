import unittest
from unittest.mock import patch

from uplift import hub_loader


class HubLoaderDependencyTests(unittest.TestCase):
    def test_check_dependencies_rejects_stale_timm(self):
        with patch.object(hub_loader, "_get_installed_version", return_value="1.0.23"):
            with self.assertRaisesRegex(ImportError, r"timm>=1\.0\.24"):
                hub_loader._check_dependencies("dinov3-splus16")

    def test_check_dependencies_accepts_minimum_timm(self):
        with patch.object(hub_loader, "_get_installed_version", return_value="1.0.24"):
            hub_loader._check_dependencies("dinov3-splus16")

    def test_version_comparison_handles_patch_versions(self):
        self.assertTrue(hub_loader._is_version_at_least("1.0.24", "1.0.24"))
        self.assertTrue(hub_loader._is_version_at_least("1.0.25", "1.0.24"))
        self.assertFalse(hub_loader._is_version_at_least("1.0.23", "1.0.24"))


if __name__ == "__main__":
    unittest.main()
