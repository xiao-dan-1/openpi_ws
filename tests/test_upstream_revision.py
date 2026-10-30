"""Revision compatibility keeps the baseline pinned and rejects unknown code."""
import subprocess
import unittest
from unittest.mock import patch

from pipeline.common import config


class UpstreamRevisionTest(unittest.TestCase):
    def test_original_baseline_is_supported(self):
        with patch.object(config.subprocess, "check_output", side_effect=[config.UPSTREAM_COMMIT + "\n", ""]):
            self.assertEqual(config.verify_upstream_revision(), config.UPSTREAM_COMMIT)

    def test_original_baseline_with_modified_runtime_code_is_rejected(self):
        with patch.object(config.subprocess, "check_output", side_effect=[
            config.UPSTREAM_COMMIT, "scripts/train.py\n",
        ]) as git:
            with self.assertRaisesRegex(ValueError, "Unreviewed changes"):
                config.verify_upstream_revision()
            self.assertIn("HEAD", git.call_args.args[0])

    def test_reviewed_local_descendant_is_supported(self):
        local = next(iter(config.SUPPORTED_LOCAL_COMMITS))
        with patch.object(config.subprocess, "check_output", side_effect=[local + "\n", ""]), \
             patch.object(config.subprocess, "run") as ancestry:
            self.assertEqual(config.verify_upstream_revision(), local)
            ancestry.assert_called_once()

    def test_unknown_descendant_is_rejected(self):
        with patch.object(config.subprocess, "check_output", return_value="f" * 40):
            with self.assertRaisesRegex(ValueError, "reviewed local revision"):
                config.verify_upstream_revision()

    def test_reviewed_revision_with_modified_runtime_code_is_rejected(self):
        local = next(iter(config.SUPPORTED_LOCAL_COMMITS))
        with patch.object(config.subprocess, "check_output", side_effect=[local, "src/openpi/models/pi0.py\n"]), \
             patch.object(config.subprocess, "run"):
            with self.assertRaisesRegex(ValueError, "Unreviewed changes"):
                config.verify_upstream_revision()

    def test_missing_baseline_ancestry_is_rejected(self):
        local = next(iter(config.SUPPORTED_LOCAL_COMMITS))
        with patch.object(config.subprocess, "check_output", return_value=local), \
             patch.object(config.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "git")):
            with self.assertRaises(subprocess.CalledProcessError):
                config.verify_upstream_revision()


if __name__ == "__main__":
    unittest.main()
