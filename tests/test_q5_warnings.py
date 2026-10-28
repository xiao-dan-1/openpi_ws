"""Q5 hides one known migration notice while preserving other diagnostics."""
import importlib
import unittest
import warnings


class VideoWarningTests(unittest.TestCase):
    def test_only_exact_video_migration_notice_is_filtered(self):
        message = ("The video decoding and encoding capabilities of torchvision "
                   "are deprecated from version 0.22 and will be removed in version 0.24. "
                   "We recommend that you migrate to TorchCodec")
        module = "torchvision.io._video_deprecation_warning"
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            import pipeline.q5
            importlib.reload(pipeline.q5)
            for _ in range(3):
                warnings.warn_explicit(message, UserWarning, "video.py", 1, module=module)
            self.assertEqual(len(captured), 0)
            warnings.warn_explicit("A different video warning", UserWarning, "video.py", 2, module=module)
            warnings.warn_explicit(message, RuntimeWarning, "video.py", 3, module=module)
            warnings.warn_explicit(message, UserWarning, "other.py", 1, module="another_package")
            self.assertEqual(len(captured), 3)


if __name__ == "__main__":
    unittest.main()
