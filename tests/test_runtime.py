"""Runtime artifact tests that do not require the full MovieLens dataset."""

import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from src.runtime import load_runtime_artifact
from src.pipeline import MovieLensRecommendationPipeline


class TestRuntime(unittest.TestCase):
    def test_missing_artifact_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as ctx:
                load_runtime_artifact(Path(tmp) / "missing.joblib")
            self.assertIn("build_artifacts.py", str(ctx.exception))

    def test_load_pipeline_artifact(self):
        pipeline = MovieLensRecommendationPipeline(n_svd_components=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime.joblib"
            joblib.dump({"version": 1, "pipeline": pipeline}, path)
            bundle = load_runtime_artifact(path)
            self.assertEqual(bundle["version"], 1)
            self.assertIsInstance(bundle["pipeline"], MovieLensRecommendationPipeline)


if __name__ == "__main__":
    unittest.main()
