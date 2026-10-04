"""Artifact-backed FastAPI integration tests.

These tests exercise the same runtime loading path used in production:
MOVELENS_ARTIFACT_PATH -> src.runtime -> FastAPI state.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import app.api as api
from src.pipeline import MovieLensRecommendationPipeline


class TestFastAPIArtifactRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.artifact_path = Path(cls.temp_dir.name) / "runtime.joblib"

        movies = pd.DataFrame(
            {
                "movieId": list(range(1, 9)),
                "title": [f"Movie {i} ({1990 + i})" for i in range(1, 9)],
                "genres": [
                    "Adventure|Comedy",
                    "Adventure|Fantasy",
                    "Comedy|Romance",
                    "Drama",
                    "Action|Sci-Fi",
                    "Action|Thriller",
                    "Fantasy",
                    "Romance",
                ],
            }
        )
        ratings = pd.DataFrame(
            [
                {
                    "userId": user_id,
                    "movieId": movie_id,
                    "rating": float(3 + (user_id + movie_id) % 3),
                    "timestamp": 1_600_000_000 + user_id + movie_id,
                }
                for user_id in range(1, 7)
                for movie_id in range(1, 7)
            ]
        )
        tags = pd.DataFrame(
            {
                "movieId": movies["movieId"],
                "combined_tags": [
                    f"story genre{movie_id} adventure character"
                    for movie_id in movies["movieId"]
                ],
            }
        )

        pipeline = MovieLensRecommendationPipeline(
            n_svd_components=2,
            min_popularity_m=1,
        ).fit(ratings_df=ratings, movies_df=movies, tags_df=tags)

        joblib.dump(
            {
                "version": 1,
                "pipeline": pipeline,
                "movies_df": movies[["movieId", "title", "genres"]],
                "baseline_ratings": ratings["rating"].to_numpy(dtype=np.float32),
            },
            cls.artifact_path,
        )

        cls.original_env = os.environ.get("MOVIELENS_ARTIFACT_PATH")
        cls.original_state = dict(api.state)
        os.environ["MOVIELENS_ARTIFACT_PATH"] = str(cls.artifact_path)
        os.environ["MOVIELENS_RUNTIME_MODE"] = "artifact"
        api.state.clear()
        cls.client = TestClient(api.app)

    @classmethod
    def tearDownClass(cls):
        api.state.clear()
        api.state.update(cls.original_state)
        if cls.original_env is None:
            os.environ.pop("MOVIELENS_ARTIFACT_PATH", None)
        else:
            os.environ["MOVIELENS_ARTIFACT_PATH"] = cls.original_env
        cls.temp_dir.cleanup()

    def test_health_and_readiness_use_artifact(self):
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["artifact_available"])

        ready = self.client.get("/ready")
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.json()["status"], "ready")
        self.assertFalse(ready.json()["models_loaded"])

    def test_search_loads_artifact_runtime(self):
        response = self.client.get("/movies/search", params={"q": "Movie 1"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json())
        self.assertIn("title", response.json()[0])
        self.assertTrue(api.state)

    def test_recommendation_endpoints_use_loaded_artifact(self):
        item = self.client.post(
            "/recommend/item",
            json={"movie_id": 1, "top_k": 3},
        )
        self.assertEqual(item.status_code, 200)
        self.assertEqual(item.json()["movie_id"], 1)
        self.assertLessEqual(len(item.json()["recommendations"]), 3)

        user = self.client.post(
            "/recommend/user",
            json={"user_id": 1, "top_k": 3},
        )
        self.assertEqual(user.status_code, 200)
        self.assertEqual(user.json()["user_id"], 1)

        cold_start = self.client.post(
            "/recommend/cold-start",
            json={"preferred_genres": ["Adventure"], "top_k": 3},
        )
        self.assertEqual(cold_start.status_code, 200)
        self.assertEqual(cold_start.json()["status"], "SUCCESS")

    def test_monitoring_endpoint_uses_artifact_baseline(self):
        response = self.client.post(
            "/monitoring/drift",
            json={"recent_ratings": [4.0, 4.5, 5.0, 3.5]},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("drift_detected", response.json())


if __name__ == "__main__":
    unittest.main()
