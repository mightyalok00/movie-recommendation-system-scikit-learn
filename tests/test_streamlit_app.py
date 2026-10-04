"""Headless UI smoke tests for the Streamlit movie-discovery app."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.pipeline import MovieLensRecommendationPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StreamlitAppTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.artifact_path = Path(self.temp_dir.name) / "runtime.joblib"
        movies = pd.DataFrame(
            {
                "movieId": list(range(1, 9)),
                "title": [
                    f"Movie {movie_id} ({1990 + movie_id})" for movie_id in range(1, 9)
                ],
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
            self.artifact_path,
        )
        self.env = patch.dict(
            os.environ,
            {"MOVIELENS_ARTIFACT_PATH": str(self.artifact_path)},
        )
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp_dir.cleanup()

    def test_discovery_personalization_and_genre_flows_render(self):
        app = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 3)

        app.selectbox(key="seed_movie").select(1).run()
        self.assertFalse(app.exception)
        app.button(key="similar_button").click().run()
        self.assertFalse(app.exception)

        app.button(key="personal_button").click().run()
        self.assertFalse(app.exception)

        app.multiselect(key="cold_start_genres").select("Adventure").run()
        app.button(key="genre_button").click().run()
        self.assertFalse(app.exception)

    def test_missing_artifact_shows_build_instructions(self):
        self.env.stop()
        self.env = patch.dict(
            os.environ,
            {
                "MOVIELENS_ARTIFACT_PATH": str(
                    self.artifact_path.with_name("missing.joblib")
                )
            },
        )
        self.env.start()
        st.cache_resource.clear()

        app = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertTrue(any("build-artifacts" in code.value for code in app.code))


if __name__ == "__main__":
    unittest.main()
