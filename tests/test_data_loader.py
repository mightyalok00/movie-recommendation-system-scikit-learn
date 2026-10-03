"""
Unit tests for data loading and sparse preprocessing.

Tests are dataset-independent: CI must not require the user's MovieLens 32M files.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_loader import MovieLensDataLoader


class TestMovieLensDataLoader(unittest.TestCase):
    def setUp(self):
        self.loader = MovieLensDataLoader()

    def test_load_movies(self):
        df = self.loader.load_movies()
        self.assertIn("movieId", df.columns)
        self.assertIn("title", df.columns)
        self.assertIn("genres", df.columns)
        self.assertIn("clean_title", df.columns)
        self.assertGreater(len(df), 10)

    def test_load_tags(self):
        """Exercise real CSV tag loading without depending on MovieLens 32M files."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            pd.DataFrame(
                {
                    "userId": [1, 2, 3, 4, 5, 6],
                    "movieId": [10, 10, 10, 20, 20, 30],
                    "tag": ["Sci-Fi", "sci-fi", " Sci-Fi ", "Comedy", "Comedy", "Drama"],
                    "timestamp": np.arange(6, dtype=np.int64),
                }
            ).to_csv(data_dir / "tags.csv", index=False)

            df = MovieLensDataLoader(data_dir=str(data_dir)).load_tags(min_tag_freq=2)

        self.assertIn("movieId", df.columns)
        self.assertIn("combined_tags", df.columns)
        self.assertEqual(set(df["movieId"]), {10, 20})
        self.assertEqual(df.loc[df["movieId"] == 10, "combined_tags"].iloc[0], "sci-fi sci-fi sci-fi")
        self.assertEqual(df.loc[df["movieId"] == 20, "combined_tags"].iloc[0], "comedy comedy")

    def test_sparse_matrix_construction(self):
        sample_ratings = pd.DataFrame({
            "userId": [1, 1, 2, 2, 3],
            "movieId": [10, 20, 20, 30, 10],
            "rating": [4.0, 5.0, 3.5, 2.0, 4.5],
            "timestamp": [100, 101, 102, 103, 104],
        })
        matrix, u2i, i2u, m2i, i2m = self.loader.get_user_movie_sparse_matrix(sample_ratings)
        self.assertEqual(matrix.shape, (3, 3))
        self.assertEqual(matrix.nnz, 5)


if __name__ == "__main__":
    unittest.main()
