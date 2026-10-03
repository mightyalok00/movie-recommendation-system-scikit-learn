"""
Unit Tests for Data Loader and Sparse Preprocessing
===================================================
"""

import unittest
import sys
import pandas as pd
import numpy as np
from pathlib import Path

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
        self.assertGreater(len(df), 1000)

    def test_load_tags(self):
        df = self.loader.load_tags(min_tag_freq=5)
        self.assertIn("movieId", df.columns)
        self.assertIn("combined_tags", df.columns)
        self.assertGreater(len(df), 100)

    def test_sparse_matrix_construction(self):
        sample_ratings = pd.DataFrame({
            "userId": [1, 1, 2, 2, 3],
            "movieId": [10, 20, 20, 30, 10],
            "rating": [4.0, 5.0, 3.5, 2.0, 4.5],
            "timestamp": [100, 101, 102, 103, 104]
        })
        matrix, u2i, i2u, m2i, i2m = self.loader.get_user_movie_sparse_matrix(sample_ratings)
        self.assertEqual(matrix.shape, (3, 3))
        self.assertEqual(matrix.nnz, 5)


if __name__ == "__main__":
    unittest.main()
