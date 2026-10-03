"""
Unit Tests for Recommendation Models
====================================
"""

import unittest
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.content_based import ContentBasedRecommender, GenreRecommender
from src.collaborative import MatrixFactorizationSVD
from src.cold_start import ColdStartPopularityRecommender
from src.hybrid import HybridRecommender
from data_loader import MovieLensDataLoader


class TestRecommenderModels(unittest.TestCase):

    def setUp(self):
        self.movies_df = pd.DataFrame({
            "movieId": [1, 2, 3, 4],
            "title": ["Toy Story (1995)", "Jumanji (1995)", "Grumpier Old Men (1995)", "Waiting to Exhale (1995)"],
            "genres": ["Adventure|Animation|Children|Comedy|Fantasy", "Adventure|Children|Fantasy", "Comedy|Romance", "Comedy|Drama|Romance"],
            "clean_title": ["Toy Story", "Jumanji", "Grumpier Old Men", "Waiting to Exhale"],
            "combined_tags": ["pixar animation fun", "board game magic", "sequel comedy older", "drama romance women"]
        })
        self.ratings_df = pd.DataFrame({
            "userId": [1, 1, 1, 2, 2, 3, 3, 4, 4],
            "movieId": [1, 2, 3, 1, 4, 2, 3, 1, 2],
            "rating": [5.0, 4.0, 2.0, 4.5, 3.0, 5.0, 4.0, 4.0, 4.0],
            "timestamp": [10, 11, 12, 13, 14, 15, 16, 17, 18]
        })

    def test_genre_recommender(self):
        model = GenreRecommender()
        model.fit(self.movies_df)
        recs = model.recommend(item_id=1, top_k=2)
        self.assertEqual(len(recs), 2)
        self.assertNotIn(1, [r[0] for r in recs])

    def test_svd_recommender(self):
        loader = MovieLensDataLoader()
        matrix, u2i, _, m2i, _ = loader.get_user_movie_sparse_matrix(self.ratings_df)
        svd = MatrixFactorizationSVD(n_components=2)
        svd.fit(matrix, u2i, m2i, ratings_df=self.ratings_df)
        recs = svd.recommend(user_id=1, top_k=2, exclude_seen=True)
        self.assertIsInstance(recs, list)

    def test_hybrid_recommender(self):
        loader = MovieLensDataLoader()
        matrix, u2i, _, m2i, _ = loader.get_user_movie_sparse_matrix(self.ratings_df)
        svd = MatrixFactorizationSVD(n_components=2).fit(matrix, u2i, m2i, ratings_df=self.ratings_df)
        content = ContentBasedRecommender().fit(self.movies_df)
        pop = ColdStartPopularityRecommender().fit(self.ratings_df)

        hybrid = HybridRecommender(
            collaborative_model=svd,
            content_model=content,
            popularity_model=pop
        ).fit(self.movies_df, self.ratings_df)

        recs = hybrid.recommend(user_id=1, top_k=2)
        self.assertGreaterEqual(len(recs), 1)


if __name__ == "__main__":
    unittest.main()
