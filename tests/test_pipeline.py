"""
Unit Tests for Scikit-Learn Recommendation Pipeline
===================================================
"""

import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import MovieLensRecommendationPipeline
from data.loader import MovieLensDataLoader


class TestScikitLearnPipeline(unittest.TestCase):

    def setUp(self):
        self.loader = MovieLensDataLoader()
        self.movies_df = self.loader.load_movies().head(100)
        self.tags_df = self.loader.load_tags()
        self.ratings_df = pd.DataFrame({
            "userId": [1, 1, 1, 2, 2, 3, 3, 4, 5, 5],
            "movieId": [1, 260, 296, 1, 318, 260, 356, 1, 296, 318],
            "rating": [5.0, 4.0, 4.5, 3.5, 5.0, 4.0, 3.0, 4.5, 4.0, 5.0],
            "timestamp": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        })

    def test_pipeline_estimator_lifecycle(self):
        pipe = MovieLensRecommendationPipeline(
            n_svd_components=4,
            collab_weight=0.60,
            content_weight=0.30,
            popularity_weight=0.10,
            diversity_penalty=0.10
        )
        
        # Test get_params
        params = pipe.get_params()
        self.assertEqual(params["n_svd_components"], 4)
        self.assertEqual(params["collab_weight"], 0.60)
        
        # Test set_params
        pipe.set_params(collab_weight=0.70)
        self.assertEqual(pipe.collab_weight, 0.70)
        
        # Test fit
        fitted_pipe = pipe.fit(self.ratings_df, self.movies_df, self.tags_df)
        self.assertTrue(fitted_pipe.is_fitted_)
        
        # Test recommend for known user
        recs = fitted_pipe.recommend(user_id=1, top_k=5, exclude_seen=True)
        self.assertIsInstance(recs, list)
        self.assertTrue(len(recs) > 0)
        
        # Test recommend for cold-start (new user)
        cold_recs = fitted_pipe.recommend(user_id=99999, top_k=5)
        self.assertIsInstance(cold_recs, list)
        self.assertTrue(len(cold_recs) > 0)
        
        # Test predict
        pred_score = fitted_pipe.predict(user_id=1, item_id=318)
        self.assertIsInstance(pred_score, float)
        self.assertGreaterEqual(pred_score, 0.0)


if __name__ == "__main__":
    unittest.main()
