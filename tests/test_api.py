"""Dataset-independent FastAPI endpoint contract tests."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app.api as api

class TestFastAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_initializer = api.ensure_models_initialized
        cls.original_state = dict(api.state)
        content_model = Mock(); content_model.recommend.return_value = [(2, 0.91), (3, 0.82)]
        hybrid = Mock(); hybrid.recommend.return_value = [(2, 0.95), (3, 0.88)]
        svd = Mock(); svd.recommend.return_value = [(2, 0.90), (3, 0.80)]
        genre = Mock(); genre.recommend.return_value = [(2, 0.87), (3, 0.79)]
        monitor = Mock(); monitor.check_drift.return_value = {"drift_detected": False, "ks_statistic": 0.10, "p_value": 0.90, "interpretation": "No significant drift detected."}
        api.state.clear()
        api.state.update({"movies_df": [1,2,3], "movie_title_map": {1:"Toy Story",2:"Jumanji",3:"Heat"}, "movie_genre_map": {1:"Animation",2:"Adventure",3:"Crime"}, "content_model": content_model, "hybrid_model": hybrid, "svd_model": svd, "genre_prior": genre, "monitor": monitor})
        api.ensure_models_initialized = lambda: None
        cls.client = TestClient(api.app)

    @classmethod
    def tearDownClass(cls):
        api.ensure_models_initialized = cls.original_initializer
        api.state.clear(); api.state.update(cls.original_state)

    def test_health_check(self):
        r = self.client.get("/health"); self.assertEqual(r.status_code, 200); self.assertTrue(r.json()["models_loaded"])
    def test_user_recommendations(self):
        r = self.client.post("/recommend/user", json={"user_id":42,"top_k":5}); self.assertEqual(r.status_code,200); self.assertEqual(r.json()["user_id"],42)
    def test_item_similarity(self):
        r = self.client.post("/recommend/item", json={"movie_id":1,"top_k":5}); self.assertEqual(r.status_code,200); self.assertEqual(r.json()["movie_id"],1)
    def test_cold_start_onboarding(self):
        r = self.client.post("/recommend/cold-start", json={"preferred_genres":["Action"],"top_k":5}); self.assertEqual(r.status_code,200)
    def test_drift_monitoring(self):
        r = self.client.post("/monitoring/drift", json={"recent_ratings":[4.0,4.5,5.0,3.5]}); self.assertEqual(r.status_code,200); self.assertFalse(r.json()["drift_detected"])

if __name__ == "__main__": unittest.main()
