"""
Integration Tests for FastAPI Microservice Endpoints
====================================================
"""

import sys
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.api import app


class TestFastAPIEndpoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")

    def test_user_recommendations(self):
        payload = {
            "user_id": 42,
            "top_k": 5,
            "diversity_penalty": 0.15
        }
        response = self.client.post("/recommend/user", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("user_id"), 42)
        self.assertIsInstance(data.get("recommendations"), list)

    def test_item_similarity(self):
        payload = {
            "movie_id": 1,
            "top_k": 5,
            "content_engine": "unified"
        }
        response = self.client.post("/recommend/item", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("movie_id"), 1)
        self.assertIsInstance(data.get("recommendations"), list)

    def test_cold_start_onboarding(self):
        payload = {
            "preferred_genres": ["Action", "Sci-Fi"],
            "top_k": 5
        }
        response = self.client.post("/recommend/cold-start", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data.get("recommendations"), list)

    def test_drift_monitoring(self):
        payload = {
            "recent_ratings": [4.0, 4.5, 5.0, 3.5, 4.0, 4.5, 5.0, 4.0]
        }
        response = self.client.post("/monitoring/drift", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("drift_detected", data)
        self.assertIn("ks_statistic", data)


if __name__ == "__main__":
    unittest.main()
