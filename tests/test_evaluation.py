"""
Unit Tests for Ranking Metrics and Evaluator
===========================================
"""

import unittest
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation import precision_at_k, recall_at_k, ndcg_at_k, average_precision_at_k, catalog_coverage


class TestEvaluationMetrics(unittest.TestCase):

    def test_precision_at_k(self):
        rec = [1, 2, 3, 4, 5]
        gt = {2, 4, 6, 8}
        p5 = precision_at_k(rec, gt, k=5)
        self.assertEqual(p5, 2 / 5)

    def test_recall_at_k(self):
        rec = [1, 2, 3, 4, 5]
        gt = {2, 4, 6, 8}
        r5 = recall_at_k(rec, gt, k=5)
        self.assertEqual(r5, 2 / 4)

    def test_ndcg_at_k_perfect(self):
        rec = [1, 2, 3]
        gt = {1, 2, 3}
        score = ndcg_at_k(rec, gt, k=3)
        self.assertAlmostEqual(score, 1.0)

    def test_coverage(self):
        recs = [[1, 2], [2, 3], [4, 5]]
        cov = catalog_coverage(recs, total_catalog_items=10)
        self.assertEqual(cov, 0.5)


if __name__ == "__main__":
    unittest.main()
