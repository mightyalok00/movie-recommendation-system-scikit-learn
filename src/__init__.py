"""
MovieLens Recommendation System - Core ML Package
=================================================
Provides modular recommendation models, ranking evaluators, and Scikit-learn pipelines.
"""

from src.base import BaseRecommender
from src.content_based import ContentBasedRecommender, GenreRecommender, TagTFIDFRecommender, split_pipe_genres
from src.collaborative import MatrixFactorizationSVD, ItemItemCollaborativeFiltering, UserUserCollaborativeFiltering
from src.supervised import SupervisedPreferenceModel
from src.cold_start import ColdStartPopularityRecommender, GenrePriorRecommender
from src.hybrid import HybridRecommender
from src.evaluation import RecommendationEvaluator, temporal_train_test_split, precision_at_k, recall_at_k, ndcg_at_k, average_precision_at_k, catalog_coverage
from src.pipeline import MovieLensRecommendationPipeline

__all__ = [
    "BaseRecommender",
    "ContentBasedRecommender",
    "GenreRecommender",
    "TagTFIDFRecommender",
    "split_pipe_genres",
    "MatrixFactorizationSVD",
    "ItemItemCollaborativeFiltering",
    "UserUserCollaborativeFiltering",
    "SupervisedPreferenceModel",
    "ColdStartPopularityRecommender",
    "GenrePriorRecommender",
    "HybridRecommender",
    "RecommendationEvaluator",
    "temporal_train_test_split",
    "MovieLensRecommendationPipeline",
]
