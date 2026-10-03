"""
Solutions package for MovieLens 32M Questions
"""

import importlib

dataset_understanding_validation = importlib.import_module(".01_dataset_understanding_validation", package="solutions")
exploratory_data_analysis = importlib.import_module(".02_exploratory_data_analysis", package="solutions")
feature_engineering = importlib.import_module(".03_feature_engineering", package="solutions")
content_based_recommendation = importlib.import_module(".04_content_based_recommendation", package="solutions")
collaborative_filtering = importlib.import_module(".05_collaborative_filtering", package="solutions")
preference_prediction_supervised = importlib.import_module(".06_preference_prediction_supervised", package="solutions")
dimensionality_reduction = importlib.import_module(".07_dimensionality_reduction", package="solutions")
hybrid_recommendation_system = importlib.import_module(".08_hybrid_recommendation_system", package="solutions")
evaluation_and_ranking = importlib.import_module(".09_evaluation_and_ranking", package="solutions")
cold_start_and_production = importlib.import_module(".10_cold_start_and_production", package="solutions")
advanced_challenges = importlib.import_module(".11_advanced_challenges", package="solutions")

__all__ = [
    "dataset_understanding_validation",
    "exploratory_data_analysis",
    "feature_engineering",
    "content_based_recommendation",
    "collaborative_filtering",
    "preference_prediction_supervised",
    "dimensionality_reduction",
    "hybrid_recommendation_system",
    "evaluation_and_ranking",
    "cold_start_and_production",
    "advanced_challenges"
]
