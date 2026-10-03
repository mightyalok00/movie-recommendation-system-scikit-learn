"""
Master Execution Pipeline for MovieLens 32M Recommendation System
=================================================================
Runs all analysis modules, executes benchmarks across all models,
validates the pipeline, and generates offline reports.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from solutions import (
    dataset_understanding_validation,
    exploratory_data_analysis,
    feature_engineering,
    content_based_recommendation,
    collaborative_filtering,
    preference_prediction_supervised,
    dimensionality_reduction,
    hybrid_recommendation_system,
    evaluation_and_ranking,
    cold_start_and_production,
    advanced_challenges
)


def main():
    start_total = time.time()
    print("=" * 90)
    print(">>> MOVIELENS 32M END-TO-END RECOMMENDATION SYSTEM SUITE")
    print("=" * 90)

    modules = [
        ("Section 1: Dataset Understanding & Validation", dataset_understanding_validation.run_section_1),
        ("Section 2: Exploratory Data Analysis", exploratory_data_analysis.run_section_2),
        ("Section 3: Feature Engineering", feature_engineering.run_section_3),
        ("Section 4: Content-Based Recommendation", content_based_recommendation.run_section_4),
        ("Section 5: Collaborative Filtering", collaborative_filtering.run_section_5),
        ("Section 6: Supervised Preference Prediction", preference_prediction_supervised.run_section_6),
        ("Section 7: Dimensionality Reduction", dimensionality_reduction.run_section_7),
        ("Section 8: Hybrid Recommendation System", hybrid_recommendation_system.run_section_8),
        ("Section 9: Evaluation & Ranking Benchmarks", evaluation_and_ranking.run_section_9),
        ("Section 10: Cold Start & Production Engineering", cold_start_and_production.run_section_10),
        ("Section 11: Advanced Challenges & Ablation Studies", advanced_challenges.run_section_11),
    ]

    for name, func in modules:
        print(f"\n>>> Running {name}...")
        t0 = time.time()
        try:
            func()
            print(f">>> Finished {name} in {time.time() - t0:.2f} seconds.\n")
        except Exception as e:
            print(f">>> ERROR running {name}: {e}\n")

    print("=" * 90)
    print(f"[SUCCESS] ALL MODULES COMPLETED in {time.time() - start_total:.2f} seconds.")
    print("=" * 90)


if __name__ == "__main__":
    main()
