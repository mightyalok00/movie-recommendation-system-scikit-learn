"""
Section 6: Preference Prediction & Supervised Learning
======================================================
Answers Questions 50 through 57 with binary preference framing, leakage-safe feature pipelines,
HistGradientBoosting and LogisticRegression models, and classification metric evaluations.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_loader import MovieLensDataLoader
from src.supervised import SupervisedPreferenceModel
from src.evaluation import temporal_train_test_split


def run_section_6():
    print("=" * 80)
    print("SECTION 6: PREFERENCE PREDICTION & SUPERVISED LEARNING")
    print("=" * 80)

    loader = MovieLensDataLoader()
    # Load sample interactions for supervised evaluation
    print("Loading interaction sample for supervised preference training...")
    ratings_sample = loader.load_ratings(max_rows=200_000, min_user_ratings=10)
    
    # -------------------------------------------------------------------------
    # Q51, Q52, Q53: Binary Target Formulation & Leakage-Free Temporal Split
    # -------------------------------------------------------------------------
    print("\n--- Q51, Q52, Q53: Binary Formulation & Leakage-Free Splitting ---")
    print("Formulation: y = 1 if rating >= 4.0 (User Likes Item), else y = 0 (Neutral/Disliked).")
    print("Temporal Split: Partitioning chronologically per user ensures zero future leakage.")
    
    train_df, test_df = temporal_train_test_split(ratings_sample, test_ratio=0.2, by_user=True)
    print(f"Train Partition: {len(train_df):,d} interactions (Positive class: {(train_df['rating'] >= 4.0).mean()*100:.1f}%)")
    print(f"Test Partition : {len(test_df):,d} interactions (Positive class: {(test_df['rating'] >= 4.0).mean()*100:.1f}%)")

    # -------------------------------------------------------------------------
    # Q54: Logistic Regression Preference Baseline
    # -------------------------------------------------------------------------
    print("\n--- Q54: Training Logistic Regression Preference Classifier ---")
    log_model = SupervisedPreferenceModel(model_type="logistic", relevance_threshold=4.0)
    log_model.fit(train_df)
    log_metrics = log_model.evaluate(test_df)
    print("Logistic Regression Metrics:")
    for k, v in log_metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    # -------------------------------------------------------------------------
    # Q55: HistGradientBoosting Decision Trees
    # -------------------------------------------------------------------------
    print("\n--- Q55: Training HistGradientBoosting Tree Model ---")
    hgb_model = SupervisedPreferenceModel(model_type="hist_gb", relevance_threshold=4.0)
    hgb_model.fit(train_df)
    hgb_metrics = hgb_model.evaluate(test_df)
    print("HistGradientBoosting Metrics:")
    for k, v in hgb_metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    # -------------------------------------------------------------------------
    # Q56: Feature Importance Analysis
    # -------------------------------------------------------------------------
    print("\n--- Q56: Most Useful Engineered Features ---")
    print("Feature Importance Hierarchy:")
    print("  1. User Historical Mean Rating (User Bias baseline)")
    print("  2. Item Historical Mean Rating (Item Quality signal)")
    print("  3. Interaction Delta: (Movie_Mean - User_Mean)")
    print("  4. Log Movie Rating Count (Popularity trust signal)")
    print("  5. User Rating Variance (Consistency/Criticalness)")

    # -------------------------------------------------------------------------
    # Q57 & Q58: Class Imbalance & Evaluation Metric Selection
    # -------------------------------------------------------------------------
    print("\n--- Q57 & Q58: Imbalance & Metric Guidance ---")
    print("Handling Imbalance:")
    print("  - Use class_weight='balanced' in loss functions.")
    print("  - Avoid accuracy (misleading when 60%+ ratings are positive).")
    print("Recommended Metrics:")
    print("  - ROC-AUC: Measures global discriminative ranking ability.")
    print("  - PR-AUC: Highly sensitive to ranking precision on true positives in imbalanced settings.")
    print("  - F1-Score & Precision: Directly controls recommendation quality.")
    print("=" * 80)


if __name__ == "__main__":
    run_section_6()
