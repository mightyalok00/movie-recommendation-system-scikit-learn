"""
Supervised User-Movie Preference Prediction
==========================================
Reframes recommendation as binary/multi-class classification.
Engineers leakage-free user, movie, and interaction features and trains
HistGradientBoosting & LogisticRegression classifiers.
Addresses Section 6 (Q63 - Q70).
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, classification_report


class SupervisedPreferenceModel:
    """
    Supervised Preference Classifier for predicting P(Like | User, Movie).
    """

    def __init__(
        self,
        model_type: str = "hist_gb",
        relevance_threshold: float = 4.0,
        random_state: int = 42
    ):
        self.model_type = model_type
        self.relevance_threshold = relevance_threshold
        self.random_state = random_state
        
        if model_type == "logistic":
            self.clf = LogisticRegression(class_weight="balanced", max_iter=500, random_state=random_state)
        elif model_type == "hist_gb":
            self.clf = HistGradientBoostingClassifier(
                class_weight="balanced",
                max_iter=150,
                learning_rate=0.08,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Feature lookup tables calculated strictly from training set (no leakage)
        self.user_stats: Dict[int, Dict[str, float]] = {}
        self.movie_stats: Dict[int, Dict[str, float]] = {}
        self.global_user_mean: float = 3.5
        self.global_movie_mean: float = 3.5
        self.is_fitted = False

    def _compute_stats(self, train_df: pd.DataFrame):
        """
        Computes user and item aggregations strictly on the training partition.
        """
        user_agg = train_df.groupby("userId")["rating"].agg(["mean", "std", "count"]).reset_index()
        self.global_user_mean = float(train_df["rating"].mean())
        user_agg["std"] = user_agg["std"].fillna(0.0)
        self.user_stats = {
            row["userId"]: {
                "u_mean": float(row["mean"]),
                "u_std": float(row["std"]),
                "u_count": float(row["count"])
            }
            for _, row in user_agg.iterrows()
        }

        movie_agg = train_df.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
        self.global_movie_mean = self.global_user_mean
        self.movie_stats = {
            row["movieId"]: {
                "m_mean": float(row["mean"]),
                "m_count": float(row["count"])
            }
            for _, row in movie_agg.iterrows()
        }

    def _extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds tabular feature vectors:
        - user_mean, user_std, log(user_count)
        - movie_mean, log(movie_count)
        - interaction diff: (movie_mean - user_mean)
        """
        features = []
        labels = (df["rating"] >= self.relevance_threshold).astype(int).to_numpy()

        for _, row in df.iterrows():
            uid = int(row["userId"])
            mid = int(row["movieId"])

            u_stat = self.user_stats.get(uid, {"u_mean": self.global_user_mean, "u_std": 0.0, "u_count": 0.0})
            m_stat = self.movie_stats.get(mid, {"m_mean": self.global_movie_mean, "m_count": 0.0})

            u_mean = u_stat["u_mean"]
            u_std = u_stat["u_std"]
            u_count = np.log1p(u_stat["u_count"])
            m_mean = m_stat["m_mean"]
            m_count = np.log1p(m_stat["m_count"])
            diff = m_mean - u_mean

            features.append([u_mean, u_std, u_count, m_mean, m_count, diff])

        return np.array(features, dtype=np.float32), labels

    def fit(self, train_df: pd.DataFrame) -> "SupervisedPreferenceModel":
        """
        Extracts features and fits the supervised classifier.
        """
        self._compute_stats(train_df)
        X_train, y_train = self._extract_features(train_df)
        self.clf.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluates classifier on test partition across key classification metrics.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation.")

        X_test, y_test = self._extract_features(test_df)
        y_pred = self.clf.predict(X_test)
        y_prob = self.clf.predict_proba(X_test)[:, 1]

        metrics = {
            "ROC_AUC": float(roc_auc_score(y_test, y_prob)),
            "PR_AUC": float(average_precision_score(y_test, y_prob)),
            "F1_Score": float(f1_score(y_test, y_pred, zero_division=0)),
            "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        }
        return metrics
