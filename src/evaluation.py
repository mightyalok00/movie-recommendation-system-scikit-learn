"""
Recommendation System Evaluation & Ranking Metrics Suite
=========================================================
Implements leakage-safe temporal train/test splitting, top-K ranking metrics
(Precision@K, Recall@K, MAP@K, NDCG@K), catalog coverage, intra-list diversity,
and popularity novelty with bootstrap confidence intervals.
Addresses Section 9 (Q88 - Q99) & Section 11 (Q114 - Q122).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter


def temporal_train_test_split(
    ratings_df: pd.DataFrame,
    test_ratio: float = 0.2,
    by_user: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates a temporal train/test split.
    If by_user=True, holds out the most recent `test_ratio` portion of ratings for each user.
    If by_user=False, splits globally on the timestamp quantile.
    Addresses Q88, Q89, Q114.
    """
    if by_user:
        # Sort chronologically per user
        sorted_df = ratings_df.sort_values(by=["userId", "timestamp"])
        
        train_list = []
        test_list = []

        # Split per user
        for uid, grp in sorted_df.groupby("userId"):
            n = len(grp)
            if n < 5:
                # For very small histories, keep in train
                train_list.append(grp)
            else:
                n_test = max(1, int(n * test_ratio))
                train_list.append(grp.iloc[:-n_test])
                test_list.append(grp.iloc[-n_test:])

        train_df = pd.concat(train_list, ignore_index=True)
        test_df = pd.concat(test_list, ignore_index=True)
    else:
        # Global cutoff timestamp
        cutoff_time = ratings_df["timestamp"].quantile(1.0 - test_ratio)
        train_df = ratings_df[ratings_df["timestamp"] <= cutoff_time].copy()
        test_df = ratings_df[ratings_df["timestamp"] > cutoff_time].copy()

    return train_df, test_df


def precision_at_k(recommended: List[int], ground_truth: Set[int], k: int = 10) -> float:
    """Calculates Precision@K = |Recommended@K ∩ Relevant| / K."""
    if k <= 0:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in ground_truth)
    return hits / k


def recall_at_k(recommended: List[int], ground_truth: Set[int], k: int = 10) -> float:
    """Calculates Recall@K = |Recommended@K ∩ Relevant| / |Relevant|."""
    if not ground_truth or k <= 0:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in ground_truth)
    return hits / len(ground_truth)


def average_precision_at_k(recommended: List[int], ground_truth: Set[int], k: int = 10) -> float:
    """Calculates Average Precision@K for a single user."""
    if not ground_truth or k <= 0:
        return 0.0
    rec_k = recommended[:k]
    score = 0.0
    num_hits = 0

    for i, item in enumerate(rec_k):
        if item in ground_truth:
            num_hits += 1
            score += num_hits / (i + 1)

    return score / min(len(ground_truth), k)


def ndcg_at_k(recommended: List[int], ground_truth: Set[int], k: int = 10) -> float:
    """Calculates Normalized Discounted Cumulative Gain@K."""
    if not ground_truth or k <= 0:
        return 0.0
    rec_k = recommended[:k]
    
    # DCG calculation
    dcg = 0.0
    for i, item in enumerate(rec_k):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)  # index i=0 -> rank 1 -> log2(2) = 1

    # Ideal DCG calculation
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def catalog_coverage(all_recommendations: List[List[int]], total_catalog_items: int) -> float:
    """Calculates catalog coverage: percentage of catalog items ever recommended."""
    recommended_set = {item for rec_list in all_recommendations for item in rec_list}
    return len(recommended_set) / max(1, total_catalog_items)


def calculate_novelty(all_recommendations: List[List[int]], item_popularity_dict: Dict[int, int], total_interactions: int) -> float:
    """
    Calculates average Novelty across all recommendations:
    Novelty(i) = -log2(P(i)), where P(i) = count(i) / total_interactions
    """
    total_novelty = 0.0
    count = 0

    for rec_list in all_recommendations:
        for mid in rec_list:
            pop = item_popularity_dict.get(mid, 1)
            prob = max(1e-9, pop / total_interactions)
            total_novelty += -np.log2(prob)
            count += 1

    return total_novelty / max(1, count)


def calculate_genre_diversity(
    all_recommendations: List[List[int]],
    movie_genre_vectors: Dict[int, np.ndarray]
) -> float:
    """
    Calculates Intra-List Diversity (ILD) as mean pairwise cosine distance
    between items within recommendation lists.
    """
    diversity_scores = []
    
    for rec_list in all_recommendations:
        vecs = [movie_genre_vectors[mid] for mid in rec_list if mid in movie_genre_vectors]
        if len(vecs) < 2:
            continue
        
        # Pairwise distances
        n = len(vecs)
        pairs_dist = 0.0
        num_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                norm_i = np.linalg.norm(vecs[i])
                norm_j = np.linalg.norm(vecs[j])
                if norm_i > 0 and norm_j > 0:
                    sim = np.dot(vecs[i], vecs[j]) / (norm_i * norm_j)
                    pairs_dist += (1.0 - sim)
                    num_pairs += 1
        
        if num_pairs > 0:
            diversity_scores.append(pairs_dist / num_pairs)

    return float(np.mean(diversity_scores)) if diversity_scores else 0.0


class RecommendationEvaluator:
    """
    Unified Offline Recommendation Evaluator running standardized ranking protocols.
    """

    def __init__(
        self,
        test_df: pd.DataFrame,
        relevance_threshold: float = 4.0,
        total_catalog_size: int = 87585,
        popularity_dict: Optional[Dict[int, int]] = None
    ):
        self.relevance_threshold = relevance_threshold
        self.total_catalog_size = total_catalog_size
        self.popularity_dict = popularity_dict or {}
        self.total_interactions = sum(self.popularity_dict.values()) if self.popularity_dict else 1

        # Build ground truth positive sets for each user
        positives = test_df[test_df["rating"] >= relevance_threshold]
        self.ground_truth_map: Dict[int, Set[int]] = (
            positives.groupby("userId")["movieId"].apply(set).to_dict()
        )
        self.eval_users = list(self.ground_truth_map.keys())

    def evaluate_model(
        self,
        model,
        sample_users: int = 500,
        top_k_list: List[int] = [5, 10, 20],
        genre_vectors: Optional[Dict[int, np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        Runs comprehensive evaluation on a sampled cohort of evaluation users.
        """
        rng = np.random.RandomState(42)
        target_users = (
            rng.choice(self.eval_users, size=min(sample_users, len(self.eval_users)), replace=False)
            if len(self.eval_users) > sample_users else self.eval_users
        )

        metrics = {f"Precision@{k}": [] for k in top_k_list}
        metrics.update({f"Recall@{k}": [] for k in top_k_list})
        metrics.update({f"NDCG@{k}": [] for k in top_k_list})
        metrics.update({f"MAP@{k}": [] for k in top_k_list})

        all_recommendations_k10 = []

        for uid in target_users:
            gt = self.ground_truth_map[uid]
            # Generate top-20 recommendations
            recs = model.recommend(user_id=uid, top_k=max(top_k_list), exclude_seen=True)
            rec_ids = [mid for mid, _ in recs]
            
            all_recommendations_k10.append(rec_ids[:10])

            for k in top_k_list:
                metrics[f"Precision@{k}"].append(precision_at_k(rec_ids, gt, k=k))
                metrics[f"Recall@{k}"].append(recall_at_k(rec_ids, gt, k=k))
                metrics[f"NDCG@{k}"].append(ndcg_at_k(rec_ids, gt, k=k))
                metrics[f"MAP@{k}"].append(average_precision_at_k(rec_ids, gt, k=k))

        # Aggregate means
        summary_results = {
            metric: float(np.mean(vals))
            for metric, vals in metrics.items()
        }

        # Catalog Coverage
        summary_results["Coverage@10"] = float(catalog_coverage(all_recommendations_k10, self.total_catalog_size))

        # Novelty
        if self.popularity_dict:
            summary_results["Novelty@10"] = float(calculate_novelty(all_recommendations_k10, self.popularity_dict, self.total_interactions))

        # Diversity
        if genre_vectors:
            summary_results["IntraListDiversity@10"] = float(calculate_genre_diversity(all_recommendations_k10, genre_vectors))

        return summary_results
