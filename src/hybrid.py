"""
Hybrid Recommendation System
============================
Integrates Content-Based semantic matching and Collaborative Filtering latent factors
with score normalization, tunable component weighting, popularity de-biasing,
and genre-diversity MMR re-ranking.
Addresses Section 8 (Q79 - Q86), Q117, Q118.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Set

from src.base import BaseRecommender
from src.content_based import ContentBasedRecommender
from src.collaborative import MatrixFactorizationSVD
from src.cold_start import ColdStartPopularityRecommender


def min_max_normalize(scores: Dict[int, float]) -> Dict[int, float]:
    """Normalizes score dictionary values to [0, 1] range."""
    if not scores:
        return {}
    vals = list(scores.values())
    min_v, max_v = min(vals), max(vals)
    if max_v == min_v:
        return {k: 1.0 for k in scores}
    return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}


class HybridRecommender(BaseRecommender):
    """
    Weighted Hybrid Recommender uniting Collaborative SVD, Content-Based vectors,
    and Bayesian Popularity with dynamic cold-start routing and genre diversity MMR.
    """

    def __init__(
        self,
        collaborative_model: Optional[MatrixFactorizationSVD] = None,
        content_model: Optional[ContentBasedRecommender] = None,
        popularity_model: Optional[ColdStartPopularityRecommender] = None,
        collab_model: Optional[MatrixFactorizationSVD] = None,
        collab_weight: float = 0.60,
        content_weight: float = 0.30,
        popularity_weight: float = 0.10,
        diversity_penalty: float = 0.15
    ):
        super().__init__(name="WeightedHybridRecommender")
        self.collab_model = collaborative_model if collaborative_model is not None else collab_model
        self.collaborative_model = self.collab_model
        self.content_model = content_model
        self.pop_model = popularity_model
        self.popularity_model = popularity_model
        
        self.collab_weight = collab_weight
        self.content_weight = content_weight
        self.popularity_weight = popularity_weight
        self.diversity_penalty = diversity_penalty

        self.movie_genres_map: Dict[int, Set[str]] = {}
        self.user_history_map: Dict[int, Dict[int, float]] = {}

    def fit(
        self,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame
    ) -> "HybridRecommender":
        """
        Indexes movie genre sets and user rating histories for fast hybrid resolution.
        """
        # Build movie genres mapping
        for _, row in movies_df.iterrows():
            mid = int(row["movieId"])
            g_str = str(row["genres"]) if pd.notna(row["genres"]) else ""
            genres = {g.strip() for g in g_str.split("|") if g.strip()}
            self.movie_genres_map[mid] = genres

        # Build user history map (movieId -> rating)
        user_groups = ratings_df.groupby("userId")
        for uid, grp in user_groups:
            self.user_history_map[int(uid)] = dict(zip(grp["movieId"].astype(int), grp["rating"].astype(float)))

        self.is_fitted = True
        return self

    def recommend(
        self,
        user_id: Optional[int] = None,
        item_id: Optional[int] = None,
        top_k: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Generates personalized hybrid recommendations for target user.
        Routes cold-start users gracefully.
        """
        if not self.is_fitted:
            raise ValueError("HybridRecommender must be fitted before recommendation.")

        # 1. User Cold-Start Routing: If user is new or has no interaction history
        if user_id is None or user_id not in self.user_history_map:
            return self.pop_model.recommend(top_k=top_k, exclude_seen=exclude_seen)

        user_ratings = self.user_history_map[user_id]
        seen_movies = set(user_ratings.keys()) if exclude_seen else set()

        # If sparse history (< 3 ratings), lean heavily on content + popularity
        is_sparse_user = len(user_ratings) < 3

        # 2. Retrieve Candidate Recommendations from Collaborative Model
        collab_candidates = self.collab_model.recommend(user_id=user_id, top_k=top_k * 4, exclude_seen=exclude_seen)
        collab_dict = {mid: score for mid, score in collab_candidates}

        # 3. Retrieve Candidate Recommendations from Content Model
        content_candidates = self.content_model.recommend_for_user_profile(user_ratings=user_ratings, top_k=top_k * 4, exclude_seen=exclude_seen)
        content_dict = {mid: score for mid, score in content_candidates}

        # 4. Normalize component scores into [0, 1]
        norm_collab = min_max_normalize(collab_dict)
        norm_content = min_max_normalize(content_dict)

        # Union of candidate items
        all_candidates = set(norm_collab.keys()) | set(norm_content.keys())

        # If candidates are empty, fall back to Bayesian popularity
        if not all_candidates:
            return self.pop_model.recommend(top_k=top_k, seen_movies=seen_movies)

        # 5. Dynamic Weighting
        if is_sparse_user:
            w_collab = 0.15
            w_content = 0.65
            w_pop = 0.20
        else:
            w_collab = self.collab_weight
            w_content = self.content_weight
            w_pop = self.popularity_weight

        # 6. Score Fusion
        fused_scores: Dict[int, float] = {}
        for mid in all_candidates:
            c_score = norm_collab.get(mid, 0.0)
            t_score = norm_content.get(mid, 0.0)
            p_score = self.pop_model.movie_scores.get(mid, 0.0) / 5.0  # normalize rating
            
            fused_score = (w_collab * c_score) + (w_content * t_score) + (w_pop * p_score)
            fused_scores[mid] = fused_score

        # 7. Genre Diversity MMR Re-ranking (Penalize excessive genre redundancy)
        ranked_candidates = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_recommendations: List[Tuple[int, float]] = []
        selected_genres_count: Dict[str, int] = {}

        for mid, base_score in ranked_candidates:
            genres = self.movie_genres_map.get(mid, set())
            # Calculate redundancy penalty based on already chosen genres in top list
            genre_overlap = sum(selected_genres_count.get(g, 0) for g in genres)
            adjusted_score = base_score - (self.diversity_penalty * (genre_overlap / (len(genres) + 1e-5)))

            final_recommendations.append((mid, max(0.0, adjusted_score)))
            for g in genres:
                selected_genres_count[g] = selected_genres_count.get(g, 0) + 1

        # Re-sort with diversity-adjusted scores
        final_recommendations.sort(key=lambda x: x[1], reverse=True)
        return final_recommendations[:top_k]
