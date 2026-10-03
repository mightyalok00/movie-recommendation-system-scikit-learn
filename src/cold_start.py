"""
Cold-Start Handling & Robustness Module
=======================================
Implements Bayesian-smoothed popularity baselines, genre-prior onboarding recommenders,
and content-fallback routers for new users and unrated items.
Addresses Section 10 (Q100 - Q105) & Baseline questions (Q115, Q116).
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Set
from src.base import BaseRecommender


class ColdStartPopularityRecommender(BaseRecommender):
    """
    Bayesian-Smoothed Popularity Recommender (IMDB Weighted Rating formula).
    Calculates:
        Score = (v / (v + m)) * R + (m / (v + m)) * C
    where:
        v = number of ratings for movie
        m = minimum rating threshold (dampening parameter)
        R = average rating of movie
        C = global mean rating across all movies
    """

    def __init__(self, min_ratings_m: int = 50):
        super().__init__(name="BayesianPopularityBaseline")
        self.min_ratings_m = min_ratings_m
        self.ranked_movies: List[Tuple[int, float]] = []
        self.movie_scores: Dict[int, float] = {}
        self.global_mean_C: float = 3.5

    def fit(self, ratings_df: pd.DataFrame) -> "ColdStartPopularityRecommender":
        """
        Computes Bayesian weighted rating scores for all movies.
        """
        movie_stats = ratings_df.groupby("movieId")["rating"].agg(["count", "mean"]).reset_index()
        self.global_mean_C = float(ratings_df["rating"].mean())
        
        m = self.min_ratings_m
        C = self.global_mean_C
        
        v = movie_stats["count"]
        R = movie_stats["mean"]
        
        # Bayesian formula
        movie_stats["score"] = (v / (v + m)) * R + (m / (v + m)) * C
        movie_stats = movie_stats.sort_values(by="score", ascending=False)
        
        self.ranked_movies = [
            (int(row["movieId"]), float(row["score"]))
            for _, row in movie_stats.iterrows()
        ]
        self.movie_scores = dict(self.ranked_movies)
        self.is_fitted = True
        return self

    def recommend(
        self,
        user_id: Optional[int] = None,
        item_id: Optional[int] = None,
        top_k: int = 10,
        exclude_seen: bool = True,
        seen_movies: Optional[Set[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Returns top-K globally popular, high-quality movies, excluding seen items if provided.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before recommendation.")
            
        seen = seen_movies or set()
        recommendations = []
        for mid, score in self.ranked_movies:
            if mid in seen and exclude_seen:
                continue
            recommendations.append((mid, score))
            if len(recommendations) >= top_k:
                break

        return recommendations


class GenrePriorRecommender(BaseRecommender):
    """
    Onboarding Recommender for New Users.
    When a new user selects 1 or more favorite genres during registration,
    recommends highest Bayesian-rated movies strictly within those genres.
    """

    def __init__(self, popularity_model: ColdStartPopularityRecommender):
        super().__init__(name="GenrePriorOnboarding")
        self.popularity_model = popularity_model
        self.genre_to_movies: Dict[str, List[Tuple[int, float]]] = {}

    def fit(self, movies_df: pd.DataFrame) -> "GenrePriorRecommender":
        """
        Indexes movies by genre and sorts each genre bucket by Bayesian popularity score.
        """
        if not self.popularity_model.is_fitted:
            raise ValueError("Popularity model must be fitted first.")

        genre_buckets: Dict[str, List[Tuple[int, float]]] = {}
        movie_scores = self.popularity_model.movie_scores
        mids = movies_df["movieId"].to_numpy()
        genres_arr = movies_df["genres"].fillna("").to_numpy()

        for mid, genres_str in zip(mids, genres_arr):
            mid_int = int(mid)
            score = movie_scores.get(mid_int, 0.0)
            for g in str(genres_str).split("|"):
                g = g.strip()
                if g and g != "(no genres listed)":
                    if g not in genre_buckets:
                        genre_buckets[g] = []
                    genre_buckets[g].append((mid_int, score))

        # Sort each genre bucket descending by score
        for g in genre_buckets:
            genre_buckets[g].sort(key=lambda x: x[1], reverse=True)

        self.genre_to_movies = genre_buckets
        self.is_fitted = True
        return self

    def recommend(
        self,
        preferred_genres: Optional[List[str]] = None,
        top_k: int = 10,
        exclude_seen: Optional[Set[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Interleaves top movies from selected preferred genres.
        """
        if not preferred_genres:
            return self.popularity_model.recommend(top_k=top_k)

        seen = exclude_seen or set()
        selected_movies: List[Tuple[int, float]] = []
        seen_in_rec = set()

        # Round-robin selection across selected genres to ensure diversity
        pointers = {g: 0 for g in preferred_genres if g in self.genre_to_movies}
        while len(selected_movies) < top_k and any(pointers[g] < len(self.genre_to_movies[g]) for g in pointers):
            for g in preferred_genres:
                if g in pointers and pointers[g] < len(self.genre_to_movies[g]):
                    mid, score = self.genre_to_movies[g][pointers[g]]
                    pointers[g] += 1
                    if mid not in seen and mid not in seen_in_rec:
                        selected_movies.append((mid, score))
                        seen_in_rec.add(mid)
        if not selected_movies:
            return self.popularity_model.recommend(top_k=top_k, exclude_seen=bool(exclude_seen), seen_movies=seen)

        return selected_movies
