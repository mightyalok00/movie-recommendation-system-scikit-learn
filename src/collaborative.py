"""
Collaborative Filtering & Latent Factor Matrix Decomposition
============================================================
Implements TruncatedSVD Matrix Factorization, Item-Item CF, and User-User CF
using Scikit-learn algorithms on sparse user-item interaction matrices.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Set
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

from src.base import BaseRecommender


class MatrixFactorizationSVD(BaseRecommender):
    """
    Latent Factor Collaborative Recommender via TruncatedSVD.
    Addresses Q52, Q53, Q56, Q57, Q58, Q59, Q60, Q61, Q72, Q73.
    """

    def __init__(self, n_components: int = 64, random_state: int = 42):
        super().__init__(name=f"TruncatedSVD(k={n_components})")
        self.n_components = n_components
        self.random_state = random_state
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state, algorithm="randomized")
        
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None  # Shape: (n_components, n_items)
        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}
        self.movie_to_idx: Dict[int, int] = {}
        self.idx_to_movie: Dict[int, int] = {}
        self.user_seen_items: Dict[int, Set[int]] = {}
        self.global_mean: float = 0.0

    def fit(
        self,
        rating_matrix: csr_matrix,
        user_to_idx: Dict[int, int],
        movie_to_idx: Dict[int, int],
        ratings_df: Optional[pd.DataFrame] = None
    ) -> "MatrixFactorizationSVD":
        """
        Decomposes the User-Movie rating matrix into latent user and item factor matrices.
        R ≈ U · Σ · V^T = UserFactors · ItemFactors
        """
        self.user_to_idx = user_to_idx
        self.idx_to_user = {idx: uid for uid, idx in user_to_idx.items()}
        self.movie_to_idx = movie_to_idx
        self.idx_to_movie = {idx: mid for mid, idx in movie_to_idx.items()}

        # Fit TruncatedSVD on sparse matrix
        self.user_factors = self.svd.fit_transform(rating_matrix)
        self.item_factors = self.svd.components_  # (n_components, n_items)
        
        # Calculate global rating mean
        if rating_matrix.nnz > 0:
            self.global_mean = float(rating_matrix.data.mean())

        # Track seen items for fast exclusion during user recommendation
        if ratings_df is not None:
            self.user_seen_items = ratings_df.groupby("userId")["movieId"].apply(set).to_dict()

        self.is_fitted = True
        return self

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """Percentage of variance explained by each latent dimension."""
        return self.svd.explained_variance_ratio_

    @property
    def total_explained_variance_(self) -> float:
        """Cumulative explained variance captured by the latent components."""
        return float(np.sum(self.svd.explained_variance_ratio_))

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """
        Predicts the rating score for a (userId, movieId) pair via dot product.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting.")
        
        if user_id not in self.user_to_idx or movie_id not in self.movie_to_idx:
            return self.global_mean

        u_idx = self.user_to_idx[user_id]
        m_idx = self.movie_to_idx[movie_id]
        
        score = float(np.dot(self.user_factors[u_idx], self.item_factors[:, m_idx]))
        return float(np.clip(score, 0.5, 5.0))

    def recommend(
        self,
        user_id: Optional[int] = None,
        item_id: Optional[int] = None,
        top_k: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Generates top-K recommendations for a user by scoring all items via vector matrix multiplication.
        """
        if not self.is_fitted or user_id is None:
            return []

        if user_id not in self.user_to_idx:
            # Fallback for unknown user (cold start)
            return []

        u_idx = self.user_to_idx[user_id]
        user_vector = self.user_factors[u_idx]  # Shape: (n_components,)
        
        # Score all catalog movies in parallel: (1, k) @ (k, n_items) -> (n_items,)
        scores = np.dot(user_vector, self.item_factors)
        
        # Mask out seen movies
        seen_movies = self.user_seen_items.get(user_id, set())
        if exclude_seen and seen_movies:
            seen_indices = [self.movie_to_idx[m] for m in seen_movies if m in self.movie_to_idx]
            scores[seen_indices] = -1e9

        # Efficient top-K selection with argpartition
        if len(scores) <= top_k:
            top_indices = np.argsort(-scores)
        else:
            top_partition = np.argpartition(-scores, top_k)[:top_k]
            top_indices = top_partition[np.argsort(-scores[top_partition])]

        recommendations = []
        for idx in top_indices:
            movie_id = self.idx_to_movie[idx]
            score = float(scores[idx])
            if score > -1e8:
                recommendations.append((int(movie_id), score))

        return recommendations


class ItemItemCollaborativeFiltering(BaseRecommender):
    """
    Item-Item k-Nearest Neighbors Collaborative Recommender.
    Finds movies with similar co-rating patterns across users.
    Addresses Q51, Q54, Q55.
    """

    def __init__(self, n_neighbors: int = 20, metric: str = "cosine"):
        super().__init__(name="ItemItemKNN_CF")
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.nn_model = NearestNeighbors(metric=metric, algorithm="brute")
        self.movie_to_idx: Dict[int, int] = {}
        self.idx_to_movie: Dict[int, int] = {}
        self.item_matrix: Optional[csr_matrix] = None

    def fit(self, rating_matrix: csr_matrix, movie_to_idx: Dict[int, int]) -> "ItemItemCollaborativeFiltering":
        """
        Fits NearestNeighbors on transpose of user-movie matrix (Item x User).
        """
        self.movie_to_idx = movie_to_idx
        self.idx_to_movie = {idx: mid for mid, idx in movie_to_idx.items()}
        
        # Transpose: rows are items, columns are users
        self.item_matrix = rating_matrix.T.tocsr()
        self.nn_model.fit(self.item_matrix)
        self.is_fitted = True
        return self

    def recommend(
        self,
        user_id: Optional[int] = None,
        item_id: Optional[int] = None,
        top_k: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[int, float]]:
        if not self.is_fitted or item_id not in self.movie_to_idx:
            return []

        idx = self.movie_to_idx[item_id]
        target_vec = self.item_matrix[idx]
        
        if target_vec.nnz == 0:
            return []

        distances, indices = self.nn_model.kneighbors(target_vec, n_neighbors=min(top_k + 1, self.item_matrix.shape[0]))
        
        recommendations = []
        for dist, neighbor_idx in zip(distances[0], indices[0]):
            rec_movie_id = self.idx_to_movie[neighbor_idx]
            if rec_movie_id == item_id and exclude_seen:
                continue
            sim_score = float(max(0.0, 1.0 - dist))
            recommendations.append((int(rec_movie_id), sim_score))
            if len(recommendations) >= top_k:
                break

        return recommendations


class UserUserCollaborativeFiltering(BaseRecommender):
    """
    User-User k-Nearest Neighbors Collaborative Recommender.
    Identifies peers with similar rating tastes and aggregates their top-rated movies.
    Addresses Q51, Q54.
    """

    def __init__(self, n_neighbors: int = 30, metric: str = "cosine"):
        super().__init__(name="UserUserKNN_CF")
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.nn_model = NearestNeighbors(metric=metric, algorithm="brute")
        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}
        self.idx_to_movie: Dict[int, int] = {}
        self.rating_matrix: Optional[csr_matrix] = None
        self.user_seen_items: Dict[int, Set[int]] = {}

    def fit(
        self,
        rating_matrix: csr_matrix,
        user_to_idx: Dict[int, int],
        movie_to_idx: Dict[int, int],
        ratings_df: Optional[pd.DataFrame] = None
    ) -> "UserUserCollaborativeFiltering":
        self.rating_matrix = rating_matrix
        self.user_to_idx = user_to_idx
        self.idx_to_user = {idx: uid for uid, idx in user_to_idx.items()}
        self.idx_to_movie = {idx: mid for mid, idx in movie_to_idx.items()}

        if ratings_df is not None:
            self.user_seen_items = ratings_df.groupby("userId")["movieId"].apply(set).to_dict()

        self.nn_model.fit(self.rating_matrix)
        self.is_fitted = True
        return self

    def recommend(
        self,
        user_id: Optional[int] = None,
        item_id: Optional[int] = None,
        top_k: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[int, float]]:
        if not self.is_fitted or user_id not in self.user_to_idx:
            return []

        u_idx = self.user_to_idx[user_id]
        target_vec = self.rating_matrix[u_idx]

        distances, neighbor_indices = self.nn_model.kneighbors(target_vec, n_neighbors=self.n_neighbors + 1)
        
        # Exclude query user
        neighbor_indices = neighbor_indices[0][1:]
        similarities = 1.0 - distances[0][1:]

        # Aggregate neighbor ratings weighted by similarity
        seen_movies = self.user_seen_items.get(user_id, set()) if exclude_seen else set()
        candidate_scores: Dict[int, float] = {}
        sim_sums: Dict[int, float] = {}

        for n_idx, sim in zip(neighbor_indices, similarities):
            if sim <= 0:
                continue
            n_row = self.rating_matrix[n_idx]
            movie_indices = n_row.indices
            ratings = n_row.data

            for m_idx, rating in zip(movie_indices, ratings):
                mid = self.idx_to_movie[m_idx]
                if mid in seen_movies:
                    continue
                candidate_scores[mid] = candidate_scores.get(mid, 0.0) + (sim * rating)
                sim_sums[mid] = sim_sums.get(mid, 0.0) + sim

        # Normalized predicted ratings
        predictions = [
            (mid, candidate_scores[mid] / sim_sums[mid])
            for mid in candidate_scores
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]
