"""
Content-Based Recommendation Models
====================================
Implements genre-based, tag TF-IDF, and unified multi-modal content recommenders
using Scikit-learn feature extractors and NearestNeighbors indexers.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

from src.base import BaseRecommender


def split_pipe_genres(text: str) -> List[str]:
    """Top-level tokenizer for genre strings to enable pickle/joblib serialization."""
    return text.split("|")


class GenreRecommender(BaseRecommender):
    """
    Pure Genre-Based Recommender using multi-hot binary vectors and Cosine Similarity.
    Addresses Q33, Q43, Q45, Q46.
    """

    def __init__(self, metric: str = "cosine", n_neighbors: int = 20):
        super().__init__(name="GenreRecommender")
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.vectorizer = CountVectorizer(tokenizer=split_pipe_genres, token_pattern=None, lowercase=False, binary=True)
        self.nn_model = NearestNeighbors(metric=self.metric, algorithm="brute")
        self.movie_ids: np.ndarray = np.array([])
        self.movie_to_idx: Dict[int, int] = {}
        self.genre_matrix: Optional[csr_matrix] = None

    def fit(self, movies_df: pd.DataFrame) -> "GenreRecommender":
        """
        Fits multi-hot genre matrix and builds Nearest Neighbors index.
        """
        self.movie_ids = movies_df["movieId"].to_numpy()
        self.movie_to_idx = {mid: idx for idx, mid in enumerate(self.movie_ids)}
        
        # Clean genres: replace missing with empty string
        genres_series = movies_df["genres"].fillna("(no genres listed)")
        self.genre_matrix = self.vectorizer.fit_transform(genres_series)
        
        self.nn_model.fit(self.genre_matrix)
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
        Retrieves top-K similar movies based purely on genre overlap.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling recommend.")
        if item_id not in self.movie_to_idx:
            return []

        idx = self.movie_to_idx[item_id]
        target_vec = self.genre_matrix[idx]
        
        # Query top_k + 1 neighbors to account for query item itself
        distances, indices = self.nn_model.kneighbors(target_vec, n_neighbors=min(top_k + 1, len(self.movie_ids)))
        
        recommendations = []
        for dist, neighbor_idx in zip(distances[0], indices[0]):
            rec_movie_id = self.movie_ids[neighbor_idx]
            if rec_movie_id == item_id and exclude_seen:
                continue
            sim_score = float(1.0 - dist)  # Cosine distance = 1 - cosine_similarity
            recommendations.append((int(rec_movie_id), sim_score))
            if len(recommendations) >= top_k:
                break

        return recommendations


class TagTFIDFRecommender(BaseRecommender):
    """
    Tag-based Content Recommender using TF-IDF representation with sublinear scaling.
    Addresses Q36, Q37, Q38, Q44.
    """

    def __init__(
        self,
        max_features: int = 50000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        sublinear_tf: bool = True
    ):
        super().__init__(name="TagTFIDFRecommender")
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=sublinear_tf,
            stop_words="english"
        )
        self.nn_model = NearestNeighbors(metric="cosine", algorithm="brute")
        self.movie_ids: np.ndarray = np.array([])
        self.movie_to_idx: Dict[int, int] = {}
        self.tfidf_matrix: Optional[csr_matrix] = None

    def fit(self, metadata_df: pd.DataFrame) -> "TagTFIDFRecommender":
        """
        Transforms combined tag documents into TF-IDF sparse matrix and builds NN index.
        """
        self.movie_ids = metadata_df["movieId"].to_numpy()
        self.movie_to_idx = {mid: idx for idx, mid in enumerate(self.movie_ids)}
        
        tags_text = metadata_df["combined_tags"].fillna("").astype(str)
        self.tfidf_matrix = self.vectorizer.fit_transform(tags_text)
        self.nn_model.fit(self.tfidf_matrix)
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
        target_vec = self.tfidf_matrix[idx]
        
        # If the movie has no tags (zero vector), return empty recommendations
        if target_vec.nnz == 0:
            return []

        distances, indices = self.nn_model.kneighbors(target_vec, n_neighbors=min(top_k + 1, len(self.movie_ids)))
        
        recommendations = []
        for dist, neighbor_idx in zip(distances[0], indices[0]):
            rec_movie_id = self.movie_ids[neighbor_idx]
            if rec_movie_id == item_id and exclude_seen:
                continue
            sim_score = float(1.0 - dist)
            if sim_score > 0.0:
                recommendations.append((int(rec_movie_id), sim_score))
            if len(recommendations) >= top_k:
                break

        return recommendations


class ContentBasedRecommender(BaseRecommender):
    """
    Unified Multi-Modal Content Recommender combining:
    - Multi-hot Genre Features (weighted)
    - TF-IDF Tag Features (sublinear TF, unigrams + bigrams)
    - TF-IDF Title Features (character/word n-grams for series/sequel discovery)
    Addresses Q33-Q41, Q43-Q50.
    """

    def __init__(
        self,
        genre_weight: float = 1.0,
        tag_weight: float = 1.5,
        title_weight: float = 0.8,
        max_tag_features: int = 40000,
        max_title_features: int = 15000,
        metric: str = "cosine"
    ):
        super().__init__(name="UnifiedContentRecommender")
        self.genre_weight = genre_weight
        self.tag_weight = tag_weight
        self.title_weight = title_weight
        
        self.genre_vectorizer = CountVectorizer(tokenizer=split_pipe_genres, token_pattern=None, lowercase=False, binary=True)
        self.tag_vectorizer = TfidfVectorizer(
            max_features=max_tag_features,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            stop_words="english"
        )
        self.title_vectorizer = TfidfVectorizer(
            max_features=max_title_features,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            stop_words="english"
        )
        self.nn_model = NearestNeighbors(metric=metric, algorithm="brute")
        self.movie_ids: np.ndarray = np.array([])
        self.movie_to_idx: Dict[int, int] = {}
        self.feature_matrix: Optional[csr_matrix] = None

    def fit(self, content_df: pd.DataFrame) -> "ContentBasedRecommender":
        """
        Builds weighted multi-modal sparse feature matrix.
        """
        self.movie_ids = content_df["movieId"].to_numpy()
        self.movie_to_idx = {mid: idx for idx, mid in enumerate(self.movie_ids)}
        
        # 1. Multi-hot Genre matrix
        genre_mat = self.genre_vectorizer.fit_transform(content_df["genres"].fillna("(no genres listed)"))
        genre_mat = genre_mat.astype(np.float32) * self.genre_weight

        # 2. Tag TF-IDF matrix
        tag_mat = self.tag_vectorizer.fit_transform(content_df["combined_tags"].fillna(""))
        tag_mat = tag_mat.astype(np.float32) * self.tag_weight

        # 3. Title TF-IDF matrix
        title_mat = self.title_vectorizer.fit_transform(content_df["clean_title"].fillna(""))
        title_mat = title_mat.astype(np.float32) * self.title_weight

        # Stack horizontally into high-dimensional sparse representation
        self.feature_matrix = hstack([genre_mat, tag_mat, title_mat]).tocsr()
        self.nn_model.fit(self.feature_matrix)
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
        Item-to-item content recommendation.
        """
        if not self.is_fitted or item_id not in self.movie_to_idx:
            return []

        idx = self.movie_to_idx[item_id]
        target_vec = self.feature_matrix[idx]
        
        distances, indices = self.nn_model.kneighbors(target_vec, n_neighbors=min(top_k + 5, len(self.movie_ids)))
        
        recommendations = []
        for dist, neighbor_idx in zip(distances[0], indices[0]):
            rec_movie_id = self.movie_ids[neighbor_idx]
            if rec_movie_id == item_id and exclude_seen:
                continue
            sim_score = float(max(0.0, 1.0 - dist))
            recommendations.append((int(rec_movie_id), sim_score))
            if len(recommendations) >= top_k:
                break

        return recommendations

    def recommend_for_user_profile(
        self,
        user_ratings: Dict[int, float],
        top_k: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Builds an aggregated user preference vector by computing rating-weighted
        average of interacted item vectors, then queries NearestNeighbors.
        Addresses Q34, Q50, Q85.
        """
        if not self.is_fitted or not user_ratings:
            return []

        user_vec = None
        total_weight = 0.0

        for mid, rating in user_ratings.items():
            if mid in self.movie_to_idx and rating >= 3.0:
                weight = rating - 2.5  # Positive centering
                m_vec = self.feature_matrix[self.movie_to_idx[mid]] * weight
                user_vec = m_vec if user_vec is None else user_vec + m_vec
                total_weight += weight

        if user_vec is None or total_weight == 0:
            return []

        n_req = min(top_k + len(user_ratings), len(self.movie_ids))
        distances, indices = self.nn_model.kneighbors(user_vec, n_neighbors=n_req)
        
        recommendations = []
        for dist, neighbor_idx in zip(distances[0], indices[0]):
            rec_movie_id = self.movie_ids[neighbor_idx]
            if exclude_seen and rec_movie_id in user_ratings:
                continue
            sim_score = float(max(0.0, 1.0 - dist))
            recommendations.append((int(rec_movie_id), sim_score))
            if len(recommendations) >= top_k:
                break

        return recommendations
