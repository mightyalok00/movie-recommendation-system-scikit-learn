"""
Scikit-Learn Reusable Estimator Pipeline
=========================================
Encapsulates the end-to-end MovieLens recommender workflow into a custom,
Scikit-learn compliant Estimator class conforming to sklearn BaseEstimator.
Addresses Q123.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.content_based import ContentBasedRecommender
from src.collaborative import MatrixFactorizationSVD
from src.cold_start import ColdStartPopularityRecommender
from src.hybrid import HybridRecommender


class MovieLensRecommendationPipeline(BaseEstimator):
    """
    Unified Scikit-Learn Compatible Recommender Pipeline.
    Supports fit(X), predict(X), recommend(user_id, top_k), get_params(), set_params().
    """

    def __init__(
        self,
        n_svd_components: int = 64,
        collab_weight: float = 0.60,
        content_weight: float = 0.30,
        popularity_weight: float = 0.10,
        diversity_penalty: float = 0.15,
        min_popularity_m: int = 50,
        random_state: int = 42
    ):
        self.n_svd_components = n_svd_components
        self.collab_weight = collab_weight
        self.content_weight = content_weight
        self.popularity_weight = popularity_weight
        self.diversity_penalty = diversity_penalty
        self.min_popularity_m = min_popularity_m
        self.random_state = random_state

        # Internal sub-models
        self.content_model = None
        self.svd_model = None
        self.pop_model = None
        self.hybrid_model = None
        self.is_fitted_ = False

    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, tags_df: Optional[pd.DataFrame] = None):
        """
        Fits all constituent sub-models and links them inside the hybrid orchestrator.
        """
        # 1. Content Model
        content_meta = movies_df.copy()
        if tags_df is not None:
            if "combined_tags" in tags_df.columns:
                content_meta = pd.merge(content_meta, tags_df[["movieId", "combined_tags"]], on="movieId", how="left")
            elif "tag" in tags_df.columns:
                tags_agg = tags_df.groupby("movieId")["tag"].apply(lambda t: " ".join(t.dropna().astype(str))).reset_index()
                tags_agg.columns = ["movieId", "combined_tags"]
                content_meta = pd.merge(content_meta, tags_agg, on="movieId", how="left")
            else:
                content_meta["combined_tags"] = ""
        else:
            content_meta["combined_tags"] = ""
            
        content_meta["combined_tags"] = content_meta["combined_tags"].fillna("")
        content_meta["clean_title"] = content_meta["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True).str.strip()
        
        self.content_model = ContentBasedRecommender()
        self.content_model.fit(content_meta)

        # 2. Collaborative SVD Model
        from data_loader import MovieLensDataLoader
        loader = MovieLensDataLoader()
        matrix, u2i, _, m2i, _ = loader.get_user_movie_sparse_matrix(ratings_df)
        
        self.svd_model = MatrixFactorizationSVD(n_components=self.n_svd_components, random_state=self.random_state)
        self.svd_model.fit(matrix, u2i, m2i, ratings_df=ratings_df)

        # 3. Popularity Baseline
        self.pop_model = ColdStartPopularityRecommender(min_ratings_m=self.min_popularity_m)
        self.pop_model.fit(ratings_df)

        # 4. Hybrid Recommender
        self.hybrid_model = HybridRecommender(
            collaborative_model=self.svd_model,
            content_model=self.content_model,
            popularity_model=self.pop_model,
            collab_weight=self.collab_weight,
            content_weight=self.content_weight,
            popularity_weight=self.popularity_weight,
            diversity_penalty=self.diversity_penalty
        )
        self.hybrid_model.fit(movies_df=movies_df, ratings_df=ratings_df)

        self.is_fitted_ = True
        return self

    def recommend(self, user_id: int, top_k: int = 10, exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """Generates recommendations via the hybrid pipeline."""
        if not self.is_fitted_:
            raise ValueError("Pipeline must be fitted first.")
        return self.hybrid_model.recommend(user_id=user_id, top_k=top_k, exclude_seen=exclude_seen)

    def predict(self, user_item_pairs: List[Tuple[int, int]]) -> np.ndarray:
        """Predicts rating scores for a batch of (user, item) pairs."""
        if not self.is_fitted_:
            raise ValueError("Pipeline must be fitted first.")
        return np.array([self.svd_model.predict_rating(u, i) for u, i in user_item_pairs])
