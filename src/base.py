"""
Base Recommender Abstract Class
===============================
Defines the unified interface for all recommendation algorithms in the project.
Follows Scikit-learn estimator conventions (fit, transform/predict/recommend).
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np


class BaseRecommender(ABC):
    """
    Abstract Base Class establishing standard fit/recommend interfaces
    compatible with Scikit-learn design patterns.
    """

    def __init__(self, name: str = "BaseRecommender"):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, *args, **kwargs) -> "BaseRecommender":
        """
        Fit the recommendation model to training data.
        """
        pass

    @abstractmethod
    def recommend(
        self,
        user_id: Optional[int] = None,
        item_id: Optional[int] = None,
        top_k: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Generate top-K recommended movie IDs and their predicted scores.
        
        Args:
            user_id: Target user ID (for personalized models).
            item_id: Seed movie ID (for item-to-item similarity models).
            top_k: Number of recommendations to return.
            exclude_seen: Whether to filter out movies already interacted with.
            
        Returns:
            List[Tuple[int, float]]: List of (movieId, score) pairs sorted in descending order of score.
        """
        pass

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Scikit-learn compatible parameter inspection."""
        return {"name": self.name}

    def set_params(self, **parameters) -> "BaseRecommender":
        """Scikit-learn compatible parameter setting."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
