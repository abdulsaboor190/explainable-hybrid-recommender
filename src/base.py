"""Base class definition for explainable recommender models."""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple
import pandas as pd


class BaseRecommender(ABC):
    """Abstract base class for all explainable recommendation models.
    
    Provides standard interfaces for fitting models on interaction datasets,
    predicting bounded ratings, generating ranked recommendations, and providing
    human-interpretable explanations.
    """

    RANDOM_STATE: int = 42

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit the recommendation model to training interaction data.

        Parameters
        ----------
        train_df : pd.DataFrame
            DataFrame containing interaction history with required columns:
            ['user_id', 'item_id', 'rating', 'timestamp'].
        """
        pass

    @abstractmethod
    def predict(self, user: Any, item: Any) -> float:
        """Predict the rating a user would assign to a specific item.

        The predicted rating must strictly reside in the range [1.0, 5.0].
        If user or item is unobserved in training data, the implementation
        must fall back gracefully: first to user mean (if user is known),
        and then to global training mean (if user is unknown or unrated).

        Parameters
        ----------
        user : Any
            Unique identifier of the user (user_id).
        item : Any
            Unique identifier of the item (item_id).

        Returns
        -------
        float
            Predicted rating bounded within [1.0, 5.0].
        """
        pass

    @abstractmethod
    def recommend(
        self, user: Any, n: int = 10, exclude_seen: bool = True
    ) -> List[Tuple[Any, float]]:
        """Generate top-N item recommendations for a given user.

        Parameters
        ----------
        user : Any
            Unique identifier of the user for whom to recommend items.
        n : int, default=10
            Number of top items to return.
        exclude_seen : bool, default=True
            Whether to exclude items already interacted with in train_df.

        Returns
        -------
        List[Tuple[Any, float]]
            List of (item_id, predicted_rating) tuples sorted in descending order
            by predicted rating.
        """
        pass

    @abstractmethod
    def explain(self, user: Any, item: Any) -> str:
        """Generate a transparent, human-interpretable explanation for a prediction.

        Parameters
        ----------
        user : Any
            Unique identifier of the user.
        item : Any
            Unique identifier of the item.

        Returns
        -------
        str
            A clear text explanation detailing why this item is recommended
            to this user (e.g., neighbor consensus, matching content features,
            or baseline tendencies).
        """
        pass

