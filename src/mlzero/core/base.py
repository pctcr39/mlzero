"""
core/base.py — Base class for ALL algorithms in this project.

Every algorithm (supervised, unsupervised, RL) inherits from BaseModel.
This gives a consistent interface across the entire project.
"""

import numpy as np


class BaseModel:
    """
    Base class for all ML models in this project.
    Every algorithm inherits from this and implements fit() and predict().
    """

    def fit(self, X, y=None):
        """
        Learn from data.
        - Supervised:      fit(X, y)   — X=features, y=labels
        - Unsupervised:    fit(X)      — no labels needed
        - Semi-supervised: fit(X, y)   — y has some None/NaN values
        """
        raise NotImplementedError("Every model must implement fit()")

    def predict(self, X):
        """Make predictions on new data."""
        raise NotImplementedError("Every model must implement predict()")

    def score(self, X, y):
        """Default scoring. Regression models use MSE. Classification models override with accuracy."""
        predictions = self.predict(X)
        return np.mean((predictions - y) ** 2)

    def __repr__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({params})"
