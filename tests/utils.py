import numpy as np


class MockLinearRegressor:
    """A mock linear regressor with fixed weights."""

    def __init__(self, weights):
        """
        :param weights: Weights for the estimator so that the estimate
            is ``x @ weigtts``.
        """
        self.weights = weights

    def predict(self, x):
        """Return prediction for ``x``."""
        return x @ self.weights


class MockBinaryLinearClassifier:
    """A mock binary linear classifier with fixed weights."""

    def __init__(self, weights):
        """
        :param weights: Weights for the estimator so that the estimate
            is ``x @ weigtts``.
        """
        self.weights = weights

    def predict_proba(self, x):
        """Return prediction for ``x``."""
        z = x @ self.weights
        return 1 / (1 + np.exp(-z))
