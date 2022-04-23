from unittest import TestCase

import numpy as np

from xaikit.explainers.breakdown import breakdown

from .utils import MockBinaryLinearClassifier, MockLinearRegressor


class TestBreakdown(TestCase):
    """Test cases for the breakdown method."""

    @classmethod
    def setup_class(cls):
        ran = np.random.RandomState(42)
        cls.x = ran.random((100, 4))
        cls.weights = np.array([1, 2, -0.5, 0.25])

    def test_breakdown_regression(self):
        est = MockLinearRegressor(self.weights)

        x0 = np.array([0.5, 1.0, 1.0, 0.1])
        contributions = breakdown(est, self.x, x0)["breakdown"]

        x_mean = np.mean(self.x, axis=0)
        expected_contributions = (x0 - x_mean) * self.weights

        self.assertTrue(np.allclose(contributions, expected_contributions))

    def test_breakdown_regression_permute_features(self):
        est = MockLinearRegressor(self.weights)

        x0 = np.array([0.5, 1.0, 0.2, -0.5])
        contributions = breakdown(est, self.x, x0, features=[2, 0])["breakdown"]

        x_mean = np.mean(self.x, axis=0)
        theoretical_contributions = (x0 - x_mean) * self.weights
        expected_contributions = [
            theoretical_contributions[2],
            theoretical_contributions[0],
            theoretical_contributions[1] + theoretical_contributions[3],
        ]

        self.assertTrue(np.allclose(contributions, expected_contributions))

    def test_breakdown_binary_classification(self):
        est = MockBinaryLinearClassifier(self.weights)

        x0 = np.array([0.5, 1.0, 1.0, 0.1])
        contributions = breakdown(est, self.x, x0)["breakdown"]

        x_mean = np.mean(self.x, axis=0)
        contribution_signs = np.sign(contributions)
        expected_contribution_signs = np.sign((x0 - x_mean) * self.weights)

        self.assertTrue(np.all(contribution_signs == expected_contribution_signs))
