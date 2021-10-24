from unittest import TestCase

import numpy as np

from xaikit.explainers.breakdown import breakdown
from xaikit.adapters.adapter_interface import ModelAdapterInterface


class TestCeterisParibus(TestCase):
    """Test cases for the breakdown method."""

    def test_breakdown(self):
        ran = np.random.RandomState(42)
        x = ran.random((100, 4))
        weights = np.array([1, 2, -0.5, 0.25])

        class MockEstimator(ModelAdapterInterface):
            def predict(self, x):
                return x @ weights

        est = MockEstimator()

        x0 = np.array([0.5, 1.0, 1.0, 0.1])
        contributions = breakdown(est, x, x0)["breakdown"]

        x_mean = np.mean(x, axis=0)
        expected_contributions = (x0 - x_mean) * weights

        self.assertTrue(np.allclose(contributions, expected_contributions))

    def test_breakdown_features(self):
        ran = np.random.RandomState(42)
        x = ran.random((100, 4))
        weights = np.array([1, 2, -0.5, 0.25])

        class MockEstimator(ModelAdapterInterface):
            def predict(self, x):
                return x @ weights

        est = MockEstimator()

        x0 = np.array([0.5, 1.0, 0.2, -0.5])
        contributions = breakdown(est, x, x0, features=[2, 0])["breakdown"]

        x_mean = np.mean(x, axis=0)
        theoretical_contributions = (x0 - x_mean) * weights
        expected_contributions = [
            theoretical_contributions[2],
            theoretical_contributions[0],
            theoretical_contributions[1] + theoretical_contributions[3],
        ]

        self.assertTrue(np.allclose(contributions, expected_contributions))
