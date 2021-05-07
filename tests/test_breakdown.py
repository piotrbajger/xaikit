from unittest import TestCase

import numpy as np

from xaikit.breakdown import breakdown


class TestCeterisParibus(TestCase):
    """Test cases for the breakdown method."""

    def test_breakdown(self):
        """Test breakdown for a simple linear model"""
        ran = np.random.RandomState(42)
        x = ran.random((100, 2))

        class MockEstimator:
            def predict(self, x):
                return x[:, 0] + 2 * x[:, 1]

        est = MockEstimator()

        result = breakdown(est, x, x0=np.array([0.5, 1.0]))

        # Compute mean prediction
        base_y = np.mean(est.predict(x), axis=0)
        self.assertEqual(result["breakdown"][0], base_y)

        # Fix first variable
        result_1 = 0.5 + 2 * np.mean(x[:, 1], axis=0)
        self.assertAlmostEqual(result["breakdown"][1], result_1, places=10)

        # Fix both variables (=> prediction for x0)
        self.assertEqual(result["breakdown"][2], 2.5)

    def test_breakdown_features(self):
        """Test breakdown feature selection"""
        ran = np.random.RandomState(42)
        x = ran.random((100, 4))

        class MockEstimator:
            def predict(self, x):
                return x @ np.array([1, 2, -4, 1])

        est = MockEstimator()

        result = breakdown(est, x, x0=np.array([0.5, 1.0, 0.2, -0.5]), features=[0, 2])

        # Compute mean prediction
        base_y = np.mean(est.predict(x), axis=0)
        self.assertEqual(result["breakdown"][0], base_y)

        # Last entry is the prediction for x0
        self.assertEqual(result["breakdown"][-1], 1.2)

        self.assertEqual(len(result["breakdown"]), 4)
