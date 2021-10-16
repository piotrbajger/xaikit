from unittest import TestCase

import numpy as np

from xaikit.adapters.adapter_interface import ModelAdapterInterface
from xaikit.shap import shap


class TestShap(TestCase):
    """Test cases for the SHAP method."""

    def test_shap_no_interactions(self):
        ran = np.random.RandomState(42)
        x = ran.random((100, 4))
        weights = np.array([1, 2, -0.5, 0.25])

        class MockEstimator(ModelAdapterInterface):
            def predict(self, x):
                return x @ weights

        est = MockEstimator()

        x0 = np.array([0.5, 1.0, 1.0, 0.1])
        shap_result = shap(est, x, x0)

        shap_means = shap_result["shap_values_mean"]
        shap_std = shap_result["shap_values_std"]

        x_mean = np.mean(x, axis=0)
        expected_values = (x0 - x_mean) * weights

        self.assertTrue(np.allclose(expected_values, shap_means))
        self.assertTrue(np.allclose(0, shap_std))

    def test_shap_with_interactions(self):
        ran = np.random.RandomState(42)
        x = ran.random((100, 4))
        weights = np.array([1, 2, -0.5, 0.25])

        class MockEstimator(ModelAdapterInterface):
            def predict(self, x):
                return x @ weights - 0.5 * x[:, 1] * x[:, 3]

        est = MockEstimator()

        x0 = np.array([0.5, 1.0, 1.0, 0.1])
        shap_result = shap(est, x, x0)

        shap_means = shap_result["shap_values_mean"]
        shap_std = shap_result["shap_values_std"]

        x_mean = np.mean(x, axis=0)
        expected_values = (x0 - x_mean) * weights

        self.assertAlmostEqual(expected_values[0], shap_means[0])
        self.assertNotAlmostEqual(expected_values[1], shap_means[1])
        self.assertAlmostEqual(expected_values[2], shap_means[2])
        self.assertNotAlmostEqual(expected_values[3], shap_means[3])

        self.assertAlmostEqual(shap_std[0], 0)
        self.assertNotAlmostEqual(shap_std[1], 0)
        self.assertAlmostEqual(shap_std[2], 0)
        self.assertNotAlmostEqual(shap_std[3], 0)
