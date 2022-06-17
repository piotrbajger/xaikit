from unittest import TestCase

import numpy as np

from sklearn.linear_model import LinearRegression

from xaikit.explainers.ceteris_paribus import ceteris_paribus


class TestCeterisParibus(TestCase):
    """Test cases for ceteris paribus method."""

    def test_ceteris_paribus(self):
        """Test importance of each feature separately."""
        ran = np.random.RandomState(42)
        x = ran.random((100, 2))
        y = x[:, 0] + 2 * x[:, 1]

        model = LinearRegression()
        model.fit(x, y)

        result = ceteris_paribus(
            model, [0.0, 1.0], feature=0, values=[-1, -0.5, 0, 0.5, 1]
        )

        cp = result["ceteris_paribus"]
        self.assertTrue(np.max(np.abs(cp - [1, 1.5, 2.0, 2.5, 3.0])) < 1e-10)
        self.assertTrue(np.abs(result["base_prediction"] - 2.0) < 1e-10)
