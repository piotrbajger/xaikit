from unittest import TestCase

import numpy as np

from xaikit.ceteris_paribus import ceteris_paribus

from sklearn.linear_model import LinearRegression


class TestCeterisParibus(TestCase):
    """Test cases for ceteris paribus method."""

    def test_ceteris_paribus(self):
        """Test importance of each feature separately."""
        x = np.random.random((100, 2))
        y = x[:, 0] + 2 * x[:, 1]

        est = LinearRegression()
        est.fit(x, y)

        result = ceteris_paribus(est, [0., 1.], feature=0, values=[-1, -0.5, 0, 0.5, 1])

        cp = result["ceteris_paribus"]
        self.assertTrue(np.max(np.abs(cp - [1, 1.5, 2., 2.5, 3.])) < 1e-10)
        self.assertTrue(np.abs(result["base_prediction"] - 2.) < 1e-10)
