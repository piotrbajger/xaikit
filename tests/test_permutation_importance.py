from unittest import TestCase

import pandas as pd
import numpy as np

from xaikit.permutation_importance import permutation_importance

from sklearn.linear_model import LinearRegression


class TestPermutationImportance(TestCase):
    """Test cases for permutation feature importance."""

    def test_importance(self):
        """Test importance of each feature separately."""
        x = np.random.random((100, 5))
        y = x[:, 0]

        est = LinearRegression()
        est.fit(x, y)

        result = permutation_importance(est, x, y, n_repeats=10, random_state=42)
        means = result["importances_mean"]

        self.assertTrue(means.shape == (x.shape[1],))
        self.assertTrue(np.argmax(means) == 0)

    def test_importance_pandas(self):
        """Test importance of each feature separately for a pandas dataframe."""
        x = pd.DataFrame(data=np.random.random((100, 5)))
        y = x.iloc[:, 0]

        est = LinearRegression()
        est.fit(x, y)

        result = permutation_importance(est, x, y, n_repeats=10, random_state=42)
        means = result["importances_mean"]

        self.assertTrue(means.shape == (x.shape[1],))
        self.assertTrue(np.argmax(means) == 0)

    def test_importance_groups(self):
        """Test importance of feature groups."""
        x = np.random.random((100, 5))
        y = x[:, 0]

        est = LinearRegression()
        est.fit(x, y)

        result = permutation_importance(
            est,
            x,
            y,
            n_repeats=10,
            random_state=42,
            feature_groups={"Imp": [0], "Rest": [1, 2, 3, 4]},
        )
        means = result["importances_mean"]

        self.assertTrue(means.shape == (2,))
        self.assertTrue(np.argmax(means) == 0)

    def test_importance_pandas_groups(self):
        """Test importance of feature groups for a pandas dataframe."""
        x = pd.DataFrame(
            columns=[f"x{i}" for i in range(5)], data=np.random.random((100, 5))
        )

        y = x.iloc[:, 0]

        est = LinearRegression()
        est.fit(x, y)

        result = permutation_importance(
            est,
            x,
            y,
            n_repeats=10,
            random_state=42,
            feature_groups={"Imp": ["x0"], "Rest": ["x1", "x2", "x3", "x4"]},
        )
        means = result["importances_mean"]

        self.assertTrue(means.shape == (2,))
        self.assertTrue(np.argmax(means) == 0)
