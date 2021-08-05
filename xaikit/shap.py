import numpy as np
import numpy.random

from sklearn.utils import Bunch

from xaikit import utils
from xaikit import breakdown


def shap(estimator, x, x0, features=None, n_permutations=10, seed=42):
    """
    Calculates Shapley Additive Explanations (SHAP) contributions.

    :param estimator: An estimator compatible with the scikit-learn
        ``Estimator`` interface.
    :param x: An ``n x m`` feature matrix of ``n`` observations of ``m`` variables.
    :param x0: A single observation for which the breakdown plot is to be constructed.
    :param features: Controls which features are included in the result. This argument
        controls both the order in which features appear in the plot, as well as which
        features are bundled together.
    :param n_permutations: Number of permutations to compute the SHAP values.
    :param seed: Seed number for the permutations generator.
    """
    rng = np.random.RandomState(seed)

    predict = utils.get_predictor(estimator)
    x0 = utils.ensure_2d_array(x0)

    n_vars = x0.shape[-1]

    if features is None:
        features = list(range(n_vars))
    all_features = utils.append_missing_features(features, n_features=n_vars)

    n_features = len(features)

    base_y = np.mean(predict(x), axis=0)
    target_y = predict(x0)
    contributions = np.zeros((n_permutations, n_features))

    for i in range(n_permutations):
        new_x = utils.get_values(x).copy()

        feature_path = rng.permutation(range(n_features))
        permuted_features = [all_features[k] for k in feature_path]

        permuted_contributions = breakdown.contributions_along_path(
            predict, new_x, x0, permuted_features
        )
        contributions[i, feature_path] = permuted_contributions

    return Bunch(
        contributions=contributions,
        shap_values_mean=contributions.mean(axis=0),
        shap_values_std=contributions.std(axis=0),
        base_prediction=base_y,
        target_prediction=target_y,
    )
