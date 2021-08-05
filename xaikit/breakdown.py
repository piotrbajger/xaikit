import numpy as np

from sklearn.utils import Bunch

from xaikit import utils


def breakdown(estimator, x, x0, features=None):
    """
    Creates data for a breakdown plot.

    :param estimator: An estimator compatible with the scikit-learn
        ``Estimator`` interface.
    :param x: An ``n x m`` feature matrix of ``n`` observations of ``m`` variables.
    :param x0: A single observation for which the breakdown plot is to be constructed.
    :param features: Controls which features are included in the result. This argument
        controls both the order in which features appear in the plot, as well as which
        features are bundled together under the "remaining features" label.
    """
    predict = utils.get_predictor(estimator)

    x0 = utils.ensure_2d_array(x0)
    n_vars = x0.shape[-1]

    if features is None:
        features = list(range(n_vars))
    all_features = utils.append_missing_features(features, n_features=n_vars)

    new_x = utils.get_values(x).copy()
    base_y = np.mean(predict(x), axis=0)
    target_y = predict(x0)
    contributions = contributions_along_path(predict, new_x, x0, all_features)

    return Bunch(
        breakdown=contributions,
        features=features,
        base_prediction=base_y,
        target_prediction=target_y,
    )


def contributions_along_path(predict, x, x0, feature_path):
    """
    Implementation of the breakdown logic. Separated so that it can be
    re-used in SHAP calculations.
    """
    base_y = np.mean(predict(x), axis=0)
    yhats = [base_y]

    for feature in feature_path:
        x[:, feature] = x0[:, feature]
        next_y = np.mean(predict(x), axis=0)
        yhats.append(next_y)

    contributions = np.diff(yhats, axis=0)
    return contributions
