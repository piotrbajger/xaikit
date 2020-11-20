import numpy as np

from sklearn.utils import Bunch, check_random_state, check_array


def ceteris_paribus(estimator, x, feature, values, relative=False):
    """
    Creates a "ceteris paribus" data for a given feature.

    Computes predictions of an ``estimator`` for a subset ``values``
    of possible values of a given feature ``feature`` while keeping
    the remaining features fixed.

    :param estimator: A fitted estimator.
    :param x ndarray or DataFrame, shape (1, n_features): A single observation
        used as the basis for computation.
    :param feature: A feature to be varied.
    :param values: Values to be admitted by the feature.
    :param relative: If True will return differences between predictions for
        a perturbed observation and the baseline prediction for ``x``.
        Default: False.
    """
    if isinstance(x, list):
        x = np.array(x)

    if not hasattr(x, "iloc"):
        x = check_array(x.reshape(1, -1), force_all_finite="allow-nan", dtype=None)

    if hasattr(estimator, "predict_proba"):
        predict = estimator.predict_proba
    else:
        predict = estimator.predict

    if isinstance(feature, str) and hasattr(x, "columns"):
        feature = list(x.columns).index(feature)

    base_y = predict(x.reshape(1, -1))[0]

    n_values = len(values)
    x_all = np.concatenate([x for _ in range(n_values)], axis=0)
    x_all[:, feature] = values

    y = predict(x_all)

    if relative:
        y -= base_y

    return Bunch(ceteris_paribus=y, base_prediction=base_y)
