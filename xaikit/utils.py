import numpy as np
import pandas as pd


def get_predictor(estimator):
    """
    Returns a ``predict`` or ``predict_proba`` function of an estimator.

    :param estimator: An instance of scikit-learn Estimator.
    """
    # TODO: Handle non-scalar predictors (e.g. multiclass)
    if hasattr(estimator, "predict_proba"):
        return lambda x: estimator.predict_proba(x)[:, -1]
    else:
        return estimator.predict


def ensure_2d_array(x):
    """
    Ensures that an array ``x`` is two-dimensional.
    """
    return np.atleast_2d(x)


def append_missing_features(features, n_features):
    """Appends missing features to a list of features."""
    all_features = range(n_features)
    missing_features = [f for f in all_features if f not in features]

    if len(missing_features):
        features.append(missing_features)

    return features


def get_values(x):
    """
    Returns ``x.values`` for a pandas DataFrame, or ``x`` itself if it's already
    a numpy array.
    """
    if isinstance(x, pd.DataFrame):
        return x.values

    return x
