def get_predictor(estimator):
    """
    Returns a ``predict`` or ``predict_proba`` function of an estimator.

    :param estimator: An instance of scikit-learn Estimator.
    """
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba
    else:
        return estimator.predict