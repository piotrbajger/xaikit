import numpy as np

from sklearn.utils import Bunch


def breakdown(estimator, x, x0):
    """
    Creates data for a breakdown plot.

    :param estimator: An estimator compatible with the scikit-learn
        ``Estimator`` interface.
    :param x: An ``n x m`` feature matrix of ``n`` observations of ``m`` variables.
    :param x0: A single observation for which the breakdown plot is to be constructed.
    """
    if hasattr(estimator, "predict_proba"):
        predict = estimator.predict_proba
    else:
        predict = estimator.predict

    x0 = x0.reshape(1, -1)

    base_y = np.mean(predict(x))

    new_x = x.copy()

    n_features = x0.shape[-1]
    result = np.zeros(n_features + 1)

    result[0] = base_y

    for i in range(n_features):
        new_x[:, i] = x0[:, i]
        next_y = np.mean(predict(new_x))
        result[i + 1] = next_y

    return Bunch(breakdown=result)
