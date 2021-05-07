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

    x0 = x0.reshape(1, -1)
    n_vars = x0.shape[-1]

    # If no features provided, default to the full list
    if features is None:
        features = list(range(n_vars))

    # Check if any features are to be bundled together as the "remaining features"
    all_features = list(range(n_vars))
    remaining_features = [f for f in all_features if f not in features]

    n_features = len(features) + int(len(remaining_features) > 0)
    result = np.zeros(n_features + 1)

    base_y = np.mean(predict(x), axis=0)
    new_x = x.copy()

    result[0] = base_y

    for i in features:
        new_x[:, i] = x0[:, i]
        next_y = np.mean(predict(new_x), axis=0)
        result[i + 1] = next_y

    if len(remaining_features) > 0:
        new_x[:, remaining_features] = x0[:, remaining_features]
        next_y = np.mean(predict(new_x))
        result[-1] = next_y

    return Bunch(
        breakdown=result,
        features=features,
        base_prediction=result[0],
        target_prediction=result[-1],
        remaining_features=remaining_features,
    )
