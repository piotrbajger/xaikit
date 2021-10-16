from xaikit.adapters.adapter_interface import ModelAdapterInterface


class SklearnModelAdapter(ModelAdapterInterface):
    """Adapter for scikit-learn models."""

    def __init__(self, sklearn_model):
        """
        :param sklearn_model: Scikit-learn estimator.
        """
        self.model = sklearn_model

        self._set_predictor()

    def predict(self, x):
        return self._predictor(x)

    def _set_predictor(self):
        """Sets the predictor to either predict or predict_proba."""
        if hasattr(self.model, "predict_proba"):
            self._predictor = self.model.predict_proba
        else:
            self._predictor = self.model.predict
