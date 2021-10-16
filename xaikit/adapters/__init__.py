import sklearn

from xaikit.adapters.sklearn_adapter import SklearnModelAdapter


def create_model_adapter(external_model):
    """
    Brings an external model under the xAIkit interface.
    """
    if isinstance(external_model, sklearn.base.BaseEstimator):
        return SklearnModelAdapter(external_model)
