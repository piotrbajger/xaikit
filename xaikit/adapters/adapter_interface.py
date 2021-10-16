import abc


class ModelAdapterInterface(metaclass=abc.ABCMeta):
    """
    Interface for Model adapters.

    The goal of this interface is to create a shared interface for models
    from different libraries (scikit-learn, keras, etc.). Models wrapped
    in this interface can then be used with the explaining methods.

    Should not be created directly, use :py:func:`~adapters.create_model_adapter`
    factory instead.
    """

    @abc.abstractmethod
    def predict(self, x):
        pass
