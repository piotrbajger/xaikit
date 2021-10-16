from unittest import TestCase

from sklearn.linear_model import LinearRegression, LogisticRegression

from xaikit.adapters.sklearn_adapter import SklearnModelAdapter


class SklearnModelAdapterTest(TestCase):
    def test_estimator_with_predict(self):
        external_model = LinearRegression()
        adapter = SklearnModelAdapter(external_model)

        self.assertEqual(adapter._predictor.__name__, "predict")

    def test_estimator_with_predict_proba(self):
        external_model = LogisticRegression()
        adapter = SklearnModelAdapter(external_model)

        self.assertEqual(adapter._predictor.__name__, "predict_proba")
