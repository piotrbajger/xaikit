.. _getting_started:

Getting started
===============

Motivation
----------

xaikit is meant to be your eXplainable AI toolkit which provides out-of-the-box
model-agnostic lightweight tools to inspect your machine learning models.

The majority of the algorithms and ideas are based on the fantastic book
`Explanotary Model Analysis <https://github.com/pbiecek/ema>`_ by P. Biecek and T. Burzykowski.

Getting started
---------------

We will use the scikit-learn library and the Breast Cancer Wisconsin Dataset as our example.
The problem is to classify breast masses as either malignant or benign based on a number
of features extracted from medical images.

First we load the dataset:

.. code-block:: python

  from sklearn.datasets import load_breast_cancer


  breast_cancer_dataset = load_breast_cancer(as_frame=True)

  x = breast_cancer_dataset["data"]
  y = breast_cancer_dataset["target"]

  # Use a subset of features for simplicity
  features = [c for c in x.columns if c.startswith("mean")]
  x = x[features]

We then create a simple classifier. For the purpose of showcasting the xaikit module,
we will use entire dataset (instead of splitting it into training and testing subsets).

A simple logistic regression model easily achieves a 96% F1-score on this dataset.

.. code-block:: python

  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import f1_score


  model = LogisticRegression(max_iter=10_000)
  model.fit(x, y)

  y_pred = model.predict(x)

  f1 = f1_score(y, y_pred)
  print(f"F1-score: {f1*100:.2f}%")

Observation-level explainer
---------------------------

Let us now consider a prediction for a single patient:

.. code-block:: python

  patient_id = 22
  proba_pred = model.predict_proba(x)

  patient_x = x.iloc[patient_id]
  patient_y = proba_pred[patient_id, 1]

  print(f"Prediction for {patient_id}: {patient_y:.2f}.")

Using the :py:func:`~xaikit.explainers.shap.shap()`
explainer from xaikit we can examine how each feature contributes
to this prediction:

.. code-block:: python

  from xaikit.explainers.shap import shap


  result = shap(model, x, x0=patient_x)
  shap_means = result["shap_values_mean"]

  # Pretty print the results
  data_to_print = zip(x.columns, patient_x, shap_means)
  print(f"{'base':<30} {result['base_prediction']:.2f}")
  print("-" * 36)
  for feature_name, feature_val, shap_mean in data_to_print:
    print(f"{feature_name}={feature_val:.2f} {shap_mean:+.2f}")
  print("-" * 36)
  print(f"{'target':<30} {result['target_prediction']:.2f}")

Which should result in the following output:

.. code-block::

  base                                 0.63
  -----------------------------------------
  mean radius = 15.34                 -0.00
  mean texture = 14.26                +0.06
  mean perimeter = 102.50             -0.47
  mean area = 704.40                  -0.00
  mean smoothness = 0.11              -0.00
  mean compactness = 0.21             -0.01
  mean concavity = 0.21               -0.01
  mean concave points = 0.10          -0.00
  mean symmetry = 0.25                -0.00
  mean fractal dimension = 0.07       -0.00
  -----------------------------------------
  target                               0.19

From here we see that, for example, the patient's tumour having
a mean perimeter of 102.50mm results in a decrease of the
probability of the tumour being malignant by 0.47. Similarly,
having the mean texture of 14.26 increases the probability
by 0.06.

.. seealso::
    Other observation-level explainers: :py:func:`~xaikit.explainers.breakdown.breakdown()`,
    :py:func:`~xaikit.explainers.ceteris_paribus.ceteris_paribus()`.