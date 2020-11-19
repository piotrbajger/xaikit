#!/usr/bin/env python

from distutils.core import setup

setup(
    name="xaikit",
    version="0.1",
    description="eXplainable AI tools for scikit-learn.",
    author="Piotr Bajger",
    url="",
    install_requires=[
        "scikit-learn",
        "numpy",
        "pandas",
        "joblib",
    ],
    packages=["xaikit"],
)
