import pandas as pd
from sklearn.metrics import fbeta_score

import pytest
from train_model import (
    load_processed_data,
    load_model_artifacts
)


def test_data():
    data = pd.read_csv("data/census.csv", skipinitialspace=True, low_memory=False)
    
    assert set(data.columns) == {
        "age",
        "workclass",
        "fnlgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "salary"
    }


def test_load_data():
    X_train, X_test, y_train, y_test = load_processed_data()
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert 0.1 * len(X_train) < len(X_test) < len(X_train)


def test_model():
    _, X_test, _, y_test = load_processed_data()
    model, _, _ = load_model_artifacts()
    y_pred = model.predict(X_test)
    assert fbeta_score(y_pred, y_test, beta=1) > 0.65
