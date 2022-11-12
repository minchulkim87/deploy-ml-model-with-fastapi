"""
Script to perform unittests
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score

from train_model import (
    load_data,
    process_data,
    load_model_artifacts,
    CAT_FEATURES
)


def test_data() -> None:
    """
    Tests whether the raw data contains the expected columns.
    """
    data = load_data()

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


def test_processed_data() -> None:
    """
    Tests whether the pre-processed data has expected shapes.
    """
    
    data = load_data()
    _, encoder, lb = load_model_artifacts()
    train, test = train_test_split(data, test_size=0.2)
    
    X_train, y_train, _, _ = process_data(
        train, label="salary", train=False,
        categorical_features=CAT_FEATURES, encoder=encoder, lb=lb
    )
    
    X_test, y_test, _, _ = process_data(
        test, label="salary", train=False,
        categorical_features=CAT_FEATURES, encoder=encoder, lb=lb
    )
    
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_model() -> None:
    """
    Tests whether the model f1 score on test data is acceptable.
    """
    
    data = load_data()
    _, test = train_test_split(data, test_size=0.2)
    
    model, encoder, lb = load_model_artifacts()
    
    X_test, y_test, _, _ = process_data(
        test, label="salary", train=False,
        categorical_features=CAT_FEATURES, encoder=encoder, lb=lb
    )
    
    y_pred = model.predict(X_test)
    assert fbeta_score(y_pred, y_test, beta=1) > 0.65
