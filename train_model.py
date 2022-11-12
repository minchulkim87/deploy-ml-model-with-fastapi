"""
# Script to train machine learning model.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import fbeta_score
import joblib

from typing import List, Tuple, Optional


CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]


def load_data() -> pd.DataFrame:
    """Loads raw data

    Returns:
        pd.DataFrame: raw data
    """
    return pd.read_csv(
        "data/census.csv",
        skipinitialspace=True,
        low_memory=False)


def process_data(data: pd.DataFrame,
                 label: Optional[str] = None,
                 train: bool = True,
                 categorical_features: Optional[List[str]] = None,
                 encoder: Optional[OneHotEncoder] = None,
                 lb: Optional[LabelBinarizer] = None) -> Tuple[np.ndarray,
                                                               np.ndarray,
                                                               OneHotEncoder,
                                                               LabelBinarizer]:
    """processes raw data and converts categorical columns and target labels into sklearn-ready types.

    Args:
        data (pd.DataFrame): raw data
        label (str): column name of the target variable
        train (bool, optional): whether to create encoder and binarizer or not. Defaults to True.
        categorical_features (Optional[List[str]], optional): list of columns to treat as categorical. Defaults to None.
        encoder (Optional[OneHotEncoder], optional): if train is False, and there are categorical_features, provide a OneHotEncoder. Defaults to None.
        lb (Optional[LabelBinarizer], optional): if train is False, provide a LabelBinarizer. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, OneHotEncoder, LabelBinarizer]: X, y, onehotencoder, label_binarizer
    """

    if train:
        label_binarizer = LabelBinarizer()
        onehotencoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    else:
        label_binarizer = lb
        onehotencoder = encoder
    
    if isinstance(label, str):
        y = data.pop(label)

        if train:
            y = label_binarizer.fit_transform(y.values).ravel()
        else:
            y = label_binarizer.transform(y.values).ravel()
    else:
        y = None
    
    if categorical_features:
        if train:
            X = np.concatenate(
                [
                    data.drop(columns=categorical_features),
                    onehotencoder.fit_transform(data[categorical_features].values)
                ],
                axis=1
            )
        else:
            X = np.concatenate(
                [
                    data.drop(columns=categorical_features),
                    onehotencoder.transform(data[categorical_features].values)
                ],
                axis=1
            )

    else:
        X = data

    return X, y, onehotencoder, label_binarizer


def train_model(X: np.ndarray, y: np.ndarray) -> AdaBoostClassifier:
    """Trains an adaBoost model using the given data and returns the classifier.

    Args:
        X (np.ndarray): X train
        y (np.ndarray): y train

    Returns:
        AdaBoostClassifier: classifier
    """
    model = AdaBoostClassifier()
    model.fit(X, y)
    return model


def save_model_artifacts(
        model: AdaBoostClassifier,
        encoder: OneHotEncoder,
        lb: LabelBinarizer) -> None:
    """Saves model artifacts as pickles

    Args:
        model (AdaBoostClassifier): trained model
        encoder (OneHotEncoder): trained one hot encoder for categorical variables
        lb (LabelBinarizer): trained label binarizer for target variable
    """
    joblib.dump(model, 'model/model.pkl')
    joblib.dump(encoder, 'model/encoder.pkl')
    joblib.dump(lb, 'model/lb.pkl')


def load_model_artifacts() -> Tuple[AdaBoostClassifier,
                                    OneHotEncoder,
                                    LabelBinarizer]:
    """Loads the saved artifacts

    Returns:
        Tuple[AdaBoostClassifier, OneHotEncoder, LabelBinarizer]: model, encoder, lb
    """
    model = joblib.load('model/model.pkl')
    encoder = joblib.load('model/encoder.pkl')
    lb = joblib.load('model/lb.pkl')
    return model, encoder, lb


def compute_model_metrics(y: np.ndarray, pred: np.ndarray) -> float:
    """Computes and returns f1-score

    Args:
        y (np.ndarray): true labels
        pred (np.ndarray): model predicted labels

    Returns:
        float: f1-score
    """
    return fbeta_score(y, pred, beta=1, zero_division=1)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)-15s %(message)s")

    logging.info("Loading data")
    data = load_data()
    train, test = train_test_split(data, test_size=0.2)

    logging.info("Processing data")
    X_train, y_train, encoder, lb = process_data(
        train, label="salary", train=True,
        categorical_features=CAT_FEATURES
    )
    X_test, y_test, _, _ = process_data(
        test, label="salary", train=False,
        categorical_features=CAT_FEATURES, encoder=encoder, lb=lb
    )

    logging.info("Training model")
    model = train_model(X_train, y_train)

    logging.info("Saving model artifacts")
    save_model_artifacts(model, encoder, lb)

    logging.info("Scoring model on test set")
    y_pred = model.predict(X_test)
    score = compute_model_metrics(y_test, y_pred)
    logging.info(f"F1 score = {score: .3f}")
