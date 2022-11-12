# Script to train machine learning model.
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import fbeta_score
import joblib

from typing import List, Optional


CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def load_data():
    return pd.read_csv("data/census.csv", skipinitialspace=True, low_memory=False)


def process_data(data: pd.DataFrame,
                 label: str,
                 categorical_features: Optional[List[str]]=None):
    
    binarizer = LabelBinarizer()
    onehotencoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    
    y = data.pop(label)
    y = binarizer.fit_transform(y.values).ravel()
    
    if categorical_features:
        X = np.concatenate(
            [
                data.drop(columns=categorical_features),
                onehotencoder.fit_transform(data[categorical_features].values)
            ],
            axis=1
        )
    
    else:
        X= data
    
    return X, y, onehotencoder, binarizer


def train_model(X, y):
    model = AdaBoostClassifier()
    model.fit(X, y)
    return model


def save_model_artifacts(model, encoder, lb):
    joblib.dump(model, 'model/model.pkl')
    joblib.dump(encoder, 'model/encoder.pkl')
    joblib.dump(lb, 'model/lb.pkl')


def load_model_artifacts():
    model = joblib.load('model/model.pkl')
    encoder = joblib.load('model/encoder.pkl')
    lb = joblib.load('model/lb.pkl')
    return model, encoder, lb


def compute_model_metrics(y, pred):
    return fbeta_score(y, pred, beta=1, zero_division=1)


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
    
    logging.info("Loading data")
    data = load_data()

    logging.info("Processing data")
    X, y, encoder, lb = process_data(
        data, label="salary", categorical_features=CAT_FEATURES
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    logging.info("Training model")
    model = train_model(X_train, y_train)

    logging.info("Saving model artifacts")
    save_model_artifacts(model, encoder, lb)

    logging.info("Scoring model on test set")
    y_pred = model.predict(X_test)
    score = compute_model_metrics(y_test, y_pred)
    logging.info(f"F1 score = {score: .3f}")
