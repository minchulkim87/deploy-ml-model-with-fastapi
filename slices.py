"""_summary_
"""

import logging

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

from train_model import (
    load_data,
    process_data,
    load_model_artifacts,
    compute_model_metrics,
    CAT_FEATURES
)

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")


def slice_performance(data: pd.DataFrame,
                      feature: str,
                      category: str,
                      model: AdaBoostClassifier,
                      encoder: OneHotEncoder,
                      lb: LabelBinarizer) -> float:
    temp_df = data[data[feature] == category]

    X_test, y_test, _, _ = process_data(
        temp_df, label="salary", train=False,
        categorical_features=CAT_FEATURES, encoder=encoder, lb=lb)

    y_pred = model.predict(X_test)

    return compute_model_metrics(y_test, y_pred)


def test_slice_performance():
    """ Check performance on categorical features """

    data = load_data()
    _, test = train_test_split(data, test_size=0.20)
    model, encoder, lb = load_model_artifacts()
    
    slice_metrics = []
    for feature in CAT_FEATURES:
        for cat in test[feature].unique():
            score = slice_performance(test, feature, cat, model, encoder, lb)
            slice_metrics.append(f"{feature} {cat}: f1 score: {score: .3f}")

    with open('model/slice_output.txt', 'w') as file:
        for row in slice_metrics:
            file.write(row + '\n')

    logging.info("Performance metrics for slices saved to slice_output.txt")


if __name__ == '__main__':
    test_slice_performance()
