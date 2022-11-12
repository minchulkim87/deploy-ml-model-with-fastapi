# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model is an AdaBoost classifier with default parameters using the scikit-learn package.

## Intended Use

The model predicts whether the salary is above or below $50,000 / year.
The model has been developed as a part of a project for the Udacity Machine Learning DevOps Engineer Nanodegree.

## Training Data

The dataset is from the 1994 Census provided by the UCI Machine Learning Repository.
More information can be found [here](https://archive.ics.uci.edu/ml/datasets/census+income)

## Evaluation Data

The dataset was split into 80:20 train:test sets.

## Metrics

The evaluated f1-score is 0.668.

## Ethical Considerations

The data contains personal attributes such as sex and race.
We should exercise caution to not introduce bias into model performance for the different groups.

## Caveats and Recommendations

As mentioned in the Ethical Considerations around the issues that could arise from model performance for the different groups, the performance on many slices of the data are quite poor with f1-scores well below 0.5. See `model/slice_output.txt` for details. One place to start is to gather more data and use different hyperparameters to improve model performance across all slices.