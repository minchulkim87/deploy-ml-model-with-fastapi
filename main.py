"""This is the code for the API
"""

import numpy as np
import pandas as pd

from train_model import (
    load_model_artifacts,
    process_data,
    CAT_FEATURES
)

from fastapi import FastAPI
import uvicorn

from pydantic import BaseModel
from typing import Literal


app = FastAPI()


class ModelInput(BaseModel):
    age: int
    workclass: Literal['Federal-gov',
                       'State-gov',
                       'Local-gov',
                       'Private',
                       'Self-emp-inc',
                       'Self-emp-not-inc',
                       'Without-pay']
    fnlgt: int
    education: Literal[
        'Doctorate', 'Masters', 'Bachelors', 'Some-college', 'HS-grad',
        'Prof-school', 'Assoc-acdm', 'Assoc-voc',
        '12th', '11th', '10th', '9th', '7th-8th', '5th-6th', '1st-4th', 'Preschool'
    ]
    education_num: int
    marital_status: Literal[
        "Never-married",
        "Married-civ-spouse",
        "Married-spouse-absent",
        "Married-AF-spouse",
        "Divorced",
        "Separated",
        "Widowed"
    ]
    occupation: Literal["Tech-support",
                        "Craft-repair",
                        "Other-service",
                        "Sales",
                        "Exec-managerial",
                        "Prof-specialty",
                        "Handlers-cleaners",
                        "Machine-op-inspct",
                        "Adm-clerical",
                        "Farming-fishing",
                        "Transport-moving",
                        "Priv-house-serv",
                        "Protective-serv",
                        "Armed-Forces"]
    relationship: Literal[
        "Wife", "Husband", "Other-relative", "Unmarried", "Not-in-family", "Own-child"
    ]
    race: Literal[
        "Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "White", "Other"
    ]
    sex: Literal["Female", "Male"]
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: Literal[
        'United-States', 'Cuba', 'Jamaica', 'India', 'Mexico',
        'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran',
        'Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand',
        'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
        'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
        'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru',
        'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
        'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
        'Holand-Netherlands'
    ]

    class Config:
        schema_extra = {
            "age": 23,
            "workclass": "Private",
            "fnlgt": 183175,
            "education": "Some-college",
            "education_num": 10,
            "marital_status": "Divorced",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "Black",
            "sex": "Female",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 20,
            "native_country": "India"
        }


# Load artifacts
model, encoder, lb = load_model_artifacts()


# Root path
@app.get("/")
async def root():
    return {
        "message": "This app predicts whether salary is greater than $50,000 / year."
    }

# Prediction path
@app.post("/predict-income")
async def predict(input: ModelInput):

    input_data = np.array([[
        input.age,
        input.workclass,
        input.fnlgt,
        input.education,
        input.education_num,
        input.marital_status,
        input.occupation,
        input.relationship,
        input.race,
        input.sex,
        input.capital_gain,
        input.capital_loss,
        input.hours_per_week,
        input.native_country
    ]])

    original_cols = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours-per-week",
        "native-country"
    ]

    input_df = pd.DataFrame(data=input_data, columns=original_cols)

    X, _, _, _ = process_data(
        input_df, label=None, train=False, categorical_features=CAT_FEATURES, encoder=encoder, lb=lb)
    y = model.predict(X)
    pred = lb.inverse_transform(y)[0]

    return {"Income prediction": pred}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
