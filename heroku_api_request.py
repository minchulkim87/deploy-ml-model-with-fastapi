"""
Test the live API on Heroku
"""

import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")


test_data = {
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


app_url = "https://udacitymldevops.herokuapp.com/predict-income"

r = requests.post(app_url, json=test_data)
assert r.status_code == 200

logging.info("Testing Heroku app")
logging.info(f"Status code: {r.status_code}")
logging.info(f"Response body: {r.json()}")