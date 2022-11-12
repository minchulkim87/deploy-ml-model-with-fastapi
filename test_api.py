""" This module tests the root and the prediction end points """

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_get_root():
    """ Test the root page get a succesful response"""
    r = client.get("/")
    assert r.status_code == 200


def test_post_predict_up():
    """Test an example when annual salary is quite likely greater than $50,000"""

    r = client.post("/predict-income", json={
        "age": 42,
        "workclass": "Private",
        "fnlgt": 280464,
        "education": "Masters",
        "education_num": 10,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 80,
        "native_country": "United-States"
    })

    assert r.status_code == 200
    assert r.json() == {"Income prediction": ">50K"}


def test_post_predict_down():
    """Test an example when annual salary is quite likely less than $50,000"""
    r = client.post("/predict-income", json={
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
    })

    assert r.status_code == 200
    assert r.json() == {"Income prediction": "<=50K"}