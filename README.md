# Deploy ML Model with FastAPI

This project is a part of the Udacity Machine Learning DevOps Engineer Nanodegree.

The key objectives of this project is to apply best practices when it comes to deploying machine learning models.

1. Quality control using `pytest`.
2. CI/CD using `GitHub`.
3. Model API deployment using `FastAPI`.
4. Documentation in the style of Model Cards.

## Requirements

Use Python 3.8+.

To install required packages, use the shell command:

```sh
pip install -r requirements.txt
```

## Quality control using `pytest`

I used `pytest` to perform unittests. Run the tests by using the shell command:

```sh
pytest
```

The above two are automatically run on "push" to GitHub.

## Deploying the API locally using `FastAPI`

Spint up a local API using the shell command:

```
python -m main
```