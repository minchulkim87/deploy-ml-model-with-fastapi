# Deploy ML Model with FastAPI

This project is a part of the Udacity Machine Learning DevOps Engineer Nanodegree.

The key objectives of this project is to apply best practices when it comes to deploying machine learning models.

1. Dependency management using `conda`.
2. Code version control using `git`.
3. Quality control using `pytest`.
4. CI/CD using `GitHub`.
5. Model API deployment using `FastAPI`.

## Dependency management using `conda`

I used `conda` to manage package dependencies. Miniconda, a minimal installation of `conda`, can be downloaded and installed from [here](docs.conda.io/en/latest/miniconda.html) - I used Python version 3.8.

After installing, use a terminal with the conda command available (on Windows this would be the Anaconda Powershell) and navigate to this folder.

Then, to install all dependencies, use the shell command:

```sh
conda env create -f environment.yml
```

To create the conda environment. To use the environment, use the shell command:

```sh
conda activate mldeployudacity
```

## Code version control using `git`.

I used `git` manage code version control. `git` can be downloaded and installed from [here](git-scm.com).
