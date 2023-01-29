# Chronic Kidney Disease Analysis

This project is simple modularised app around CKD dataset. 
The dataset is available on [UCI ML repository](https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease). 
The dataset is a collection of 400 patients with chronic kidney disease and 100 healthy patients. 
The dataset contains 25 features and 1 target variable. 
The target variable is a binary variable with 1 representing the presence of chronic kidney disease and 0 representing the absence of chronic kidney disease.

# Preconditions

- Python 3.8
- Conda 22.9.0

# Setup

- Create a conda environment with the following command: `conda create -n <environment-name> --file requirements.txt`

# Repository structure

- `dataset` - contains the raw dataset, the dataset description and the preprocessed dataset
- `scripts` - contains the scripts used for preprocessing the dataset, and some exploratory notebooks
- `app` - contains the moduls for preprocessing and training the model
- `tests` - contains some simple unit tests for the modules in the `app` folder
- `model` - contains the trained models

# Usage

This app is a simple CLI app, which can be used to train and test the model.
To run the app use the following command: `python cmd.py <arguments>`.
The arguments are the following:
- `--path` - path to the dataset
- `--mode` - mode of the app, can be `train` or `test`
- `--tuning` - if the model should be tuned, can be `True` or `False`
- `--save_model` - if the model should be saved, can be `True` or `False`
- `--model_path` - path to save the model, or path to load the model from when mode is `test`