# Script to test model training

import pytest
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from starter.starter.ml.data import process_data
from starter.starter.ml.model import train_model, inference


@pytest.fixture
def data():
    """ Load in the cleaned data for testing """
    data = pd.read_csv("./starter/data/census_clean.csv")
    train, test = train_test_split(data, test_size=0.20, random_state=1)
    return train


@pytest.fixture
def cat_features():
    """ Define the categorical features """
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


@pytest.fixture
def processed(data, cat_features):
    X_train, y_train, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    return X_train, y_train, encoder, lb


@pytest.fixture
def model(processed):
    X_train, y_train, encoder, lb = processed
    model = train_model(X_train, y_train)
    return model


def test_process_data(processed):
    """
    Test the outputs of the process_data function:
     - X_train has the correct number of columns
     - y_train contains only zeros and ones
    """
    X_train, y_train, encoder, lb = processed
    assert X_train.shape[1] == 107
    assert set(y_train) == set([0, 1])


def test_model_build(processed):
    """
    Test the model is of the correct type
     - Gradient boosting classifier
    """
    X_train, y_train, encoder, lb = processed
    model = train_model(X_train, y_train)
    assert type(model).__name__ == 'GradientBoostingClassifier'


def test_model_output(processed, model):
    """
    Test inference on the training data
     - confirm sum of predictions on X_train is 4997
       (4997 predicted ones, rest predicted zeros)
    """
    X_train, y_train, encoder, lb = processed
    preds = inference(model, X_train)
    assert preds.sum() == 4997

