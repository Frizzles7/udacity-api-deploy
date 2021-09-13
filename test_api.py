from fastapi.testclient import TestClient
from starter.main import app
import json

client = TestClient(app)

def test_greeting():
    """ Test the greeting at the root """
    r = client.get("/")
    assert r.status_code == 200
    assert json.loads(r.text) == {"Greeting": "Welcome to the API!"}

def test_inference_zero():
    """ Test the inference post function for a prediction of zero """
    data_zero = {
        "age": 20,
        "workclass": "Private",
        "fnlgt": 215646,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Handlers-cleaners",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 20,
        "native-country": "United-States"
    }
    r = client.post("/inference", json=data_zero)
    assert r.status_code == 200
    assert json.loads(r.text) == {"prediction": "Salary <= 50K"}

def test_inference_one():
    """ Test the inference post function for a prediction of one """
    data_one = {
        "age": 41,
        "workclass": "Private",
        "fnlgt": 284582,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States"
    }
    r = client.post("/inference", json=data_one)
    assert r.status_code == 200
    assert json.loads(r.text) == {"prediction": "Salary > 50K"}

