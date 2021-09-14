import json
import requests

# sample data for testing
data_zero = {
        "age": 20,
        "workclass": "Private",
        "fnlgt": 215646,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Never-married",
        "occupation": "Handlers-cleaners",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 20,
        "native_country": "United-States"
    }
    
data_one = {
        "age": 41,
        "workclass": "Private",
        "fnlgt": 284582,
        "education": "Masters",
        "education_num": 14,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States"
    }

# submit the post requests to the api
r = requests.post("https://megan-udacity-app.herokuapp.com/inference", data=json.dumps(data_zero))
print(r.status_code)
print(r.text)

r = requests.post("https://megan-udacity-app.herokuapp.com/inference", data=json.dumps(data_one))
print(r.status_code)
print(r.text)

