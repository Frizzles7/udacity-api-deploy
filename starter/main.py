from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from starter.starter.ml.data import process_data
from starter.starter.ml.model import inference

# from dvc_on_heroku_instructions.md
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()

class Features(BaseModel):
    age: int = 41
    workclass: str = 'Private'
    fnlgt: int = 284582
    education: str = 'Masters'
    education_num: int = 14
    marital_status: str = 'Married-civ-spouse'
    occupation: str = 'Exec-managerial'
    relationship: str = 'Wife'
    race: str = 'White'
    sex: str = 'Female'
    capital_gain: int = 0
    capital_loss: int = 0
    hours_per_week: int = 50
    native_country: str = 'United-States'
     
class Prediction(BaseModel):
    prediction: str

@app.get("/")
async def welcome():
    return {"Greeting": "Welcome to the API!"}

@app.post("/inference", response_model=Prediction, status_code=200)
def run_prediction(data: Features):
    # ingest Features into dataframe for processing
    df = pd.DataFrame([{"age": data.age,
                        "workclass": data.workclass,
                        "fnlgt": data.fnlgt,
                        "education": data.education,
                        "education-num": data.education_num,
                        "marital-status": data.marital_status,
                        "occupation": data.occupation,
                        "relationship": data.relationship,
                        "race": data.race,
                        "sex": data.sex,
                        "capital-gain": data.capital_gain,
                        "capital-loss": data.capital_loss,
                        "hours-per-week": data.hours_per_week,
                        "native-country": data.native_country}])
    
    # load model artifacts needed for processing and inference
    model = joblib.load("starter/model/model.pkl")
    encoder = joblib.load("starter/model/encoder.pkl")
    lb = joblib.load("starter/model/lb.pkl")
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    
    # process the data to get it into the correct format for inference
    X, _, _, _ = process_data(df, 
                              categorical_features=cat_features, 
                              training=False, 
                              encoder=encoder, 
                              lb=lb)

    # generate predictions
    preds = inference(model, X)
    
    # convert to human readable output
    if preds == 0:
        prediction_text = "Salary <= 50K"
    else:
        prediction_text = "Salary > 50K"
        
    # create response to return
    prediction_response = {"prediction": prediction_text}
    
    return prediction_response
    
