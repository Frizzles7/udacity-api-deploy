import pandas as pd
import joblib
import json
from ml.model import compute_model_metrics, inference
from ml.data import process_data


def performance_by_slice(data, model, encoder, lb, cat_features):
    """
    Calculate the performance of the model on slices of the categorical features
    and save results to `slice_output.txt`
    
    Inputs:
        data: dataframe with the data to slice
        model: trained model to use to determine performance
        encoder: encoder used to process data
        lb: label binarizer used to process data
        cat_features: list of the column names of categorical features
    """
    
    results = {}
    for col in cat_features:
        col_results = {}
        for category in data[col].unique():
            data_temp = data[data[col] == category]
            X, y, _, _ = process_data(
                data_temp, 
                categorical_features=cat_features, 
                label="salary", 
                training=False, 
                encoder=encoder, 
                lb=lb)
            preds = inference(model, X)
            precision, recall, fbeta = compute_model_metrics(y, preds)
            col_results[category] = [precision, recall, fbeta]
        results[col] = col_results
    
    # write output in results dictionary to file
    with open('starter/starter/slice_output.txt', 'w') as f:
        json.dump(results, f, indent=2)

    
if __name__ == "__main__":
    data = pd.read_csv("starter/data/census_clean.csv")
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
    performance_by_slice(data, model, encoder, lb, cat_features)
    
