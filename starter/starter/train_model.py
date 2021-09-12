# Script to train machine learning model


from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code
from ml.data import process_data
from ml.model import train_model
import pandas as pd
import joblib

# Add code to load in the data
data = pd.read_csv("../data/census_clean.csv")

# Perform train-test split
train, test = train_test_split(data, test_size=0.20, random_state=1)

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

# Process the training data
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data
X_text, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, 
    encoder=encoder, lb=lb
)

# Train and save the model, encoder, and lb
model = train_model(X_train, y_train)
joblib.dump(model, "model.pkl")
joblib.dump(encoder, "encoder.pkl")
joblib.dump(lb, "lb.pkl")

