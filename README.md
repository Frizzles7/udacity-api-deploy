# Overview
In this project, I use census data to build a model. I use DVC to track the data and artifacts created. I use an S3 bucket as my DVC remote. I setup Github Actions to pull artifacts using DVC, install dependencies, run flake8, and run pytest.

Next I create an API to serve the predictions using FastAPI and deploy it using Heroku.

# Environment Set up
* Download and install Miniconda if you do not have conda already.
    * conda create -n [envname] "python=3.8" scikit-learn dvc dvc-s3 pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge

# Repositories
* Clone this repository and into your project directory, and then initialize DVC.
* For the DVC remote, this project uses S3. Setup your S3 bucket, install the AWS CLI tool, create an IAM user, obtain your Access Key Id and Secret Access Key.
* Configure the AWS CLI to use your Access Key Id and Secret Access Key.
* Store your Access Key Id and Secret Access Key as secrets in Github for Github Actions. In the repository settings, this can be done under Secrets. Create new repository secrets, one for Access Key Id and one for Secret Access Key.

# Running the Model
* The raw data was cleaned by removing all spaces in a text editor to create the cleaned version of the data.
* Model training can be run from the top level of the project using the DVC pipeline as follows:
```bash
dvc run -n model_training -d starter/starter/train_model.py -d starter/data/census_clean.csv -o starter/model/model.pkl -o starter/model/encoder.pkl -o starter/model/lb.pkl python starter/starter/train_model.py
```
* Model performance on slices of the data can be run from the top level of the project using the DVC pipeline as follows:
```bash
dvc run -n performance_slices -d starter/starter/performance_slices.py -d starter/data/census_clean.csv -d starter/model/model.pkl -d starter/model/encoder.pkl -d starter/model/lb.pkl -o starter/starter/slice_output.txt python starter/starter/performance_slices.py
```

# API
* The API is available at https://megan-udacity-app.herokuapp.com/ and can be used in the following ways:
    * Documentation is available at https://megan-udacity-app.herokuapp.com/docs - see this documentation for an example
    * The root domain displays a welcome message at https://megan-udacity-app.herokuapp.com/
    * Posts to the API can be completed at https://megan-udacity-app.herokuapp.com/inference

Note that I setup the Procfile for Heroku per https://stackoverflow.com/questions/59391560/how-to-run-uvicorn-in-heroku.
