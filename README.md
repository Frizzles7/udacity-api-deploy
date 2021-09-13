# Overview
In this project, I use census data to build a model. I use DVC to track the data and artifacts created. I use an S3 bucket as my DVC remote. I setup Github Actions to pull artifacts using DVC, install dependencies, run flake8, and run pytest.

Next I create an API.

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


# to be done
# API Creation
*  Create a RESTful API using FastAPI this must implement:
    * GET on the root giving a welcome message.
    * POST that does model inference.
    * Type hinting must be used.
    * Use a Pydantic model to ingest the body from POST. This model should contain an example.
   	 * Hint: the data has names with hyphens and Python does not allow those as variable names. Do not modify the column names in the csv and instead use the functionality of FastAPI/Pydantic/etc to deal with this.
* Write 3 unit tests to test the API (one for the GET and two for POST, one that tests each prediction).

# API Deployment
* Create a free Heroku account (for the next steps you can either use the web GUI or download the Heroku CLI).
* Create a new app and have it deployed from your GitHub repository.
    * Enable automatic deployments that only deploy if your continuous integration passes.
    * Hint: think about how paths will differ in your local environment vs. on Heroku.
    * Hint: development in Python is fast! But how fast you can iterate slows down if you rely on your CI/CD to fail before fixing an issue. I like to run flake8 locally before I commit changes.
* Write a script that uses the requests module to do one POST on your live API.
