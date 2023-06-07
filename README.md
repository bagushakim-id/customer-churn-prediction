# Customer Churn Prediction

I worked on a project using deep learning models, specifically the Sequential API and Functional API, with the goal of predicting whether a customer will churn or not. The project involved evaluating model performance by testing it on previously unseen data.

## Data Source
You can access the data on `dataset` folder.

## Folder Description
- `dataset` : directory used to store data
- `deployment`: the deployment folder contains several files and scripts necessary for deploying a machine learning model and its associated web application
- `model` : contains a selection of files important to the project's model development. The brief descriptions of each of these files are as follows
  - `churn_model.h5`: contains deep learning models that have been trained and saved in pickle format. This model will be used in the prediction stage to predict customer churn based on the input data.
  - `final_pipeline.pkl`: represents the final version of a trained model and associated data preprocessing steps that have been combined into a cohesive pipeline for ease of use and deployment. The pipeline include ColumnTransformer. By saving the pipeline as a pickle file, it can be easily loaded and used for making predictions on new data without the need to retrain or reconfigure the individual components.
- `script`: contains the Python notebook files used to run the code in this project. There are two files contained in this folder:
  - `churn_inference.ipynb`: notebook file used to make inferences on the model that has been trained.
  - `main-notebook.ipynb`: notebook file that contains complete code for processing and analyzing data, training models, and evaluating model performance.

## Tools and Libraries Used
- numpy
- pandas
- seaborn
- matplotlib
- feature engine
- scikit-learn
- tensorflow
- keras
- scipy
