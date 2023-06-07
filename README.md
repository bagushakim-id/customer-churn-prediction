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

## Overall Conclusion
Based on the EDA (Exploratory Data Analysis) and model evaluation, the following conclusions can be drawn:

1. **Data Balance:** The data is already balanced, meaning there is no need for additional data balancing techniques to address class imbalance issues.
2. **Age and Churn:** The highest churn rate occurs among customers who are 38 years old, indicating that this age group may have a higher likelihood of churning.
3. **Complaint Status:** The "Not Applicable" complaint status dominates over other complaint statuses and has a higher potential for churn. This suggests that customers with no complaints may be more likely to churn.
4. **Reasons for Churn:** The main reasons for churn include poor website experience, unsatisfactory customer service, and low product quality. Addressing these issues could help reduce churn and improve customer retention.
5. **Model Evaluation:** The f1-score metric was chosen for model evaluation to minimize false negatives and positives. The sequential API with the AdaMax optimizer was selected due to its more stable f1-score values than the functional API, which showed more fluctuations.
6. **Model Selection:** The chosen sequential API model is considered a good fit for the data and can be used for predicting churn on new data.
7. **Further Improvement:** For enhancing model performance, feature selection can be implemented to exclude irrelevant features that correlate little with the target variable. Additionally, trying higher epochs during training can be explored to observe any potential improvement in the loss graph.

Based on the analysis, it is recommended to focus on addressing the reasons for churn, such as improving website experience, customer service, and product quality. The selected model can be used for predicting churn, and further enhancements can be made by refining the feature set and experimenting with higher epochs during training.
