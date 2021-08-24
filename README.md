# Determining Heart Disease

Using the [UCI heart disease dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease), I want to determine if patients have heart disease or not given the features in this dataset, which are detailed below. I used logistic regression to classify and predict, whose hyperparameters are tuned using azure hyperdrive. This was then compared against azure's automl search to identify the most accurate model, which was then operationalized by deploying to ACI with Azure application insights.

## Project Set Up and Installation
First import the [UCI heart disease dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease) to Azure datasets and name it `uci_heart_disease`. Run all the cells in both the Jupyter notebooks, starting with hyperdrive, which will leverage the `train.py` script. Then, run the automl notebook that will execute automl runs (note that I have a beefy cluster from work, so the parallel runs can be reduced to be cost efficient) and save the best model. It will then use this model to deploy with ACI and test. 

## Dataset

### Overview
I'm using the [UCI heart disease dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease) 

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
