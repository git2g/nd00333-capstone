import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error, confusion_matrix
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.core import Dataset

def main():
    # Parse arguments from invoker
    parser = argparse.ArgumentParser()
    parser.add_argument('--C', type=float, default=1.0, help="regularize to apply a penalty to increasing the magnitude of parameter values to reduce overfitting. Must be a positive float. Smaller values specify stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="maximum number of iterations for the solvers to converge.")
    parser.add_argument('--n_jobs', type=int, default=1, help="number of CPU cores used when parallelizing over classes")
    parser.add_argument("--input-data", type=str, help="input dataset of UCIs health data in csv format")

    args = parser.parse_args()

    run = Run.get_context()
    run.log("--C (Regularization Strength):", np.float(args.C))
    run.log("--max_iter (Max iterations:", np.int(args.max_iter))
    run.log("--n_jobs (CPU cores:", np.int(args.n_jobs))

    # retrieve workspace and dataset
    ws = run.experiment.workspace
    dataset = Dataset.get_by_id(ws, id=args.input_data)

    # clean data
    df = dataset.to_pandas_dataframe()
    x = df.dropna()
    y = df.pop('target')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    model = LogisticRegression(C=args.C, max_iter=args.max_iter, n_jobs=args.n_jobs).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("accuracy", np.float(accuracy))
    value = {
       "schema_type": "confusion_matrix",
       "schema_version": "v1",
       "data": {
           "class_labels": ["0", "1"],
           "matrix": confusion_matrix(y_test, model.predict(x_test)).tolist()
       }
    }
    run.log_confusion_matrix(name='Confusion Matrix', value=value)
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=model, filename='outputs/model.pkl')

if __name__ == '__main__':
    main()
