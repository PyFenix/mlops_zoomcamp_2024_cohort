import os
import numpy as np
import pickle
import click
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved",
)
@click.option(
    "--local",
    is_flag=True,
    help="Run the script locally without connecting to an MLflow server",
)
def run_train(data_path: str, local: bool):
    if local:
        # Define the local tracking URI within the project directory
        tracking_uri = os.path.join(os.path.dirname(__file__), "../mlruns")
        os.makedirs(tracking_uri, exist_ok=True)
        mlflow.set_tracking_uri(f"file://{os.path.abspath(tracking_uri)}")
    else:
        mlflow.set_tracking_uri("http://localhost:5000")

    mlflow.set_experiment("mlops_zoomcamp")

    # Enable autologging, but we will manually log some parts
    mlflow.sklearn.autolog(log_input_examples=False, log_model_signatures=False)

    with mlflow.start_run() as run:
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        # Ensure X_train and X_val are converted to appropriate format if needed
        if isinstance(X_train, np.ndarray):
            X_train = np.array(X_train)
        if isinstance(X_val, np.ndarray):
            X_val = np.array(X_val)

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        print(f"RMSE: {rmse}")
        min_samples_split = rf.get_params().get("min_samples_split")
        print(f"min_samples_split: {min_samples_split}")

        # Manually log parameters and metrics
        mlflow.log_param("max_depth", rf.max_depth)
        mlflow.log_param("random_state", rf.random_state)
        mlflow.log_metric("rmse", rmse)

        # Log the model manually
        mlflow.sklearn.log_model(rf, artifact_path="model")


if __name__ == "__main__":
    run_train()
