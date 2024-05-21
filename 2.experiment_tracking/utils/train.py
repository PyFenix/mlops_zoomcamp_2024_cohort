import os
import pickle
import click
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved",
)
def run_train(data_path: str):
    # Set the tracking URI to a directory within the project
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("mlops_zoomcamp")

    # Enable autologging, but we will manually log some parts
    mlflow.sklearn.autolog(log_input_examples=False, log_model_signatures=False)

    with mlflow.start_run() as run:
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)
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