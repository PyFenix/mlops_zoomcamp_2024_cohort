import pickle
import pandas as pd
import argparse
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


CATEGORICAL_FEATURES = ["PULocationID", "DOLocationID"]


def read_data(filename, categorical=CATEGORICAL_FEATURES):
    """
    Read data from a Parquet file and preprocess it.

    Args:
        filename (str): The path to the Parquet file.
        categorical (list): List of categorical feature names.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    # Check if S3_ENDPOINT_URL is set
    s3_endpoint_url = os.getenv("S3_ENDPOINT_URL")

    if s3_endpoint_url:
        # Set storage options to use localstack S3
        storage_options = {"client_kwargs": {"endpoint_url": s3_endpoint_url}}
        df = pd.read_parquet(filename, storage_options=storage_options)
    else:
        # Read directly from the file
        df = pd.read_parquet(filename)

    df = prepare_data(df, categorical)
    return df


def prepare_data(df, categorical):
    """
    Prepare data by calculating the duration and filtering rows.

    Args:
        df (pd.DataFrame): The input DataFrame.
        categorical (list): List of categorical feature names.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df


def get_input_path(year, month):
    default_input_pattern = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    input_pattern = os.getenv("INPUT_FILE_PATTERN", default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = "s3://nyc-duration/out/taxi_type=yellow_year={year:04d}_month={month:02d}.parquet"
    output_pattern = os.getenv("OUTPUT_FILE_PATTERN", default_output_pattern)
    return output_pattern.format(year=year, month=month)


def load_model(model_path):
    """
    Load the model and dictionary vectorizer from a file.

    Args:
        model_path (str): The path to the model file.

    Returns:
        tuple: A tuple containing the dictionary vectorizer and the model.
    """
    with open(model_path, "rb") as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


def prepare_data(df, categorical):

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df


def make_predictions(df, dv, model, categorical):
    """
    Make predictions using the model and the dictionary vectorizer.

    Args:
        df (pd.DataFrame): The input DataFrame containing trip data.
        dv (DictVectorizer): The dictionary vectorizer.
        model (LinearRegression): The trained linear regression model.
        categorical (list): List of categorical feature names.

    Returns:
        pd.Series: The predictions.
    """
    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    return y_pred


def main(year, month):
    """
    Main function to load the model, read data, make predictions, and save the results.

    Args:
        year (int): The year of the data to process.
        month (int): The month of the data to process.
    """
    model_path = "models/model.bin"

    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    dv, model = load_model(model_path)
    df = read_data(input_file, CATEGORICAL_FEATURES)

    y_pred = make_predictions(df, dv, model, CATEGORICAL_FEATURES)

    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

    # Create a DataFrame with the predictions and 'ride_id'
    results_df = pd.DataFrame({"ride_id": df["ride_id"], "predicted_duration": y_pred})
    print(
        f"The average predicted duration for {month:02d} of {year:04d} is:\n{results_df['predicted_duration'].mean()}"
    )

    s3_endpoint_url = os.getenv("S3_ENDPOINT_URL")
    if s3_endpoint_url:
        storage_options = {"client_kwargs": {"endpoint_url": s3_endpoint_url}}
        results_df.to_parquet(
            output_file,
            storage_options=storage_options,
            engine="pyarrow",
            compression=None,
            index=False,
        )
    else:
        results_df.to_parquet(
            output_file, engine="pyarrow", compression=None, index=False
        )

    print("Predictions saved to", output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NYC Taxi Duration Prediction")
    parser.add_argument("--year", type=int, required=True, help="Year of the trip data")
    parser.add_argument(
        "--month", type=int, required=True, help="Month of the trip data"
    )
    args = parser.parse_args()
    main(args.year, args.month)
