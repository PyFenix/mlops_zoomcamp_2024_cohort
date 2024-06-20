import pickle
import pandas as pd
import argparse

# import requests
# from sklearn.feature_extraction import DictVectorizer
# from sklearn.linear_model import LinearRegression

# Define categorical features
categorical = ["PULocationID", "DOLocationID"]


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


def read_data(filename):
    """
    Read data from a Parquet file and preprocess it.

    Args:
        filename (str): The path to the Parquet file.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    df = pd.read_parquet(filename)

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df


def make_predictions(df, dv, model):
    """
    Make predictions using the model and the dictionary vectorizer.

    Args:
        df (pd.DataFrame): The input DataFrame containing trip data.
        dv (DictVectorizer): The dictionary vectorizer.
        model (LinearRegression): The trained linear regression model.

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
    model_path = "model.bin"
    data_url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    output_file = f"results_df_{year:04d}_{month:02d}.parquet"

    dv, model = load_model(model_path)
    df = read_data(data_url)

    y_pred = make_predictions(df, dv, model)

    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

    # Create a DataFrame with the predictions and 'ride_id'
    results_df = pd.DataFrame({"ride_id": df["ride_id"], "predicted_duration": y_pred})
    print(
        f"The average predicted duration for {month:02d} of {year:04d} is:\n{results_df['predicted_duration'].mean()}"
    )

    # Save the results to a Parquet file
    results_df.to_parquet(output_file, engine="pyarrow", compression=None, index=False)
    print("Predictions saved to", output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NYC Taxi Duration Prediction")
    parser.add_argument("--year", type=int, required=True, help="Year of the trip data")
    parser.add_argument(
        "--month", type=int, required=True, help="Month of the trip data"
    )
    args = parser.parse_args()
    main(args.year, args.month)
