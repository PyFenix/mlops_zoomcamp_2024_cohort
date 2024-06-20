import pickle
import pandas as pd
from flask import Flask, request, jsonify


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


def predict(year, month):
    """
    Make predictions using the model and the dictionary vectorizer.

    Args:
        df (pd.DataFrame): The input DataFrame containing trip data.
        dv (DictVectorizer): The dictionary vectorizer.
        model (LinearRegression): The trained linear regression model.

    Returns:
        pd.Series: The predictions.
    """
    model_path = "model.bin"

    data_url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"

    df = read_data(data_url)
    dv, model = load_model(model_path)

    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    # df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

    # Create a DataFrame with the predictions and 'ride_id'
    # results_df = pd.DataFrame({"ride_id": df["ride_id"], "predicted_duration": y_pred})
    print(
        f"The average predicted duration for {month:02d} of {year:04d} is:\n{round(y_pred.mean(), 3)}"
    )
    return round(y_pred.mean(), 3)


app = Flask("duration-prediction")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        data = request.get_json()
        print(f"Received data: {data}")
        year = data.get("year")
        month = data.get("month")

        if year is None or month is None:
            return jsonify({"error": "Year and month must be provided"}), 400

        try:
            year = int(year)
            month = int(month)
        except ValueError:
            return jsonify({"error": "Year and month must be integers"}), 400

        if not (1 <= month <= 12):
            return jsonify({"error": "Month must be between 1 and 12"}), 400

        avg_duration = predict(year, month)
        result = {"average_duration": avg_duration}
        return jsonify(result)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
