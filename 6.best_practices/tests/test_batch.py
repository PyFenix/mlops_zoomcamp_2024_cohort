import batch
from datetime import datetime
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
# Define categorical features
CATEGORICAL_FEATURES = os.getenv("CATEGORICAL_FEATURES")

# Define categorical features
CATEGORICAL_FEATURES = batch.CATEGORICAL_FEATURES


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]
    columns = [
        "PULocationID",
        "DOLocationID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
    ]
    df = pd.DataFrame(data, columns=columns)

    expected_data = [
        (-1, -1, dt(1, 1), dt(1, 10), 9.0),
        (1, 1, dt(1, 2), dt(1, 10), 8.0),
    ]
    expected_columns = [
        "PULocationID",
        "DOLocationID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "duration",
    ]
    expected_df = pd.DataFrame(expected_data, columns=expected_columns)
    expected_df[CATEGORICAL_FEATURES] = expected_df[CATEGORICAL_FEATURES].astype(str)

    result_df = batch.prepare_data(df, CATEGORICAL_FEATURES)

    assert result_df.to_dict(orient="list") == expected_df.to_dict(orient="list")


if __name__ == "__main__":
    test_prepare_data()
