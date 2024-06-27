import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

year = 2023
month = 1


# Helper function to create datetime
def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


# Create the dataframe
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
df_input = pd.DataFrame(data, columns=columns)

# Input file path pattern
input_file = os.getenv("INPUT_FILE_PATTERN").format(year=year, month=month)

# Storage options
options = {"client_kwargs": {"endpoint_url": "http://localhost:4566"}}

# Save the dataframe to S3
df_input.to_parquet(
    input_file, engine="pyarrow", compression=None, index=False, storage_options=options
)
# Run the batch.py script for January 2023
os.system("python batch.py --year 2023 --month 1")

# Output file path
output_file = os.getenv("OUTPUT_FILE_PATTERN").format(year=year, month=month)

# Read the saved data from S3
df_output = pd.read_parquet(output_file, storage_options=options)

# Verify the result
sum_predicted_durations = df_output["predicted_duration"].sum()
print(
    f"The sum of predicted durations for the test dataframe is: {sum_predicted_durations}"
)
