import io
import pandas as pd
import requests
from pandas import DataFrame
from io import BytesIO
from typing import List

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test




@data_loader
def ingest_files(**kwargs) -> pd.DataFrame:
    """
    Ingest data from a specified URL and load it into a pandas DataFrame.

    This function downloads a Parquet file from a given URL, reads it into a 
    pandas DataFrame, and then returns the concatenated DataFrame. The URL is
    constructed based on a specified year and month.

    Args:
        **kwargs: Arbitrary keyword arguments.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the data from the Parquet file.

    Raises:
        Exception: If the HTTP request to the URL fails.
    """
    year = 2023
    month = 3
    
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(response.text)

    df = pd.read_parquet(BytesIO(response.content))
    
    return df

@test
def test_output(df) -> None:
    """
    Test the output of the ingest_files function.

    This function checks if the DataFrame returned by the ingest_files function
    is not None. It raises an assertion error if the DataFrame is None.

    Args:
        df (pd.DataFrame): The DataFrame to be tested.

    Raises:
        AssertionError: If the DataFrame is None.
    """
    assert df is not None, 'The output is undefined'