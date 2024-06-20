import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(df, *args, **kwargs):
    """
    Transform the input DataFrame by adding a trip duration column and filtering rows.

    This function adds a new column `duration` to the DataFrame, which calculates the trip duration 
    in minutes. It then filters out trips with durations less than or equal to 1 minute or greater 
    than 60 minutes. Finally, it converts the `PULocationID` and `DOLocationID` columns to strings.

    Args:
        df (pd.DataFrame): The input DataFrame containing trip data.
        args: Additional arguments from upstream blocks (if applicable).
        kwargs: Additional keyword arguments from upstream blocks (if applicable).

    Returns:
        pd.DataFrame: The transformed DataFrame with a new `duration` column, filtered rows, 
        and updated data types for `PULocationID` and `DOLocationID`.
    """
    df = df.assign(duration=(df['tpep_dropoff_datetime']-df['tpep_pickup_datetime']).apply(lambda td: td.total_seconds() / 60))
    df = df[(df['duration']>1) & (df['duration']<=60)]
    df = df.astype({'PULocationID': 'str', 'DOLocationID': 'str'})

    return df


@test
def test_output(output, *args) -> None:
    """
    Test the output of the transform function.

    This function checks if the output from the transform function is not None. 
    It raises an assertion error if the output is None.

    Args:
        output (pd.DataFrame): The output DataFrame from the transform function.
        args: Additional arguments (if any).

    Raises:
        AssertionError: If the output is None.
    """
    assert output is not None, 'The output is undefined'