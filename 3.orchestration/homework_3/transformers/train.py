if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

@transformer
def transform(df, *args, **kwargs):
    """
    Transform the input DataFrame by vectorizing categorical features and training a linear regression model.

    This function transforms the input DataFrame by converting specified categorical features into a
    format suitable for machine learning algorithms using a dictionary vectorizer. It then trains a
    linear regression model on the transformed features and the target variable (duration).

    Args:
        df (pd.DataFrame): The input DataFrame containing trip data.
        args: Additional arguments from upstream blocks (if applicable).
        kwargs: Additional keyword arguments from upstream blocks (if applicable).

    Returns:
        model (LinearRegression): The trained linear regression model.
        vec (DictVectorizer): The dictionary vectorizer fitted to the input data.

    Raises:
        None
    """
    list_of_dicts = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')
    # Initialize the DictVectorizer
    vec = DictVectorizer()

    # Fit and transform the data
    X_train = vec.fit_transform(list_of_dicts)

    # Extract the target variable
    y_train = df['duration'].values

    # Train a plain linear regression model with default parameters
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Print the intercept of the trained model
    print(model.intercept_)

    return model, vec


@test
def test_output(output, *args) -> None:
    """
    Test the output of the transform function.

    This function checks if the output from the transform function is not None.
    It raises an assertion error if the output is None.

    Args:
        output (tuple): A tuple containing the trained model and the dictionary vectorizer.
        args: Additional arguments (if any).

    Raises:
        AssertionError: If the output is None.
    """
    assert output is not None, 'The output is undefined'
    # assert isinstance(output, tuple) and len(output) == 2, 'The output should be a tuple containing the model and vectorizer'
    # model, vec = output
    # assert isinstance(model, LinearRegression), 'The first element of the output should be a LinearRegression model'
    # assert isinstance(vec, DictVectorizer), 'The second element of the output should be a DictVectorizer'
