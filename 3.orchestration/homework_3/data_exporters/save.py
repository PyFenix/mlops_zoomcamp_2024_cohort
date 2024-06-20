import pickle
import mlflow

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("nyc-taxi-experiment")

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Export model and dictionary vectorizer to MLflow and local file system.

    This function exports a trained model and its corresponding dictionary vectorizer.
    It logs the dictionary vectorizer as a binary artifact and the model using MLflow's
    `log_model` function. The function initiates an MLflow run, saves the vectorizer
    locally, logs it as an artifact, and then logs the model to MLflow.

    Args:
        model: The trained machine learning model to be exported.
        vec: The dictionary vectorizer associated with the model.
        args: Additional arguments from upstream blocks (if applicable).
        kwargs: Additional keyword arguments from upstream blocks (if applicable).

    Returns:
        None

    Raises:
        None
    """
    model, vec = data
    # Specify your data exporting logic here
    with mlflow.start_run():
        with open('dict_vectorizer.bin', 'wb') as f_out:
            pickle.dump(vec, f_out)
        mlflow.log_artifact('dict_vectorizer.bin')

        mlflow.sklearn.log_model(model, 'model')
    print('OK')