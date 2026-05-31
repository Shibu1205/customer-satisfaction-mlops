import logging
import pandas as pd
import mlflow
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from zenml.client import Client
from .config import ModelNameConfig

experiment_tracker=Client().active_stack.experiment_tracker
@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ModelNameConfig,
) -> RegressorMixin:

    try:

        model = None

        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog

            model = LinearRegressionModel()

            trained_model = model.train(
                X_train,
                y_train
            )

            return trained_model

        else:
            raise ValueError(
                "Model {} not supported".format(
                    config.model_name
                )
            )

    except Exception as e:

        logging.error(
            "Error in training model: {}".format(e)
        )

        raise e