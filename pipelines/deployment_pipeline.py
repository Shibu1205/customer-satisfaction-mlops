import json
import numpy as np
import pandas as pd

from pydantic import BaseModel

from materializer.custom_materializer import cs_materializer
from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model
from steps.config import ModelNameConfig

from .utils import get_data_for_test

from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW

from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)

from zenml.integrations.mlflow.services import (
    MLFlowDeploymentService,
)

from zenml.integrations.mlflow.steps import (
    mlflow_model_deployer_step,
)


# Docker settings
docker_settings = DockerSettings(
    required_integrations=[MLFLOW]
)


# -----------------------------
# Dynamic Importer Step
# -----------------------------
@step(enable_cache=False)
def dynamic_importer() -> str:
    """
    Loads batch inference data.
    """
    data = get_data_for_test()
    return data


# -----------------------------
# Deployment Trigger Config
# -----------------------------
class DeploymentTriggerConfig(BaseModel):
    min_accuracy: float = 0.0


# -----------------------------
# Deployment Trigger Step
# -----------------------------
@step(enable_cache=False)
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
) -> bool:
    """
    Determines whether model should be deployed.
    """
    return accuracy >= config.min_accuracy


# -----------------------------
# MLFlow Loader Config
# -----------------------------
class MLFlowDeploymentLoaderStepParameters(BaseModel):
    pipeline_name: str
    step_name: str
    running: bool = True


# -----------------------------
# Prediction Service Loader
# -----------------------------
@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:

    mlflow_model_deployer_component = (
        MLFlowModelDeployer.get_active_model_deployer()
    )

    existing_services = (
        mlflow_model_deployer_component.find_model_server(
            pipeline_name=pipeline_name,
            pipeline_step_name=pipeline_step_name,
            model_name=model_name,
            running=running,
        )
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by "
            f"step '{pipeline_step_name}' in pipeline "
            f"'{pipeline_name}' for model '{model_name}' "
            f"is currently running."
        )

    return existing_services[0]


# -----------------------------
# Predictor Step
# -----------------------------
@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    """
    Run inference against deployed model service.
    """

    service.start(timeout=10)

    data = json.loads(data)

    # Remove metadata
    data.pop("columns", None)
    data.pop("index", None)

    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]

    df = pd.DataFrame(
        data["data"],
        columns=columns_for_df,
    )

    json_list = json.loads(
        json.dumps(
            list(df.T.to_dict().values())
        )
    )

    data_array = np.array(json_list)

    prediction = service.predict(data_array)

    return prediction


# =========================================================
# Continuous Deployment Pipeline
# =========================================================
@pipeline(
    enable_cache=False,
    settings={"docker": docker_settings},
)
def continuous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0.0,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):

    # Step 1 - Ingest data
    df = ingest_df(data_path=data_path)

    # Step 2 - Clean data
    X_train, X_test, y_train, y_test = clean_df(df=df)

    # Step 3 - Train model
    model = train_model(
        X_train=X_train,
        y_train=y_train,
        config=ModelNameConfig(
            model_name="LinearRegression"
        ),
    )

    # Step 4 - Evaluate model
    r2_score, rmse = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    # Step 5 - Trigger deployment
    deployment_decision = deployment_trigger(
        accuracy=r2_score,
        config=DeploymentTriggerConfig(
            min_accuracy=min_accuracy
        ),
    )

    # Step 6 - Deploy model
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )


# =========================================================
# Inference Pipeline
# =========================================================
@pipeline(
    enable_cache=False,
    settings={"docker": docker_settings},
)
def inference_pipeline(
    pipeline_name: str,
    pipeline_step_name: str,
):

    # Load batch data
    batch_data = dynamic_importer()

    # Load deployed model service
    model_deployment_service = (
        prediction_service_loader(
            pipeline_name=pipeline_name,
            pipeline_step_name=pipeline_step_name,
            running=True,
        )
    )

    # Make predictions
    prediction = predictor(
        service=model_deployment_service,
        data=batch_data,
    )

    return prediction
