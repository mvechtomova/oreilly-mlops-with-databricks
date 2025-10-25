from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.models.model import ModelInfo
from mlflow.pyfunc import PythonModelContext
from mlflow.utils.environment import _mlflow_conda_env

from hotel_booking.config import Tags


def adjust_price(price: float) -> float:
    """Adjust the price by a fixed percentage."""
    return round(price * 1.05, 2)


class HotelBookingModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context: PythonModelContext) -> None:
        self.model = mlflow.sklearn.load_model(context.artifacts["lightgbm-pipeline"])

    def predict(
        self, context: PythonModelContext, model_input: pd.DataFrame | np.ndarray
    ) -> dict:
        predictions = self.model.predict(model_input)
        return {"Total price per night": [adjust_price(pred) for pred in predictions]}

    def log_register_model(
        self,
        wrapped_model_info: ModelInfo,
        pyfunc_model_name: str,
        experiment_name: str,
        tags: Tags,
        code_paths: list,
    ) -> None:
        """Log and register the Pyfunc model."""
        wrapped_model_uri = wrapped_model_info.model_uri
        input_example_dict = mlflow.artifacts.load_dict(
            f"{wrapped_model_info.artifact_path}/input_example.json"
        )
        input_example = pd.DataFrame(
            input_example_dict["data"], columns=input_example_dict["columns"]
        )
        additional_pip_deps = []
        for package in code_paths:
            whl_name = package.split("/")[-1]
            additional_pip_deps.append(f"code/{whl_name}")
        conda_env = _mlflow_conda_env(additional_pip_deps=additional_pip_deps)

        mlflow.set_experiment(experiment_name=experiment_name)
        with mlflow.start_run(
            run_name=f"wrapper-lightgbm-{datetime.now().strftime('%Y-%m-%d')}",
            tags=tags.to_dict(),
        ):
            signature = infer_signature(
                model_input=input_example,
                model_output={"Total price per night": [100.00]},
            )
            self.model_info = mlflow.pyfunc.log_model(
                python_model=self,
                name="pyfunc-wrapper",
                artifacts={"lightgbm-pipeline": wrapped_model_uri},
                signature=signature,
                code_paths=code_paths,
                conda_env=conda_env,
                input_example=input_example,
            )
        client = MlflowClient()
        registered_model = mlflow.register_model(
            model_uri=self.model_info.model_uri,
            name=pyfunc_model_name,
            tags=tags.to_dict(),
        )
        client.set_registered_model_alias(
            name=pyfunc_model_name,
            alias="latest-model",
            version=registered_model.version,
        )
        return registered_model.version
