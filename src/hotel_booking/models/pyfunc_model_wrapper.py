from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModelContext

from hotel_booking.config import Tags


class HotelBookingModelWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context: PythonModelContext) -> None:
        self.model = mlflow.sklearn.load_model(
            context.artifacts["lightgbm-pipeline"]
        )

    def predict(self, context: PythonModelContext, model_input: pd.DataFrame | np.ndarray) -> dict:
        predictions = self.model.predict(model_input)
        return {"Total price per night": [round(pred*1.05, 2) for pred in predictions]}

    def log_register_model(self, wrapped_model_uri: str, pyfunc_model_name: str,
                           experiment_name: str, tags: Tags) -> None:

        mlflow.set_experiment(experiment_name=experiment_name)
        with mlflow.start_run(run_name=f"wrapper-lightgbm-{datetime.now().strftime('%Y-%m-%d')}",
            tags=tags.to_dict()):

            signature = infer_signature(model_input=self.model.input_example,
                                        model_output={'Total price per night': [100.00]})
            model_info = mlflow.pyfunc.log_model(
                python_model=self,
                name="pyfunc-wrapper",
                artifacts={
                    "lightgbm-pipeline": wrapped_model_uri},
                signature=signature,
            )
        client = MlflowClient()
        registered_model = mlflow.register_model(
                model_uri=model_info.model_uri,
                name=pyfunc_model_name,
                tags=tags.to_dict(),
            )
        client.set_registered_model_alias(
            name=pyfunc_model_name,
            alias="latest-model",
            version=registered_model.model_version,
        )
        return registered_model.version
