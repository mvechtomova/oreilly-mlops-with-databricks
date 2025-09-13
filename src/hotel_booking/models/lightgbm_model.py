from datetime import datetime

import mlflow
import pandas as pd
import pyspark
from lightgbm import LGBMRegressor
from loguru import logger
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from hotel_booking.config import ProjectConfig, Tags


class LightGBMModel:
    """A basic model class for house price prediction using LightGBM.

    This class handles data loading, feature preparation, model training, and MLflow logging.
    """
    def __init__(self, config: ProjectConfig) -> None:
        """Initialize the LightGBM model."""
        self.config = config
        self.cat_features = self.config.cat_features
        self.parameters = self.config.parameters
        self.pipeline = None
        self.metrics = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, parameters: dict= None) -> None:
        """Prepare features and train the model.

        :param X_train: Training features as a DataFrame
        :param y_train: Training target as a Series
        """

        class CatToIntTransformer(BaseEstimator, TransformerMixin):
            """
            Transformer that encodes categorical columns as integer codes for LightGBM.
            Unknown categories at transform time are encoded as -1.
            """
            def __init__(self, cat_features: list[str]) -> None:
                """Initialize the transformer with categorical feature names."""
                self.cat_features = cat_features
                self.cat_maps_ = {}

            def fit(self, X: pd.DataFrame, y=None) -> None:
                """Fit the transformer to the DataFrame X."""
                self.fit_transform(X)
                return self

            def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
                """Fit and transform the DataFrame X."""
                X = X.copy()
                for col in self.cat_features:
                    c = pd.Categorical(X[col])
                    # Build mapping: {category: code}
                    self.cat_maps_[col] = dict(zip(c.categories, range(len(c.categories)), strict=False))
                    X[col] = X[col].map(lambda val, col=col: self.cat_maps_[col].get(val, -1)).astype("category")
                return X

            def transform(self, X: pd.DataFrame) -> pd.DataFrame:
                """Transform the DataFrame X by encoding categorical features as integers."""
                X = X.copy()
                for col in self.cat_features:
                    X[col] = X[col].map(lambda val, col=col: self.cat_maps_[col].get(val, -1)).astype("category")
                return X

        if parameters is not None:
            self.parameters = parameters
        preprocessor = ColumnTransformer(
            transformers=[("cat", CatToIntTransformer(self.cat_features), self.cat_features)],
            remainder="passthrough"
        )
        self.pipeline = Pipeline(
            steps=[("preprocessor", preprocessor),
                   ("regressor", LGBMRegressor(**self.parameters))])
        logger.info("🚀 Starting training...")
        self.pipeline.fit(X_train, y_train)

    def log_model(self, experiment_name: str,
                  tags: Tags,
                  X_test: pd.DataFrame,
                  y_test: pd.Series,
                  train_set_spark: pyspark.sql.DataFrame,
                  train_query: str,
                  test_set_spark: pyspark.sql.DataFrame,
                  test_query: str) -> None:
        """Log the model to MLflow."""
        tags = tags.to_dict()
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"lightgbm-training-{datetime.now().strftime('%Y-%m-%d')}",
                      description="LightGBM model training", tags=tags):
            mlflow.log_params(self.parameters)
            signature = infer_signature(
                model_input=X_test, model_output=self.pipeline.predict(X_test)
            )

            training = mlflow.data.from_spark(
                df=train_set_spark,
                sql=train_query
            )
            testing = mlflow.data.from_spark(
                df=test_set_spark,
                sql=test_query
            )
            mlflow.log_input(training, context="training")
            mlflow.log_input(testing, context="testing")

            self.model_info = mlflow.sklearn.log_model(
                sk_model=self.pipeline,
                name="lightgbm-pipeline",
                signature=signature,
                input_example=X_test[0:1]
            )
            eval_data = X_test.copy()
            eval_data[self.config.target] = y_test

            # This will log the evaluation metrics
            result = mlflow.models.evaluate(
                    self.model_info.model_uri,
                    eval_data,
                    targets=self.config.target,
                    model_type="regressor",
                    evaluators=["default"],
                )
            self.metrics = result.metrics
            return self.model_info

    def register_model(self: "LightGBMModel", model_name: str, tags: Tags) -> None:
        """Register the model in MLflow Model Registry."""
        client = MlflowClient()
        registered_model = mlflow.register_model(
                model_uri=self.model_info.model_uri,
                name=model_name,
                tags=tags.to_dict(),
            )
        client.set_registered_model_alias(
            name=model_name,
            alias="latest-model",
            version=registered_model.version,
        )
        return registered_model.version
