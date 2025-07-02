import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from hotel_booking.config import ProjectConfig


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

class LightGBMModel:
    """A basic model class for house price prediction using LightGBM.

    This class handles data loading, feature preparation, model training, and MLflow logging.
    """
    def __init__(self, config: ProjectConfig) -> None:
        """Initialize the LightGBM model."""
        self.config = config
        self.cat_features = self.config.cat_features

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, parameters: dict= None) -> None:
        """Prepare features and train the model.

        :param X_train: Training features as a DataFrame
        :param y_train: Training target as a Series
        """
        if parameters is None:
            parameters = self.config.parameters
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

    def compute_metrics(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Compute regression metrics.

        :param X_test: Test features
        :param y_test: True target values
        :return: Dictionary with mse, rmse, mae, and r2_score
        """
        y_pred = self.pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mean_absolute_error(y_test, y_pred),
            "r2_score": r2_score(y_test, y_pred),
        }
