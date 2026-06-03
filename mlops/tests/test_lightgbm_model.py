"""Unit tests for model training and the categorical transformer (no MLflow)."""
# test fixtures (mocker, cfg, ...) are conventionally un-annotated
# ruff: noqa: ANN001, ANN201, ANN202

from hotel_booking.models.lightgbm_model import LightGBMModel


def test_train_builds_pipeline_and_predicts(training_df, cfg) -> None:
    X = training_df[cfg.num_features + cfg.cat_features]
    y = training_df[cfg.target]
    model = LightGBMModel(config=cfg)
    model.train(X, y)
    assert model.pipeline is not None
    preds = model.pipeline.predict(X)
    # We assert shape/runnability, not accuracy.
    assert len(preds) == len(X)


def test_cat_transformer_encodes_unknown_categories_as_minus_one(
    training_df, cfg
) -> None:
    # Train, then predict on a row with a category never seen in training.
    X = training_df[cfg.num_features + cfg.cat_features]
    y = training_df[cfg.target]
    model = LightGBMModel(config=cfg)
    model.train(X, y)

    unseen = X.iloc[[0]].copy()
    unseen["room_type"] = "Room_Type_999"  # not in the training categories
    # Pipeline must not raise; unknown category is mapped to -1 internally.
    preds = model.pipeline.predict(unseen)
    assert len(preds) == 1


def test_train_honors_explicit_parameters(training_df, cfg) -> None:
    X = training_df[cfg.num_features + cfg.cat_features]
    y = training_df[cfg.target]
    model = LightGBMModel(config=cfg)
    model.train(
        X, y, parameters={"n_estimators": 5, "max_depth": 2, "learning_rate": 0.1}
    )
    assert model.parameters["n_estimators"] == 5
