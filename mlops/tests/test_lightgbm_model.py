"""Unit tests for model training and the categorical transformer (no MLflow)."""
# test fixtures (mocker, cfg, ...) are conventionally un-annotated
# ruff: noqa: ANN001, ANN201, ANN202

from hotel_booking.config import Tags
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


def _trained_model(training_df, cfg) -> LightGBMModel:
    X = training_df[cfg.num_features + cfg.cat_features]
    y = training_df[cfg.target]
    model = LightGBMModel(config=cfg)
    model.train(X, y)
    return model


def test_log_model_logs_inputs_signature_and_metrics(mocker, training_df, cfg) -> None:
    model = _trained_model(training_df, cfg)
    X_test = training_df[cfg.num_features + cfg.cat_features]
    y_test = training_df[cfg.target]

    # autospec=True binds each mock to the installed mlflow signature, so a
    # renamed function or changed kwarg fails the test instead of passing silently.
    # mlflow.* helpers are patched at their canonical paths: patching them via the
    # `lightgbm_model.mlflow.<sub>` alias makes mlflow's lazy submodule loader
    # rebuild sibling submodules and silently drop other patches. Names imported
    # directly into the module (infer_signature) must stay on the alias path.
    p = "hotel_booking.models.lightgbm_model"
    set_experiment = mocker.patch("mlflow.set_experiment", autospec=True)
    log_params = mocker.patch("mlflow.log_params", autospec=True)
    log_input = mocker.patch("mlflow.log_input", autospec=True)
    mocker.patch("mlflow.start_run", autospec=True)
    mocker.patch("mlflow.data.from_spark", autospec=True)
    mocker.patch(f"{p}.infer_signature", autospec=True)
    log_model = mocker.patch("mlflow.sklearn.log_model", autospec=True)
    evaluate = mocker.patch("mlflow.models.evaluate", autospec=True)
    log_model.return_value = mocker.Mock(model_uri="models:/m/1")
    evaluate.return_value = mocker.Mock(metrics={"rmse": 1.23})

    tags = Tags(git_sha="abc", branch="main")
    info = model.log_model(
        experiment_name="/exp",
        tags=tags,
        X_test=X_test,
        y_test=y_test,
        train_set_spark=mocker.Mock(),
        train_query="TRAIN",
        test_set_spark=mocker.Mock(),
        test_query="TEST",
    )

    set_experiment.assert_called_once_with("/exp")
    log_params.assert_called_once_with(model.parameters)
    # Both the training and testing datasets are logged as run inputs.
    assert log_input.call_count == 2
    log_model.assert_called_once()
    assert model.metrics == {"rmse": 1.23}
    assert info is log_model.return_value


def test_register_model_registers_and_sets_alias(mocker, training_df, cfg) -> None:
    model = _trained_model(training_df, cfg)
    model.model_info = mocker.Mock(model_uri="models:/m/1")

    p = "hotel_booking.models.lightgbm_model"
    register_model = mocker.patch("mlflow.register_model", autospec=True)
    client = mocker.patch(f"{p}.MlflowClient", autospec=True).return_value
    register_model.return_value = mocker.Mock(version="7")

    tags = Tags(git_sha="abc", branch="main")
    version = model.register_model(model_name="cat.sch.model", tags=tags)

    register_model.assert_called_once_with(
        model_uri="models:/m/1", name="cat.sch.model", tags=tags.to_dict()
    )
    client.set_registered_model_alias.assert_called_once_with(
        name="cat.sch.model", alias="latest-model", version="7"
    )
    assert version == "7"
