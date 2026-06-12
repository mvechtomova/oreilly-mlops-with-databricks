"""Unit tests for the pyfunc wrapper: predict post-processing and log/register."""
# test fixtures (mocker, cfg, ...) are conventionally un-annotated
# ruff: noqa: ANN001, ANN201, ANN202

import numpy as np

from hotel_booking.config import Tags
from hotel_booking.models.pyfunc_model_wrapper import HotelBookingModelWrapper


def test_load_context_loads_pipeline_from_artifact(mocker) -> None:
    # autospec binds the mock to mlflow.sklearn.load_model's real signature.
    # Patched at the canonical path (not the module alias) to avoid mlflow's lazy
    # submodule loader dropping the patch.
    load_model = mocker.patch("mlflow.sklearn.load_model", autospec=True)
    context = mocker.Mock(artifacts={"lightgbm-pipeline": "models:/m/1"})

    wrapper = HotelBookingModelWrapper()
    wrapper.load_context(context)

    load_model.assert_called_once_with("models:/m/1")
    assert wrapper.model is load_model.return_value


def test_predict_applies_commission_and_labels_output(mocker) -> None:
    wrapper = HotelBookingModelWrapper()
    wrapper.model = mocker.Mock()
    wrapper.model.predict.return_value = np.array([100.0, 200.0])

    out = wrapper.predict(context=None, model_input=mocker.Mock())

    # adjust_price adds 5% and rounds to 2 dp.
    assert out == {"Total price per night": [105.0, 210.0]}


def test_log_register_model_logs_registers_and_aliases(mocker) -> None:
    wrapper = HotelBookingModelWrapper()

    # autospec=True on every patch ties the mocks to the installed mlflow API.
    # mlflow.* helpers use canonical paths (the module-alias path makes mlflow's
    # lazy loader drop sibling patches); names imported into the module
    # (infer_signature, MlflowClient, _mlflow_conda_env) use the alias path.
    p = "hotel_booking.models.pyfunc_model_wrapper"
    load_dict = mocker.patch("mlflow.artifacts.load_dict", autospec=True)
    load_dict.return_value = {"data": [[1, 2]], "columns": ["a", "b"]}
    set_experiment = mocker.patch("mlflow.set_experiment", autospec=True)
    mocker.patch("mlflow.start_run", autospec=True)
    mocker.patch(f"{p}.infer_signature", autospec=True)
    conda = mocker.patch(f"{p}._mlflow_conda_env", autospec=True)
    log_model = mocker.patch("mlflow.pyfunc.log_model", autospec=True)
    register_model = mocker.patch("mlflow.register_model", autospec=True)
    client = mocker.patch(f"{p}.MlflowClient", autospec=True).return_value
    log_model.return_value = mocker.Mock(model_uri="models:/pyfunc/1")
    register_model.return_value = mocker.Mock(version="3")

    wrapped_info = mocker.Mock(
        model_uri="models:/wrapped/1", artifact_path="run/artifacts"
    )
    tags = Tags(git_sha="abc", branch="main")

    version = wrapper.log_register_model(
        wrapped_model_info=wrapped_info,
        pyfunc_model_name="cat.sch.pyfunc",
        experiment_name="/exp",
        tags=tags,
        code_paths=["dist/hotel_booking-0.1.0-py3-none-any.whl"],
    )

    # The wheel from code_paths is turned into a code/<whl> pip dependency.
    deps = conda.call_args.kwargs["additional_pip_deps"]
    assert deps == ["code/hotel_booking-0.1.0-py3-none-any.whl"]

    set_experiment.assert_called_once_with(experiment_name="/exp")
    log_model.assert_called_once()
    register_model.assert_called_once_with(
        model_uri="models:/pyfunc/1", name="cat.sch.pyfunc", tags=tags.to_dict()
    )
    client.set_registered_model_alias.assert_called_once_with(
        name="cat.sch.pyfunc", alias="latest-model", version="3"
    )
    assert version == "3"
