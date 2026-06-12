"""Unit tests for utility helpers. External clients are mocked via `mocker`."""
# test fixtures (mocker, cfg, ...) are conventionally un-annotated
# ruff: noqa: ANN001, ANN201, ANN202

import pytest

from hotel_booking.models.pyfunc_model_wrapper import adjust_price
from hotel_booking.utils.common import (
    create_parser,
    get_delta_table_version,
    set_mlflow_tracking_uri,
)


def test_adjust_price_applies_five_percent_and_rounds() -> None:
    assert adjust_price(100.0) == 105.0
    assert adjust_price(99.99) == 104.99  # 104.9895 -> rounded to 2 dp


def test_get_delta_table_version_reads_history(mocker) -> None:
    delta_table = mocker.Mock()
    delta_table.history.return_value.select.return_value.first.return_value = [12]
    mocker.patch(
        "hotel_booking.utils.common.DeltaTable.forName",
        return_value=delta_table,
        autospec=True,
    )

    version = get_delta_table_version(spark=mocker.Mock(), full_table_name="c.s.t")
    assert version == "12"


def test_set_mlflow_tracking_uri_noop_inside_databricks(mocker) -> None:
    # On a Databricks runtime the function must not reconfigure MLflow.
    mocker.patch.dict("os.environ", {"DATABRICKS_RUNTIME_VERSION": "17.3"}, clear=False)
    set_uri = mocker.patch(
        "hotel_booking.utils.common.mlflow.set_tracking_uri", autospec=True
    )
    set_mlflow_tracking_uri()
    set_uri.assert_not_called()


def test_set_mlflow_tracking_uri_configures_profile_locally(mocker) -> None:
    mocker.patch.dict("os.environ", {"PROFILE": "my-profile"}, clear=True)
    mocker.patch("hotel_booking.utils.common.load_dotenv", autospec=True)
    set_uri = mocker.patch(
        "hotel_booking.utils.common.mlflow.set_tracking_uri", autospec=True
    )
    set_reg = mocker.patch(
        "hotel_booking.utils.common.mlflow.set_registry_uri", autospec=True
    )

    set_mlflow_tracking_uri()

    set_uri.assert_called_once_with("databricks://my-profile")
    set_reg.assert_called_once_with("databricks-uc://my-profile")


def test_create_parser_common_subcommand() -> None:
    args = create_parser(["common", "--root_path", "/root", "--env", ".env"])
    assert args.command == "common"
    assert args.root_path == "/root"
    assert args.env == ".env"


def test_create_parser_train_deploy_model_subcommand() -> None:
    args = create_parser(
        [
            "train_deploy_model",
            "--root_path",
            "/root",
            "--env",
            ".env",
            "--branch",
            "main",
            "--git_sha",
            "abc123",
            "--run_id",
            "42",
            "--job_id",
            "99",
        ]
    )
    assert args.command == "train_deploy_model"
    assert args.branch == "main"
    assert args.git_sha == "abc123"
    assert args.run_id == "42"
    assert args.job_id == "99"


def test_create_parser_requires_a_subcommand() -> None:
    # dest="command" is required, so an empty argv must exit with an error.
    with pytest.raises(SystemExit):
        create_parser([])


def test_create_parser_train_deploy_model_requires_branch() -> None:
    with pytest.raises(SystemExit):
        create_parser(["train_deploy_model", "--root_path", "/root", "--env", ".env"])
