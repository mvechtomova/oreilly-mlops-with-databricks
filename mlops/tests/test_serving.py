"""Unit tests for the create-vs-update branch in serve_model (WorkspaceClient mocked)."""
# test fixtures (mocker, cfg, ...) are conventionally un-annotated
# ruff: noqa: ANN001, ANN201, ANN202

import pytest

from hotel_booking.models.serving import serve_model


@pytest.fixture
def run_serve_model(mocker):
    """Run serve_model against a mocked workspace whose endpoints have given names.

    Returns the mocked WorkspaceClient so the test can assert which branch ran.
    """

    def run(existing_names: list[str]):
        workspace = mocker.Mock()
        # `name` is a reserved Mock constructor kwarg, so set it after construction.
        endpoints = []
        for endpoint_name in existing_names:
            endpoint = mocker.Mock()
            endpoint.name = endpoint_name
            endpoints.append(endpoint)
        workspace.serving_endpoints.list.return_value = endpoints
        mocker.patch(
            "hotel_booking.models.serving.WorkspaceClient", return_value=workspace
        )
        serve_model(
            entity_name="mlops_dev.hotel_booking.model",
            entity_version="1",
            tags={"key": "project", "value": "hotel-booking"},
            endpoint_name="hotel-booking-endpoint",
            catalog_name="mlops_dev",
            schema_name="hotel_booking",
            table_name_prefix="hotel_booking",
        )
        return workspace

    return run


def test_creates_endpoint_when_absent(run_serve_model) -> None:
    workspace = run_serve_model(existing_names=["some-other-endpoint"])
    workspace.serving_endpoints.create.assert_called_once()
    workspace.serving_endpoints.update_config.assert_not_called()


def test_updates_endpoint_when_present(run_serve_model) -> None:
    workspace = run_serve_model(existing_names=["hotel-booking-endpoint"])
    workspace.serving_endpoints.update_config.assert_called_once()
    workspace.serving_endpoints.create.assert_not_called()
