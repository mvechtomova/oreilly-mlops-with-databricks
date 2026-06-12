"""Unit tests for the custom deployment exceptions."""
# test fixtures (mocker, cfg, ...) are conventionally un-annotated
# ruff: noqa: ANN001, ANN201, ANN202

import pytest

from hotel_booking.utils.exceptions import (
    DeploymentNotApprovedError,
    MissingDeploymentTagError,
)


@pytest.mark.parametrize(
    "exc_cls",
    [MissingDeploymentTagError, DeploymentNotApprovedError],
)
def test_custom_exceptions_are_raisable_and_carry_message(exc_cls) -> None:
    # Both are plain Exception subclasses used as deployment gate signals.
    assert issubclass(exc_cls, Exception)
    with pytest.raises(exc_cls, match="boom"):
        raise exc_cls("boom")
