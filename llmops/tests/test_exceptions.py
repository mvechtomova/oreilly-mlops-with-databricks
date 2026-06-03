"""Unit tests for the custom deployment exceptions."""
# ruff: noqa: ANN001, ANN201, ANN202

import pytest

from arxiv_curator.utils.exceptions import (
    DeploymentNotApprovedError,
    MissingDeploymentTagError,
)


@pytest.mark.parametrize("exc", [DeploymentNotApprovedError, MissingDeploymentTagError])
def test_custom_errors_are_raisable_exceptions(exc) -> None:
    assert issubclass(exc, Exception)
    with pytest.raises(exc, match="boom"):
        raise exc("boom")
