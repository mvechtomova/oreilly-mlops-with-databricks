"""Unit tests for the create-vs-get branches in VectorSearchManager.

The VectorSearchClient is mocked, so no endpoint or index is created.
"""
# test fixtures (mocker, cfg, ...) are conventionally un-annotated
# ruff: noqa: ANN001, ANN201, ANN202

import pytest

from arxiv_curator.vector_search import VectorSearchManager

ENDPOINT = "vector-search-arxiv-endpoint"


@pytest.fixture
def build_manager(mocker, cfg):
    """Build a VectorSearchManager with the VectorSearchClient mocked.

    `endpoint_exists` / `index_exists` drive which branch runs. Returns the
    manager and the mocked client.
    """

    def build(endpoint_exists=False, index_exists=False):
        client = mocker.Mock()
        client.endpoint_exists.return_value = endpoint_exists
        client.index_exists.return_value = index_exists
        mocker.patch(
            "arxiv_curator.vector_search.VectorSearchClient", return_value=client
        )
        return VectorSearchManager(config=cfg), client

    return build


def test_init_uses_catalog_and_schema_from_config(build_manager, cfg) -> None:
    manager, _ = build_manager()
    assert manager.index_name == f"{cfg.catalog}.{cfg.schema}.arxiv_index"


def test_creates_endpoint_when_absent(build_manager) -> None:
    manager, client = build_manager(endpoint_exists=False)
    manager.create_endpoint_if_not_exists()
    client.endpoint_exists.assert_called_once_with(ENDPOINT)
    client.create_endpoint_and_wait.assert_called_once()


def test_skips_endpoint_when_present(build_manager) -> None:
    manager, client = build_manager(endpoint_exists=True)
    manager.create_endpoint_if_not_exists()
    client.create_endpoint_and_wait.assert_not_called()


def test_creates_index_when_absent(build_manager) -> None:
    manager, client = build_manager(endpoint_exists=True, index_exists=False)
    manager.create_or_get_index()
    client.create_delta_sync_index.assert_called_once()
    client.get_index.assert_not_called()


def test_gets_index_when_present(build_manager, cfg) -> None:
    index_name = f"{cfg.catalog}.{cfg.schema}.arxiv_index"
    manager, client = build_manager(endpoint_exists=True, index_exists=True)
    manager.create_or_get_index()
    client.index_exists.assert_called_once_with(
        endpoint_name=ENDPOINT, index_name=index_name
    )
    client.get_index.assert_called_once_with(index_name=index_name)
    client.create_delta_sync_index.assert_not_called()
