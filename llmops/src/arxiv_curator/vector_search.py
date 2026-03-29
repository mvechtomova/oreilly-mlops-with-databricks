from typing import Any

from databricks.vector_search.client import VectorSearchClient
from loguru import logger

from arxiv_curator.config import ProjectConfig


class VectorSearchManager:
    """
    VectorSearchManager handles the setup and management of vector search
    endpoints and indexes for arxiv paper chunks.
    """

    def __init__(
        self,
        config: ProjectConfig,
        endpoint_name: str = "vector-search-arxiv-endpoint",
        embedding_model: str = "databricks-gte-large-en",
    ) -> None:
        """
        Initialize VectorSearchManager with configuration.

        Args:
            config: ProjectConfig object with catalog and schema configurations
            endpoint_name: Name of the vector search endpoint
            embedding_model: Name of the embedding model endpoint to use
        """
        self.config = config
        self.endpoint_name = endpoint_name
        self.embedding_model = embedding_model
        self.catalog_name = config.catalog_name
        self.schema_name = config.schema_name

        self.client = VectorSearchClient()
        self.index_name = f"{self.catalog_name}.{self.schema_name}.arxiv_index"

    def create_endpoint_if_not_exists(self) -> None:
        """
        Create vector search endpoint if it doesn't already exist.
        Uses STANDARD endpoint type.
        """
        endpoint_exists = any(
            item.name == self.endpoint_name for item in self.client.list_endpoints()
        )

        if not endpoint_exists:
            logger.info(f"Creating vector search endpoint: {self.endpoint_name}")
            self.client.create_endpoint_and_wait(
                name=self.endpoint_name, endpoint_type="STANDARD"
            )
            logger.info(f"Vector search endpoint created: {self.endpoint_name}")
        else:
            logger.info(f"Vector search endpoint already exists: {self.endpoint_name}")

    def create_or_get_index(self) -> Any:
        """
        Create delta sync index if it doesn't exist, or get existing index.
        The index is created on the arxiv_chunks table with automatic
        embedding generation.

        Returns:
            Vector search index object
        """
        self.create_endpoint_if_not_exists()

        index_exists = any(
            item.name == self.index_name
            for item in self.client.list_indexes(self.endpoint_name)
        )

        if not index_exists:
            logger.info(f"Creating vector search index: {self.index_name}")
            source_table = f"{self.catalog_name}.{self.schema_name}.arxiv_chunks"

            index = self.client.create_delta_sync_index(
                endpoint_name=self.endpoint_name,
                source_table_name=source_table,
                index_name=self.index_name,
                pipeline_type="TRIGGERED",
                primary_key="id",
                embedding_source_column="text",
                embedding_model_endpoint_name=self.embedding_model,
            )
            logger.info(f"Vector search index created: {self.index_name}")
        else:
            logger.info(f"Vector search index already exists: {self.index_name}")
            index = self.client.get_index(index_name=self.index_name)

        return index

    def sync_index(self) -> None:
        """
        Sync the vector search index with the latest data from the source table.
        This triggers the index to update with any new or modified chunks.
        """
        index = self.create_or_get_index()
        logger.info(f"Syncing vector search index: {self.index_name}")
        index.sync()
