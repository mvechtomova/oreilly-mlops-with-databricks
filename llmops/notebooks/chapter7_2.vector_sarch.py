# Databricks notebook source
from databricks.vector_search.client import VectorSearchClient

from arxiv_curator.config import ProjectConfig

cfg = ProjectConfig.from_yaml("../project_config.yml")
catalog= cfg.catalog
schema = cfg.schema

vsc = VectorSearchClient()
vector_search_endpoint_name = "vector-search-arxiv-endpoint"
if not vsc.endpoint_exists(name=vector_search_endpoint_name):
    vsc.create_endpoint_and_wait(
        name=vector_search_endpoint_name,
        endpoint_type="STANDARD",
        usage_policy_id=cfg.usage_policy_id)

# COMMAND ----------

vs_index_fullname = f"{catalog}.{schema}.arxiv_index"
embedding_model_endpoint = "databricks-gte-large-en"

if not vsc.index_exists(
    endpoint_name=vector_search_endpoint_name,
    index_name=vs_index_fullname,
):
    index = vsc.create_delta_sync_index(
        endpoint_name=vector_search_endpoint_name,
        source_table_name=f"{catalog}.{schema}.arxiv_chunks",
        index_name=vs_index_fullname,
        pipeline_type="TRIGGERED",
        primary_key="id",
        embedding_source_column="text",
        embedding_model_endpoint_name=embedding_model_endpoint,
        usage_policy_id=cfg.usage_policy_id
    )
else:
    index = vsc.get_index(index_name=vs_index_fullname)
    index.sync()


# COMMAND ----------
index = vsc.get_index(
    endpoint_name=vector_search_endpoint_name,
    index_name=vs_index_fullname,
)
index.wait_until_ready()

# COMMAND ----------
from databricks.vector_search.reranker import DatabricksReranker

results = index.similarity_search(
    query_text="Chunking strategies for document processing",
    columns=["text", "id"],
    filters={'year': "2026"},
    num_results=5,
    query_type = "hybrid",
    reranker=DatabricksReranker(
        columns_to_rerank=["text", "title", "summary"])
    )
