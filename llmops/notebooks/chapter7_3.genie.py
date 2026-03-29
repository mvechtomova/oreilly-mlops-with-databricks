# Databricks notebook source

from arxiv_curator.config import ProjectConfig

cfg = ProjectConfig.from_yaml("../project_config.yml")
catalog = cfg.catalog
schema = cfg.schema


# COMMAND ----------
# create warehouse

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql
from databricks.sdk.service.sql import CreateWarehouseRequestWarehouseType


w = WorkspaceClient()

created = w.warehouses.create(
    name="__2XS_arxiv_warehouse",
    cluster_size="2X-Small",
    max_num_clusters=1,
    auto_stop_mins=5,
    warehouse_type=CreateWarehouseRequestWarehouseType("PRO"),
    enable_serverless_compute=True,
    tags=sql.EndpointTags(
        custom_tags=[sql.EndpointTagPair(key="project", value="arxiv_curator")]
    ),
).result()

warehouse_id = created.id

# COMMAND ----------
import json

serialized_space = {
    "version": 1,
    "data_sources": {
        "tables": [
            {
                "identifier": f"{catalog}.{schema}.arxiv_papers",
                "column_configs": [
                    {"column_name": "authors"},
                    {"column_name": "ingest_ts", "get_example_values": True},
                    {"column_name": "paper_id", "get_example_values": True},
                    {
                        "column_name": "pdf_url",
                        "get_example_values": True,
                        "build_value_dictionary": True,
                    },
                    {"column_name": "processed", "get_example_values": True},
                    {"column_name": "published", "get_example_values": True},
                    {
                        "column_name": "summary",
                        "get_example_values": True,
                        "build_value_dictionary": True,
                    },
                    {
                        "column_name": "title",
                        "get_example_values": True,
                        "build_value_dictionary": True,
                    },
                    {
                        "column_name": "volume_path",
                        "get_example_values": True,
                        "build_value_dictionary": True,
                    },
                ],
            }
        ]
    },
}

space = w.genie.create_space(
    warehouse_id=warehouse_id,
    serialized_space=json.dumps(serialized_space),
    title="arxiv-curator-space",
)

# COMMAND ----------
space_id = space.space_id

space = w.genie.get_space(space_id=space_id, include_serialized_space=True)
json.loads(space.serialized_space)

# COMMAND ----------
conversation = w.genie.start_conversation_and_wait(
    space_id=space.space_id,
    content="Find the last 10 papers published")

conversation.as_dict()

# COMMAND ----------
message = w.genie.create_message_and_wait(
    space_id=space.space_id,
    conversation_id=conversation.conversation_id,
    content="Return the list of authors of the last 10 papers published")

message.as_dict()
