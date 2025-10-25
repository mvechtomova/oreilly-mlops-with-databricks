from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    AiGatewayConfig,
    AiGatewayInferenceTableConfig,
    EndpointCoreConfigInput,
    EndpointTag,
    ServedEntityInput,
)


def serve_model(
    entity_name: str,
    entity_version: str,
    tags: dict,
    endpoint_name: str,
    catalog_name: str,
    schema_name: str,
    monitoring_table_suffix: str,
    scale_to_zero_enabled: bool = True,
    workload_size: str = "Small",
) -> None:
    served_entities = [
        ServedEntityInput(
            entity_name=entity_name,
            scale_to_zero_enabled=scale_to_zero_enabled,
            workload_size=workload_size,
            entity_version=entity_version,
        )
    ]

    ai_gateway_cfg = AiGatewayConfig(
        inference_table_config=AiGatewayInferenceTableConfig(
            enabled=True,
            catalog_name=catalog_name,
            schema_name=schema_name,
            monitoring_table_suffix=monitoring_table_suffix,
        )
    )

    workspace = WorkspaceClient()
    endpoint_exists = any(
        item.name == endpoint_name for item in workspace.serving_endpoints.list()
    )

    if not endpoint_exists:
        workspace.serving_endpoints.create(
            name=endpoint_name,
            config=EndpointCoreConfigInput(
                served_entities=served_entities,
            ),
            ai_gateway=ai_gateway_cfg,
            tags=[EndpointTag.from_dict(tags)],
        )
    else:
        workspace.serving_endpoints.update_config(
            name=endpoint_name, served_entities=served_entities
        )
