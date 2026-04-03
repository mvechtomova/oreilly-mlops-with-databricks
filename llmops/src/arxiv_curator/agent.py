import asyncio
import json
import os
import warnings
from collections.abc import Generator
from datetime import datetime
from typing import Any
from uuid import uuid4

import backoff
import mlflow
import nest_asyncio
import openai
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow import MlflowClient
from mlflow.entities import SpanType
from mlflow.models.resources import (
    DatabricksGenieSpace,
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksTable,
    DatabricksVectorSearchIndex,
)
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)

from arxiv_curator.config import ProjectConfig
from arxiv_curator.mcp import create_mcp_tools
from arxiv_curator.memory import LakebaseMemory


class ArxivAgent(ResponsesAgent):
    def __init__(
        self,
        llm_endpoint: str,
        system_prompt: str,
        catalog: str,
        schema: str,
        genie_space_id: str | None = None,
        lakebase_project_id: str | None = None,
    ):
        """Initializes the Arxiv Agent."""
        nest_asyncio.apply()

        self.system_prompt = system_prompt
        self.llm_endpoint = llm_endpoint
        self.workspace_client = WorkspaceClient()
        self.model_serving_client = (
            self.workspace_client.serving_endpoints.get_open_ai_client()
        )

        # Initialize Lakebase memory if configured
        self.memory: LakebaseMemory | None = None
        if lakebase_project_id:
            self.memory = LakebaseMemory(
                project_id=lakebase_project_id,
            )

        # Create tools from config
        host = self.workspace_client.config.host
        tools = asyncio.run(create_mcp_tools(
            w=self.workspace_client,
            url_list=[
                f"{host}/api/2.0/mcp/vector-search/{catalog}/{schema}",
                f"{host}/api/2.0/mcp/genie/{genie_space_id}",
            ],
        ))
        self._tools_dict = {tool.name: tool for tool in tools}

    def get_tool_specs(self) -> list[dict]:
        """Returns tool specifications in the format OpenAI expects."""
        return [tool_info.spec for tool_info in self._tools_dict.values()]

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """Executes the specified tool with the given arguments."""
        if tool_name not in self._tools_dict:
            available = list(self._tools_dict.keys())
            return f"Error: Unknown tool '{tool_name}'. Available tools: {available}"
        return self._tools_dict[tool_name].exec_fn(**args)

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def call_llm(
        self,
        messages: list[dict[str, Any]],
    ) -> Generator[dict[str, Any], None, None]:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="PydanticSerializationUnexpectedValue"
            )
            stream = self.model_serving_client.chat.completions.create(
                model=self.llm_endpoint,
                messages=to_chat_completions_input(messages),
                tools=self.get_tool_specs(),
                stream=True,
            )
            with mlflow.start_span(name="call_llm", span_type=SpanType.LLM) as span:
                span.set_inputs({"messages": messages})
                last_chunk: dict[str, Any] = {}
                for chunk in stream:
                    chunk_dict = chunk.to_dict()
                    last_chunk = chunk_dict
                    yield chunk_dict
                llm_request_id = stream.response.headers.get("x-request-id")
                outputs: dict[str, Any] = {
                    "model": last_chunk.get("model"),
                    "usage": last_chunk.get("usage"),
                }
                if llm_request_id:
                    outputs["llm_request_id"] = llm_request_id
                span.set_outputs(outputs)

    def handle_tool_call(
        self, tool_call: dict[str, Any], messages: list[dict[str, Any]]
    ) -> ResponsesAgentStreamEvent:
        """
        Execute tool calls, add them to the running message history,
        and return a ResponsesStreamEvent w/ tool output
        """
        args = json.loads(tool_call["arguments"])
        result = str(self.execute_tool(tool_name=tool_call["name"], args=args))

        tool_call_output = self.create_function_call_output_item(
            tool_call["call_id"], result
        )
        messages.append(tool_call_output)
        return ResponsesAgentStreamEvent(
            type="response.output_item.done", item=tool_call_output
        )

    @mlflow.trace(span_type=SpanType.RETRIEVER, name="memory_load")
    def load_memory(self, session_id: str) -> list[dict[str, Any]]:
        """Load previous messages from Lakebase memory."""
        if self.memory:
            return self.memory.load_messages(session_id)
        return []

    @mlflow.trace(span_type=SpanType.CHAIN, name="memory_save")
    def save_memory(
        self, session_id: str, messages: list[dict[str, Any]]
    ) -> None:
        """Save new messages to Lakebase memory."""
        self.memory.save_messages(session_id, messages)

    def _extract_output_items(
        self, events: list[ResponsesAgentStreamEvent],
    ) -> list[dict[str, Any]]:
        """Extract and serialize output items from stream events."""
        return [
            e.item if isinstance(e.item, dict) else e.item.model_dump()
            for e in events
            if e.type == "response.output_item.done"
        ]

    def _run_tool_loop(
        self,
        messages: list[dict[str, Any]],
        max_iter: int = 10,
    ) -> list[ResponsesAgentStreamEvent]:
        """Run the LLM ↔ tool loop until the model stops or max_iter."""
        events: list[ResponsesAgentStreamEvent] = []
        for _ in range(max_iter):
            last_msg = messages[-1]
            if last_msg.get("role") == "assistant":
                break
            elif last_msg.get("type") == "function_call":
                events.append(self.handle_tool_call(last_msg, messages))
            else:
                events.extend(
                    output_to_responses_items_stream(
                        chunks=self.call_llm(messages),
                        aggregator=messages,
                    ),
                )
        else:
            events.append(
                ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=self.create_text_output_item(
                        "Max iterations reached. Stopping.",
                        str(uuid4()),
                    ),
                ),
            )
        return events

    @mlflow.trace(span_type=SpanType.CHAIN)
    def call_and_run_tools(
        self,
        request_input: list[dict[str, Any]],
        previous_messages: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> list[ResponsesAgentStreamEvent]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
        ]
        if previous_messages:
            messages.extend(previous_messages)
        messages.extend(request_input)

        mlflow.update_current_trace(
            tags={
                "git_sha": os.getenv("GIT_SHA", "local"),
                "model_serving_endpoint_name": os.getenv(
                    "MODEL_SERVING_ENDPOINT_NAME", "local"),
                "model_version": os.getenv("MODEL_VERSION", "local"),
            },
            metadata=(
                {"mlflow.trace.session": session_id} if session_id else {}),
            client_request_id=request_id,
        )

        events = self._run_tool_loop(messages)

        if session_id and self.memory:
            self.save_memory(
                session_id,
                request_input + self._extract_output_items(events),
            )
        return events

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        events = list(self.predict_stream(request))
        return ResponsesAgentResponse(
            output=self._extract_output_items(events),
            custom_outputs=request.custom_inputs,
        )

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        custom = request.custom_inputs or {}
        session_id = custom.get("session_id")
        request_id = custom.get("request_id")

        previous_messages = (
            self.load_memory(session_id)
            if session_id and self.memory else []
        )

        request_input = [i.model_dump() for i in request.input]
        events = self.call_and_run_tools(
            request_input=request_input,
            previous_messages=previous_messages,
            request_id=request_id,
            session_id=session_id,
        )
        yield from events

def log_register_agent(
    cfg: ProjectConfig,
    git_sha: str,
    run_id: str,
    agent_code_path: str,
    model_name: str,
    evaluation_metrics: dict | None = None,
) -> mlflow.entities.model_registry.RegisteredModel:
    """
    Log and register an MLflow agent model to Unity Catalog.

    Args:
        cfg: Project configuration containing catalog, schema, and other settings.
        git_sha: Git commit SHA for tracking.
        run_id: Run identifier for tracking.
        model_name: Model path in Unity Catalog.
        agent_code_path: Path to the agent Python file.
        evaluation_metrics: Optional evaluation metrics to log.

    Returns:
        RegisteredModel object from Unity Catalog.
    """

    resources = [
        DatabricksServingEndpoint(endpoint_name=cfg.llm_endpoint),
        DatabricksGenieSpace(genie_space_id=cfg.genie_space_id),
        DatabricksVectorSearchIndex(
            index_name=f"{cfg.catalog}.{cfg.schema}.arxiv_index"
        ),
        DatabricksTable(table_name=f"{cfg.catalog}.{cfg.schema}.arxiv_papers"),
        DatabricksSQLWarehouse(warehouse_id=cfg.warehouse_id),
        DatabricksServingEndpoint(endpoint_name="databricks-bge-large-en"),
    ]

    model_config = {
        "catalog": cfg.catalog,
        "schema": cfg.schema,
        "genie_space_id": cfg.genie_space_id,
        "system_prompt": cfg.system_prompt,
        "llm_endpoint": cfg.llm_endpoint,
        "lakebase_project_id": cfg.lakebase_project_id,
    }

    test_request = {
    "input": [
        {"role": "user",
         "content": "What are recent papers about LLMs and reasoning?"}
    ]
    }

    mlflow.set_experiment(cfg.experiment_path)
    ts = datetime.now().strftime("%Y-%m-%d")

    with mlflow.start_run(
        run_name=f"arxiv-mcp-agent-{ts}",
        tags={"git_sha": git_sha, "run_id": run_id},
    ):
        model_info = mlflow.pyfunc.log_model(
            name="agent",
            python_model=agent_code_path,
            resources=resources,
            input_example=test_request,
            model_config=model_config,
        )
        if evaluation_metrics:
            mlflow.log_metrics(evaluation_metrics)

    logger.info(f"Registering model: {model_name}")
    registered_model = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=model_name,
        env_pack="databricks_model_serving",
        tags={"git_sha": git_sha, "run_id": run_id}
    )
    logger.info(f"Registered version: {registered_model.version}")

    client = MlflowClient()
    logger.info("Setting alias 'latest-model'")
    client.set_registered_model_alias(
        name=model_name,
        alias="latest-model",
        version=registered_model.version,
    )
    return registered_model
