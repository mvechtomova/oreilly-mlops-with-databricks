from collections.abc import Callable

from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient
from pydantic import BaseModel


class ToolInfo(BaseModel):
    """
    Class representing a tool for the agent.
    - "name" (str): The name of the tool.
    - "spec" (dict): JSON description of the tool (matches OpenAI Responses format)
    - "exec_fn" (Callable): Function that implements the tool logic
    """

    name: str
    spec: dict
    exec_fn: Callable

def create_managed_exec_fn(
    server_url: str, tool_name: str, w: WorkspaceClient
) -> Callable:
    def exec_fn(**kwargs):
        client = DatabricksMCPClient(server_url=server_url, workspace_client=w)
        response = client.call_tool(tool_name, kwargs)
        return "".join([c.text for c in response.content])

    return exec_fn


async def create_mcp_tools(w: WorkspaceClient, url_list: list[str]) -> list[ToolInfo]:
    tools = []
    for server_url in url_list:
        mcp_client = DatabricksMCPClient(server_url=server_url, workspace_client=w)
        mcp_tools = mcp_client.list_tools()
        for mcp_tool in mcp_tools:
            input_schema = mcp_tool.inputSchema.copy() if mcp_tool.inputSchema else {}
            tool_spec = {
                "type": "function",
                "function": {
                    "name": mcp_tool.name,
                    "parameters": input_schema,
                    "description": mcp_tool.description or f"Tool: {mcp_tool.name}",
                },
            }
            exec_fn = create_managed_exec_fn(server_url, mcp_tool.name, w)
            tools.append(ToolInfo(name=mcp_tool.name, spec=tool_spec, exec_fn=exec_fn))
    return tools
