"""Unit tests for the MCP tool helpers (DatabricksMCPClient mocked)."""
# test fixtures (mocker, cfg, ...) are conventionally un-annotated
# ruff: noqa: ANN001, ANN201, ANN202

import asyncio

from arxiv_curator.mcp import ToolInfo, create_managed_exec_fn, create_mcp_tools


def test_managed_exec_fn_joins_response_text(mocker) -> None:
    response = mocker.Mock()
    response.content = [mocker.Mock(text="Hello, "), mocker.Mock(text="world")]
    client = mocker.Mock()
    client.call_tool.return_value = response
    mocker.patch("arxiv_curator.mcp.DatabricksMCPClient", return_value=client)

    exec_fn = create_managed_exec_fn(
        server_url="https://srv", tool_name="search", w=mocker.Mock()
    )
    result = exec_fn(query="agents")

    assert result == "Hello, world"
    client.call_tool.assert_called_once_with("search", {"query": "agents"})


def test_create_mcp_tools_builds_tool_infos(mocker) -> None:
    tool = mocker.Mock()
    tool.name = "search_papers"
    tool.description = "Search arxiv"
    tool.inputSchema = {"type": "object", "properties": {}}

    client = mocker.Mock()
    client.list_tools.return_value = [tool]
    mocker.patch("arxiv_curator.mcp.DatabricksMCPClient", return_value=client)

    tools = asyncio.run(create_mcp_tools(mocker.Mock(), ["https://srv"]))

    assert len(tools) == 1
    info = tools[0]
    assert isinstance(info, ToolInfo)
    assert info.name == "search_papers"
    assert info.spec["function"]["name"] == "search_papers"
    assert info.spec["function"]["description"] == "Search arxiv"
    assert info.spec["function"]["parameters"] == tool.inputSchema
    assert callable(info.exec_fn)


def test_create_mcp_tools_defaults_missing_description(mocker) -> None:
    tool = mocker.Mock()
    tool.name = "fetch"
    tool.description = None
    tool.inputSchema = None  # falls back to an empty schema

    client = mocker.Mock()
    client.list_tools.return_value = [tool]
    mocker.patch("arxiv_curator.mcp.DatabricksMCPClient", return_value=client)

    tools = asyncio.run(create_mcp_tools(mocker.Mock(), ["https://srv"]))

    assert tools[0].spec["function"]["description"] == "Tool: fetch"
    assert tools[0].spec["function"]["parameters"] == {}
