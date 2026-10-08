"""Unit tests for Model Context Protocol (MCP) client, transports, and tool registry (FR-MCP-1, FR-MCP-2, FR-MCP-7)."""

import pytest

from libs.common.errors import ConflictError, GovernanceError, NotFoundError, ValidationError
from libs.guardrails.governance import GovernanceGate
from libs.guardrails.tokens import ApprovalTokenManager
from libs.mcp_client import (
    InMemoryMCPTransport,
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPClient,
    MCPErrorCode,
    MCPListToolsResult,
    MCPTextContent,
    MCPToolCallResult,
    MCPToolDefinition,
    MCPToolInputSchema,
    MCPToolRegistry,
)


def create_mock_mcp_handler() -> InMemoryMCPTransport:
    """Helper creating an in-memory MCP server handler with standard tools."""
    tools = [
        MCPToolDefinition(
            name="search_web",
            description="Search the web for real-time information",
            inputSchema=MCPToolInputSchema(
                type="object",
                properties={"query": {"type": "string"}, "num_results": {"type": "integer"}},
                required=["query"],
            ),
        ),
        MCPToolDefinition(
            name="run_sql_read",
            description="Execute read-only SQL query against analytics database",
            inputSchema=MCPToolInputSchema(
                type="object",
                properties={"sql": {"type": "string"}},
                required=["sql"],
            ),
        ),
        MCPToolDefinition(
            name="run_sql_write",
            description="Execute high-risk database mutation query",
            inputSchema=MCPToolInputSchema(
                type="object",
                properties={"sql": {"type": "string"}},
                required=["sql"],
            ),
        ),
    ]

    def handler(request: JSONRPCRequest) -> JSONRPCResponse:
        if request.method == "tools/list":
            return JSONRPCResponse(
                id=request.id,
                result=MCPListToolsResult(tools=tools).model_dump(),
            )

        if request.method == "tools/call":
            params = request.params
            name = params.get("name")
            args = params.get("arguments", {})

            if name == "search_web":
                return JSONRPCResponse(
                    id=request.id,
                    result=MCPToolCallResult(
                        content=[MCPTextContent(text=f"Search results for: {args.get('query')}")],
                        metadata={"provider": "mock_search"},
                    ).model_dump(),
                )

            if name == "run_sql_read":
                return JSONRPCResponse(
                    id=request.id,
                    result=MCPToolCallResult(
                        content=[MCPTextContent(text="[{'id': 1, 'count': 42}]")],
                        metadata={"rows_returned": 1},
                    ).model_dump(),
                )

            if name == "run_sql_write":
                return JSONRPCResponse(
                    id=request.id,
                    result=MCPToolCallResult(
                        content=[MCPTextContent(text="Rows affected: 1")],
                        metadata={"mutation_applied": True},
                    ).model_dump(),
                )

            return JSONRPCResponse(
                id=request.id,
                error=JSONRPCError(code=MCPErrorCode.METHOD_NOT_FOUND, message=f"Tool '{name}' not found"),
            )

        return JSONRPCResponse(
            id=request.id,
            error=JSONRPCError(code=MCPErrorCode.METHOD_NOT_FOUND, message=f"Method '{request.method}' not found"),
        )

    return InMemoryMCPTransport(handler=handler, server_name="mock-server")


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.unit
def test_jsonrpc_models_and_serialization() -> None:
    """Validate JSON-RPC 2.0 envelopes and error code specifications."""
    req = JSONRPCRequest(id=1, method="tools/list")
    assert req.jsonrpc == "2.0"
    assert req.method == "tools/list"

    res_ok = JSONRPCResponse(id=1, result={"tools": []})
    assert res_ok.is_success
    assert res_ok.error is None

    res_err = JSONRPCResponse(
        id=1,
        error=JSONRPCError(code=MCPErrorCode.METHOD_NOT_FOUND, message="Not found"),
    )
    assert not res_err.is_success
    assert res_err.error is not None
    assert res_err.error.code == -32601


@pytest.mark.unit
def test_mcp_client_list_tools_and_caching() -> None:
    """Validate tool discovery and caching in MCPClient."""
    transport = create_mock_mcp_handler()
    client = MCPClient(transport=transport, server_name="test-server")

    tools = client.list_tools()
    assert len(tools) == 3
    tool_names = [t.name for t in tools]
    assert "search_web" in tool_names
    assert "run_sql_read" in tool_names
    assert "run_sql_write" in tool_names

    # Second call uses in-memory cache
    tools2 = client.list_tools()
    assert len(tools2) == 3


@pytest.mark.unit
def test_mcp_client_argument_schema_validation() -> None:
    """Validate that missing required arguments raises ValidationError."""
    transport = create_mock_mcp_handler()
    client = MCPClient(transport=transport, server_name="test-server")

    # Missing mandatory "query" field
    with pytest.raises(ValidationError) as exc:
        client.call_tool("search_web", arguments={"num_results": 5})

    assert "Missing required parameters" in str(exc.value)


@pytest.mark.unit
def test_mcp_client_tool_not_found() -> None:
    """Validate calling a nonexistent tool raises NotFoundError."""
    transport = create_mock_mcp_handler()
    client = MCPClient(transport=transport, server_name="test-server")

    with pytest.raises(NotFoundError) as exc:
        client.call_tool("nonexistent_tool", arguments={})

    assert "nonexistent_tool" in str(exc.value)


@pytest.mark.unit
def test_mcp_client_call_tool_execution() -> None:
    """Validate successful tool execution and output extraction."""
    transport = create_mock_mcp_handler()
    client = MCPClient(transport=transport, server_name="test-server")

    result = client.call_tool("search_web", arguments={"query": "AI Agents"})
    assert not result.isError
    assert "Search results for: AI Agents" in result.get_text_output()
    assert result.metadata["provider"] == "mock_search"


@pytest.mark.unit
def test_mcp_client_governance_gate_enforcement() -> None:
    """Validate that high-risk Tier 2 tool calls require signed approval tokens."""
    token_manager = ApprovalTokenManager(secret_key="cluster-test-secret-32-chars-long")
    gate = GovernanceGate(token_manager=token_manager)
    transport = create_mock_mcp_handler()
    client = MCPClient(transport=transport, server_name="test-server", governance_gate=gate)

    tenant_id = "tenant-prod"
    run_id = "run-mcp-999"
    write_args = {"sql": "UPDATE settings SET maintenance = true;"}

    # 1. Tier 0 read executes cleanly without token
    read_res = client.call_tool(
        "run_sql_read",
        arguments={"sql": "SELECT 1;"},
        tenant_id=tenant_id,
        run_id=run_id,
    )
    assert not read_res.isError
    assert "rows_returned" in read_res.metadata

    # 2. Tier 2 write without token is BLOCKED by Governance Gate
    with pytest.raises(GovernanceError) as exc:
        client.call_tool(
            "run_sql_write",
            arguments=write_args,
            tenant_id=tenant_id,
            run_id=run_id,
            approval_token=None,
        )
    assert "rejected by governance gate" in str(exc.value)

    # 3. Issue valid HMAC-SHA256 token bound to exact arguments
    token = token_manager.issue_token(
        run_id=run_id,
        tenant_id=tenant_id,
        action="run_sql_write",
        arguments=write_args,
    )

    # 4. Tier 2 write with signed token EXECUTES successfully!
    write_res = client.call_tool(
        "run_sql_write",
        arguments=write_args,
        tenant_id=tenant_id,
        run_id=run_id,
        approval_token=token,
    )
    assert not write_res.isError
    assert "Rows affected: 1" in write_res.get_text_output()


@pytest.mark.unit
def test_mcp_tool_registry_routing() -> None:
    """Validate multi-server tool aggregation and routing in MCPToolRegistry."""
    # Server 1: Web
    t1 = InMemoryMCPTransport(
        handler=lambda req: JSONRPCResponse(
            id=req.id,
            result=MCPListToolsResult(
                tools=[
                    MCPToolDefinition(
                        name="web_search",
                        description="Web search",
                        inputSchema=MCPToolInputSchema(required=["q"]),
                    )
                ]
            ).model_dump()
            if req.method == "tools/list"
            else MCPToolCallResult(content=[MCPTextContent(text="Web hit")]).model_dump(),
        ),
        server_name="web-server",
    )

    # Server 2: SQL
    t2 = InMemoryMCPTransport(
        handler=lambda req: JSONRPCResponse(
            id=req.id,
            result=MCPListToolsResult(
                tools=[
                    MCPToolDefinition(
                        name="sql_query",
                        description="SQL query",
                        inputSchema=MCPToolInputSchema(required=["sql"]),
                    )
                ]
            ).model_dump()
            if req.method == "tools/list"
            else MCPToolCallResult(content=[MCPTextContent(text="SQL hit")]).model_dump(),
        ),
        server_name="sql-server",
    )

    client1 = MCPClient(transport=t1, server_name="web-server")
    client2 = MCPClient(transport=t2, server_name="sql-server")

    registry = MCPToolRegistry()
    registry.register_client(client1)
    registry.register_client(client2)

    all_tools = registry.list_all_tools()
    assert len(all_tools) == 2
    tool_names = {t.name for t in all_tools}
    assert tool_names == {"web_search", "sql_query"}

    # Route web tool
    res1 = registry.call_tool("web_search", arguments={"q": "test"})
    assert res1.get_text_output() == "Web hit"

    # Route sql tool
    res2 = registry.call_tool("sql_query", arguments={"sql": "SELECT 1"})
    assert res2.get_text_output() == "SQL hit"


@pytest.mark.unit
def test_mcp_tool_registry_detects_collision() -> None:
    """Validate that duplicate tool names across servers raise ConflictError."""
    dup_tool = MCPToolDefinition(name="duplicate_tool", description="Test")

    t1 = InMemoryMCPTransport(
        handler=lambda req: JSONRPCResponse(
            id=req.id, result=MCPListToolsResult(tools=[dup_tool]).model_dump()
        ),
        server_name="server-1",
    )
    t2 = InMemoryMCPTransport(
        handler=lambda req: JSONRPCResponse(
            id=req.id, result=MCPListToolsResult(tools=[dup_tool]).model_dump()
        ),
        server_name="server-2",
    )

    client1 = MCPClient(transport=t1, server_name="server-1")
    client2 = MCPClient(transport=t2, server_name="server-2")

    registry = MCPToolRegistry()
    registry.register_client(client1)

    with pytest.raises(ConflictError) as exc:
        registry.register_client(client2)

    assert "Tool collision detected" in str(exc.value)
