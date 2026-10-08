"""MCP Tool Registry aggregating tools across distributed MCP servers (FR-MCP-1, FR-MCP-2).

================================================================================
EDUCATIONAL ARCHITECTURE NOTES:
Why an Aggregated Tool Registry?
--------------------------------------------------------------------------------
1. Centralized Discovery, Decentralized Execution:
   The supervisor and specialist agents need a unified catalog of tools to
   plan and execute actions. However, the tools are physically hosted on
   different servers (Web Search, SQL Sandbox, Report Generator).
   The `MCPToolRegistry` acts as an intelligent API Gateway for tools:
   - Queries `tools/list` across all registered MCP clients.
   - Maintains an in-memory routing table mapping `tool_name -> MCPClient`.
   - Dispatches `call_tool(name, ...)` directly to the responsible server.

2. Conflict Detection & Tool Namespacing:
   If two independent servers register a tool with the same name, the registry
   detects the collision at startup and prevents ambiguous tool routing.
================================================================================
"""

from typing import Any

from libs.common.errors import ConflictError, NotFoundError
from libs.common.logging import get_logger
from libs.mcp_client.client import MCPClient
from libs.mcp_client.models import MCPToolCallResult, MCPToolDefinition

logger = get_logger("mcp.registry")


class MCPToolRegistry:
    """Aggregates and routes tool calls across multiple distributed MCP servers."""

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._servers: dict[str, MCPClient] = {}
        self._tool_route_map: dict[str, MCPClient] = {}
        self._tool_definitions: dict[str, MCPToolDefinition] = {}

    def register_client(self, client: MCPClient) -> None:
        """Register an MCP client and discover its tools into the routing table.

        Args:
            client: Connected MCPClient instance.
        """
        server_name = client.server_name
        self._servers[server_name] = client

        # Discover tools and populate routing table
        tools = client.list_tools()
        for tool in tools:
            if tool.name in self._tool_route_map:
                existing_server = self._tool_route_map[tool.name].server_name
                raise ConflictError(
                    f"Tool collision detected: '{tool.name}' is already registered by server '{existing_server}'",
                    details={"tool": tool.name, "server": server_name, "existing": existing_server},
                )

            self._tool_route_map[tool.name] = client
            self._tool_definitions[tool.name] = tool

        logger.info(
            "Registered MCP client in registry",
            server=server_name,
            tool_count=len(tools),
            tools=[t.name for t in tools],
        )

    def get_tool_definition(self, name: str) -> MCPToolDefinition:
        """Fetch definition metadata for a tool."""
        if name not in self._tool_definitions:
            raise NotFoundError(
                f"Tool '{name}' is not registered in the MCP tool registry",
                resource_type="mcp_tool",
                resource_id=name,
            )
        return self._tool_definitions[name]

    def list_all_tools(self) -> list[MCPToolDefinition]:
        """Return all tool definitions available across all connected MCP servers."""
        return list(self._tool_definitions.values())

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        tenant_id: str = "tenant-default",
        run_id: str = "run-default",
        approval_token: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> MCPToolCallResult:
        """Route and execute a tool call to its hosting MCP server.

        Args:
            name: Target tool identifier.
            arguments: Tool arguments.
            tenant_id: Tenant scope for isolation.
            run_id: Active orchestration run ID.
            approval_token: Optional cryptographic token for Tier 2 mutations.
            timeout_seconds: Remote execution timeout.

        Returns:
            MCPToolCallResult from the hosting server.
        """
        if name not in self._tool_route_map:
            raise NotFoundError(
                f"Tool '{name}' is not registered in the MCP tool registry",
                resource_type="mcp_tool",
                resource_id=name,
            )

        client = self._tool_route_map[name]
        return client.call_tool(
            name=name,
            arguments=arguments,
            tenant_id=tenant_id,
            run_id=run_id,
            approval_token=approval_token,
            timeout_seconds=timeout_seconds,
        )

    async def acall_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        tenant_id: str = "tenant-default",
        run_id: str = "run-default",
        approval_token: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> MCPToolCallResult:
        """Asynchronously route and execute a tool call."""
        if name not in self._tool_route_map:
            raise NotFoundError(
                f"Tool '{name}' is not registered in the MCP tool registry",
                resource_type="mcp_tool",
                resource_id=name,
            )

        client = self._tool_route_map[name]
        return await client.acall_tool(
            name=name,
            arguments=arguments,
            tenant_id=tenant_id,
            run_id=run_id,
            approval_token=approval_token,
            timeout_seconds=timeout_seconds,
        )

    def close_all(self) -> None:
        """Close all registered MCP client transports and release resources."""
        for client in self._servers.values():
            try:
                client.close()
            except Exception as e:
                logger.warning("Error closing MCP client", server=client.server_name, error=str(e))
