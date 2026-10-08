"""Model Context Protocol (MCP) client implementation (FR-MCP-1, FR-MCP-2, FR-MCP-7).

================================================================================
EDUCATIONAL ARCHITECTURE NOTES:
Why the MCP Client is the Platform's Non-Negotiable Tool Plane Gate
--------------------------------------------------------------------------------
1. The Principle of Isolation:
   Agents (Document RAG, Web Research, SQL Analytics, Data Processing, Reports)
   MUST NEVER directly import database drivers (`psycopg2`, `sqlalchemy`),
   scraping libraries (`requests`, `playwright`), or code executors.
   Instead, every agent consumes tools exclusively by dispatching JSON-RPC
   invocations through this `MCPClient`.

2. The Governance Gate Bridge (P0 Non-Negotiable Pillar):
   Before an action is dispatched to the transport layer, the `MCPClient`
   can verify permissions against the platform's `GovernanceGate`.
   If an agent attempts a Tier 2 write (e.g. `run_sql_write`) without a valid
   cryptographic approval token, the client intercepts and blocks the call
   before it ever reaches the server!

3. Telemetry and Trace Propagation:
   Distributed tracing across services requires passing the W3C trace context
   and platform correlation ID (`X-Correlation-ID`) into the JSON-RPC envelope
   under the `_meta` parameter object, ensuring complete end-to-end trace waterfalls.
================================================================================
"""

from typing import Any
from uuid import uuid4

from libs.common.errors import (
    ExternalServiceError,
    GovernanceError,
    NotFoundError,
    ValidationError,
)
from libs.common.logging import get_logger
from libs.common.telemetry import inject_trace_context
from libs.guardrails.governance import GovernanceGate
from libs.mcp_client.models import (
    JSONRPCRequest,
    JSONRPCResponse,
    MCPErrorCode,
    MCPListToolsResult,
    MCPToolCallResult,
    MCPToolDefinition,
)
from libs.mcp_client.transport import MCPTransport

logger = get_logger("mcp.client")


class MCPClient:
    """Standardized client for communicating with an MCP server."""

    def __init__(
        self,
        transport: MCPTransport,
        server_name: str = "mcp-server",
        governance_gate: GovernanceGate | None = None,
    ) -> None:
        """Initialize MCP client.

        Args:
            transport: Concrete communication transport (Stdio, In-Memory, SSE).
            server_name: Human-readable name for logging and metrics.
            governance_gate: Optional governance gate for enforcing risk-tier checks.
        """
        self.transport = transport
        self.server_name = server_name
        self.governance_gate = governance_gate
        self._cached_tools: dict[str, MCPToolDefinition] | None = None

    def list_tools(self, force_refresh: bool = False) -> list[MCPToolDefinition]:
        """Discover tools exposed by the MCP server (`tools/list`).

        Args:
            force_refresh: True to bypass cache and re-query the server.

        Returns:
            List of declared MCP tool definitions.
        """
        if self._cached_tools is not None and not force_refresh:
            return list(self._cached_tools.values())

        req_id = f"req-{uuid4().hex[:8]}"
        trace_ctx = inject_trace_context()
        request = JSONRPCRequest(
            id=req_id,
            method="tools/list",
            params={"_meta": trace_ctx},
        )

        logger.debug("Listing MCP tools", server=self.server_name, req_id=req_id)
        response: JSONRPCResponse = self.transport.send_request(request)

        if not response.is_success or response.result is None:
            err_msg = response.error.message if response.error else "Unknown error"
            raise ExternalServiceError(
                f"Failed to list tools from MCP server '{self.server_name}': {err_msg}",
                service_name=self.server_name,
            )

        tools_result = MCPListToolsResult.model_validate(response.result)
        self._cached_tools = {tool.name: tool for tool in tools_result.tools}
        logger.info(
            "Discovered MCP tools",
            server=self.server_name,
            tool_count=len(self._cached_tools),
            tool_names=list(self._cached_tools.keys()),
        )
        return list(self._cached_tools.values())

    async def alist_tools(self, force_refresh: bool = False) -> list[MCPToolDefinition]:
        """Asynchronously discover tools exposed by the MCP server."""
        if self._cached_tools is not None and not force_refresh:
            return list(self._cached_tools.values())

        req_id = f"req-{uuid4().hex[:8]}"
        trace_ctx = inject_trace_context()
        request = JSONRPCRequest(
            id=req_id,
            method="tools/list",
            params={"_meta": trace_ctx},
        )

        response: JSONRPCResponse = await self.transport.asend_request(request)

        if not response.is_success or response.result is None:
            err_msg = response.error.message if response.error else "Unknown error"
            raise ExternalServiceError(
                f"Failed to list tools from MCP server '{self.server_name}': {err_msg}",
                service_name=self.server_name,
            )

        tools_result = MCPListToolsResult.model_validate(response.result)
        self._cached_tools = {tool.name: tool for tool in tools_result.tools}
        return list(self._cached_tools.values())

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        tenant_id: str = "tenant-default",
        run_id: str = "run-default",
        approval_token: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> MCPToolCallResult:
        """Invoke an MCP tool with schema validation and governance checks (FR-MCP-7).

        Args:
            name: Registered tool name.
            arguments: Dictionary of arguments.
            tenant_id: Tenant scope for isolation and policy lookup.
            run_id: Active orchestration run identifier.
            approval_token: Cryptographic HMAC token for Tier 2 mutations.
            timeout_seconds: Timeout for remote execution.

        Returns:
            MCPToolCallResult containing structured output content.
        """
        # 1. Ensure tools discovered
        if self._cached_tools is None:
            self.list_tools()
        assert self._cached_tools is not None

        if name not in self._cached_tools:
            raise NotFoundError(
                f"Tool '{name}' not found on MCP server '{self.server_name}'",
                resource_type="mcp_tool",
                resource_id=name,
            )

        tool_def = self._cached_tools[name]

        # 2. Argument validation against inputSchema
        self._validate_arguments(tool_def, arguments)

        # 3. Governance Gate Enforcement (P0 Non-Negotiable Pillar)
        if self.governance_gate is not None:
            decision = self.governance_gate.evaluate(
                run_id=run_id,
                tenant_id=tenant_id,
                action=name,
                arguments=arguments,
                approval_token=approval_token,
            )
            if not decision.allowed:
                raise GovernanceError(
                    f"Action '{name}' rejected by governance gate: {decision.reason}",
                    action=name,
                    details={"remediation_hint": decision.remediation_hint}
                    if decision.remediation_hint
                    else None,
                )

        # 4. Prepare JSON-RPC request envelope
        req_id = f"call-{uuid4().hex[:8]}"
        trace_ctx = inject_trace_context()
        meta_payload = {
            **trace_ctx,
            "tenant_id": tenant_id,
            "run_id": run_id,
            "approval_token": approval_token,
        }

        request = JSONRPCRequest(
            id=req_id,
            method="tools/call",
            params={
                "name": name,
                "arguments": arguments,
                "_meta": meta_payload,
            },
        )

        logger.info(
            "Dispatching MCP tool call",
            server=self.server_name,
            tool=name,
            req_id=req_id,
            tenant_id=tenant_id,
        )
        response: JSONRPCResponse = self.transport.send_request(
            request, timeout_seconds=timeout_seconds
        )

        # 5. Handle response & error mappings
        if not response.is_success or response.result is None:
            err = response.error
            if err:
                if err.code == MCPErrorCode.INVALID_PARAMS:
                    raise ValidationError(f"Invalid arguments for tool '{name}': {err.message}")
                if err.code == MCPErrorCode.UNAUTHORIZED:
                    raise GovernanceError(f"Unauthorized tool execution: {err.message}", action=name)
                raise ExternalServiceError(
                    f"Tool '{name}' execution failed: {err.message}",
                    service_name=self.server_name,
                    details=err.data if isinstance(err.data, dict) else {"data": err.data},
                )
            raise ExternalServiceError(
                f"Tool '{name}' execution returned empty response",
                service_name=self.server_name,
            )

        return MCPToolCallResult.model_validate(response.result)

    async def acall_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        tenant_id: str = "tenant-default",
        run_id: str = "run-default",
        approval_token: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> MCPToolCallResult:
        """Asynchronously invoke an MCP tool with schema validation and governance checks."""
        if self._cached_tools is None:
            await self.alist_tools()
        assert self._cached_tools is not None

        if name not in self._cached_tools:
            raise NotFoundError(
                f"Tool '{name}' not found on MCP server '{self.server_name}'",
                resource_type="mcp_tool",
                resource_id=name,
            )

        tool_def = self._cached_tools[name]
        self._validate_arguments(tool_def, arguments)

        if self.governance_gate is not None:
            decision = self.governance_gate.evaluate(
                run_id=run_id,
                tenant_id=tenant_id,
                action=name,
                arguments=arguments,
                approval_token=approval_token,
            )
            if not decision.allowed:
                raise GovernanceError(
                    f"Action '{name}' rejected by governance gate: {decision.reason}",
                    action=name,
                    details={"remediation_hint": decision.remediation_hint}
                    if decision.remediation_hint
                    else None,
                )

        req_id = f"call-{uuid4().hex[:8]}"
        trace_ctx = inject_trace_context()
        meta_payload = {
            **trace_ctx,
            "tenant_id": tenant_id,
            "run_id": run_id,
            "approval_token": approval_token,
        }

        request = JSONRPCRequest(
            id=req_id,
            method="tools/call",
            params={
                "name": name,
                "arguments": arguments,
                "_meta": meta_payload,
            },
        )

        response: JSONRPCResponse = await self.transport.asend_request(
            request, timeout_seconds=timeout_seconds
        )

        if not response.is_success or response.result is None:
            err = response.error
            if err:
                if err.code == MCPErrorCode.INVALID_PARAMS:
                    raise ValidationError(f"Invalid arguments for tool '{name}': {err.message}")
                if err.code == MCPErrorCode.UNAUTHORIZED:
                    raise GovernanceError(f"Unauthorized tool execution: {err.message}", action=name)
                raise ExternalServiceError(
                    f"Tool '{name}' execution failed: {err.message}",
                    service_name=self.server_name,
                )
            raise ExternalServiceError(
                f"Tool '{name}' execution returned empty response",
                service_name=self.server_name,
            )

        return MCPToolCallResult.model_validate(response.result)

    def _validate_arguments(self, tool_def: MCPToolDefinition, arguments: dict[str, Any]) -> None:
        """Verify that all required properties declared in inputSchema are provided."""
        schema = tool_def.inputSchema
        missing_fields = [f for f in schema.required if f not in arguments]
        if missing_fields:
            raise ValidationError(
                f"Missing required parameters for tool '{tool_def.name}': {missing_fields}",
                details={"missing": missing_fields, "schema": schema.model_dump()},
            )

    def close(self) -> None:
        """Close client and terminate underlying transport."""
        self.transport.close()
