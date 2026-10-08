"""Model Context Protocol (MCP) data models and JSON-RPC 2.0 specifications (FR-MCP-1, FR-MCP-2).

================================================================================
EDUCATIONAL ARCHITECTURE NOTES:
Why Model Context Protocol (MCP) for the Agent Tool Plane?
--------------------------------------------------------------------------------
1. Decoupling Agent Reasoning from Tool Execution:
   In traditional naive agent frameworks, Python functions are directly imported
   and invoked inside the agent runtime. This creates catastrophic issues:
   - Dependency bloat: The agent process needs SQL drivers, browser binaries,
     pandas, PDF renderers, and custom C-libraries in a single bloated image.
   - Security blast radius: If an agent runs untrusted code or a malicious query,
     it shares memory and environment variables (including API keys) with the supervisor.
   - Inflexibility: Tool implementations cannot be independently scaled or deployed
     in separate secure pods/sandboxes.

2. The MCP Protocol Standard (Anthropic / Open Protocol):
   MCP defines an open, JSON-RPC 2.0 protocol over stdio or HTTP/SSE:
   - Tools are discovered dynamically via `tools/list`.
   - Tools are invoked via `tools/call` with strict JSON Schema argument validation.
   - Servers return standardized content blocks (`text`, `image`, `resource`).
   - Every agent interacts with tools uniformly through standard RPC client calls.
================================================================================
"""

from enum import IntEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class MCPErrorCode(IntEnum):
    """Standard JSON-RPC 2.0 and MCP protocol error codes."""

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    TOOL_EXECUTION_ERROR = -32000
    UNAUTHORIZED = -32001


class JSONRPCError(BaseModel):
    """JSON-RPC 2.0 error payload."""

    code: int = Field(description="Numeric error code")
    message: str = Field(description="Human-readable error description")
    data: Any | None = Field(default=None, description="Structured error diagnostics or traceback")


class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 request envelope sent from MCP client to server."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int = Field(description="Unique request correlation identifier")
    method: str = Field(description="Remote procedure name (e.g. 'tools/list', 'tools/call')")
    params: dict[str, Any] = Field(default_factory=dict, description="Method-specific arguments")


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 response envelope returned from MCP server to client."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int = Field(description="Matching request correlation identifier")
    result: Any | None = Field(default=None, description="Method execution result payload")
    error: JSONRPCError | None = Field(default=None, description="Error payload if method failed")

    @property
    def is_success(self) -> bool:
        """Helper checking if response contains a valid result without error."""
        return self.error is None


# =============================================================================
# Tool Schema Models (FR-MCP-2)
# =============================================================================


class MCPToolInputSchema(BaseModel):
    """JSON Schema definition describing the expected input parameters of an MCP tool."""

    type: Literal["object"] = "object"
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary mapping parameter names to JSON schema property definitions",
    )
    required: list[str] = Field(
        default_factory=list,
        description="List of mandatory argument names required for execution",
    )


class MCPToolDefinition(BaseModel):
    """Declaration of an available capability exposed by an MCP server."""

    name: str = Field(description="Unique tool identifier (e.g. 'search_web', 'run_sql_query')")
    description: str = Field(description="Detailed natural language description of tool behavior")
    inputSchema: MCPToolInputSchema = Field(
        default_factory=MCPToolInputSchema,
        description="JSON schema defining parameters accepted by this tool",
    )


class MCPListToolsResult(BaseModel):
    """Result payload returned by `tools/list`."""

    tools: list[MCPToolDefinition] = Field(
        default_factory=list,
        description="List of tools available on the target MCP server",
    )


# =============================================================================
# Tool Execution Models (FR-MCP-7)
# =============================================================================


class MCPTextContent(BaseModel):
    """Plain-text content block returned by an MCP tool invocation."""

    type: Literal["text"] = "text"
    text: str = Field(description="Textual output from the tool execution")


class MCPImageContent(BaseModel):
    """Base64-encoded image content block (e.g. generated chart or visualization)."""

    type: Literal["image"] = "image"
    data: str = Field(description="Base64-encoded binary image data")
    mimeType: str = Field(default="image/png", description="MIME content type")


class MCPToolCallResult(BaseModel):
    """Structured response from `tools/call` containing output blocks and status."""

    content: list[MCPTextContent | MCPImageContent] = Field(
        default_factory=list,
        description="Ordered sequence of output content items returned by the tool",
    )
    isError: bool = Field(
        default=False,
        description="True if tool execution encountered a domain error or exception",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metadata (latency, cost, source provenance, token consumption)",
    )

    def get_text_output(self) -> str:
        """Concatenate all text content blocks into a single string."""
        return "\n".join(
            block.text for block in self.content if isinstance(block, MCPTextContent)
        )
