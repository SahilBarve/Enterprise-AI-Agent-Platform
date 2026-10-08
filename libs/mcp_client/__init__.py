"""Model Context Protocol (MCP) Client Library.

Provides client-side abstractions, pluggable transports, and tool registries
for interacting with independent Model Context Protocol (MCP) servers.
"""

from libs.mcp_client.client import MCPClient
from libs.mcp_client.models import (
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPErrorCode,
    MCPImageContent,
    MCPListToolsResult,
    MCPTextContent,
    MCPToolCallResult,
    MCPToolDefinition,
    MCPToolInputSchema,
)
from libs.mcp_client.registry import MCPToolRegistry
from libs.mcp_client.transport import (
    InMemoryMCPTransport,
    MCPTransport,
    StdioMCPTransport,
)

__all__ = [
    "InMemoryMCPTransport",
    "JSONRPCError",
    "JSONRPCRequest",
    "JSONRPCResponse",
    "MCPClient",
    "MCPErrorCode",
    "MCPImageContent",
    "MCPListToolsResult",
    "MCPTextContent",
    "MCPToolCallResult",
    "MCPToolDefinition",
    "MCPToolInputSchema",
    "MCPToolRegistry",
    "MCPTransport",
    "StdioMCPTransport",
]
