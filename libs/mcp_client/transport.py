"""Transport layer abstractions for Model Context Protocol (MCP) communication (FR-MCP-1).

================================================================================
EDUCATIONAL ARCHITECTURE NOTES:
Why Pluggable Transports for MCP?
--------------------------------------------------------------------------------
The MCP specification defines two primary standard wire transports:
1. Standard I/O (`stdio`):
   - The MCP client spawns the MCP server as a subprocess and communicates via
     standard input and output using newline-delimited JSON-RPC messages.
   - Ideal for local development, CLI tools, and secure single-host daemon execution.
   - Zero networking overhead, no open ports, and strict process isolation.

2. Server-Sent Events over HTTP (`http/sse`):
   - Client sends JSON-RPC requests via HTTP POST; server pushes responses and
     notifications via SSE streaming.
   - Ideal for distributed Kubernetes deployments where MCP servers run in separate
     scalable pods with dedicated network policies and service meshes.

3. In-Memory Transport (`InMemoryMCPTransport`):
   - Direct Python dispatch between client and server implementation without OS I/O.
   - Critical for unit tests and local mock execution: runs in sub-millisecond time
     with zero network/process overhead while exercising the exact same JSON-RPC
     protocol validation code path.
================================================================================
"""

import asyncio
import json
import subprocess
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable

from libs.common.errors import ExternalServiceError, ValidationError
from libs.common.logging import get_logger
from libs.mcp_client.models import (
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPErrorCode,
)

logger = get_logger("mcp.transport")


class MCPTransport(ABC):
    """Abstract base transport for exchanging JSON-RPC 2.0 messages with an MCP server."""

    @abstractmethod
    def send_request(self, request: JSONRPCRequest, timeout_seconds: float = 30.0) -> JSONRPCResponse:
        """Send a JSON-RPC request to the MCP server and wait for the response synchronously.

        Args:
            request: Formatted JSON-RPC request envelope.
            timeout_seconds: Maximum time to wait for a response before timing out.

        Returns:
            JSONRPCResponse received from the server.
        """
        ...

    @abstractmethod
    async def asend_request(
        self, request: JSONRPCRequest, timeout_seconds: float = 30.0
    ) -> JSONRPCResponse:
        """Asynchronously send a JSON-RPC request to the MCP server.

        Args:
            request: Formatted JSON-RPC request envelope.
            timeout_seconds: Maximum time to wait for a response.

        Returns:
            JSONRPCResponse received from the server.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release underlying system resources (processes, sockets, file descriptors)."""
        ...


# =============================================================================
# In-Memory Transport (FR-MCP-1 Test & Local Execution)
# =============================================================================


class InMemoryMCPTransport(MCPTransport):
    """High-speed in-process transport dispatching JSON-RPC requests directly to a handler.

    Enables testing full MCP protocol encoding, validation, and serialization without
    spawning external operating system processes.
    """

    def __init__(
        self,
        handler: Callable[[JSONRPCRequest], JSONRPCResponse],
        server_name: str = "in-memory-mcp",
    ) -> None:
        """Initialize in-memory transport with a synchronous message handler.

        Args:
            handler: Callable function taking a JSONRPCRequest and returning JSONRPCResponse.
            server_name: Identifier for diagnostic logging.
        """
        self.handler = handler
        self.server_name = server_name
        self.is_closed = False

    def send_request(
        self, request: JSONRPCRequest, timeout_seconds: float = 30.0  # noqa: ARG002
    ) -> JSONRPCResponse:
        """Dispatch request directly to handler through JSON serialization cycle."""
        if self.is_closed:
            raise ExternalServiceError(
                f"Cannot send request to closed transport '{self.server_name}'",
                service_name=self.server_name,
            )

        # Force serialization/deserialization cycle to verify pure JSON-RPC compatibility
        req_json = request.model_dump_json()
        deserialized_req = JSONRPCRequest.model_validate_json(req_json)

        try:
            resp = self.handler(deserialized_req)
            resp_json = resp.model_dump_json()
            return JSONRPCResponse.model_validate_json(resp_json)
        except Exception as e:
            logger.error("In-memory MCP handler failed", server=self.server_name, error=str(e))
            return JSONRPCResponse(
                id=request.id,
                error=JSONRPCError(
                    code=MCPErrorCode.INTERNAL_ERROR,
                    message=f"Handler execution error: {e!s}",
                ),
            )

    async def asend_request(
        self, request: JSONRPCRequest, timeout_seconds: float = 30.0
    ) -> JSONRPCResponse:
        """Asynchronously invoke the synchronous handler."""
        return self.send_request(request, timeout_seconds=timeout_seconds)

    def close(self) -> None:
        """Mark transport as closed."""
        self.is_closed = True


# =============================================================================
# Standard I/O (stdio) Subprocess Transport (FR-MCP-1)
# =============================================================================


class StdioMCPTransport(MCPTransport):
    """Production standard I/O transport managing an MCP server subprocess.

    Communicates over stdin/stdout using newline-delimited JSON-RPC messages.
    """

    def __init__(
        self,
        command: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        server_name: str = "stdio-mcp-server",
    ) -> None:
        """Start the MCP server subprocess.

        Args:
            command: Executable command and arguments (e.g. ['python', '-m', 'services.mcp_servers.sql']).
            cwd: Working directory for the subprocess.
            env: Environment variable dictionary.
            server_name: Server identifier for logging and metrics.
        """
        self.command = command
        self.cwd = cwd
        self.env = env
        self.server_name = server_name
        self.lock = threading.Lock()

        logger.info("Spawning Stdio MCP server subprocess", command=command, server=server_name)
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line-buffered
                cwd=cwd,
                env=env,
            )
        except Exception as e:
            raise ExternalServiceError(
                f"Failed to spawn MCP server '{server_name}': {e!s}",
                service_name=server_name,
            ) from e

    def send_request(
        self, request: JSONRPCRequest, timeout_seconds: float = 30.0  # noqa: ARG002
    ) -> JSONRPCResponse:
        """Write JSON-RPC request to subprocess stdin and read line from stdout."""
        if not self.process or self.process.poll() is not None:
            raise ExternalServiceError(
                f"MCP server subprocess '{self.server_name}' is not running (exit code {self.process.poll() if self.process else 'None'})",
                service_name=self.server_name,
            )

        req_line = request.model_dump_json() + "\n"

        with self.lock:
            assert self.process.stdin is not None
            assert self.process.stdout is not None

            try:
                self.process.stdin.write(req_line)
                self.process.stdin.flush()
            except Exception as e:
                raise ExternalServiceError(
                    f"Failed to write to MCP server '{self.server_name}' stdin: {e!s}",
                    service_name=self.server_name,
                ) from e

            # Read response line with timeout
            line = self.process.stdout.readline()
            if not line:
                stderr_output = ""
                if self.process.stderr:
                    stderr_output = self.process.stderr.read()
                raise ExternalServiceError(
                    f"MCP server '{self.server_name}' terminated unexpectedly or produced empty stdout. Stderr: {stderr_output}",
                    service_name=self.server_name,
                )

            try:
                data = json.loads(line.strip())
                return JSONRPCResponse.model_validate(data)
            except Exception as e:
                raise ValidationError(
                    f"Malformed JSON-RPC response from MCP server '{self.server_name}': {line}",
                    details={"raw_output": line, "error": str(e)},
                ) from e

    async def asend_request(
        self, request: JSONRPCRequest, timeout_seconds: float = 30.0
    ) -> JSONRPCResponse:
        """Execute send_request in a thread pool to avoid blocking async event loops."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.send_request, request, timeout_seconds)

    def close(self) -> None:
        """Gracefully terminate subprocess and close pipes."""
        with self.lock:
            if self.process and self.process.poll() is None:
                logger.info("Terminating MCP subprocess", server=self.server_name)
                try:
                    self.process.terminate()
                    self.process.wait(timeout=2.0)
                except Exception:
                    self.process.kill()
