"""Model Context Protocol (MCP) server for SQL Analytics (FR-SQL-1 to FR-SQL-4, FR-MCP-1).

================================================================================
EDUCATIONAL ARCHITECTURE NOTES:
Why Expose SQL Analytics via an Independent MCP Server?
--------------------------------------------------------------------------------
1. Centralized Data Access Boundary:
   - In traditional architectures, database connection strings and credentials
     are spread across multiple agent containers.
   - By running SQL Analytics as an independent MCP server:
     a) Database credentials stay isolated inside this single microservice.
     b) Network policies allow only this MCP server pod to connect to PostgreSQL/RDS.
     c) The agent supervisor runtime has zero database connection libraries or credentials.

2. Enforcing the P0 Governance Gate on Write Actions:
   - `sql_execute_read` allows fast, safe read-only queries.
   - `sql_execute_write` is flagged as a Tier 2 High-Risk action.
   - When the agent calls `sql_execute_write`, the `MCPClient` verifies that a signed
     approval token bound to the exact query hash is present before dispatching.
================================================================================
"""

import json
import sys
from typing import Any

from sqlalchemy import Engine, create_engine

from libs.common.logging import get_logger
from libs.mcp_client.models import (
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPErrorCode,
    MCPListToolsResult,
    MCPTextContent,
    MCPToolCallResult,
    MCPToolDefinition,
    MCPToolInputSchema,
)
from libs.mcp_client.transport import InMemoryMCPTransport
from services.mcp_servers.sql.executor import SQLExecutor
from services.mcp_servers.sql.introspector import SQLSchemaIntrospector
from services.mcp_servers.sql.models import SQLExecutionResult, SQLIntrospectionResponse
from services.mcp_servers.sql.validator import SafeSQLValidator

logger = get_logger("mcp.sql.server")


class SQLAnalyticsMCPServer:
    """Production MCP Server exposing SQL introspection, dry-run explain, and query execution."""

    def __init__(
        self,
        engine: Engine | None = None,
        db_url: str | None = None,
        allow_writes: bool = False,
    ) -> None:
        """Initialize the SQL Analytics MCP server.

        Args:
            engine: Optional pre-configured SQLAlchemy Engine.
            db_url: Optional database connection URL (defaults to in-memory SQLite if none).
            allow_writes: Whether DML write statements are permitted.
        """
        if engine is not None:
            self.engine = engine
        elif db_url is not None:
            self.engine = create_engine(db_url)
        else:
            # Default to in-memory SQLite for testing/standalone dev
            self.engine = create_engine("sqlite:///:memory:")

        self.validator = SafeSQLValidator(allow_writes=allow_writes)
        self.introspector = SQLSchemaIntrospector(self.engine)
        self.executor = SQLExecutor(
            self.engine, validator=self.validator, allow_writes=allow_writes
        )

        # Declare available MCP tool definitions (FR-MCP-2)
        self.tools: dict[str, MCPToolDefinition] = {
            "sql_introspect_schema": MCPToolDefinition(
                name="sql_introspect_schema",
                description=(
                    "Reflect the relational database catalog: discover available tables, columns, "
                    "data types, primary keys, and foreign key relationships for schema linking."
                ),
                inputSchema=MCPToolInputSchema(
                    properties={},
                    required=[],
                ),
            ),
            "sql_explain_query": MCPToolDefinition(
                name="sql_explain_query",
                description=(
                    "Perform a dry-run EXPLAIN on a SQL statement to inspect the execution plan, "
                    "verify syntax validity, and catch unindexed full-table scans before execution."
                ),
                inputSchema=MCPToolInputSchema(
                    properties={
                        "sql": {
                            "type": "string",
                            "description": "SQL query to analyze via dry-run EXPLAIN",
                        },
                    },
                    required=["sql"],
                ),
            ),
            "sql_execute_read": MCPToolDefinition(
                name="sql_execute_read",
                description=(
                    "Execute a validated read-only SQL query against the database. Automatically enforces "
                    "row caps, statement timeouts, and blocks any DDL or mutation attempts."
                ),
                inputSchema=MCPToolInputSchema(
                    properties={
                        "sql": {
                            "type": "string",
                            "description": "Safe SELECT SQL statement to execute",
                        },
                        "max_rows": {
                            "type": "integer",
                            "description": "Maximum rows to return (default: 100, capped at 1000)",
                            "default": 100,
                        },
                    },
                    required=["sql"],
                ),
            ),
            "sql_execute_write": MCPToolDefinition(
                name="sql_execute_write",
                description=(
                    "Execute a DML database mutation (INSERT, UPDATE, DELETE). This is a Tier 2 High-Risk "
                    "action strictly requiring a cryptographically signed approval token from human-in-the-loop governance."
                ),
                inputSchema=MCPToolInputSchema(
                    properties={
                        "sql": {
                            "type": "string",
                            "description": "DML statement to execute within an isolated transaction",
                        },
                    },
                    required=["sql"],
                ),
            ),
        }

    def handle_request(self, request: JSONRPCRequest) -> JSONRPCResponse:
        """Handle incoming JSON-RPC 2.0 requests."""
        method = request.method
        req_id = request.id
        params = request.params

        logger.debug("Received SQL MCP request", method=method, req_id=req_id)

        try:
            if method == "tools/list":
                tool_list = MCPListToolsResult(tools=list(self.tools.values()))
                return JSONRPCResponse(id=req_id, result=tool_list.model_dump())

            if method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})

                if not tool_name or tool_name not in self.tools:
                    return JSONRPCResponse(
                        id=req_id,
                        error=JSONRPCError(
                            code=MCPErrorCode.METHOD_NOT_FOUND,
                            message=f"Tool '{tool_name}' not found on SQL Analytics MCP server",
                        ),
                    )

                result = self._dispatch_tool(tool_name, arguments)
                return JSONRPCResponse(id=req_id, result=result.model_dump())

            return JSONRPCResponse(
                id=req_id,
                error=JSONRPCError(
                    code=MCPErrorCode.METHOD_NOT_FOUND,
                    message=f"Unsupported MCP method: '{method}'",
                ),
            )

        except Exception as e:
            logger.error("SQL MCP server error", error=str(e), method=method)
            return JSONRPCResponse(
                id=req_id,
                error=JSONRPCError(
                    code=MCPErrorCode.TOOL_EXECUTION_ERROR,
                    message=f"SQL server execution error: {e!s}",
                ),
            )

    def _dispatch_tool(self, tool_name: str, arguments: dict[str, Any]) -> MCPToolCallResult:
        """Route tool invocation to appropriate executor method."""
        if tool_name == "sql_introspect_schema":
            resp: SQLIntrospectionResponse = self.introspector.introspect()
            return MCPToolCallResult(
                content=[MCPTextContent(text=resp.model_dump_json(indent=2))],
                metadata={"total_tables": resp.total_tables, "dialect": resp.dialect},
            )

        if tool_name == "sql_explain_query":
            sql = arguments.get("sql", "")
            explain_dict = self.executor.explain_query(sql)
            is_err = not explain_dict.get("is_valid", False)
            return MCPToolCallResult(
                content=[MCPTextContent(text=json.dumps(explain_dict, indent=2))],
                isError=is_err,
                metadata={"sql": sql, "is_valid": not is_err},
            )

        if tool_name == "sql_execute_read":
            sql = arguments.get("sql", "")
            max_rows = int(arguments.get("max_rows", 100))
            exec_result: SQLExecutionResult = self.executor.execute_read(sql, max_rows=max_rows)
            return MCPToolCallResult(
                content=[MCPTextContent(text=exec_result.model_dump_json(indent=2))],
                isError=exec_result.is_error,
                metadata={
                    "row_count": exec_result.row_count,
                    "execution_time_ms": exec_result.execution_time_ms,
                    "is_truncated": exec_result.is_truncated,
                },
            )

        if tool_name == "sql_execute_write":
            sql = arguments.get("sql", "")
            write_result: SQLExecutionResult = self.executor.execute_write(sql)
            return MCPToolCallResult(
                content=[MCPTextContent(text=write_result.model_dump_json(indent=2))],
                isError=write_result.is_error,
                metadata={
                    "row_count": write_result.row_count,
                    "execution_time_ms": write_result.execution_time_ms,
                },
            )

        raise ValueError(f"Unknown tool: {tool_name}")

    def get_in_memory_transport(self) -> InMemoryMCPTransport:
        """Return high-performance in-memory transport connected directly to this server instance."""
        return InMemoryMCPTransport(
            handler=self.handle_request,
            server_name="sql-analytics-mcp",
        )

    def run_stdio(self) -> None:
        """Run standard I/O message loop for stdio subprocess transport."""
        logger.info("SQL Analytics MCP server started in stdio mode")
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                raw_req = json.loads(line)
                request = JSONRPCRequest.model_validate(raw_req)
                response = self.handle_request(request)
                sys.stdout.write(response.model_dump_json() + "\n")
                sys.stdout.flush()
            except Exception as e:
                err_resp = JSONRPCResponse(
                    id="unknown",
                    error=JSONRPCError(code=MCPErrorCode.PARSE_ERROR, message=str(e)),
                )
                sys.stdout.write(err_resp.model_dump_json() + "\n")
                sys.stdout.flush()


if __name__ == "__main__":
    server = SQLAnalyticsMCPServer()
    server.run_stdio()
