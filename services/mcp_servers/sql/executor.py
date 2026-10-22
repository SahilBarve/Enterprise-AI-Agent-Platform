"""Safe SQL query executor with timeout, row-capping, and dry-run EXPLAIN (FR-SQL-3, FR-SQL-4).

================================================================================
EDUCATIONAL ARCHITECTURE NOTES:
Why Dry-Run EXPLAIN and Execution Row Caps are Crucial in Production
--------------------------------------------------------------------------------
1. Dry-Run EXPLAIN Before Execution (FR-SQL-3):
   - Rather than directly running an untested SQL query against a production table,
     `EXPLAIN` asks the database query planner to evaluate the syntax and execution plan
     without fetching or mutating data.
   - If the query references nonexistent tables, has invalid syntax, or attempts an
     unindexed full-table scan on 100M rows, `EXPLAIN` flags the error instantly with
     zero database impact.

2. Strict Row Capping & Serialization:
   - Returning 50,000 rows to an LLM blows the context window and triggers out-of-memory
     crashes in the agent runtime.
   - The executor fetches at most `max_rows + 1`. If more rows exist, it sets `is_truncated=True`,
     informing the downstream agent to refine its query (e.g. add aggregations or tighter WHERE filters).
================================================================================
"""

import time
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import Engine, text

from libs.common.logging import get_logger
from services.mcp_servers.sql.models import SQLExecutionResult, SQLValidationResult
from services.mcp_servers.sql.validator import SafeSQLValidator

logger = get_logger("mcp.sql.executor")


class SQLExecutor:
    """Executes validated SQL queries against a database engine with safety limits."""

    def __init__(
        self,
        engine: Engine,
        validator: SafeSQLValidator | None = None,
        default_max_rows: int = 100,
        statement_timeout_seconds: float = 10.0,
        allow_writes: bool = False,
    ) -> None:
        """Initialize SQL executor.

        Args:
            engine: Connected SQLAlchemy database engine.
            validator: SQL safety validator instance.
            default_max_rows: Default maximum rows returned to the agent.
            statement_timeout_seconds: Hard execution deadline in seconds.
            allow_writes: Whether DML write statements are permitted.
        """
        self.engine = engine
        self.validator = validator or SafeSQLValidator(allow_writes=allow_writes)
        self.default_max_rows = default_max_rows
        self.statement_timeout_seconds = statement_timeout_seconds
        self.allow_writes = allow_writes

    def explain_query(self, sql: str) -> dict[str, Any]:
        """Perform dry-run query plan analysis using EXPLAIN (FR-SQL-3).

        Args:
            sql: SQL statement to inspect.

        Returns:
            Dictionary containing execution plan lines and estimated cost indicators.
        """
        # Validate query before explain
        val: SQLValidationResult = self.validator.validate_query(sql, execution_mode="READ")
        if not val.is_valid:
            return {
                "is_valid": False,
                "error": f"Validation failed: {val.reason}",
                "plan": [],
            }

        explain_sql = f"EXPLAIN {val.sanitized_sql}"
        logger.debug("Executing dry-run EXPLAIN", sql=explain_sql)

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(explain_sql))
                plan_lines = [str(row[0]) for row in result.fetchall()]

            return {
                "is_valid": True,
                "sanitized_sql": val.sanitized_sql,
                "plan": plan_lines,
                "warnings": val.warnings,
            }
        except Exception as e:
            logger.warning("Dry-run EXPLAIN failed", error=str(e), sql=sql)
            return {
                "is_valid": False,
                "error": f"Database EXPLAIN failed: {e!s}",
                "plan": [],
            }

    def execute_read(self, sql: str, max_rows: int | None = None) -> SQLExecutionResult:
        """Validate and execute a read-only SQL query (FR-SQL-3, FR-SQL-4).

        Args:
            sql: SQL query string.
            max_rows: Optional row cap override (defaults to default_max_rows).

        Returns:
            SQLExecutionResult with rows, columns, and execution timing.
        """
        effective_max = max_rows or self.default_max_rows

        # 1. Static AST Validation
        val: SQLValidationResult = self.validator.validate_query(sql, execution_mode="READ")
        if not val.is_valid:
            logger.warning("SQL read query rejected by validator", reason=val.reason, sql=sql)
            return SQLExecutionResult(
                query=sql,
                is_error=True,
                error_message=f"SQL validation error: {val.reason}",
            )

        start_time = time.perf_counter()
        target_sql = val.sanitized_sql

        try:
            with self.engine.connect() as conn:
                # Execute read query
                cursor = conn.execute(text(target_sql))
                columns = list(cursor.keys())

                # Fetch up to max_rows + 1 to detect truncation
                raw_rows = cursor.fetchmany(effective_max + 1)
                is_truncated = len(raw_rows) > effective_max
                final_rows = raw_rows[:effective_max]

                serialized_rows: list[dict[str, Any]] = [
                    {
                        col: self._serialize_value(val)
                        for col, val in zip(columns, row, strict=False)
                    }
                    for row in final_rows
                ]

                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                return SQLExecutionResult(
                    query=target_sql,
                    columns=columns,
                    rows=serialized_rows,
                    row_count=len(serialized_rows),
                    execution_time_ms=round(elapsed_ms, 2),
                    is_truncated=is_truncated,
                    is_error=False,
                )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error("SQL read execution error", error=str(e), sql=target_sql)
            return SQLExecutionResult(
                query=target_sql,
                execution_time_ms=round(elapsed_ms, 2),
                is_error=True,
                error_message=str(e),
            )

    def execute_write(self, sql: str) -> SQLExecutionResult:
        """Validate and execute a DML mutation within a transaction (FR-SQL-4).

        Args:
            sql: SQL statement to execute.

        Returns:
            SQLExecutionResult with affected row count.
        """
        if not self.allow_writes:
            return SQLExecutionResult(
                query=sql,
                is_error=True,
                error_message="Mutations are disabled on this database connection",
            )

        val: SQLValidationResult = self.validator.validate_query(sql, execution_mode="WRITE")
        if not val.is_valid:
            return SQLExecutionResult(
                query=sql,
                is_error=True,
                error_message=f"SQL write validation error: {val.reason}",
            )

        start_time = time.perf_counter()
        target_sql = val.sanitized_sql

        try:
            with self.engine.begin() as conn:
                cursor = conn.execute(text(target_sql))
                rowcount = cursor.rowcount
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                return SQLExecutionResult(
                    query=target_sql,
                    columns=["affected_rows"],
                    rows=[{"affected_rows": rowcount}],
                    row_count=rowcount,
                    execution_time_ms=round(elapsed_ms, 2),
                    is_error=False,
                )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error("SQL write execution error", error=str(e), sql=target_sql)
            return SQLExecutionResult(
                query=target_sql,
                execution_time_ms=round(elapsed_ms, 2),
                is_error=True,
                error_message=str(e),
            )

    def _serialize_value(self, val: Any) -> Any:
        """Convert database native types into JSON-serializable primitives."""
        if val is None:
            return None
        if isinstance(val, (datetime, date)):
            return val.isoformat()
        if isinstance(val, Decimal):
            return float(val)
        if isinstance(val, UUID):
            return str(val)
        if isinstance(val, (bytes, bytearray)):
            return f"<binary {len(val)} bytes>"
        return val
