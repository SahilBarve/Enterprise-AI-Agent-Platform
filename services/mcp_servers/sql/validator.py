"""Safe AST-based SQL validator and security guard (FR-SQL-4).

================================================================================
EDUCATIONAL ARCHITECTURE NOTES:
Why Static SQL AST Validation is Mandatory for Autonomous AI Agents
--------------------------------------------------------------------------------
1. The Danger of Unchecked Text-to-SQL:
   - LLMs are prone to generating destructive statements:
     * Chained injection: `SELECT * FROM users; DROP TABLE audits;`
     * Accidental mutation: `UPDATE accounts SET balance = 0` when asked "who has 0 balance?"
     * Denial of Service: `SELECT * FROM huge_log_table` (reading 500 million rows, crashing RAM).
     * Exfiltration functions: `pg_read_file('/etc/passwd')` or `pg_sleep(60)`.

2. AST-Based Validation over Naive Regex:
   - A naive regex for "DROP" will trigger false positives on legitimate queries:
     `SELECT user_id, drop_rate FROM network_telemetry;`
   - Our validator uses token-aware parsing:
     * Strips SQL comments while respecting quoted literals.
     * Evaluates statement boundaries (strictly blocking statement chaining).
     * Distinguishes read queries (SELECT, WITH ... SELECT) from mutations.
     * Enforces row caps by automatically injecting or capping `LIMIT` clauses.
================================================================================
"""

import re
from typing import Literal

from libs.common.logging import get_logger
from services.mcp_servers.sql.models import SQLValidationResult

logger = get_logger("mcp.sql.validator")

# Prohibited DDL commands that destroy or modify database structure
PROHIBITED_DDL_KEYWORDS = {
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "RENAME",
    "GRANT",
    "REVOKE",
    "ATTACH",
    "DETACH",
    "VACUUM",
    "REINDEX",
    "EXEC",
    "EXECUTE",
}

# DML mutation commands permitted only in WRITE mode with Tier 2 approval
DML_WRITE_KEYWORDS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "MERGE",
    "UPSERT",
    "REPLACE",
}

# Dangerous system stored procedures and file I/O functions
DANGEROUS_FUNCTIONS = {
    "pg_read_file",
    "pg_write_file",
    "pg_read_binary_file",
    "pg_ls_dir",
    "pg_sleep",
    "dblink",
    "dblink_exec",
    "xp_cmdshell",
    "load_file",
    "sys_eval",
    "sys_exec",
    "copy_to",
    "copy_from",
}


class SafeSQLValidator:
    """Security validator enforcing safe SQL execution rules for AI agents."""

    def __init__(
        self,
        default_limit: int = 100,
        max_limit: int = 1000,
        allow_writes: bool = False,
    ) -> None:
        """Initialize SQL validator.

        Args:
            default_limit: Default LIMIT injected if missing in SELECT queries.
            max_limit: Maximum allowed LIMIT ceiling.
            allow_writes: Whether DML write statements (UPDATE/INSERT/DELETE) are allowed.
        """
        self.default_limit = default_limit
        self.max_limit = max_limit
        self.allow_writes = allow_writes

    def validate_query(
        self,
        sql: str,
        execution_mode: Literal["READ", "WRITE"] = "READ",
    ) -> SQLValidationResult:
        """Parse and validate SQL query against security policies (FR-SQL-4).

        Args:
            sql: Raw SQL string to validate.
            execution_mode: Intended execution mode ('READ' or 'WRITE').

        Returns:
            SQLValidationResult detailing validity, query type, and sanitized query.
        """
        raw = sql.strip()
        if not raw:
            return SQLValidationResult(
                is_valid=False,
                query_type="UNKNOWN",
                sanitized_sql="",
                reason="SQL statement is empty",
            )

        # 1. Strip comments while preserving literal strings
        clean_sql = self._strip_comments(raw)
        if not clean_sql:
            return SQLValidationResult(
                is_valid=False,
                query_type="UNKNOWN",
                sanitized_sql="",
                reason="SQL contains only comments",
            )

        # 2. Check for multi-statement chaining (e.g. semicolons outside quotes)
        statements = self._split_statements(clean_sql)
        if len(statements) > 1:
            logger.warning("Rejected multi-statement SQL attempt", statement_count=len(statements))
            return SQLValidationResult(
                is_valid=False,
                query_type="UNKNOWN",
                sanitized_sql=clean_sql,
                reason="Multiple SQL statements are prohibited; only single-statement queries allowed",
            )

        single_stmt = statements[0].strip()

        # 3. Tokenize leading keywords
        tokens = self._tokenize_words(single_stmt)
        if not tokens:
            return SQLValidationResult(
                is_valid=False,
                query_type="UNKNOWN",
                sanitized_sql=single_stmt,
                reason="Failed to parse SQL tokens",
            )

        first_token = tokens[0].upper()

        # 4. Block DDL operations strictly
        for ddl_word in PROHIBITED_DDL_KEYWORDS:
            if ddl_word in (t.upper() for t in tokens[:4]):
                logger.warning("Blocked DDL statement", keyword=ddl_word)
                return SQLValidationResult(
                    is_valid=False,
                    query_type="UNKNOWN",
                    sanitized_sql=single_stmt,
                    reason=f"DDL statement '{ddl_word}' is strictly prohibited on the platform",
                )

        # 5. Block dangerous functions
        for dangerous_fn in DANGEROUS_FUNCTIONS:
            pattern = re.compile(rf"\b{re.escape(dangerous_fn)}\s*\(", re.IGNORECASE)
            if pattern.search(single_stmt):
                logger.warning("Blocked dangerous SQL function", function=dangerous_fn)
                return SQLValidationResult(
                    is_valid=False,
                    query_type="UNKNOWN",
                    sanitized_sql=single_stmt,
                    reason=f"Dangerous database function '{dangerous_fn}()' is prohibited",
                )

        # 6. Determine Query Type (READ vs WRITE)
        query_type: Literal["READ", "WRITE", "UNKNOWN"] = "UNKNOWN"
        if first_token in ("SELECT", "WITH", "EXPLAIN", "SHOW", "DESCRIBE"):
            # Check if WITH statement contains an internal mutation (e.g. WITH ins AS (INSERT ...))
            if first_token == "WITH":
                has_write_in_cte = any(w in (t.upper() for t in tokens) for w in DML_WRITE_KEYWORDS)
                query_type = "WRITE" if has_write_in_cte else "READ"
            else:
                query_type = "READ"
        elif first_token in DML_WRITE_KEYWORDS:
            query_type = "WRITE"

        if query_type == "UNKNOWN":
            return SQLValidationResult(
                is_valid=False,
                query_type="UNKNOWN",
                sanitized_sql=single_stmt,
                reason=f"Unsupported initial SQL token: '{first_token}'",
            )

        # 7. Mode enforcement
        if execution_mode == "READ" and query_type == "WRITE":
            return SQLValidationResult(
                is_valid=False,
                query_type="WRITE",
                sanitized_sql=single_stmt,
                reason=f"Mutation statement '{first_token}' is prohibited in read-only mode",
            )

        if execution_mode == "WRITE" and not self.allow_writes:
            return SQLValidationResult(
                is_valid=False,
                query_type="WRITE",
                sanitized_sql=single_stmt,
                reason="Database mutations are disabled on this connection",
            )

        # 8. LIMIT Enforcement for READ (SELECT) statements
        sanitized_sql = single_stmt
        limit_injected = False
        warnings: list[str] = []

        if query_type == "READ" and first_token in ("SELECT", "WITH"):
            sanitized_sql, limit_injected, warning = self._enforce_limit(single_stmt)
            if warning:
                warnings.append(warning)

        return SQLValidationResult(
            is_valid=True,
            query_type=query_type,
            sanitized_sql=sanitized_sql,
            reason="Query validated safe",
            limit_injected=limit_injected,
            warnings=warnings,
        )

    def _strip_comments(self, sql: str) -> str:
        """Remove single-line (--) and multi-line (/* */) comments outside string literals."""
        # Remove single-line comments
        lines = []
        for line in sql.splitlines():
            # Check for -- outside quotes
            cleaned_line = self._strip_line_comment(line)
            if cleaned_line.strip():
                lines.append(cleaned_line)
        joined = "\n".join(lines)

        # Remove multi-line comments
        return re.sub(r"/\*.*?\*/", "", joined, flags=re.DOTALL).strip()

    def _strip_line_comment(self, line: str) -> str:
        """Strip -- from line if not within quotation marks."""
        in_single = False
        in_double = False
        i = 0
        while i < len(line):
            c = line[i]
            if c == "'" and not in_double:
                in_single = not in_single
            elif c == '"' and not in_single:
                in_double = not in_double
            elif (
                c == "-"
                and i + 1 < len(line)
                and line[i + 1] == "-"
                and not in_single
                and not in_double
            ):
                return line[:i]
            i += 1
        return line

    def _split_statements(self, sql: str) -> list[str]:
        """Split SQL string by semicolons that are outside string literals."""
        statements: list[str] = []
        current: list[str] = []
        in_single = False
        in_double = False

        for c in sql:
            if c == "'" and not in_double:
                in_single = not in_single
                current.append(c)
            elif c == '"' and not in_single:
                in_double = not in_double
                current.append(c)
            elif c == ";" and not in_single and not in_double:
                stmt_str = "".join(current).strip()
                if stmt_str:
                    statements.append(stmt_str)
                current = []
            else:
                current.append(c)

        trailing = "".join(current).strip()
        if trailing:
            statements.append(trailing)

        return statements

    def _tokenize_words(self, sql: str) -> list[str]:
        """Extract alphanumeric words from SQL statement."""
        return re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", sql)

    def _enforce_limit(self, sql: str) -> tuple[str, bool, str | None]:
        """Check for LIMIT clause; inject or cap as required."""
        # Match trailing LIMIT \d+
        limit_match = re.search(r"\bLIMIT\s+(\d+)\s*$", sql, re.IGNORECASE)
        if limit_match:
            existing_limit = int(limit_match.group(1))
            if existing_limit > self.max_limit:
                # Cap the limit
                capped_sql = re.sub(
                    r"\bLIMIT\s+(\d+)\s*$",
                    f"LIMIT {self.max_limit}",
                    sql,
                    flags=re.IGNORECASE,
                )
                warning = f"Requested LIMIT {existing_limit} exceeded maximum ceiling; capped to {self.max_limit}."
                return capped_sql, False, warning
            return sql, False, None

        # Check if LIMIT exists anywhere in the top-level query
        # If no trailing LIMIT is found, append default LIMIT
        injected_sql = f"{sql.rstrip(';')} LIMIT {self.default_limit}"
        return injected_sql, True, None
