"""Unit tests for SQL Analytics MCP Server, AST validator, and safe executor (FR-SQL-1 to FR-SQL-4, FR-MCP-1)."""

import json

from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine

from libs.guardrails.governance import GovernanceGate
from libs.mcp_client.client import MCPClient
from services.mcp_servers.sql.models import SQLExecutionResult, SQLIntrospectionResponse
from services.mcp_servers.sql.server import SQLAnalyticsMCPServer
from services.mcp_servers.sql.validator import SafeSQLValidator


def test_sql_validator_blocks_ddl_and_multi_statements() -> None:
    """Verify that validator rejects DDL operations and statement chaining (FR-SQL-4)."""
    validator = SafeSQLValidator()

    # 1. Multi-statement chaining
    res_multi = validator.validate_query("SELECT id FROM users; DROP TABLE audits;")
    assert not res_multi.is_valid
    assert "multiple" in res_multi.reason.lower()

    # 2. Direct DDL DROP
    res_drop = validator.validate_query("DROP TABLE users;")
    assert not res_drop.is_valid
    assert "prohibited" in res_drop.reason.lower()

    # 3. Direct DDL TRUNCATE
    res_trunc = validator.validate_query("TRUNCATE TABLE logs;")
    assert not res_trunc.is_valid
    assert "prohibited" in res_trunc.reason.lower()

    # 4. Dangerous file function
    res_fn = validator.validate_query("SELECT pg_read_file('/etc/passwd');")
    assert not res_fn.is_valid
    assert "dangerous" in res_fn.reason.lower()


def test_sql_validator_enforces_limit_injection_and_capping() -> None:
    """Verify automatic LIMIT injection and ceiling enforcement for SELECT queries (FR-SQL-4)."""
    validator = SafeSQLValidator(default_limit=50, max_limit=200)

    # 1. Injects default LIMIT 50 when missing
    res_missing = validator.validate_query("SELECT * FROM employees WHERE active = 1")
    assert res_missing.is_valid
    assert res_missing.limit_injected is True
    assert "LIMIT 50" in res_missing.sanitized_sql

    # 2. Preserves valid user LIMIT
    res_valid = validator.validate_query("SELECT * FROM employees LIMIT 20")
    assert res_valid.is_valid
    assert res_valid.limit_injected is False
    assert "LIMIT 20" in res_valid.sanitized_sql

    # 3. Caps excessively large LIMIT
    res_excessive = validator.validate_query("SELECT * FROM employees LIMIT 5000")
    assert res_excessive.is_valid
    assert "LIMIT 200" in res_excessive.sanitized_sql
    assert len(res_excessive.warnings) > 0


def test_sql_validator_differentiates_read_and_write() -> None:
    """Verify that mutation statements are rejected in READ mode."""
    validator = SafeSQLValidator(allow_writes=True)

    # Read statement in READ mode
    res_read = validator.validate_query("SELECT count(*) FROM orders", execution_mode="READ")
    assert res_read.is_valid
    assert res_read.query_type == "READ"

    # Mutation statement in READ mode
    res_mut = validator.validate_query(
        "UPDATE users SET active = 0 WHERE id = 1", execution_mode="READ"
    )
    assert not res_mut.is_valid
    assert "mutation" in res_mut.reason.lower()

    # Mutation statement in WRITE mode
    res_write = validator.validate_query(
        "UPDATE users SET active = 0 WHERE id = 1", execution_mode="WRITE"
    )
    assert res_write.is_valid
    assert res_write.query_type == "WRITE"


def test_mcp_sql_server_introspect_explain_and_read() -> None:
    """Verify MCP tools discovery, introspection, EXPLAIN, and read execution via in-memory transport."""
    # Setup test database with schema and seed data
    engine = create_engine("sqlite:///:memory:")
    meta = MetaData()
    users_t = Table(
        "users",
        meta,
        Column("id", Integer, primary_key=True),
        Column("username", String(50), nullable=False),
        Column("role", String(20), default="analyst"),
    )
    meta.create_all(engine)

    with engine.begin() as conn:
        conn.execute(
            users_t.insert(),
            [
                {"id": 1, "username": "alice", "role": "admin"},
                {"id": 2, "username": "bob", "role": "engineer"},
                {"id": 3, "username": "charlie", "role": "analyst"},
            ],
        )

    # Spin up MCP Server and MCP Client
    server = SQLAnalyticsMCPServer(engine=engine, allow_writes=True)
    transport = server.get_in_memory_transport()
    client = MCPClient(transport=transport, server_name="sql-test")

    # 1. Discover tools
    tools = client.list_tools()
    tool_names = {t.name for t in tools}
    assert "sql_introspect_schema" in tool_names
    assert "sql_explain_query" in tool_names
    assert "sql_execute_read" in tool_names
    assert "sql_execute_write" in tool_names

    # 2. Introspect schema (FR-SQL-2)
    intro_result = client.call_tool("sql_introspect_schema", {})
    intro_data = SQLIntrospectionResponse.model_validate_json(intro_result.get_text_output())
    assert intro_data.total_tables >= 1
    table_names = [t.name for t in intro_data.tables]
    assert "users" in table_names

    # 3. Dry-run EXPLAIN (FR-SQL-3)
    explain_result = client.call_tool(
        "sql_explain_query",
        {"sql": "SELECT username FROM users WHERE role = 'admin'"},
    )
    assert not explain_result.isError
    explain_payload = json.loads(explain_result.get_text_output())
    assert explain_payload.get("is_valid") is True
    assert len(explain_payload.get("plan", [])) > 0

    # 4. Execute Read Query (FR-SQL-3)
    read_result = client.call_tool(
        "sql_execute_read",
        {"sql": "SELECT id, username, role FROM users ORDER BY id ASC"},
    )
    assert not read_result.isError
    exec_data = SQLExecutionResult.model_validate_json(read_result.get_text_output())
    assert exec_data.row_count == 3
    assert exec_data.columns == ["id", "username", "role"]
    assert exec_data.rows[0]["username"] == "alice"

    client.close()


def test_mcp_sql_server_write_governance_enforcement() -> None:
    """Verify that write queries enforce the Governance Gate approval token (P0 Pillar)."""
    engine = create_engine("sqlite:///:memory:")
    meta = MetaData()
    Table("audit_log", meta, Column("id", Integer, primary_key=True), Column("event", String(50)))
    meta.create_all(engine)

    server = SQLAnalyticsMCPServer(engine=engine, allow_writes=True)
    gate = GovernanceGate()
    transport = server.get_in_memory_transport()
    client = MCPClient(transport=transport, server_name="sql-gov-test", governance_gate=gate)

    write_sql = "INSERT INTO audit_log (id, event) VALUES (1, 'deploy_triggered')"

    # 1. Unapproved call must raise GovernanceError (Tier 2 Action)
    import pytest

    from libs.common.errors import GovernanceError

    with pytest.raises(GovernanceError) as exc_info:
        client.call_tool(
            name="sql_execute_write",
            arguments={"sql": write_sql},
            approval_token=None,
        )
    assert "rejected by governance gate" in str(exc_info.value).lower()

    # 2. Issue cryptographic token and resume
    valid_token = gate.token_manager.issue_token(
        run_id="run-default",
        tenant_id="tenant-default",
        action="sql_execute_write",
        arguments={"sql": write_sql},
    )

    approved_result = client.call_tool(
        name="sql_execute_write",
        arguments={"sql": write_sql},
        approval_token=valid_token,
    )
    assert not approved_result.isError
    write_data = SQLExecutionResult.model_validate_json(approved_result.get_text_output())
    assert write_data.row_count == 1

    client.close()
