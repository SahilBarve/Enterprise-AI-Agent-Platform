"""Unit tests for SQL Analytics Agent with schema linking and self-correction (FR-SQL-3, FR-SQL-5, FR-SQL-8)."""

from sqlalchemy import Column, Float, Integer, MetaData, String, Table, create_engine

from libs.agents.sql_analytics import SQLAnalyticsAgent
from libs.mcp_client.client import MCPClient
from services.mcp_servers.sql.server import SQLAnalyticsMCPServer


def test_sql_agent_end_to_end_read_pipeline() -> None:
    """Verify that SQLAnalyticsAgent executes Text-to-SQL end-to-end via MCP tools."""
    # 1. Setup in-memory test database
    engine = create_engine("sqlite:///:memory:")
    meta = MetaData()
    orders_t = Table(
        "orders",
        meta,
        Column("order_id", Integer, primary_key=True),
        Column("customer_name", String(50)),
        Column("amount", Float),
        Column("status", String(20)),
    )
    meta.create_all(engine)

    with engine.begin() as conn:
        conn.execute(
            orders_t.insert(),
            [
                {
                    "order_id": 101,
                    "customer_name": "Acme Corp",
                    "amount": 1250.50,
                    "status": "completed",
                },
                {"order_id": 102, "customer_name": "Globex", "amount": 850.00, "status": "pending"},
                {
                    "order_id": 103,
                    "customer_name": "Initech",
                    "amount": 420.00,
                    "status": "completed",
                },
            ],
        )

    # 2. Spin up MCP Server and MCP Client
    server = SQLAnalyticsMCPServer(engine=engine, allow_writes=False)
    transport = server.get_in_memory_transport()
    client = MCPClient(transport=transport, server_name="sql-agent-test")

    # 3. Instantiate Agent
    agent = SQLAnalyticsAgent(mcp_client=client, max_correction_retries=2)

    # 4. Execute analytical query
    result = agent.query("How many orders are in the database?")

    assert "orders" in result.question.lower()
    assert result.row_count >= 1
    assert "count" in result.sql_query.lower() or "orders" in result.sql_query.lower()
    assert "Analytical Query Insights" in result.summary_insights
    assert result.execution_time_ms > 0
    assert result.retries_attempted == 0

    client.close()


def test_sql_agent_self_correction_on_nonexistent_table() -> None:
    """Verify that agent self-corrects when initial SQL references a missing table (FR-SQL-5)."""
    engine = create_engine("sqlite:///:memory:")
    meta = MetaData()
    users_t = Table(
        "customers",
        meta,
        Column("id", Integer, primary_key=True),
        Column("name", String(50)),
    )
    meta.create_all(engine)

    with engine.begin() as conn:
        conn.execute(users_t.insert(), [{"id": 1, "name": "Jane"}])

    server = SQLAnalyticsMCPServer(engine=engine, allow_writes=False)
    transport = server.get_in_memory_transport()
    client = MCPClient(transport=transport, server_name="sql-repair-test")

    # Subclass agent to deliberately produce an erroneous table name on attempt 1
    class FaultySQLAgent(SQLAnalyticsAgent):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]
            self.attempt = 0

        def _generate_sql(self, question: str, schema_context: str) -> str:
            self.attempt += 1
            if self.attempt == 1:
                return "SELECT * FROM missing_clients_table"
            return super()._generate_sql(question, schema_context)

    agent = FaultySQLAgent(mcp_client=client, max_correction_retries=2)

    result = agent.query("List all customers")

    # Should have recovered via self-correction loop!
    assert result.retries_attempted >= 1
    assert result.row_count == 1

    client.close()


def test_sql_agent_chart_recommendation_logic() -> None:
    """Verify chart type selection logic based on column schema and records (FR-SQL-8)."""
    engine = create_engine("sqlite:///:memory:")
    server = SQLAnalyticsMCPServer(engine=engine)
    client = MCPClient(transport=server.get_in_memory_transport(), server_name="chart-test")
    agent = SQLAnalyticsAgent(mcp_client=client)

    # 1. Time series data -> line chart
    chart_line = agent._recommend_chart(
        columns=["month", "revenue"],
        rows=[{"month": "2026-01", "revenue": 1000.0}],
    )
    assert chart_line == "line"

    # 2. Category + numeric -> bar chart
    chart_bar = agent._recommend_chart(
        columns=["category", "count"],
        rows=[{"category": "electronics", "count": 42}],
    )
    assert chart_bar == "bar"

    # 3. Only textual / single column -> table
    chart_table = agent._recommend_chart(
        columns=["name"],
        rows=[{"name": "Alice"}],
    )
    assert chart_table == "table"

    client.close()
