"""Data models for SQL Analytics MCP Server (FR-SQL-1 to FR-SQL-4)."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class SQLColumnInfo(BaseModel):
    """Metadata describing a database column."""

    name: str = Field(description="Column identifier")
    type_name: str = Field(description="Data type (e.g. VARCHAR, INTEGER, TIMESTAMP)")
    nullable: bool = Field(default=True, description="Whether NULL values are permitted")
    primary_key: bool = Field(default=False, description="Whether column belongs to primary key")
    description: str | None = Field(default=None, description="Semantic or business description")


class SQLForeignKeyInfo(BaseModel):
    """Metadata describing a foreign key relationship across tables."""

    constrained_columns: list[str] = Field(description="Local column names in this table")
    referred_table: str = Field(description="Foreign target table name")
    referred_columns: list[str] = Field(description="Foreign target column names")


class SQLTableInfo(BaseModel):
    """Metadata describing a database table or view."""

    name: str = Field(description="Table or view name")
    schema_name: str | None = Field(default=None, description="Database schema/namespace")
    columns: list[SQLColumnInfo] = Field(
        default_factory=list, description="Ordered column definitions"
    )
    foreign_keys: list[SQLForeignKeyInfo] = Field(
        default_factory=list, description="Outbound foreign key relationships"
    )
    description: str | None = Field(
        default=None, description="Table summary or business description"
    )


class SQLIntrospectionResponse(BaseModel):
    """Structured response from the `sql_introspect_schema` tool."""

    dialect: str = Field(description="Database dialect (e.g. postgresql, sqlite, mysql)")
    tables: list[SQLTableInfo] = Field(
        default_factory=list, description="Available tables and schemas"
    )
    total_tables: int = Field(default=0, description="Count of introspected tables")


class SQLValidationResult(BaseModel):
    """Result of static AST parsing and security validation (FR-SQL-4)."""

    is_valid: bool = Field(description="Whether query passes security rules")
    query_type: Literal["READ", "WRITE", "UNKNOWN"] = Field(
        default="UNKNOWN", description="Classification of the SQL statement"
    )
    sanitized_sql: str = Field(description="Cleaned SQL with mandatory LIMIT injected if missing")
    reason: str = Field(default="", description="Validation failure explanation if invalid")
    limit_injected: bool = Field(
        default=False, description="Whether LIMIT was automatically appended"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Non-fatal warnings or cost advisories"
    )


class SQLExecutionResult(BaseModel):
    """Structured response from query execution."""

    query: str = Field(description="Executed SQL string")
    columns: list[str] = Field(default_factory=list, description="Result column header names")
    rows: list[dict[str, Any]] = Field(
        default_factory=list, description="Result records formatted as dictionaries"
    )
    row_count: int = Field(default=0, description="Number of rows returned")
    execution_time_ms: float = Field(
        default=0.0, description="Server execution time in milliseconds"
    )
    is_truncated: bool = Field(default=False, description="Whether result set exceeded max row cap")
    is_error: bool = Field(default=False, description="True if query execution raised an error")
    error_message: str | None = Field(
        default=None, description="Database error traceback if failed"
    )
