"""Database schema introspection and semantic layer extraction (FR-SQL-2).

================================================================================
EDUCATIONAL ARCHITECTURE NOTES:
Why Schema Introspection & Schema Linking are Essential for Text-to-SQL
--------------------------------------------------------------------------------
1. Preventing Hallucinated Table & Column Names:
   - LLMs trained on open data have strong assumptions (e.g. assuming a table
     is named `users` with a `created_at` column).
   - In real enterprise databases, tables may be named `t_usr_dim` and timestamps
     named `dw_insert_ts`.
   - Dynamic introspection reflects the exact current database catalog so the
     agent generates precise SQL.

2. Schema Linking (Token Efficiency & Attention Focus) (FR-SQL-2):
   - In production databases with 100+ tables, dumping the entire DDL into the
     LLM prompt wastes 20,000+ tokens and degrades accuracy ("lost in the middle").
   - `schema_linking()` matches the user's natural language question keywords
     against table names, column names, and business descriptions, selecting only
     the 2-5 relevant tables required for the query.
================================================================================
"""

from sqlalchemy import Engine, inspect

from libs.common.logging import get_logger
from services.mcp_servers.sql.models import (
    SQLColumnInfo,
    SQLForeignKeyInfo,
    SQLIntrospectionResponse,
    SQLTableInfo,
)

logger = get_logger("mcp.sql.introspector")


class SQLSchemaIntrospector:
    """Introspects relational database schemas and formats them for LLM schema linking."""

    def __init__(self, engine: Engine, custom_descriptions: dict[str, str] | None = None) -> None:
        """Initialize introspector with a SQLAlchemy engine.

        Args:
            engine: Connected SQLAlchemy database engine.
            custom_descriptions: Optional semantic layer mapping table names to business summaries.
        """
        self.engine = engine
        self.custom_descriptions = custom_descriptions or {}
        self._cached_schema: SQLIntrospectionResponse | None = None

    def introspect(self, force_refresh: bool = False) -> SQLIntrospectionResponse:
        """Reflect tables, columns, data types, and foreign keys from the database.

        Args:
            force_refresh: True to bypass cache and re-query the database.

        Returns:
            SQLIntrospectionResponse containing all reflected tables.
        """
        if self._cached_schema is not None and not force_refresh:
            return self._cached_schema

        dialect_name = self.engine.dialect.name
        inspector = inspect(self.engine)
        table_names = inspector.get_table_names()

        tables: list[SQLTableInfo] = []

        for table_name in table_names:
            # 1. Fetch Columns
            raw_columns = inspector.get_columns(table_name)
            columns: list[SQLColumnInfo] = []
            for col in raw_columns:
                columns.append(
                    SQLColumnInfo(
                        name=col["name"],
                        type_name=str(col["type"]),
                        nullable=bool(col.get("nullable", True)),
                        primary_key=bool(col.get("primary_key", False)),
                        description=col.get("comment"),
                    )
                )

            # 2. Fetch Foreign Keys
            raw_fks = inspector.get_foreign_keys(table_name)
            foreign_keys: list[SQLForeignKeyInfo] = []
            for fk in raw_fks:
                foreign_keys.append(
                    SQLForeignKeyInfo(
                        constrained_columns=fk.get("constrained_columns", []),
                        referred_table=fk.get("referred_table", ""),
                        referred_columns=fk.get("referred_columns", []),
                    )
                )

            table_desc = self.custom_descriptions.get(table_name)
            tables.append(
                SQLTableInfo(
                    name=table_name,
                    schema_name=None,
                    columns=columns,
                    foreign_keys=foreign_keys,
                    description=table_desc,
                )
            )

        response = SQLIntrospectionResponse(
            dialect=dialect_name,
            tables=tables,
            total_tables=len(tables),
        )
        self._cached_schema = response
        logger.info(
            "Completed database schema introspection",
            dialect=dialect_name,
            table_count=len(tables),
            tables=[t.name for t in tables],
        )
        return response

    def schema_linking(self, question: str, max_tables: int = 5) -> list[SQLTableInfo]:
        """Filter schema tables to only those relevant to the user question (FR-SQL-2).

        Args:
            question: User inquiry string.
            max_tables: Maximum number of tables to include.

        Returns:
            List of prioritized SQLTableInfo candidates.
        """
        schema = self.introspect()
        if not schema.tables:
            return []

        lower_q = question.lower()
        scored_tables: list[tuple[float, SQLTableInfo]] = []

        for table in schema.tables:
            score = 0.0
            table_lower = table.name.lower()

            # Direct table name match
            if table_lower in lower_q:
                score += 3.0

            # Partial word overlap with table name
            for word in table_lower.replace("_", " ").split():
                if len(word) > 2 and word in lower_q:
                    score += 1.5

            # Column matches
            for col in table.columns:
                col_lower = col.name.lower()
                if col_lower in lower_q:
                    score += 1.0

            # Description matches
            if table.description and any(w in lower_q for w in table.description.lower().split()):
                score += 1.0

            scored_tables.append((score, table))

        # Sort by relevance score descending
        scored_tables.sort(key=lambda x: x[0], reverse=True)

        # If no tables matched specifically, return the top N default tables
        selected = [t for _, t in scored_tables if _ > 0][:max_tables]
        if not selected:
            selected = [t for _, t in scored_tables][:max_tables]

        return selected

    def format_schema_for_prompt(self, tables: list[SQLTableInfo] | None = None) -> str:
        """Format table schemas into compact, token-efficient DDL blocks for LLM prompt context.

        Args:
            tables: Optional subset of tables to format. If None, formats all introspected tables.

        Returns:
            Formatted string describing tables, columns, primary keys, and relationships.
        """
        if tables is None:
            schema = self.introspect()
            tables = schema.tables

        lines: list[str] = []
        for table in tables:
            col_parts: list[str] = []
            for col in table.columns:
                pk_str = " PRIMARY KEY" if col.primary_key else ""
                null_str = "" if col.nullable else " NOT NULL"
                col_parts.append(f"  {col.name} {col.type_name}{pk_str}{null_str}")

            fk_parts: list[str] = []
            for fk in table.foreign_keys:
                if fk.referred_table and fk.constrained_columns:
                    fk_parts.append(
                        f"  FOREIGN KEY ({', '.join(fk.constrained_columns)}) REFERENCES {fk.referred_table}({', '.join(fk.referred_columns)})"
                    )

            all_defs = col_parts + fk_parts
            desc_comment = f" -- {table.description}" if table.description else ""
            lines.append(f"CREATE TABLE {table.name} ({desc_comment}")
            lines.append(",\n".join(all_defs))
            lines.append(");")

        return "\n\n".join(lines)
