"""Web Search and Content Extraction MCP Server package (FR-WEB-1, FR-WEB-2)."""

from services.mcp_servers.web_search.models import (
    WebPageFetchResponse,
    WebSearchResponse,
    WebSearchResultItem,
)
from services.mcp_servers.web_search.security import is_safe_url, sanitize_web_content

__all__ = [
    "WebPageFetchResponse",
    "WebSearchResponse",
    "WebSearchResultItem",
    "is_safe_url",
    "sanitize_web_content",
]
