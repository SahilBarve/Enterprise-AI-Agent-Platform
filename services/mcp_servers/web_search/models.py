"""Data models for Web Search and Content Extraction MCP Server (FR-WEB-1, FR-WEB-6)."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class WebSearchResultItem(BaseModel):
    """Individual search result item from an external search provider."""

    title: str = Field(description="Webpage title")
    url: str = Field(description="Target HTTP/HTTPS URL")
    snippet: str = Field(description="Summary excerpt from search engine")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Provider relevance score")
    published_date: str | None = Field(default=None, description="Published date if available")


class WebSearchResponse(BaseModel):
    """Structured response from the `web_search` tool."""

    query: str = Field(description="Search query string")
    results: list[WebSearchResultItem] = Field(
        default_factory=list, description="List of matching web items"
    )
    total_results: int = Field(default=0, description="Total matching items found")
    provider: str = Field(
        default="mock", description="Search provider name (duckduckgo, tavily, mock)"
    )


class WebPageFetchResponse(BaseModel):
    """Structured response from the `web_fetch` tool."""

    url: str = Field(description="Target URL fetched")
    title: str = Field(description="Extracted page title")
    content: str = Field(
        description="Cleaned, sanitized plain text or markdown extracted from HTML"
    )
    raw_length: int = Field(description="Character length of extracted content")
    fetched_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(), description="ISO timestamp"
    )
    is_safe: bool = Field(default=True, description="SSRF and security check result")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Extracted meta tags and headers"
    )
