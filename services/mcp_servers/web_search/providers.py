"""Pluggable search providers for Web Search MCP Server (FR-WEB-1).

================================================================================
EDUCATIONAL ARCHITECTURE NOTES:
Why Pluggable Search Providers with Fallback?
--------------------------------------------------------------------------------
1. Decoupling from Single-Vendor Outages:
   - External search APIs (Tavily, SerpAPI, Brave) frequently experience rate limits,
     billing halts, or network degradation.
   - The `FallbackSearchProvider` chains multiple providers: if Tavily encounters
     a 429 rate limit or authentication failure, it automatically falls back to
     DuckDuckGo or the local mock provider.

2. Deterministic Testing & Offline Development:
   - Real search engines return nondeterministic results that change hour by hour,
     breaking test assertions.
   - The `MockSearchProvider` delivers predictable, instant, offline responses,
     allowing CI test suites to run in milliseconds without external HTTP calls.
================================================================================
"""

from abc import ABC, abstractmethod
from urllib.parse import quote_plus

import httpx

from libs.common.logging import get_logger
from services.mcp_servers.web_search.models import (
    WebSearchResponse,
    WebSearchResultItem,
)

logger = get_logger("mcp.web_search.providers")


class SearchProvider(ABC):
    """Abstract interface for web search engine adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        ...

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> WebSearchResponse:
        """Execute web search query.

        Args:
            query: Natural language or keyword search query.
            max_results: Maximum number of search result items to return.

        Returns:
            WebSearchResponse containing ranked items.
        """
        ...


class MockSearchProvider(SearchProvider):
    """Deterministic in-memory search provider for unit tests and local development."""

    def __init__(
        self, predefined_results: dict[str, list[WebSearchResultItem]] | None = None
    ) -> None:
        """Initialize mock provider with optional custom result mappings."""
        self._custom_results: dict[str, list[WebSearchResultItem]] = predefined_results or {}

    @property
    def name(self) -> str:
        return "mock"

    def register_mock_results(self, query_keyword: str, results: list[WebSearchResultItem]) -> None:
        """Register specific mock results returned when query matches keyword."""
        self._custom_results[query_keyword.lower()] = results

    def search(self, query: str, max_results: int = 5) -> WebSearchResponse:
        """Return matching mock results or generate synthetic domain-specific entries."""
        lower_query = query.lower()

        # Check registered results
        for kw, items in self._custom_results.items():
            if kw in lower_query:
                return WebSearchResponse(
                    query=query,
                    results=items[:max_results],
                    total_results=len(items[:max_results]),
                    provider=self.name,
                )

        # Generate realistic synthetic results based on query keywords
        results: list[WebSearchResultItem] = [
            WebSearchResultItem(
                title=f"Official Overview: {query.title()}",
                url=f"https://docs.enterprise-tech.org/{quote_plus(query.lower()[:30])}",
                snippet=f"Comprehensive documentation and authoritative overview regarding {query}. Details architecture, best practices, and enterprise usage.",
                score=0.95,
                published_date="2026-01-15",
            ),
            WebSearchResultItem(
                title=f"Deep Dive Analysis & Benchmarks: {query.title()}",
                url=f"https://tech-insights.io/reports/{quote_plus(query.lower()[:20])}",
                snippet=f"Industry benchmarks and comparative analysis for {query}. Highlights performance trade-offs, scaling limits, and operational metrics.",
                score=0.88,
                published_date="2026-03-20",
            ),
            WebSearchResultItem(
                title=f"Community Guide & Architecture Patterns for {query.title()}",
                url=f"https://dev-community.org/guides/{quote_plus(query.lower()[:20])}",
                snippet=f"Community guide covering implementation pitfalls, troubleshooting, and real-world production configurations for {query}.",
                score=0.79,
                published_date="2026-02-10",
            ),
        ]

        trimmed = results[:max_results]
        return WebSearchResponse(
            query=query,
            results=trimmed,
            total_results=len(trimmed),
            provider=self.name,
        )


class DuckDuckGoSearchProvider(SearchProvider):
    """DuckDuckGo HTML / Lite search engine provider."""

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self.timeout_seconds = timeout_seconds

    @property
    def name(self) -> str:
        return "duckduckgo"

    def search(self, query: str, max_results: int = 5) -> WebSearchResponse:
        """Search DuckDuckGo using lite HTML endpoint."""
        url = "https://html.duckduckgo.com/html/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {"q": query, "b": ""}

        try:
            with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True) as client:
                resp = client.post(url, headers=headers, data=data)
                resp.raise_for_status()

            from bs4 import BeautifulSoup

            soup = BeautifulSoup(resp.text, "html.parser")
            items: list[WebSearchResultItem] = []

            for result in soup.find_all("div", class_="result"):
                title_elem = result.find("a", class_="result__a")
                snippet_elem = result.find("a", class_="result__snippet")
                if not title_elem:
                    continue

                raw_url = str(title_elem.get("href", ""))
                # DuckDuckGo wraps URLs in ud= query parameter
                clean_url = raw_url
                if "uddg=" in raw_url:
                    from urllib.parse import parse_qs, unquote

                    qs = parse_qs(raw_url.split("?", 1)[-1])
                    if "uddg" in qs:
                        clean_url = unquote(qs["uddg"][0])

                title = title_elem.get_text(strip=True)
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                if clean_url.startswith("http"):
                    items.append(
                        WebSearchResultItem(
                            title=title,
                            url=clean_url,
                            snippet=snippet,
                            score=0.85,
                        )
                    )
                if len(items) >= max_results:
                    break

            return WebSearchResponse(
                query=query,
                results=items,
                total_results=len(items),
                provider=self.name,
            )
        except Exception as e:
            logger.warning("DuckDuckGo search failed", query=query, error=str(e))
            raise


class TavilySearchProvider(SearchProvider):
    """Tavily search API provider for AI agents."""

    def __init__(self, api_key: str, timeout_seconds: float = 10.0) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    @property
    def name(self) -> str:
        return "tavily"

    def search(self, query: str, max_results: int = 5) -> WebSearchResponse:
        """Call Tavily search API."""
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_raw_content": False,
        }

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()

            items: list[WebSearchResultItem] = []
            for res in data.get("results", []):
                items.append(
                    WebSearchResultItem(
                        title=res.get("title", ""),
                        url=res.get("url", ""),
                        snippet=res.get("content", ""),
                        score=float(res.get("score", 0.8)),
                        published_date=res.get("published_date"),
                    )
                )

            return WebSearchResponse(
                query=query,
                results=items,
                total_results=len(items),
                provider=self.name,
            )
        except Exception as e:
            logger.warning("Tavily search failed", query=query, error=str(e))
            raise


class FallbackSearchProvider(SearchProvider):
    """Resilient search provider chaining multiple providers in priority order (FR-WEB-1)."""

    def __init__(self, providers: list[SearchProvider]) -> None:
        """Initialize with an ordered list of providers to try sequentially."""
        if not providers:
            raise ValueError("At least one search provider is required for FallbackSearchProvider")
        self.providers = providers

    @property
    def name(self) -> str:
        return f"fallback({','.join(p.name for p in self.providers)})"

    def search(self, query: str, max_results: int = 5) -> WebSearchResponse:
        """Attempt search on each provider sequentially until one succeeds."""
        errors: list[str] = []
        for provider in self.providers:
            try:
                logger.debug("Attempting search with provider", provider=provider.name, query=query)
                return provider.search(query=query, max_results=max_results)
            except Exception as e:
                err_str = f"{provider.name}: {e!s}"
                logger.info(
                    "Search provider failed, trying next fallback",
                    provider=provider.name,
                    error=str(e),
                )
                errors.append(err_str)

        raise RuntimeError(f"All search providers failed for query '{query}': {'; '.join(errors)}")
