"""Model Context Protocol (MCP) server for Web Search and Content Extraction (FR-WEB-1, FR-MCP-1).

================================================================================
EDUCATIONAL ARCHITECTURE NOTES:
Why Expose Web Research as an Independent MCP Server?
--------------------------------------------------------------------------------
1. Complete Isolation of Web Networking:
   - Fetching web pages involves untrusted networks, DNS queries, HTML parsing,
     and potentially large payloads.
   - By running Web Search as a dedicated MCP server:
     a) Network policies can restrict outgoing HTTP traffic ONLY to this server pod.
     b) The main supervisor agent pod has zero access to external HTTP egress,
        preventing any data exfiltration from the supervisor.
     c) Scraping libraries (BeautifulSoup, httpx) are contained within this server.

2. MCP Tool Declarations (FR-MCP-2):
   - Exposes two atomic tools:
     * `web_search`: Queries external search engines and returns ranked URLs + snippets.
     * `web_fetch`: Takes a URL, verifies SSRF safety, downloads HTML, extracts text,
       and wraps it inside `<untrusted_web_content>` tags.

3. Result Caching with TTL (FR-WEB-7):
   - External search queries are cached in memory (or Redis) to avoid hammering
     search providers for the same query across multiple agent sub-steps.
================================================================================
"""

import json
import sys
import time
from typing import Any

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
from services.mcp_servers.web_search.crawler import WebCrawler
from services.mcp_servers.web_search.models import WebPageFetchResponse, WebSearchResponse
from services.mcp_servers.web_search.providers import (
    MockSearchProvider,
    SearchProvider,
)

logger = get_logger("mcp.web_search.server")


class WebSearchMCPServer:
    """Production MCP Server exposing `web_search` and `web_fetch` tools over JSON-RPC 2.0."""

    def __init__(
        self,
        provider: SearchProvider | None = None,
        crawler: WebCrawler | None = None,
        cache_ttl_seconds: float = 300.0,
    ) -> None:
        """Initialize the Web Search MCP server.

        Args:
            provider: Pluggable search provider (defaults to MockSearchProvider).
            crawler: Web crawler instance for page fetching.
            cache_ttl_seconds: Duration to cache search queries in seconds.
        """
        self.provider = provider or MockSearchProvider()
        self.crawler = crawler or WebCrawler()
        self.cache_ttl_seconds = cache_ttl_seconds
        # In-memory query cache: query -> (timestamp, WebSearchResponse)
        self._search_cache: dict[str, tuple[float, WebSearchResponse]] = {}

        # Declare available tool schemas (FR-MCP-2)
        self.tools: dict[str, MCPToolDefinition] = {
            "web_search": MCPToolDefinition(
                name="web_search",
                description=(
                    "Search the public web for real-time information, documentation, news, or technical articles. "
                    "Returns a list of ranked search results containing titles, URLs, and summaries."
                ),
                inputSchema=MCPToolInputSchema(
                    properties={
                        "query": {
                            "type": "string",
                            "description": "Search query keywords or natural language question",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to retrieve (default: 5, max: 10)",
                            "default": 5,
                        },
                    },
                    required=["query"],
                ),
            ),
            "web_fetch": MCPToolDefinition(
                name="web_fetch",
                description=(
                    "Fetch and extract clean readable text from a public web page with SSRF protection, "
                    "robots.txt politeness, and indirect prompt-injection sanitization."
                ),
                inputSchema=MCPToolInputSchema(
                    properties={
                        "url": {
                            "type": "string",
                            "description": "Target HTTP or HTTPS webpage URL to fetch",
                        },
                    },
                    required=["url"],
                ),
            ),
        }

    def handle_request(self, request: JSONRPCRequest) -> JSONRPCResponse:
        """Handle incoming JSON-RPC 2.0 request.

        Args:
            request: JSON-RPC request envelope.

        Returns:
            JSON-RPC response envelope with result or error.
        """
        method = request.method
        req_id = request.id
        params = request.params

        logger.debug("Received MCP request", method=method, req_id=req_id)

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
                            message=f"Tool '{tool_name}' not recognized by web search server",
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
            logger.error("Web Search MCP server error", error=str(e), method=method)
            return JSONRPCResponse(
                id=req_id,
                error=JSONRPCError(
                    code=MCPErrorCode.TOOL_EXECUTION_ERROR,
                    message=f"Execution error: {e!s}",
                ),
            )

    def _dispatch_tool(self, tool_name: str, arguments: dict[str, Any]) -> MCPToolCallResult:
        """Dispatch tool execution to appropriate internal handler."""
        if tool_name == "web_search":
            query = arguments.get("query", "")
            max_results = int(arguments.get("max_results", 5))
            return self._execute_web_search(query=query, max_results=max_results)

        if tool_name == "web_fetch":
            url = arguments.get("url", "")
            return self._execute_web_fetch(url=url)

        raise ValueError(f"Unknown tool: {tool_name}")

    def _execute_web_search(self, query: str, max_results: int) -> MCPToolCallResult:
        """Execute web search with TTL caching (FR-WEB-1, FR-WEB-7)."""
        cache_key = f"{query.strip().lower()}:{max_results}"
        now = time.time()

        # Check in-memory cache
        if cache_key in self._search_cache:
            cached_time, cached_resp = self._search_cache[cache_key]
            if now - cached_time < self.cache_ttl_seconds:
                logger.debug("Web search cache hit", query=query)
                return MCPToolCallResult(
                    content=[
                        MCPTextContent(
                            text=cached_resp.model_dump_json(indent=2),
                        )
                    ],
                    metadata={"cached": True, "provider": cached_resp.provider},
                )

        # Call search provider
        search_resp = self.provider.search(query=query, max_results=max_results)
        self._search_cache[cache_key] = (now, search_resp)

        return MCPToolCallResult(
            content=[
                MCPTextContent(
                    text=search_resp.model_dump_json(indent=2),
                )
            ],
            metadata={
                "cached": False,
                "provider": search_resp.provider,
                "total_results": search_resp.total_results,
            },
        )

    def _execute_web_fetch(self, url: str) -> MCPToolCallResult:
        """Execute page crawl and HTML extraction (FR-WEB-2, FR-WEB-3, FR-WEB-8)."""
        fetch_resp: WebPageFetchResponse = self.crawler.fetch_page(url)

        is_error = not fetch_resp.is_safe or "error" in fetch_resp.metadata
        return MCPToolCallResult(
            content=[
                MCPTextContent(
                    text=fetch_resp.model_dump_json(indent=2),
                )
            ],
            isError=is_error,
            metadata={
                "url": url,
                "is_safe": fetch_resp.is_safe,
                "char_length": fetch_resp.raw_length,
                "fetched_at": fetch_resp.fetched_at,
            },
        )

    def get_in_memory_transport(self) -> InMemoryMCPTransport:
        """Create an in-memory transport connected directly to this server instance."""
        return InMemoryMCPTransport(
            handler=self.handle_request,
            server_name="web-search-mcp",
        )

    def run_stdio(self) -> None:
        """Run standard I/O loop for stdio transport subprocess execution."""
        logger.info("Web Search MCP server started in stdio mode")
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
    server = WebSearchMCPServer()
    server.run_stdio()
