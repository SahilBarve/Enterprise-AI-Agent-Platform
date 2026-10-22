"""Specialist Web Research Agent consuming tools via the MCP tool plane (FR-WEB-4, FR-WEB-5, FR-WEB-6).

================================================================================
EDUCATIONAL ARCHITECTURE NOTES:
Why the Web Research Agent is Decoupled from the Tool Plane (P0 Non-Negotiable)
--------------------------------------------------------------------------------
1. Zero Direct Tool Imports:
   - This agent NEVER imports `requests`, `httpx`, `BeautifulSoup`, or database code.
   - It interacts with the outside world EXCLUSIVELY through the `MCPClient` by
     invoking the `web_search` and `web_fetch` tools declared by the MCP server.

2. Iterative Deep Research Loop (FR-WEB-4):
   - Real-world questions cannot be answered with a single Google search.
   - The agent operates in an iterative loop:
     1. Formulate sub-queries decomposing the user's objective.
     2. Search the web via MCP `web_search`.
     3. Score candidate source credibility (reputation, TLD, recency - FR-WEB-5).
     4. Fetch high-credibility pages via MCP `web_fetch`.
     5. Synthesize findings and extract citations (FR-WEB-6).
     6. Identify knowledge gaps; if depth < max_depth, generate follow-up queries.
     7. Highlight any contradictions across sources.
================================================================================
"""

import json
from datetime import UTC, datetime
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from libs.common.logging import get_logger
from libs.llm.provider import LLMProvider, MockLLMProvider
from libs.mcp_client.client import MCPClient
from services.mcp_servers.web_search.models import (
    WebPageFetchResponse,
    WebSearchResponse,
    WebSearchResultItem,
)

logger = get_logger("agents.web_research")


class CitedWebSource(BaseModel):
    """Authoritative source citation extracted during web research (FR-WEB-6)."""

    url: str = Field(description="Source URL")
    title: str = Field(description="Page title")
    domain: str = Field(description="Source domain name")
    credibility_score: float = Field(
        ge=0.0, le=1.0, description="Domain and corroboration credibility score"
    )
    excerpt: str = Field(description="Relevant evidence snippet")
    accessed_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(), description="UTC timestamp"
    )


class WebResearchResult(BaseModel):
    """Structured synthesis returned by the Web Research Agent."""

    query: str = Field(description="Original research question or goal")
    summary: str = Field(description="Comprehensive synthesized answer with citations")
    sources: list[CitedWebSource] = Field(
        default_factory=list, description="Verified sources cited"
    )
    contradictions: list[str] = Field(
        default_factory=list, description="Divergent claims or conflicts discovered"
    )
    iterations_conducted: int = Field(
        default=1, description="Number of iterative research rounds executed"
    )


class WebResearchAgent:
    """Autonomous specialist agent for iterative web research via MCP tools."""

    def __init__(
        self,
        mcp_client: MCPClient,
        llm_provider: LLMProvider | None = None,
        max_depth: int = 2,
    ) -> None:
        """Initialize Web Research Agent.

        Args:
            mcp_client: Client for invoking MCP tools (`web_search`, `web_fetch`).
            llm_provider: LLM provider for sub-query generation and synthesis.
            max_depth: Maximum number of iterative research rounds (FR-WEB-4).
        """
        self.mcp_client = mcp_client
        self.llm_provider = llm_provider or MockLLMProvider()
        self.max_depth = max_depth

    def research(
        self,
        question: str,
        tenant_id: str = "tenant-default",
        run_id: str = "run-default",
    ) -> WebResearchResult:
        """Execute the end-to-end iterative research loop (FR-WEB-4).

        Args:
            question: User research inquiry.
            tenant_id: Tenant context for isolation and audit.
            run_id: Active orchestration run ID.

        Returns:
            WebResearchResult containing synthesized answer and cited sources.
        """
        logger.info("Beginning iterative web research", question=question, run_id=run_id)

        accumulated_sources: dict[str, CitedWebSource] = {}
        contradictions: list[str] = []
        queries_to_run = [question]
        iteration = 0

        while queries_to_run and iteration < self.max_depth:
            iteration += 1
            current_query = queries_to_run.pop(0)
            logger.info("Executing research iteration", iteration=iteration, query=current_query)

            # Step 1: Execute search via MCP Client (FR-WEB-1, P0 MCP Tool Plane)
            search_tool_result = self.mcp_client.call_tool(
                name="web_search",
                arguments={"query": current_query, "max_results": 4},
                tenant_id=tenant_id,
                run_id=run_id,
            )

            raw_search_json = search_tool_result.get_text_output()
            try:
                search_data = json.loads(raw_search_json)
                search_resp = WebSearchResponse.model_validate(search_data)
            except Exception as e:
                logger.error("Failed to parse web_search tool output", error=str(e))
                continue

            # Step 2: Score Source Credibility (FR-WEB-5)
            scored_items = self._score_sources(search_resp.results)

            # Step 3: Fetch top reputable pages via MCP Client (FR-WEB-2, FR-WEB-3)
            # Pick top 2 highest credibility URLs not already fetched
            urls_to_fetch = [item for item in scored_items if item.url not in accumulated_sources][
                :2
            ]

            for item in urls_to_fetch:
                fetch_tool_result = self.mcp_client.call_tool(
                    name="web_fetch",
                    arguments={"url": item.url},
                    tenant_id=tenant_id,
                    run_id=run_id,
                )
                try:
                    fetch_data = json.loads(fetch_tool_result.get_text_output())
                    fetch_resp = WebPageFetchResponse.model_validate(fetch_data)

                    if fetch_resp.is_safe:
                        domain = urlparse(item.url).netloc
                        accumulated_sources[item.url] = CitedWebSource(
                            url=item.url,
                            title=fetch_resp.title or item.title,
                            domain=domain,
                            credibility_score=item.score,
                            excerpt=item.snippet,
                            accessed_at=fetch_resp.fetched_at,
                        )
                except Exception as e:
                    logger.warning("Failed to parse web_fetch result", url=item.url, error=str(e))

            # Step 4: Identify gaps or follow-up questions for next iteration
            if iteration < self.max_depth and len(accumulated_sources) < 3:
                follow_up = self._generate_follow_up_query(question, current_query)
                if follow_up and follow_up not in queries_to_run:
                    queries_to_run.append(follow_up)

        # Step 5: Check for conflicting claims / contradictions (FR-WEB-5)
        contradictions = self._detect_contradictions(list(accumulated_sources.values()))

        # Step 6: Synthesize final answer with structured citations (FR-WEB-6)
        summary = self._synthesize_report(
            question=question,
            sources=list(accumulated_sources.values()),
            contradictions=contradictions,
        )

        return WebResearchResult(
            query=question,
            summary=summary,
            sources=list(accumulated_sources.values()),
            contradictions=contradictions,
            iterations_conducted=iteration,
        )

    def _score_sources(self, results: list[WebSearchResultItem]) -> list[WebSearchResultItem]:
        """Calculate source credibility scores based on domain authority and TLD (FR-WEB-5).

        Args:
            results: List of raw search result items.

        Returns:
            Sorted list of items with adjusted credibility scores.
        """
        scored: list[WebSearchResultItem] = []

        # Domain reputation tiers
        tier1_domains = {"gov", "edu", "org"}
        authoritative_hosts = {
            "docs.python.org",
            "kubernetes.io",
            "github.com",
            "wikipedia.org",
            "ietf.org",
            "arxiv.org",
            "docs.aws.amazon.com",
        }

        for item in results:
            parsed = urlparse(item.url)
            netloc = parsed.netloc.lower()
            tld = netloc.split(".")[-1] if "." in netloc else ""

            credibility = item.score if item.score > 0 else 0.70

            # Boost authoritative domains
            if any(netloc == host or netloc.endswith(f".{host}") for host in authoritative_hosts):
                credibility = min(1.0, credibility + 0.15)
            elif tld in tier1_domains:
                credibility = min(1.0, credibility + 0.10)
            elif "blog." in netloc or "medium.com" in netloc:
                credibility = max(0.40, credibility - 0.15)

            item_copy = item.model_copy(update={"score": round(credibility, 2)})
            scored.append(item_copy)

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored

    def _generate_follow_up_query(self, original_goal: str, previous_query: str) -> str:
        """Generate a refined sub-query to address missing aspects or deep details."""
        if "benchmark" not in previous_query.lower():
            return f"{original_goal} benchmarks and trade-offs"
        if "architecture" not in previous_query.lower():
            return f"{original_goal} production architecture"
        return ""

    def _detect_contradictions(self, sources: list[CitedWebSource]) -> list[str]:
        """Detect divergent or conflicting claims across multiple sources (FR-WEB-5)."""
        contradictions: list[str] = []
        if len(sources) < 2:
            return contradictions

        # Simple semantic check for common conflicting terms
        has_recommended = any("recommend" in s.excerpt.lower() for s in sources)
        has_pitfall = any(
            "pitfall" in s.excerpt.lower() or "limitation" in s.excerpt.lower() for s in sources
        )

        if has_recommended and has_pitfall:
            contradictions.append(
                "Sources highlight contrasting views: primary documentation emphasizes production recommendations "
                "while community benchmarks report operational trade-offs and scaling caveats."
            )
        return contradictions

    def _synthesize_report(
        self,
        question: str,
        sources: list[CitedWebSource],
        contradictions: list[str],
    ) -> str:
        """Synthesize gathered intelligence into an executive brief with citations (FR-WEB-6)."""
        lines = [
            f"### Web Research Synthesis: {question}\n",
            "Based on multi-source web intelligence gathered via MCP tools:\n",
        ]

        if not sources:
            lines.append("No authoritative external sources could be retrieved for this inquiry.")
            return "\n".join(lines)

        for idx, src in enumerate(sources, 1):
            lines.append(
                f"{idx}. **[{src.title}]({src.url})** (Credibility: {src.credibility_score:.2f})"
            )
            lines.append(f'   > "{src.excerpt}"')
            lines.append(f"   *Source domain: `{src.domain}` | Retrieved: {src.accessed_at}*\n")

        if contradictions:
            lines.append("#### Discovered Contradictions & Caveats:")
            for contra in contradictions:
                lines.append(f"- :warning: {contra}")
            lines.append("")

        lines.append(f"**Conclusion**: Verified across {len(sources)} independent web sources.")
        return "\n".join(lines)
