"""Unit tests for WebResearchAgent consuming tools via MCP tool plane (FR-WEB-4, FR-WEB-5, FR-WEB-6)."""

import pytest

from libs.agents.web_research import WebResearchAgent
from libs.mcp_client.client import MCPClient
from services.mcp_servers.web_search.crawler import WebCrawler
from services.mcp_servers.web_search.models import WebSearchResultItem
from services.mcp_servers.web_search.providers import MockSearchProvider
from services.mcp_servers.web_search.server import WebSearchMCPServer


def test_web_research_agent_end_to_end_mcp_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that WebResearchAgent executes research strictly via MCPClient tools (P0 MCP Gate)."""
    # 1. Setup mock search provider with rich domain mix
    mock_provider = MockSearchProvider()
    mock_provider.register_mock_results(
        "kubernetes",
        [
            WebSearchResultItem(
                title="Kubernetes HPA Architecture",
                url="https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/",
                snippet="Official documentation on configuring HPA algorithms and custom metrics.",
                score=0.90,
            ),
            WebSearchResultItem(
                title="Personal Blog: My K8s Tips",
                url="https://mytechblog.com/posts/k8s-tips",
                snippet="Informal tips and personal setup recommendations for pod scaling.",
                score=0.75,
            ),
            WebSearchResultItem(
                title="Production Scaling Guide",
                url="https://docs.enterprise-tech.org/hpa-production",
                snippet="Enterprise architectures recommending metric server thresholds and cooldown periods.",
                score=0.85,
            ),
        ],
    )

    # 2. Mock web crawler HTML fetch responses
    crawler = WebCrawler(respect_robots=False)

    class MockResponse:
        status_code = 200
        text = "<html><body><article><p>Detailed extracted technical text regarding HPA setup.</p></article></body></html>"

        def raise_for_status(self) -> None:
            pass

    import httpx

    def mock_get(*_args: object, **_kwargs: object) -> MockResponse:
        return MockResponse()

    def mock_safe(*_args: object, **_kwargs: object) -> tuple[bool, str]:
        return (True, "OK")

    monkeypatch.setattr(httpx.Client, "get", mock_get)
    monkeypatch.setattr("services.mcp_servers.web_search.crawler.is_safe_url", mock_safe)

    # 3. Spin up MCP Server and MCP Client
    server = WebSearchMCPServer(provider=mock_provider, crawler=crawler)
    transport = server.get_in_memory_transport()
    client = MCPClient(transport=transport, server_name="web-research-agent-test")

    # 4. Instantiate Agent with MCP client
    agent = WebResearchAgent(mcp_client=client, max_depth=1)

    result = agent.research("Kubernetes pod autoscaling best practices")

    # 5. Verify research synthesis and citations
    assert "Kubernetes" in result.query
    assert len(result.sources) >= 1
    assert "Web Research Synthesis" in result.summary

    # Verify credibility scoring boosted authoritative host (kubernetes.io) (FR-WEB-5)
    k8s_source = next((s for s in result.sources if "kubernetes.io" in s.url), None)
    assert k8s_source is not None
    assert k8s_source.credibility_score >= 0.95
    assert k8s_source.domain == "kubernetes.io"
    assert k8s_source.accessed_at != ""

    client.close()


def test_web_research_agent_detects_contradictions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that contradictory excerpts are surfaced in the research report (FR-WEB-5)."""
    mock_provider = MockSearchProvider()
    mock_provider.register_mock_results(
        "database",
        [
            WebSearchResultItem(
                title="Postgres Indexing Manual",
                url="https://docs.postgresql.org/index",
                snippet="We recommend B-Tree indexes for all high-cardinality primary query workloads.",
                score=0.90,
            ),
            WebSearchResultItem(
                title="Postgres Index Bottlenecks Report",
                url="https://tech-benchmarks.org/pg-perf",
                snippet="Severe pitfall observed: excessive B-Tree indexing causes heavy write latency and write amplification limitation.",
                score=0.88,
            ),
        ],
    )

    crawler = WebCrawler(respect_robots=False)

    class MockResponse:
        status_code = 200
        text = "<html><body><p>Database indexing overview.</p></body></html>"

        def raise_for_status(self) -> None:
            pass

    import httpx

    def mock_get(*_args: object, **_kwargs: object) -> MockResponse:
        return MockResponse()

    def mock_safe(*_args: object, **_kwargs: object) -> tuple[bool, str]:
        return (True, "OK")

    monkeypatch.setattr(httpx.Client, "get", mock_get)
    monkeypatch.setattr("services.mcp_servers.web_search.crawler.is_safe_url", mock_safe)

    server = WebSearchMCPServer(provider=mock_provider, crawler=crawler)
    transport = server.get_in_memory_transport()
    client = MCPClient(transport=transport, server_name="contradiction-test")

    agent = WebResearchAgent(mcp_client=client, max_depth=1)
    result = agent.research("Postgres database indexing strategies")

    assert len(result.sources) == 2
    assert len(result.contradictions) > 0
    assert "Discovered Contradictions" in result.summary

    client.close()
