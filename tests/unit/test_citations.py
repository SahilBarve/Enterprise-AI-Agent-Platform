"""Unit tests for CitationEngine, citation extraction, and groundedness checking (FR-RAG-26, FR-RAG-27, FR-RAG-28)."""

import pytest

from libs.retrieval.citations import CitationEngine
from libs.retrieval.models import SearchResult


@pytest.fixture
def citation_engine() -> CitationEngine:
    return CitationEngine()


@pytest.fixture
def sample_candidates() -> list[SearchResult]:
    return [
        SearchResult(
            chunk_id="chk-1",
            document_id="doc-arch",
            tenant_id="tenant-1",
            collection_id="col-1",
            content="Microservices communicate over gRPC and emit OpenTelemetry traces.",
            score=0.95,
            page_number=3,
            section_path=["Architecture", "Observability"],
        ),
        SearchResult(
            chunk_id="chk-2",
            document_id="doc-security",
            tenant_id="tenant-1",
            collection_id="col-1",
            content="High-risk writes require signed HMAC-SHA256 tokens approved by administrators.",
            score=0.90,
            page_number=7,
            section_path=["Security", "Governance"],
        ),
    ]


@pytest.mark.unit
def test_format_context_and_citations(
    citation_engine: CitationEngine,
    sample_candidates: list[SearchResult],
) -> None:
    """Validate prompt context formatting and citation provenance generation."""
    formatted_context, citations = citation_engine.format_context(sample_candidates)

    assert "[Source 1]" in formatted_context
    assert "[Source 2]" in formatted_context
    assert "doc-arch" in formatted_context
    assert "Microservices communicate over gRPC" in formatted_context

    assert len(citations) == 2
    assert citations[0].source_number == 1
    assert citations[0].chunk_id == "chk-1"
    assert citations[0].page_number == 3
    assert citations[0].section_path == ["Architecture", "Observability"]

    assert citations[1].source_number == 2
    assert citations[1].chunk_id == "chk-2"
    assert citations[1].page_number == 7


@pytest.mark.unit
def test_extract_citations(
    citation_engine: CitationEngine,
    sample_candidates: list[SearchResult],
) -> None:
    """Validate extraction of valid [N] citation markers and detection of unmapped references."""
    _, citations = citation_engine.format_context(sample_candidates)

    answer = (
        "Microservices emit OpenTelemetry traces [1]. "
        "Furthermore, admin approval requires signed tokens [2]. "
        "An imaginary feature was also described [9]."
    )

    matched, unmapped = citation_engine.extract_citations(answer, citations)

    assert len(matched) == 2
    assert matched[0].source_number == 1
    assert matched[1].source_number == 2

    assert unmapped == [9]  # Source 9 does not exist


@pytest.mark.unit
def test_is_refusal_detection(citation_engine: CitationEngine) -> None:
    """Validate refusal detection for insufficient evidence (FR-RAG-28)."""
    assert citation_engine.is_refusal("I don't know based on the provided documents.") is True
    assert citation_engine.is_refusal("There is insufficient evidence in the documents.") is True
    assert citation_engine.is_refusal("The server operates on port 8080.") is False


@pytest.mark.unit
def test_verify_groundedness(
    citation_engine: CitationEngine,
    sample_candidates: list[SearchResult],
) -> None:
    """Validate sentence-level claim groundedness verification (FR-RAG-27)."""
    _, citations = citation_engine.format_context(sample_candidates)

    # 1. Fully grounded answer
    grounded_answer = "Microservices emit OpenTelemetry traces [1]."
    conf_1, unsupported_1 = citation_engine.verify_groundedness(grounded_answer, citations)
    assert conf_1 == 1.0
    assert len(unsupported_1) == 0

    # 2. Answer with an ungrounded hallucination
    mixed_answer = (
        "Microservices emit OpenTelemetry traces [1]. "
        "The company was founded in year 1820 on Mars [1]."
    )
    conf_2, unsupported_2 = citation_engine.verify_groundedness(mixed_answer, citations)
    assert conf_2 < 1.0
    assert len(unsupported_2) == 1
    assert "Mars" in unsupported_2[0]
