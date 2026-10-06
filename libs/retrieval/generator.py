"""Grounded RAG answer generator with citation tracking and refusal behavior (FR-RAG-26, FR-RAG-27, FR-RAG-28)."""

from libs.common.logging import get_logger
from libs.llm.provider import LLMMessage, LLMProvider, MockLLMProvider
from libs.retrieval.citations import CitationEngine, GroundedAnswer
from libs.retrieval.hybrid import HybridRetriever
from libs.retrieval.models import RetrievalQuery
from libs.retrieval.reranker import CrossEncoderReranker

logger = get_logger("retrieval.generator")

RAG_SYSTEM_PROMPT = """You are a trusted enterprise AI assistant.
Answer the user's question accurately and objectively, strictly grounded in the provided numbered context sources.
Guidelines:
1. Always substantiate factual statements with inline citation brackets like [1], [2], or [1, 2].
2. If the provided sources do not contain sufficient evidence to answer the question, state: "I don't know based on the provided documents."
3. Do not assume or extrapolate beyond the text. Do not make up citations."""


class RAGGenerator:
    """Orchestrates hybrid retrieval, reranking, context assembly, and grounded answer synthesis."""

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker | None = None,
        citation_engine: CitationEngine | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker or CrossEncoderReranker(use_mock=True)
        self.citation_engine = citation_engine or CitationEngine()
        self.llm_provider = llm_provider or MockLLMProvider()

    def generate(
        self,
        query: str,
        tenant_id: str,
        collection_id: str,
        top_k: int = 5,
        min_relevance: float = 0.01,
        expand_parent: bool = True,
        collection_name: str | None = None,
    ) -> GroundedAnswer:
        """Execute end-to-end RAG pipeline from query to cited, grounded answer."""
        logger.info(
            "Starting RAG generation",
            tenant_id=tenant_id,
            collection_id=collection_id,
            query=query,
        )

        # 1. Hybrid retrieval (dense + sparse with RRF)
        retrieval_query = RetrievalQuery(
            query=query,
            tenant_id=tenant_id,
            collection_id=collection_id,
            top_k=top_k * 4,  # Retrieve candidate pool for reranker
            expand_parent=expand_parent,
        )
        candidates = self.retriever.search(
            query=retrieval_query,
            collection_name=collection_name,
        )

        # 2. Cross-encoder reranking
        reranked = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
            min_score=min_relevance,
        )

        # 3. Insufficient evidence guard (FR-RAG-28)
        if not reranked:
            logger.info("No relevant candidates found above threshold; returning refusal")
            return GroundedAnswer(
                answer="I don't know based on the provided documents.",
                citations=[],
                confidence_score=1.0,
                is_refusal=True,
                unsupported_claims=[],
                unmapped_citations=[],
                tokens_used=0,
            )

        # 4. Assemble context & prompt
        formatted_context, available_citations = self.citation_engine.format_context(
            reranked
        )

        user_content = (
            f"Sources:\n{formatted_context}\n\n"
            f"Question: {query}\n\n"
            "Answer with citations:"
        )

        messages = [
            LLMMessage(role="system", content=RAG_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_content),
        ]

        # 5. Execute LLM completion
        response = self.llm_provider.complete(messages=messages, temperature=0.0)
        raw_answer = response.content.strip()

        # 6. Check for model-generated refusal
        if self.citation_engine.is_refusal(raw_answer):
            return GroundedAnswer(
                answer=raw_answer,
                citations=[],
                confidence_score=1.0,
                is_refusal=True,
                unsupported_claims=[],
                unmapped_citations=[],
                tokens_used=response.prompt_tokens + response.completion_tokens,
            )

        # 7. Extract citations and verify groundedness (FR-RAG-26, FR-RAG-27)
        matched_citations, unmapped = self.citation_engine.extract_citations(
            raw_answer, available_citations
        )
        confidence, unsupported = self.citation_engine.verify_groundedness(
            raw_answer, matched_citations
        )

        return GroundedAnswer(
            answer=raw_answer,
            citations=matched_citations,
            confidence_score=confidence,
            is_refusal=False,
            unsupported_claims=unsupported,
            unmapped_citations=unmapped,
            tokens_used=response.prompt_tokens + response.completion_tokens,
        )
