"""Grounded RAG answer generator with citation tracking and refusal behavior (FR-RAG-26, FR-RAG-27, FR-RAG-28).

This module coordinates the complete Document RAG pipeline:
1. Retrieval: Hybrid search (dense + sparse BM25 merged via RRF).
2. Reranking: Cross-encoder scoring to promote the most relevant chunks and penalize duplicates.
3. Evidence Guard: If no chunks pass the relevance threshold, short-circuits to an "I don't know"
   refusal without wasting LLM tokens or risking hallucinations.
4. Prompt Assembly: Formats numbered sources [Source 1], [Source 2] with strict grounding instructions.
5. Generation: Calls the configured LLM provider (LiteLLM / OpenAI / Ollama / Mock).
6. Post-Processing: Extracts inline citations [1], [2], verifies claim groundedness, flags unsupported
   statements, and returns a GroundedAnswer object.
"""

from libs.common.logging import get_logger
from libs.llm.provider import LLMMessage, LLMProvider, MockLLMProvider
from libs.retrieval.citations import CitationEngine, GroundedAnswer
from libs.retrieval.hybrid import HybridRetriever
from libs.retrieval.models import RetrievalQuery
from libs.retrieval.reranker import CrossEncoderReranker

logger = get_logger("retrieval.generator")

# Strict enterprise grounding prompt to prevent model speculation
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
        """Initialize the RAG generation pipeline.

        Args:
            retriever: HybridRetriever instance for Qdrant search.
            reranker: CrossEncoderReranker instance for candidate refinement.
            citation_engine: CitationEngine for provenance tracking and auditing.
            llm_provider: LLMProvider instance (LiteLLM or MockLLMProvider).
        """
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
        """Execute end-to-end RAG pipeline from user query to cited, grounded answer.

        Step-by-step pipeline:
        1. Fetch candidate pool: Retrieves top_k * 4 candidates using hybrid dense + sparse search.
        2. Rerank candidates: Passes candidates through cross-encoder for fine-grained re-scoring.
        3. Check evidence sufficiency: If zero candidates meet `min_relevance`, returns refusal (FR-RAG-28).
        4. Assemble numbered context: Formats top_k candidates into numbered blocks with citations.
        5. Invoke LLM: Prompts model with strict grounding rules.
        6. Audit answer: Extracts citation numbers, verifies claim-level grounding, and returns result.

        Args:
            query: User's natural language question.
            tenant_id: Mandatory tenant ID for isolation.
            collection_id: Target knowledge collection.
            top_k: Final number of chunks to include in prompt context (default: 5).
            min_relevance: Score cutoff below which evidence is considered missing.
            expand_parent: Whether to expand small child chunks into large parent context chunks.
            collection_name: Optional explicit collection name override.

        Returns:
            GroundedAnswer containing text, citations, confidence score, and refusal flags.
        """
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
            top_k=top_k * 4,  # Retrieve wider candidate pool for cross-encoder reranker
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
        # Prevents hallucinations when query is out-of-domain or collection has no relevant docs
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
