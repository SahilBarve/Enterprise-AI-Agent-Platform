"""Grounded RAG answer generator with caching, CRAG grading, compression, and citations (FR-RAG-17 to FR-RAG-28).

================================================================================
EDUCATIONAL ARCHITECTURE NOTES:
Complete Optimized Generation Pipeline:
--------------------------------------------------------------------------------
This coordinator integrates all Phase 1 and Phase 2 components into a single
high-performance, resilient RAG pipeline:

1. Two-Tier Semantic Cache (FR-RAG-18 to FR-RAG-22):
   Checks if this query (or a semantically equivalent query with similarity >= 0.92)
   was answered previously. If hit, returns the cached answer in <10ms, skipping
   expensive vector retrieval and LLM inference.
2. Hybrid Retrieval & Reranking (FR-RAG-11, FR-RAG-14):
   Executes dense + BM25 sparse search merged with Reciprocal Rank Fusion (RRF $k=60$),
   followed by Cross-Encoder scoring.
3. Adaptive Corrective RAG (CRAG) Grader (FR-RAG-17):
   Evaluates candidate evidence into 3 confidence bands:
   - CORRECT: High relevance -> Proceed to generation.
   - AMBIGUOUS: Marginal relevance -> Rewrites query and re-retrieves supplementary candidates.
   - INCORRECT: Low relevance -> Returns honest refusal immediately, preventing hallucinations.
4. Context Compression & Token Budgeting (FR-RAG-23 to FR-RAG-25):
   Prunes irrelevant filler sentences, enforces token budgets, and applies U-shaped
   reordering so top-ranked chunks occupy prompt attention boundaries ("Lost in the Middle").
5. LLM Synthesis & Groundedness Audit (FR-RAG-26 to FR-RAG-28):
   Generates answer with [1], [2] citations, verifies claim-level grounding against
   source excerpts, and caches successful responses.
================================================================================
"""

from libs.common.logging import get_logger
from libs.llm.provider import LLMMessage, LLMProvider, MockLLMProvider
from libs.retrieval.cache import SemanticCache
from libs.retrieval.citations import CitationEngine, GroundedAnswer
from libs.retrieval.compressor import ContextCompressor
from libs.retrieval.crag import AdaptiveCRAGRouter, CRAGConfidence
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
    """Orchestrates caching, hybrid retrieval, CRAG routing, compression, and grounded answer synthesis."""

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker | None = None,
        citation_engine: CitationEngine | None = None,
        llm_provider: LLMProvider | None = None,
        cache: SemanticCache | None = None,
        compressor: ContextCompressor | None = None,
        crag_router: AdaptiveCRAGRouter | None = None,
    ) -> None:
        """Initialize the optimized RAG generation pipeline.

        Args:
            retriever: HybridRetriever instance for Qdrant search.
            reranker: CrossEncoderReranker instance for candidate refinement.
            citation_engine: CitationEngine for provenance tracking and auditing.
            llm_provider: LLMProvider instance (LiteLLM or MockLLMProvider).
            cache: Optional SemanticCache for Tier 1 / Tier 2 caching.
            compressor: Optional ContextCompressor for token budgeting and sentence pruning.
            crag_router: Optional AdaptiveCRAGRouter for evidence grading and query rewriting.
        """
        self.retriever = retriever
        self.reranker = reranker or CrossEncoderReranker(use_mock=True)
        self.citation_engine = citation_engine or CitationEngine()
        self.llm_provider = llm_provider or MockLLMProvider()
        self.cache = cache
        self.compressor = compressor or ContextCompressor()
        self.crag_router = crag_router or AdaptiveCRAGRouter()

    def generate(
        self,
        query: str,
        tenant_id: str,
        collection_id: str,
        top_k: int = 5,
        min_relevance: float = 0.01,
        expand_parent: bool = True,
        collection_name: str | None = None,
        use_cache: bool = True,
        max_context_tokens: int | None = None,
    ) -> GroundedAnswer:
        """Execute end-to-end optimized RAG pipeline from user query to cited, grounded answer.

        Pipeline Stages:
        1. Cache Lookup: Checks exact and semantic cache in Redis/memory.
        2. Initial Retrieval: Fetches candidates via dense + sparse hybrid search with RRF.
        3. Reranking: Refines candidates using Cross-Encoder.
        4. CRAG Evaluation: Grades retrieval quality; rewrites and retries if ambiguous;
           refuses if incorrect.
        5. Context Compression: Prunes irrelevant sentences, reorders to mitigate 'lost in the middle',
           and enforces token budgets.
        6. LLM Completion: Invokes provider with strict grounding prompt.
        7. Audit & Cache: Verifies claims, parses citations, and caches verified answers.

        Returns:
            GroundedAnswer containing text, citations, confidence score, and token metrics.
        """
        logger.info(
            "Starting RAG generation",
            tenant_id=tenant_id,
            collection_id=collection_id,
            query=query,
        )

        # -------------------------------------------------------------
        # Stage 1: Two-Tier Semantic Cache Lookup (FR-RAG-18)
        # -------------------------------------------------------------
        if self.cache and use_cache:
            cached_entry = self.cache.get(
                query=query,
                tenant_id=tenant_id,
                collection_id=collection_id,
            )
            if cached_entry:
                logger.info(
                    "Returning cached RAG answer",
                    cache_tier=cached_entry.cache_tier,
                    similarity=cached_entry.similarity,
                    query=query,
                )
                return cached_entry.answer

        # -------------------------------------------------------------
        # Stage 2: Hybrid Retrieval & Reranking (FR-RAG-11, FR-RAG-14)
        # -------------------------------------------------------------
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

        reranked = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
            min_score=min_relevance,
        )

        # -------------------------------------------------------------
        # Stage 3: Adaptive Corrective RAG (CRAG) Grader (FR-RAG-17)
        # -------------------------------------------------------------
        if self.crag_router:
            crag_decision = self.crag_router.evaluate_retrieval(query, reranked)

            if crag_decision.confidence == CRAGConfidence.INCORRECT:
                logger.info(
                    "CRAG confidence INCORRECT: halting generation to prevent hallucination",
                    score=crag_decision.score,
                )
                return GroundedAnswer(
                    answer="I don't know based on the provided documents.",
                    citations=[],
                    confidence_score=1.0,
                    is_refusal=True,
                    unsupported_claims=[],
                    unmapped_citations=[],
                    tokens_used=0,
                )

            elif (
                crag_decision.confidence == CRAGConfidence.AMBIGUOUS
                and crag_decision.rewritten_query
            ):
                logger.info(
                    "CRAG confidence AMBIGUOUS: retrieving with rewritten query",
                    rewritten_query=crag_decision.rewritten_query,
                )
                supp_query = RetrievalQuery(
                    query=crag_decision.rewritten_query,
                    tenant_id=tenant_id,
                    collection_id=collection_id,
                    top_k=top_k * 2,
                    expand_parent=expand_parent,
                )
                supp_candidates = self.retriever.search(
                    query=supp_query,
                    collection_name=collection_name,
                )
                # Combine and re-rerank unified candidate pool
                combined_candidates = candidates + supp_candidates
                reranked = self.reranker.rerank(
                    query=query,
                    candidates=combined_candidates,
                    top_k=top_k,
                    min_score=min_relevance,
                )

        # Insufficient evidence fallback if reranking yielded zero results
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

        # -------------------------------------------------------------
        # Stage 4: Context Compression & Token Budgeting (FR-RAG-23 to FR-RAG-25)
        # -------------------------------------------------------------
        if self.compressor:
            compressed_context = self.compressor.compress(
                query=query,
                results=reranked,
                max_tokens=max_context_tokens,
            )
            final_chunks = compressed_context.results
        else:
            final_chunks = reranked

        # -------------------------------------------------------------
        # Stage 5: Assemble Prompt Context & Invoke LLM
        # -------------------------------------------------------------
        formatted_context, available_citations = self.citation_engine.format_context(
            final_chunks
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

        response = self.llm_provider.complete(messages=messages, temperature=0.0)
        raw_answer = response.content.strip()

        # -------------------------------------------------------------
        # Stage 6: Audit Answer Grounding (FR-RAG-26 to FR-RAG-28)
        # -------------------------------------------------------------
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

        matched_citations, unmapped = self.citation_engine.extract_citations(
            raw_answer, available_citations
        )
        confidence, unsupported = self.citation_engine.verify_groundedness(
            raw_answer, matched_citations
        )

        grounded_answer = GroundedAnswer(
            answer=raw_answer,
            citations=matched_citations,
            confidence_score=confidence,
            is_refusal=False,
            unsupported_claims=unsupported,
            unmapped_citations=unmapped,
            tokens_used=response.prompt_tokens + response.completion_tokens,
        )

        # -------------------------------------------------------------
        # Stage 7: Store in Semantic Cache (FR-RAG-22)
        # -------------------------------------------------------------
        if self.cache and use_cache:
            self.cache.set(
                query=query,
                answer=grounded_answer,
                tenant_id=tenant_id,
                collection_id=collection_id,
            )

        return grounded_answer
