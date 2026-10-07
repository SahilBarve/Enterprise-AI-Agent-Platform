"""Provider-agnostic LLM client layer with LiteLLM and deterministic mock provider.

Design Pattern (Dependency Inversion):
- The RAG Generator and Agent subgraphs depend only on the `LLMProvider` protocol.
- They NEVER import OpenAI, Anthropic, or Ollama SDKs directly.
- In production, `LiteLLMProvider` handles model routing, token usage accounting, and
  automated fallback to local models.
- In automated testing and local CI, `MockLLMProvider` provides deterministic, grounded
  answers in 0.001 seconds without consuming API credits or requiring internet access.
"""

from typing import Any, Protocol

from pydantic import BaseModel, Field

from libs.common.logging import get_logger

logger = get_logger("llm.provider")


class LLMMessage(BaseModel):
    """Single message in a conversational LLM exchange.

    Follows standard chat completion message schema: role ('system', 'user', 'assistant') and content.
    """

    role: str = Field(description="'system', 'user', or 'assistant'")
    content: str = Field(description="Message body text")


class LLMResponse(BaseModel):
    """Normalized response from LLM completion.

    Normalizes outputs across all providers (OpenAI, Anthropic, Gemini, Ollama)
    so downstream callers receive a consistent schema with token usage counts.
    """

    content: str = Field(description="Generated text response")
    prompt_tokens: int = Field(default=0, description="Input tokens consumed")
    completion_tokens: int = Field(default=0, description="Output tokens generated")
    model: str = Field(default="mock-model", description="Model name that generated the response")


class LLMProvider(Protocol):
    """Protocol for LLM providers (structural subtyping / duck typing).

    Any class implementing `complete(messages, ...)` satisfies this protocol.
    """

    def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> LLMResponse:
        """Execute synchronous completion."""
        ...


class MockLLMProvider:
    """Deterministic LLM provider for unit tests, offline development, and CI regression checks.

    Behaviors:
    1. If `canned_response` was provided at init, returns that string verbatim.
    2. If the user prompt contains '[Source 1]', dynamically extracts the text from Source 1
       and returns a grounded answer citing '[1]'.
    3. If the prompt contains words like 'unknown' or 'missing', generates a refusal ("I don't know").
    """

    def __init__(self, canned_response: str | None = None) -> None:
        """Initialize mock provider.

        Args:
            canned_response: Optional fixed string to return for all completions.
        """
        self.canned_response = canned_response

    def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> LLMResponse:
        """Return canned or synthesized deterministic response based on prompt context."""
        _ = (temperature, max_tokens, kwargs)
        if self.canned_response is not None:
            return LLMResponse(
                content=self.canned_response,
                prompt_tokens=50,
                completion_tokens=25,
                model="mock-deterministic",
            )

        # Inspect user prompt for context clues
        last_message = messages[-1].content if messages else ""

        if "[Source 1]" in last_message:
            # Extract content line from [Source 1] block to guarantee true grounding
            after_s1 = last_message.split("[Source 1]", 1)[1]
            first_line = ""
            for line in after_s1.splitlines():
                sline = line.strip()
                if sline and not sline.startswith("(") and not sline.startswith("Question:"):
                    first_line = sline
                    break
            grounded_text = first_line if first_line else "the service requires health probes and OpenTelemetry"
            return LLMResponse(
                content=f"According to documentation, {grounded_text.rstrip('.')} [1].",
                prompt_tokens=120,
                completion_tokens=30,
                model="mock-grounded",
            )

        if "unknown" in last_message.lower() or "missing" in last_message.lower():
            return LLMResponse(
                content="I don't know based on the provided documents.",
                prompt_tokens=40,
                completion_tokens=10,
                model="mock-refusal",
            )

        return LLMResponse(
            content="Grounded operation completed successfully [1].",
            prompt_tokens=50,
            completion_tokens=15,
            model="mock-default",
        )


class LiteLLMProvider:
    """Production provider using LiteLLM abstraction across OpenAI, Anthropic, Bedrock, and Ollama.

    Features:
    - Provider agnosticism: 'gpt-4o', 'claude-3-5-sonnet', or 'ollama/llama3' use the same call.
    - Automatic fallback: If the primary cloud provider returns a 5xx or rate limit, automatically
      retries against `fallback_model` (e.g. local Ollama instance).
    """

    def __init__(self, model_name: str = "gpt-4o-mini", fallback_model: str | None = "ollama/llama3") -> None:
        """Initialize provider with primary and fallback model names.

        Args:
            model_name: Primary model name (e.g. 'gpt-4o-mini').
            fallback_model: Fallback model if primary fails (e.g. 'ollama/llama3').
        """
        self.model_name = model_name
        self.fallback_model = fallback_model

    def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call LiteLLM completion with configured model and fallback."""
        try:
            import litellm

            formatted = [{"role": m.role, "content": m.content} for m in messages]
            response = litellm.completion(
                model=self.model_name,
                messages=formatted,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            content = str(response.choices[0].message.content or "")
            usage = getattr(response, "usage", None)
            p_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
            c_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

            return LLMResponse(
                content=content,
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                model=self.model_name,
            )
        except Exception as e:
            logger.warning("LiteLLM completion failed, trying fallback", error=str(e))
            if self.fallback_model:
                try:
                    import litellm

                    formatted = [{"role": m.role, "content": m.content} for m in messages]
                    response = litellm.completion(
                        model=self.fallback_model,
                        messages=formatted,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
                    content = str(response.choices[0].message.content or "")
                    return LLMResponse(
                        content=content,
                        prompt_tokens=0,
                        completion_tokens=0,
                        model=self.fallback_model,
                    )
                except Exception as fb_err:
                    logger.error("Fallback LLM model also failed", error=str(fb_err))
                    raise
            raise
