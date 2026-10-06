"""Provider-agnostic LLM client layer with LiteLLM and deterministic mock provider."""

from typing import Any, Protocol

from pydantic import BaseModel, Field

from libs.common.logging import get_logger

logger = get_logger("llm.provider")


class LLMMessage(BaseModel):
    """Single message in a conversational LLM exchange."""

    role: str = Field(description="'system', 'user', or 'assistant'")
    content: str = Field(description="Message body text")


class LLMResponse(BaseModel):
    """Normalized response from LLM completion."""

    content: str = Field(description="Generated text response")
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    model: str = Field(default="mock-model")


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

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
    """Deterministic LLM provider for unit tests, offline development, and CI regression checks."""

    def __init__(self, canned_response: str | None = None) -> None:
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
    """Production provider using LiteLLM abstraction across OpenAI, Anthropic, Bedrock, and Ollama."""

    def __init__(self, model_name: str = "gpt-4o-mini", fallback_model: str | None = "ollama/llama3") -> None:
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
