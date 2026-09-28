"""LLM access: a small client for OpenAI-compatible APIs (NVIDIA NIM by default),
plus helpers for getting reliable JSON out of a model.
"""

import json
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import httpx2
import openai
from pydantic import BaseModel, ValidationError

from app.config import Settings

# --- Errors -------------------------------------------------------------------
# Each carries a message that is safe to show to the end user.


class LLMError(Exception):
    """Base class for LLM failures."""


class LLMConfigError(LLMError):
    """Missing/invalid API key or unknown model: the server needs fixing."""


class LLMRateLimitError(LLMError):
    """The provider's (free) rate limit was hit."""


class LLMUnavailableError(LLMError):
    """Timeout, connection failure, or a provider-side error."""


class LLMOutputError(LLMError):
    """The model kept returning output we could not use."""


# --- Client -------------------------------------------------------------------


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens


@dataclass
class LLMResponse:
    text: str
    usage: Usage


class LLMClient(Protocol):
    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> LLMResponse: ...


class NIMClient:
    """Chat client for NVIDIA NIM. NIM speaks the OpenAI API, so we reuse the
    official `openai` library and only change the base URL."""

    def __init__(
        self,
        settings: Settings,
        *,
        max_retries: int = 2,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        if settings.nvidia_api_key is None:
            raise LLMConfigError(
                "The server has no NVIDIA_API_KEY. Get a free key at build.nvidia.com "
                "and put it in the .env file."
            )
        self._client = openai.AsyncOpenAI(
            base_url=settings.nim_base_url,
            api_key=settings.nvidia_api_key.get_secret_value(),
            timeout=settings.request_timeout_s,
            max_retries=max_retries,  # the library backs off and retries 429/5xx itself
            http_client=http_client,
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        try:
            resp = await self._client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except openai.RateLimitError as e:
            raise LLMRateLimitError(
                "The free NVIDIA NIM rate limit was reached. Please wait a minute and try again."
            ) from e
        except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
            raise LLMConfigError("NVIDIA NIM rejected the API key. Check NVIDIA_API_KEY.") from e
        except openai.NotFoundError as e:
            raise LLMConfigError(
                f"Model '{model}' was not found on NVIDIA NIM. "
                "Check the model name on build.nvidia.com."
            ) from e
        except (openai.APIConnectionError, openai.APIStatusError) as e:
            # APIConnectionError also covers timeouts.
            raise LLMUnavailableError(
                "NVIDIA NIM is not responding right now. Please try again shortly."
            ) from e

        text = resp.choices[0].message.content if resp.choices else None
        usage = Usage(
            calls=1,
            prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
        )
        return LLMResponse(text=text or "", usage=usage)


# --- JSON helpers ---------------------------------------------------------------

T = TypeVar("T", bound=BaseModel)


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model reply.

    Models often wrap JSON in ```json fences or add a sentence before/after it,
    so we start at the first '{' and let the decoder find where the object ends.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in the reply")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("the reply is not a JSON object")
    return obj


def _describe_error(e: Exception) -> str:
    if isinstance(e, ValidationError):
        parts = [
            f"{'.'.join(str(p) for p in err['loc']) or 'root'}: {err['msg']}"
            for err in e.errors()[:3]
        ]
        return "; ".join(parts)
    return str(e)[:300]


async def complete_json(
    llm: LLMClient,
    messages: list[dict[str, str]],
    schema: type[T],
    *,
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    retries: int = 1,
) -> tuple[T, Usage]:
    """Ask for JSON matching `schema`. If the reply is broken, show the model its
    own reply plus the error and ask once more before giving up."""
    usage = Usage()
    history = list(messages)
    for _ in range(retries + 1):
        resp = await llm.chat(history, model=model, temperature=temperature, max_tokens=max_tokens)
        usage.add(resp.usage)
        try:
            return schema.model_validate(parse_json_object(resp.text)), usage
        except (ValueError, ValidationError) as e:  # JSONDecodeError is a ValueError
            history = [
                *messages,
                {"role": "assistant", "content": resp.text},
                {
                    "role": "user",
                    "content": f"Your reply was not valid ({_describe_error(e)}). "
                    "Reply again with only the corrected JSON object.",
                },
            ]
    raise LLMOutputError("The model returned an unusable response. Please try again.")
