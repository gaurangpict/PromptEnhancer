"""A scripted stand-in for the LLM, so tests run offline and give the same result every time."""

import json
from typing import Any

from app.llm import LLMResponse, Usage


class FakeLLM:
    def __init__(self, *replies: str | dict[str, Any] | Exception) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "model": model})
        if not self._replies:
            raise AssertionError("FakeLLM ran out of scripted replies")
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return LLMResponse(text=text, usage=Usage(calls=1, prompt_tokens=100, completion_tokens=50))


def gap(
    id: str,
    impact: int,
    confidence: int,
    question: str | None = None,
    kind: str = "yesno",
    options: list[dict[str, str]] | None = None,
    default: int = 0,
) -> dict[str, Any]:
    """A valid raw gap as the model would return it; override fields to break it."""
    if options is None:
        options = {
            "yesno": [
                {"label": "Yes", "effect": f"Include {id}."},
                {"label": "No", "effect": f"Do not include {id}."},
            ],
            "either": [
                {"label": "A", "effect": f"Use option A for {id}."},
                {"label": "B", "effect": f"Use option B for {id}."},
            ],
            "choice": [
                {"label": "Short", "effect": f"Keep {id} short."},
                {"label": "Medium", "effect": f"Keep {id} medium."},
                {"label": "Long", "effect": f"Make {id} long."},
            ],
        }[kind]
    return {
        "id": id,
        "dimension": "scope",
        "impact": impact,
        "confidence": confidence,
        "question": question or f"Include {id.replace('_', ' ')}?",
        "kind": kind,
        "options": options,
        "default": default,
    }


def analysis(*gaps: dict[str, Any], task_type: str = "writing", known: list[str] | None = None):
    return {"task_type": task_type, "known": known or [], "gaps": list(gaps)}
