"""The question engine: prompt → questions to ask + visible assumptions.

One LLM call finds and scores the gaps; everything after that is plain code,
so the behaviour is predictable and testable. See docs/question-design.md.
"""

import re
from dataclasses import dataclass, field
from typing import Any, get_args

from pydantic import ValidationError

from app.config import Settings
from app.llm import LLMClient, Usage, complete_json
from app.prompts import analysis_messages
from app.schemas import Decision, QuestionsResponse, RawAnalysis, TaskType

TASK_TYPES: tuple[str, ...] = get_args(TaskType)


@dataclass
class Analysis:
    task_type: str
    known: list[str]
    questions: list[Decision]
    assumptions: list[Decision]
    ignored: list[Decision] = field(default_factory=list)
    # Why decisions were dropped or demoted. Useful in evals and for tuning.
    issues: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    def to_response(self) -> QuestionsResponse:
        return QuestionsResponse(
            task_type=self.task_type,  # type: ignore[arg-type]
            known=self.known,
            questions=self.questions,
            assumptions=self.assumptions,
        )


async def analyze(prompt: str, llm: LLMClient, settings: Settings) -> Analysis:
    raw, usage = await complete_json(
        llm,
        analysis_messages(prompt),
        RawAnalysis,
        model=settings.question_model,
        temperature=0.2,
        max_tokens=3000,
    )
    decisions, issues = parse_decisions(raw.gaps)
    decisions, dropped = drop_useless(decisions)
    questions, assumptions, ignored, demoted = select(decisions, settings)

    task_type = raw.task_type.strip().lower()
    return Analysis(
        task_type=task_type if task_type in TASK_TYPES else "other",
        known=[k.strip() for k in raw.known if isinstance(k, str) and k.strip()],
        questions=questions,
        assumptions=assumptions,
        ignored=ignored,
        issues=issues + dropped + demoted,
        usage=usage,
    )


# --- Steps -------------------------------------------------------------------


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip(" .!?")


def parse_decisions(raw_gaps: list[dict[str, Any]]) -> tuple[list[Decision], list[str]]:
    """Validate each gap on its own; a malformed one is dropped, not fatal."""
    decisions: list[Decision] = []
    issues: list[str] = []
    for i, gap in enumerate(raw_gaps):
        gap = dict(gap)
        gap["id"] = _slug(str(gap.get("id") or "")) or f"gap_{i + 1}"
        try:
            decisions.append(Decision.model_validate(gap))
        except ValidationError as e:
            first = e.errors()[0]
            issues.append(f"{gap['id']}: dropped, invalid ({first['msg']})")
    return decisions, issues


def drop_useless(decisions: list[Decision]) -> tuple[list[Decision], list[str]]:
    """Drop decisions whose answer changes nothing, and duplicates."""
    kept: list[Decision] = []
    issues: list[str] = []
    # Most valuable first, so a duplicate loses to the better-scored copy.
    for d in sorted(decisions, key=lambda d: d.value, reverse=True):
        effects = {_norm(o.effect) for o in d.options}
        if len(effects) < len(d.options):
            issues.append(f"{d.id}: dropped, options have the same effect")
            continue
        duplicate = next(
            (k for k in kept if k.id == d.id or _norm(k.question) == _norm(d.question)), None
        )
        if duplicate:
            issues.append(f"{d.id}: dropped, duplicate of {duplicate.id}")
            continue
        kept.append(d)
    return kept, issues


def phrasing_problem(d: Decision, settings: Settings) -> str | None:
    """Rules a regex can check reliably. The rest is judged in evals."""
    if len(d.question.split()) > settings.max_question_words:
        return f"longer than {settings.max_question_words} words"
    if len(d.question) > settings.max_question_chars:
        return f"longer than {settings.max_question_chars} characters"
    if d.kind == "yesno" and re.search(r"\bor\b", d.question, re.IGNORECASE):
        return "yes/no question offering alternatives (should be 'either')"
    return None


def select(
    decisions: list[Decision], settings: Settings
) -> tuple[list[Decision], list[Decision], list[Decision], list[str]]:
    """Decide ask / assume / ignore for each decision (code, not the model)."""
    ask: list[Decision] = []
    assume: list[Decision] = []
    ignore: list[Decision] = []
    issues: list[str] = []

    for d in decisions:
        if d.impact <= settings.ignore_max_impact:
            ignore.append(d)
        elif d.impact >= settings.ask_min_impact and d.confidence <= settings.ask_max_confidence:
            problem = phrasing_problem(d, settings)
            if problem:
                issues.append(f"{d.id}: assumed instead of asked, {problem}")
                assume.append(d)
            else:
                ask.append(d)
        else:
            assume.append(d)

    ask.sort(key=lambda d: d.value, reverse=True)
    overflow = ask[settings.max_questions :]
    ask = ask[: settings.max_questions]
    for d in overflow:
        issues.append(
            f"{d.id}: assumed instead of asked, over the {settings.max_questions}-question limit"
        )
    assume.extend(overflow)
    assume.sort(key=lambda d: d.impact, reverse=True)
    return ask, assume, ignore, issues
