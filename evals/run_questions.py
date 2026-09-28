"""Evaluate the question engine on evals/questions.jsonl.

    python -m evals.run_questions --tag baseline
    python -m evals.run_questions --tag baseline --only email-leave,blog-ai

Needs NVIDIA_API_KEY. Uses about 2 LLM calls per case (analysis + judge), so a
full run uses ~30 calls of the free quota. Results go to evals/results/<tag>.json;
compare two runs by comparing their "summary" sections.

What is measured (see docs/question-design.md):
- code checks: question count in the expected range, words per question, and
  how many decisions the code checks had to drop or demote ("issues")
- LLM judge: coverage of the expected gaps, questions about things the prompt
  already states, and the phrasing rules a regex can't check (one idea per
  question, neutral, specific to the prompt, same language as the prompt)
"""

import argparse
import asyncio
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from app.analyzer import Analysis, analyze
from app.config import Settings, get_settings
from app.llm import LLMClient, LLMError, NIMClient, complete_json

HERE = Path(__file__).parent
DATASET = HERE / "questions.jsonl"
RESULTS = HERE / "results"


# --- Judge ------------------------------------------------------------------------


class GapVerdict(BaseModel):
    gap: str
    covered_by: Literal["question", "assumption", "missing"]


class StatedVerdict(BaseModel):
    fact: str
    asked_again: bool


class QuestionVerdict(BaseModel):
    id: str
    one_idea: bool
    neutral: bool
    specific: bool
    same_language: bool


class JudgeResult(BaseModel):
    gaps: list[GapVerdict]
    stated: list[StatedVerdict]
    questions: list[QuestionVerdict]


JUDGE_SYSTEM = """You grade a tool that reads a user's prompt for an AI assistant and decides what to ask the user before improving the prompt. Some decisions are asked as questions; others are assumed with a default. Be strict and literal. Return only one JSON object, with no text before or after it."""


def judge_messages(case: dict[str, Any], result: Analysis) -> list[dict[str, str]]:
    questions = [
        {"id": d.id, "question": d.question, "options": [o.label for o in d.options]}
        for d in result.questions
    ]
    assumptions = [
        {"id": d.id, "decision": d.question, "assumed": d.options[d.default].effect}
        for d in result.assumptions
    ]
    expect = case["expect"]
    user = f"""<user_prompt>
{case["prompt"]}
</user_prompt>

<questions>
{json.dumps(questions, ensure_ascii=False, indent=1)}
</questions>

<assumptions>
{json.dumps(assumptions, ensure_ascii=False, indent=1)}
</assumptions>

<expected_gaps>
{json.dumps(expect["gaps"], ensure_ascii=False)}
</expected_gaps>

<stated_facts>
{json.dumps(expect["stated"], ensure_ascii=False)}
</stated_facts>

Grade the tool's output:
1. For each expected gap: is it decided by a question, by an assumption, or missing? It counts as covered when a question or assumption settles the same thing, even in different words.
2. For each stated fact: does any question ask about it again? (Assumptions do not count.)
3. For each question:
   - one_idea: it asks about exactly one thing.
   - neutral: it does not hint at the answer it expects.
   - specific: it uses the prompt's own topic, not generic wording like "What is the purpose?".
   - same_language: it is written in the same language as the user's prompt.

Return JSON in this shape, one entry per expected gap, stated fact and question, in the given order:
{{"gaps": [{{"gap": "...", "covered_by": "question|assumption|missing"}}], "stated": [{{"fact": "...", "asked_again": false}}], "questions": [{{"id": "...", "one_idea": true, "neutral": true, "specific": true, "same_language": true}}]}}"""
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


# --- Running ------------------------------------------------------------------------


def load_cases(only: set[str] | None = None) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    return [c for c in cases if not only or c["id"] in only]


async def run_case(
    case: dict[str, Any], llm: LLMClient, judge: LLMClient, settings: Settings
) -> dict[str, Any]:
    row: dict[str, Any] = {"id": case["id"]}
    start = time.perf_counter()
    try:
        result = await analyze(case["prompt"], llm, settings)
    except LLMError as e:
        return {**row, "error": str(e)}
    row["seconds"] = round(time.perf_counter() - start, 2)

    low, high = case["expect"]["questions"]
    n = len(result.questions)
    row.update(
        questions=[d.question for d in result.questions],
        assumptions=[d.options[d.default].effect for d in result.assumptions],
        issues=result.issues,
        n_questions=n,
        count_ok=low <= n <= high,
        avg_words=round(statistics.mean(len(d.question.split()) for d in result.questions), 1)
        if n
        else 0,
        tokens=result.usage.prompt_tokens + result.usage.completion_tokens,
    )

    try:
        verdict, _ = await complete_json(
            judge,
            judge_messages(case, result),
            JudgeResult,
            model=settings.judge_model,
            temperature=0.0,
            max_tokens=2000,
        )
    except LLMError as e:
        return {**row, "judge_error": str(e)}

    expected = case["expect"]["gaps"]
    covered = sum(v.covered_by != "missing" for v in verdict.gaps[: len(expected)])
    rules = [v.one_idea and v.neutral and v.specific and v.same_language for v in verdict.questions]
    row.update(
        coverage=round(covered / len(expected), 2) if expected else None,
        missing=[v.gap for v in verdict.gaps if v.covered_by == "missing"],
        redundant=[v.fact for v in verdict.stated if v.asked_again],
        rules_ok=round(sum(rules) / len(rules), 2) if rules else None,
        rule_failures=[
            v.model_dump()
            for v in verdict.questions
            if not (v.one_idea and v.neutral and v.specific and v.same_language)
        ],
    )
    return row


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return round(statistics.mean(present), 2) if present else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if "error" not in r]
    judged = [r for r in ok if "judge_error" not in r]
    return {
        "cases": len(rows),
        "errors": len(rows) - len(ok),
        "judge_errors": len(ok) - len(judged),
        "count_in_range": _mean([float(r["count_ok"]) for r in ok]),
        "avg_questions": _mean([r["n_questions"] for r in ok]),
        "avg_words_per_question": _mean([r["avg_words"] for r in ok if r["n_questions"]]),
        "gap_coverage": _mean([r["coverage"] for r in judged]),
        "redundant_questions": sum(len(r["redundant"]) for r in judged),
        "rule_pass_rate": _mean([r["rules_ok"] for r in judged]),
        "code_check_issues": sum(len(r["issues"]) for r in ok),
        "avg_seconds": _mean([r["seconds"] for r in ok]),
        "total_tokens": sum(r["tokens"] for r in ok),
    }


def print_report(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    def fmt(v: Any) -> str:
        return "-" if v is None else str(v)

    print(f"{'case':<22}{'q':>3} {'range':>6}{'cover':>7}{'redund':>7}{'rules':>7}{'issues':>7}")
    for r in rows:
        if "error" in r:
            print(f"{r['id']:<22} ERROR: {r['error']}")
            continue
        print(
            f"{r['id']:<22}{r['n_questions']:>3} {'ok' if r['count_ok'] else 'NO':>6}"
            f"{fmt(r.get('coverage')):>7}{len(r.get('redundant', [])):>7}"
            f"{fmt(r.get('rules_ok')):>7}{len(r['issues']):>7}"
        )
    print()
    for key, value in summary.items():
        print(f"  {key:<24} {fmt(value)}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tag", required=True, help="name for this run, e.g. baseline")
    parser.add_argument("--only", help="comma-separated case ids")
    args = parser.parse_args()

    settings = get_settings()
    client = NIMClient(settings)  # same client for both; the model differs per call
    cases = load_cases(set(args.only.split(",")) if args.only else None)

    rows = []
    for case in cases:  # one at a time, to stay inside the free rate limit
        print(f"running {case['id']}...", flush=True)
        rows.append(await run_case(case, client, client, settings))

    summary = summarize(rows)
    print_report(rows, summary)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{args.tag}.json"
    out.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "time": datetime.now(UTC).isoformat(timespec="seconds"),
                "settings": settings.model_dump(
                    include={
                        "question_model",
                        "judge_model",
                        "max_questions",
                        "ask_min_impact",
                        "ask_max_confidence",
                        "ignore_max_impact",
                    }
                ),
                "summary": summary,
                "cases": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\nsaved {out}")


if __name__ == "__main__":
    asyncio.run(main())
