"""Checks that the eval runner works end to end, using fakes (no quota used)."""

from app.config import Settings
from evals.run_questions import load_cases, run_case, summarize
from tests.fakes import FakeLLM, analysis, gap


def test_dataset_is_well_formed():
    cases = load_cases()
    assert len(cases) >= 15
    assert len({c["id"] for c in cases}) == len(cases)
    for c in cases:
        low, high = c["expect"]["questions"]
        assert 0 <= low <= high <= 5, c["id"]
        assert isinstance(c["expect"]["gaps"], list)
        assert isinstance(c["expect"]["stated"], list)


async def test_run_case_and_summary():
    settings = Settings(nvidia_api_key="nvapi-test", _env_file=None)
    case = {
        "id": "email-leave",
        "prompt": "write an email to my boss about leave",
        "expect": {"questions": [1, 3], "gaps": ["type of leave", "tone"], "stated": ["boss"]},
    }
    llm = FakeLLM(
        analysis(gap("leave_type", 5, 2, question="Planned leave or sick day?", kind="either"))
    )
    judge = FakeLLM(
        {
            "gaps": [
                {"gap": "type of leave", "covered_by": "question"},
                {"gap": "tone", "covered_by": "missing"},
            ],
            "stated": [{"fact": "boss", "asked_again": False}],
            "questions": [
                {
                    "id": "leave_type",
                    "one_idea": True,
                    "neutral": True,
                    "specific": True,
                    "same_language": True,
                }
            ],
        }
    )

    row = await run_case(case, llm, judge, settings)
    assert row["count_ok"] is True
    assert row["coverage"] == 0.5
    assert row["missing"] == ["tone"]
    assert row["rules_ok"] == 1.0
    assert judge.calls[0]["model"] == settings.judge_model

    summary = summarize([row, {"id": "broken", "error": "rate limit"}])
    assert summary["errors"] == 1
    assert summary["gap_coverage"] == 0.5
