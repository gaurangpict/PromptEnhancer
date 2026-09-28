"""Data shapes: what the model returns, and what the API returns."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

TaskType = Literal["code", "writing", "analysis", "learning", "other"]
Kind = Literal["yesno", "either", "choice"]

# How many options each question kind must have (see docs/question-design.md).
OPTION_COUNTS: dict[str, tuple[int, int]] = {"yesno": (2, 2), "either": (2, 2), "choice": (3, 4)}


class Option(BaseModel):
    label: str = Field(min_length=1, max_length=60)
    # The exact instruction added to the final prompt when this option is chosen.
    effect: str = Field(min_length=1, max_length=300)

    @field_validator("label", "effect", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> Any:
        return v.strip() if isinstance(v, str) else v


class Decision(BaseModel):
    """One gap in the user's prompt, written as a question with options.

    Asked decisions are answered by the user; assumed ones use `default` and
    are shown so the user can flip them.
    """

    id: str = Field(min_length=1, max_length=40)
    dimension: str = "other"
    impact: int = Field(ge=1, le=5)
    confidence: int = Field(ge=1, le=5)
    question: str = Field(min_length=1, max_length=200)
    kind: Kind
    options: list[Option]
    default: int = 0

    @field_validator("question", "dimension", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> Any:
        return v.strip() if isinstance(v, str) else v

    @model_validator(mode="after")
    def _check_options(self) -> "Decision":
        low, high = OPTION_COUNTS[self.kind]
        if not low <= len(self.options) <= high:
            expected = str(low) if low == high else f"{low}-{high}"
            raise ValueError(f"'{self.kind}' needs {expected} options, got {len(self.options)}")
        if not 0 <= self.default < len(self.options):
            raise ValueError(f"default {self.default} is not a valid option index")
        return self

    @property
    def value(self) -> int:
        """How much asking this is worth: high impact and low confidence first."""
        return self.impact * (6 - self.confidence)


# --- Model output ----------------------------------------------------------------


class RawAnalysis(BaseModel):
    """The analysis model's reply. Gaps stay as raw dicts here so one malformed
    gap can be dropped on its own instead of failing the whole reply."""

    task_type: str = "other"
    known: list[str] = Field(default_factory=list)
    gaps: list[dict[str, Any]] = Field(default_factory=list)


# --- API -------------------------------------------------------------------------


class QuestionsRequest(BaseModel):
    prompt: str

    @field_validator("prompt")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("prompt must not be empty")
        return v


class QuestionsResponse(BaseModel):
    task_type: TaskType
    known: list[str]
    questions: list[Decision]  # ask the user, most valuable first
    assumptions: list[Decision]  # decided automatically, shown so the user can flip them
