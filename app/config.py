"""Settings, read from environment variables or a `.env` file."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- NVIDIA NIM ---------------------------------------------------------
    nvidia_api_key: SecretStr | None = None
    nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    # Model names change on build.nvidia.com from time to time; override in .env.
    question_model: str = "meta/llama-3.3-70b-instruct"
    # Evals only: a different, stronger model grades the output (see evals/).
    judge_model: str = "meta/llama-3.1-405b-instruct"
    request_timeout_s: float = 60.0

    # --- Input limits -------------------------------------------------------
    max_prompt_chars: int = 4000

    # --- Question selection (see docs/question-design.md) --------------------
    max_questions: int = Field(5, ge=0)
    # Ask when impact >= ask_min_impact AND confidence <= ask_max_confidence.
    ask_min_impact: int = Field(3, ge=1, le=5)
    ask_max_confidence: int = Field(3, ge=1, le=5)
    # Gaps with impact <= ignore_max_impact are dropped entirely.
    ignore_max_impact: int = Field(1, ge=0, le=5)

    # --- Question phrasing checks --------------------------------------------
    max_question_words: int = 15
    max_question_chars: int = 110


@lru_cache
def get_settings() -> Settings:
    return Settings()
