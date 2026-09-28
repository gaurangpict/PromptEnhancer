"""FastAPI app. Run with: uvicorn app.main:app --reload"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.analyzer import analyze
from app.config import Settings, get_settings
from app.llm import (
    LLMClient,
    LLMConfigError,
    LLMError,
    LLMOutputError,
    LLMRateLimitError,
    LLMUnavailableError,
    NIMClient,
)
from app.schemas import QuestionsRequest, QuestionsResponse

app = FastAPI(
    title="PromptEnhancer",
    version="0.1.0",
    description="Asks a few one-tap questions, then turns a rough prompt into a precise one.",
)


@lru_cache
def _nim_client() -> NIMClient:
    return NIMClient(get_settings())


def get_llm() -> LLMClient:
    return _nim_client()


SettingsDep = Annotated[Settings, Depends(get_settings)]
LLMDep = Annotated[LLMClient, Depends(get_llm)]

_STATUS_FOR_ERROR: dict[type[LLMError], int] = {
    LLMConfigError: 500,
    LLMRateLimitError: 429,
    LLMUnavailableError: 503,
    LLMOutputError: 502,
}


@app.exception_handler(LLMError)
async def _llm_error(_: Request, exc: LLMError) -> JSONResponse:
    status = next((s for cls, s in _STATUS_FOR_ERROR.items() if isinstance(exc, cls)), 500)
    return JSONResponse(status_code=status, content={"detail": str(exc)})


@app.get("/api/health")
async def health(settings: SettingsDep) -> dict[str, object]:
    return {
        "status": "ok",
        "llm_configured": settings.nvidia_api_key is not None,
        "question_model": settings.question_model,
    }


@app.post("/api/questions", response_model=QuestionsResponse)
async def questions(req: QuestionsRequest, llm: LLMDep, settings: SettingsDep) -> QuestionsResponse:
    if len(req.prompt) > settings.max_prompt_chars:
        raise HTTPException(
            status_code=422,
            detail=f"Prompt is too long ({len(req.prompt)} characters, "
            f"max {settings.max_prompt_chars}).",
        )
    analysis = await analyze(req.prompt, llm, settings)
    return analysis.to_response()
