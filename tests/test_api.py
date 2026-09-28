import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.llm import LLMConfigError, LLMRateLimitError, LLMUnavailableError
from app.main import app, get_llm
from tests.fakes import FakeLLM, analysis, gap


@pytest.fixture
def settings() -> Settings:
    return Settings(nvidia_api_key="nvapi-test", max_prompt_chars=50, _env_file=None)


@pytest.fixture
def client(settings):
    app.dependency_overrides[get_settings] = lambda: settings
    yield TestClient(app)
    app.dependency_overrides.clear()


def use_llm(llm) -> None:
    app.dependency_overrides[get_llm] = lambda: llm


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["llm_configured"] is True


def test_questions_returns_questions_and_assumptions(client):
    use_llm(
        FakeLLM(
            analysis(
                gap("leave_type", impact=5, confidence=2),
                gap("length", impact=3, confidence=5),
                known=["recipient: boss"],
            )
        )
    )
    resp = client.post("/api/questions", json={"prompt": "email my boss about leave"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["task_type"] == "writing"
    assert [q["id"] for q in body["questions"]] == ["leave_type"]
    assert [a["id"] for a in body["assumptions"]] == ["length"]
    assert body["questions"][0]["options"][0]["effect"]


@pytest.mark.parametrize("prompt", ["", "   "])
def test_blank_prompt_is_rejected(client, prompt):
    use_llm(FakeLLM())
    assert client.post("/api/questions", json={"prompt": prompt}).status_code == 422


def test_too_long_prompt_is_rejected_before_calling_the_model(client):
    llm = FakeLLM()
    use_llm(llm)
    resp = client.post("/api/questions", json={"prompt": "x" * 51})
    assert resp.status_code == 422
    assert "too long" in resp.json()["detail"]
    assert llm.calls == []


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (LLMRateLimitError("slow down"), 429),
        (LLMUnavailableError("down"), 503),
        (LLMConfigError("no key"), 500),
    ],
)
def test_llm_errors_become_friendly_http_errors(client, error, status):
    use_llm(FakeLLM(error))
    resp = client.post("/api/questions", json={"prompt": "hello"})
    assert resp.status_code == status
    assert resp.json() == {"detail": str(error)}


def test_missing_api_key_gives_a_clear_error(client, monkeypatch, tmp_path):
    from app.main import _nim_client

    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    get_settings.cache_clear()
    _nim_client.cache_clear()

    resp = client.post("/api/questions", json={"prompt": "hello"})
    assert resp.status_code == 500
    assert "NVIDIA_API_KEY" in resp.json()["detail"]
    get_settings.cache_clear()


def test_unusable_model_output_is_a_502(client):
    use_llm(FakeLLM("garbage", "more garbage"))
    resp = client.post("/api/questions", json={"prompt": "hello"})
    assert resp.status_code == 502
