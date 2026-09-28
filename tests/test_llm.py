import httpx2
import pytest
from pydantic import BaseModel

from app.config import Settings
from app.llm import (
    LLMConfigError,
    LLMOutputError,
    LLMRateLimitError,
    LLMUnavailableError,
    NIMClient,
    complete_json,
    parse_json_object,
)
from tests.fakes import FakeLLM

# --- parse_json_object ----------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        'Sure! Here is the JSON:\n{"a": 1}\nHope this helps {really}.',
    ],
)
def test_parse_json_object_tolerates_wrapping(text):
    assert parse_json_object(text) == {"a": 1}


@pytest.mark.parametrize("text", ["no json here", "[1, 2]", '{"a": 1'])
def test_parse_json_object_rejects_bad_replies(text):
    with pytest.raises(ValueError):
        parse_json_object(text)


# --- complete_json -------------------------------------------------------------


class Thing(BaseModel):
    name: str
    count: int


async def test_complete_json_first_try():
    llm = FakeLLM({"name": "x", "count": 2})
    thing, usage = await complete_json(llm, [{"role": "user", "content": "hi"}], Thing, model="m")
    assert thing == Thing(name="x", count=2)
    assert usage.calls == 1


async def test_complete_json_retries_with_the_error():
    llm = FakeLLM("not json at all", {"name": "x", "count": 2})
    thing, usage = await complete_json(llm, [{"role": "user", "content": "hi"}], Thing, model="m")
    assert thing.count == 2
    assert usage.calls == 2
    retry_messages = llm.calls[1]["messages"]
    assert retry_messages[-2] == {"role": "assistant", "content": "not json at all"}
    assert "not valid" in retry_messages[-1]["content"]


async def test_complete_json_retry_names_the_schema_problem():
    llm = FakeLLM({"name": "x"}, {"name": "x", "count": 1})
    await complete_json(llm, [{"role": "user", "content": "hi"}], Thing, model="m")
    assert "count" in llm.calls[1]["messages"][-1]["content"]


async def test_complete_json_gives_up_after_retries():
    llm = FakeLLM("bad", "still bad")
    with pytest.raises(LLMOutputError):
        await complete_json(llm, [{"role": "user", "content": "hi"}], Thing, model="m")


# --- NIMClient (real openai library, fake HTTP) ------------------------------------


def _client(handler) -> NIMClient:
    settings = Settings(nvidia_api_key="nvapi-test", _env_file=None)
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return NIMClient(settings, max_retries=0, http_client=http)


def test_nim_client_requires_a_key():
    with pytest.raises(LLMConfigError):
        NIMClient(Settings(nvidia_api_key=None, _env_file=None))


async def test_nim_client_returns_text_and_usage():
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path.endswith("/chat/completions")
        assert request.headers["authorization"] == "Bearer nvapi-test"
        return httpx2.Response(
            200,
            json={
                "id": "1",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "hello"},
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            },
        )

    resp = await _client(handler).chat([{"role": "user", "content": "hi"}], model="m")
    assert resp.text == "hello"
    assert (resp.usage.prompt_tokens, resp.usage.completion_tokens) == (7, 3)


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (429, LLMRateLimitError),
        (401, LLMConfigError),
        (404, LLMConfigError),
        (500, LLMUnavailableError),
    ],
)
async def test_nim_client_maps_http_errors(status, error):
    client = _client(lambda request: httpx2.Response(status, json={"error": "nope"}))
    with pytest.raises(error):
        await client.chat([{"role": "user", "content": "hi"}], model="m")


async def test_nim_client_maps_connection_errors():
    def handler(request):
        raise httpx2.ConnectError("down", request=request)

    with pytest.raises(LLMUnavailableError):
        await _client(handler).chat([{"role": "user", "content": "hi"}], model="m")
