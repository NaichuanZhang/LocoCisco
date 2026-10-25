"""The HTTP providers against a mock OpenAI-compatible server (httpx.MockTransport, no network): request
shapes, response parsing, and the whole strands tool-call loop (tools sent as Chat Completions functions,
streamed `tool_calls` deltas parsed back into a tool run) the way vLLM / llama.cpp answer."""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from strands import Agent, tool

from unspoken.contracts import RoleSpec
from unspoken.models.audio import pcm16_to_wav, wav_to_pcm16
from unspoken.models.openai_compat import (OpenAIAudioASR, OpenAIEmbedder, OpenAISpeechTTS, chat_params,
                                           scripted_model, strands_openai_model)

URL = "http://127.0.0.1:8010/v1"


def mocked(provider_cls, spec: RoleSpec, handler):
    obj = provider_cls(spec)
    obj._make_client = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler), headers={"Authorization": "Bearer local"})
    return obj


# --------------------------------------------------------------------------- asr
async def test_asr_posts_multipart_wav_and_parses_verbose_json():
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"] = str(req.url), req.headers.get("authorization")
        body = req.content
        seen["has_wav"] = b"RIFF" in body and b'filename="audio.wav"' in body
        seen["fields"] = {k: body.split(f'name="{k}"'.encode())[1].split(b"\r\n\r\n")[1].split(b"\r\n")[0].decode()
                          for k in ("model", "response_format", "language")}
        return httpx.Response(200, json={"text": " Send the deck by Friday. ", "language": "english",
                                         "words": [{"word": " Send", "start": 0.1, "end": 0.4}],
                                         "segments": [{"no_speech_prob": 0.2}]})

    asr = mocked(OpenAIAudioASR, RoleSpec("asr", "openai-audio", url=URL, model="large-v3-turbo", options={"api_key": "k"}), handler)
    t = await asr.transcribe(bytes(16000), 16000, lang="en")
    assert seen["url"] == f"{URL}/audio/transcriptions" and seen["has_wav"] and seen["auth"] == "Bearer local"
    assert seen["fields"] == {"model": "large-v3-turbo", "response_format": "verbose_json", "language": "en"}
    assert t.text == "Send the deck by Friday." and t.lang == "en" and abs(t.conf - 0.8) < 1e-9
    assert t.words[0].text == "Send" and t.words[0].end == 0.4 and t.latency_ms >= 0
    await asr.aclose()


async def test_asr_plain_text_and_json_responses_and_errors():
    plain = mocked(OpenAIAudioASR, RoleSpec("asr", "openai-audio", url=URL, options={"response_format": "text"}),
                   lambda r: httpx.Response(200, text="hello there\n", headers={"content-type": "text/plain"}))
    assert (await plain.transcribe(bytes(320), 16000, lang="zh")).text == "hello there"
    js = mocked(OpenAIAudioASR, RoleSpec("asr", "openai-audio", url=URL),
                lambda r: httpx.Response(200, json={"text": "ni hao", "language": "zh"}))
    t = await js.transcribe(bytes(320), 16000)
    assert t.text == "ni hao" and t.lang == "zh" and t.conf == 1.0 and t.words == []
    bad = mocked(OpenAIAudioASR, RoleSpec("asr", "openai-audio", url=URL), lambda r: httpx.Response(500, text="boom"))
    with pytest.raises(httpx.HTTPStatusError):
        await bad.transcribe(bytes(320), 16000)


async def test_healthy_uses_models_endpoint():
    codes = iter([200, 401, 503])
    p = mocked(OpenAIEmbedder, RoleSpec("embed", "openai", url=URL, model="e"), lambda r: httpx.Response(next(codes)))
    assert await p.healthy() and not await p.healthy() and not await p.healthy()
    down = mocked(OpenAIEmbedder, RoleSpec("embed", "openai", url=URL, model="e"),
                  lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused")))
    assert not await down.healthy()


async def test_file_credentials_authenticate_without_entering_request_options(tmp_path):
    secret = tmp_path / "local-key"
    secret.write_text("test-only-private-key\n")
    spec = RoleSpec("llm", "openai", url=URL, model="local-model",
                    options={"api_key_file": str(secret), "max_tokens": 50})
    model = strands_openai_model(spec)
    assert model.client_args["api_key"] == "test-only-private-key"
    assert chat_params(spec) == {"temperature": 0.0, "max_tokens": 50}
    provider = OpenAIEmbedder(spec)
    assert provider.client.headers["Authorization"] == "Bearer test-only-private-key"
    assert provider.extra_options() == {"max_tokens": 50}
    await provider.aclose()


def test_environment_credentials_fail_closed_and_stay_out_of_payloads(monkeypatch):
    spec = RoleSpec("llm", "openai", url=URL, model="local-model",
                    options={"api_key_env": "LOCOCISCO_TEST_KEY"})
    monkeypatch.delenv("LOCOCISCO_TEST_KEY", raising=False)
    with pytest.raises(ValueError, match="Missing or invalid"):
        strands_openai_model(spec)
    monkeypatch.setenv("LOCOCISCO_TEST_KEY", "test-env-key")
    assert strands_openai_model(spec).client_args["api_key"] == "test-env-key"
    assert chat_params(spec) == {"temperature": 0.0}


async def test_health_option_resolves_against_the_server_root_or_base():
    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(str(req.url))
        return httpx.Response(200)

    ready = mocked(OpenAIAudioASR, RoleSpec("asr", "openai-audio", url="http://127.0.0.1:8770/v1", options={"health": "/ready", "timeout_s": 20}), handler)
    assert ready.health == "http://127.0.0.1:8770/ready" and ready.timeout == 20.0 and await ready.healthy()
    models = mocked(OpenAIAudioASR, RoleSpec("asr", "openai-audio", url="http://127.0.0.1:8770/v1"), handler)
    full = mocked(OpenAIAudioASR, RoleSpec("asr", "openai-audio", url="http://127.0.0.1:8770/v1", options={"health": "http://127.0.0.1:9/healthz"}), handler)
    await models.healthy(); await full.healthy()
    assert seen == ["http://127.0.0.1:8770/ready", "http://127.0.0.1:8770/v1/models", "http://127.0.0.1:9/healthz"]


async def test_asr_passes_server_specific_options_as_form_fields():
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        body = req.content
        seen["diarization"] = body.split(b'name="diarization"')[1].split(b"\r\n\r\n")[1].split(b"\r\n")[0].decode()
        seen["has_health"] = b'name="health"' in body or b'name="timeout_s"' in body
        return httpx.Response(200, json={"text": "ok"})

    asr = mocked(OpenAIAudioASR, RoleSpec("asr", "openai-audio", url=URL, model="parakeet-tdt-0.6b-v3",
                                          options={"response_format": "verbose_json", "language": "en", "diarization": False,
                                                   "timeout_s": 20, "health": "/ready"}), handler)
    assert (await asr.transcribe(bytes(320), 16000)).text == "ok"
    assert seen == {"diarization": "false", "has_health": False}


def test_url_is_required_for_server_providers():
    with pytest.raises(ValueError, match="needs a url"):
        OpenAIEmbedder(RoleSpec("embed", "openai"))
    with pytest.raises(ValueError, match="needs a model id"):
        strands_openai_model(RoleSpec("llm", "openai", url=URL))


# --------------------------------------------------------------------------- tts / embed
async def test_tts_posts_json_and_returns_bytes():
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["json"] = json.loads(req.content)
        return httpx.Response(200, content=pcm16_to_wav(bytes(640), 24000), headers={"content-type": "audio/wav"})

    tts = mocked(OpenAISpeechTTS, RoleSpec("tts", "openai-speech", url=URL, model="kokoro", options={"voice": "af_heart", "speed": 1.1}), handler)
    wav = await tts.synthesize("ok")
    assert seen["json"] == {"input": "ok", "response_format": "wav", "model": "kokoro", "voice": "af_heart", "speed": 1.1}
    assert wav_to_pcm16(wav) == (bytes(640), 24000)
    await tts.synthesize("hi", voice="bm_george")
    assert seen["json"]["voice"] == "bm_george"


async def test_embedder_batches_and_orders_by_index():
    calls: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        calls.append(body)
        n = len(body["input"])
        data = [{"index": i, "embedding": [float(i), 1.0]} for i in reversed(range(n))]   # out of order on purpose
        return httpx.Response(200, json={"data": data, "model": body["model"]})

    e = mocked(OpenAIEmbedder, RoleSpec("embed", "openai", url=URL, model="bge", options={"batch": 2, "dimensions": 2}), handler)
    vecs = await e.embed(["a", "b", "c"])
    assert vecs == [[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]]
    assert [c["input"] for c in calls] == [["a", "b"], ["c"]] and calls[0]["dimensions"] == 2 and calls[0]["model"] == "bge"


# --------------------------------------------------------------------------- strands chat model
@pytest.mark.parametrize("tools", [None, []])
def test_no_tool_request_omits_empty_tools_for_vllm(tools):
    model = strands_openai_model(RoleSpec("llm", "openai", url=URL, model="local-model"))
    request = model.format_request([{"role": "user", "content": [{"text": "Reply OK."}]}], tools,
                                   tool_choice={"auto": {}})
    assert "tools" not in request
    assert "tool_choice" not in request


def test_chat_params_and_request_shape():
    spec = RoleSpec("llm", "openai", url=URL, model="qwen3-1.7b",
                    options={"max_tokens": 300, "temperature": 0, "timeout": 120, "top_p": 0.9,
                             "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}})
    assert chat_params(spec) == {"temperature": 0.0, "max_tokens": 300, "top_p": 0.9,
                                 "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    assert chat_params(spec, max_tokens=1)["max_tokens"] == 1
    model = strands_openai_model(spec)
    assert model.client_args == {"base_url": URL, "api_key": "local", "timeout": 120.0, "max_retries": 1}
    spec_tool = {"name": "lights_on", "description": "Turn the lights on", "inputSchema": {"json": {"type": "object", "properties": {"room": {"type": "string", "enum": ["living"]}}, "required": ["room"]}}}
    req = model.format_request([{"role": "user", "content": [{"text": "hi"}]}], [spec_tool], "be brief")
    assert req["model"] == "qwen3-1.7b" and req["stream"] is True and req["temperature"] == 0.0 and req["max_tokens"] == 300
    assert req["tools"] == [{"type": "function", "function": {"name": "lights_on", "description": "Turn the lights on",
                                                             "parameters": spec_tool["inputSchema"]["json"]}}]
    assert req["messages"][0] == {"role": "system", "content": "be brief"}
    assert req["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}} and req["top_p"] == 0.9


def test_non_standard_chat_options_go_into_extra_body():
    spec = RoleSpec("llm", "openai", url=URL, model="qwen", options={"api_key": "none", "max_tokens": 400, "timeout_s": 120,
                                                                     "tool_choice": "auto", "reasoning": False, "health": "models"})
    params = chat_params(spec)
    assert params == {"temperature": 0.0, "max_tokens": 400, "tool_choice": "auto", "extra_body": {"reasoning": False}}
    model = strands_openai_model(spec)
    assert model.client_args["timeout"] == 120.0 and model.client_args["api_key"] == "none"
    req = model.format_request([{"role": "user", "content": [{"text": "hi"}]}], [], None)
    assert "tool_choice" not in req and req["extra_body"] == {"reasoning": False}


def sse(*events: dict[str, Any]) -> bytes:
    return b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in events) + b"data: [DONE]\n\n"


def chunk(delta: dict[str, Any], finish: str | None = None, usage: dict[str, int] | None = None) -> dict[str, Any]:
    return {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "qwen3-1.7b",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **({"usage": usage} if usage else {})}


class FakeOpenAIServer:
    """Answers like llama.cpp --jinja / vLLM: the first request streams a tool call in pieces, the second
    (after the tool result) streams a text reply. Records every request body."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        assert req.url.path.endswith("/chat/completions") and req.headers["authorization"] == "Bearer local"
        body = json.loads(req.content)
        self.requests.append(body)
        has_tool_result = any(m["role"] == "tool" for m in body["messages"])
        if not has_tool_result:
            events = [chunk({"role": "assistant", "content": None}),
                      chunk({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                             "function": {"name": "lights_on", "arguments": ""}}]}),
                      chunk({"tool_calls": [{"index": 0, "function": {"arguments": "{\"room\": "}}]}),
                      chunk({"tool_calls": [{"index": 0, "function": {"arguments": "\"living\"}"}}]}),
                      chunk({}, finish="tool_calls"),
                      {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "qwen3-1.7b", "choices": [],
                       "usage": {"prompt_tokens": 410, "completion_tokens": 19, "total_tokens": 429}}]
        else:
            events = [chunk({"role": "assistant", "content": "Lights"}), chunk({"content": " coming on."}),
                      chunk({}, finish="stop"),
                      {"id": "c2", "object": "chat.completion.chunk", "created": 1, "model": "qwen3-1.7b", "choices": [],
                       "usage": {"prompt_tokens": 450, "completion_tokens": 5, "total_tokens": 455}}]
        return httpx.Response(200, content=sse(*events), headers={"content-type": "text/event-stream"})


async def test_strands_agent_runs_a_streamed_tool_call_from_an_openai_compatible_server():
    server = FakeOpenAIServer()
    spec = RoleSpec("llm", "openai", url=URL, model="qwen3-1.7b", options={"max_tokens": 100})
    model = strands_openai_model(spec)
    # strands opens a fresh openai.AsyncOpenAI per request and closes it (with its http client), so hand it a
    # pre-built client (the `client=` path of OpenAIModel) that it reuses and never closes.
    import openai
    model._custom_client = openai.AsyncOpenAI(base_url=URL, api_key="local",
                                              http_client=httpx.AsyncClient(transport=httpx.MockTransport(server)))
    calls: list[dict[str, Any]] = []

    @tool
    def lights_on(room: str) -> str:
        """Turn the lights on in a room."""
        calls.append({"room": room})
        return f"lights on in {room}"

    agent = Agent(model=model, tools=[lights_on], system_prompt="You run a house.", callback_handler=None)
    result = await agent.invoke_async("I can't see a thing in here.")
    assert calls == [{"room": "living"}] and str(result).strip() == "Lights coming on."
    assert len(server.requests) == 2
    first, second = server.requests
    assert first["model"] == "qwen3-1.7b" and first["max_tokens"] == 100 and first["stream"] is True
    assert first["tools"][0]["function"]["name"] == "lights_on" and "room" in first["tools"][0]["function"]["parameters"]["properties"]
    assert first["messages"][0] == {"role": "system", "content": "You run a house."}
    tool_msg = next(m for m in second["messages"] if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call_1" and "lights on in living" in json.dumps(tool_msg["content"])
    assistant = next(m for m in second["messages"] if m["role"] == "assistant" and m.get("tool_calls"))
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"room": "living"}
    usage = result.metrics.accumulated_usage
    assert usage["inputTokens"] == 860 and usage["outputTokens"] == 24


def test_scripted_model_from_options():
    m = scripted_model(RoleSpec("llm", "scripted", options={"tool_calls": [["lights_on", {"room": "living"}], ["no_action"]],
                                                            "reply": "ok"}))
    assert m.tool_calls == [("lights_on", {"room": "living"}), ("no_action", {})] and m.reply == "ok"
    m2 = scripted_model(RoleSpec("llm", "scripted", options={"tool_calls": '[["tv_off", {}]]', "fail": "down"}))
    assert m2.tool_calls == [("tv_off", {})] and isinstance(m2.fail, RuntimeError)
