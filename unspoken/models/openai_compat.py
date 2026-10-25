"""Server-backed providers that speak the OpenAI HTTP API, which vLLM, llama.cpp, faster-whisper-server,
speaches, Kokoro-FastAPI and friends all expose:

  llm/vlm  "openai"        strands OpenAIModel -> POST {url}/chat/completions (tools + streaming tool calls)
  asr      "openai-audio"  POST {url}/audio/transcriptions (multipart WAV, "model" field)
  tts      "openai-speech" POST {url}/audio/speech -> WAV bytes
  embed    "openai"        POST {url}/embeddings

`url` is the base that already ends in /v1. Options every server provider takes: api_key (default "local":
local servers ignore it), timeout or timeout_s (seconds), health (what healthy() GETs: a path such as
"/ready" is resolved against the server root, "models" against the base, or a full URL; default
"{url}/models"), plus the provider specifics documented on each class. The openai / strands / boto3
imports are local to the builders, so the registry never loads them for a profile that does not use them.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from ..contracts import RoleSpec, Transcript, Word
from .audio import pcm16_to_wav

DEFAULT_TIMEOUT = 60.0
DEFAULT_API_KEY = "local"


def _base(spec: RoleSpec) -> str:
    if not spec.url:
        raise ValueError(f"role {spec.role!r} provider {spec.provider!r} needs a url")
    return spec.url.rstrip("/")


def api_key_for(spec: RoleSpec) -> str:
    if spec.options.get("api_key_file"):
        try:
            key = Path(str(spec.options["api_key_file"])).expanduser().read_text().strip()
        except OSError:
            raise ValueError(f"Cannot read the local credential file for {spec.role}") from None
    elif spec.options.get("api_key_env"):
        key = os.environ.get(str(spec.options["api_key_env"]), "").strip()
    else:
        key = str(spec.options.get("api_key") or DEFAULT_API_KEY).strip()
    if not key or "\n" in key or "\r" in key:
        raise ValueError(f"Missing or invalid local API credential for {spec.role}")
    return key


def _headers(spec: RoleSpec) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key_for(spec)}"}


_COMMON_OPTIONS = {"api_key", "api_key_file", "api_key_env", "timeout", "timeout_s", "health"}


def timeout_of(spec: RoleSpec, default: float = DEFAULT_TIMEOUT) -> float:
    return float(spec.options.get("timeout", spec.options.get("timeout_s", default)))


def health_url(spec: RoleSpec) -> str:
    """options.health: "/ready" -> server root + /ready (NeMo-Speech.cpp), "models" -> {url}/models, or a full URL."""
    base = _base(spec)
    h = str(spec.options.get("health") or "models")
    if "://" in h:
        return h
    if h.startswith("/"):
        parts = urlsplit(base)
        return urlunsplit((parts.scheme, parts.netloc, h, "", ""))
    return f"{base}/{h}"


class _HttpProvider:
    consumed: frozenset[str] = frozenset()      # options the subclass reads itself (the rest may pass through)

    def __init__(self, spec: RoleSpec) -> None:
        self.spec = spec
        self.base = _base(spec)
        self.model = spec.model
        self.timeout = timeout_of(spec)
        self.health = health_url(spec)
        self._client: httpx.AsyncClient | None = None

    def extra_options(self) -> dict[str, Any]:
        """Options neither the base class nor the subclass consumes: server-specific fields passed through."""
        return {k: v for k, v in self.spec.options.items() if k not in _COMMON_OPTIONS and k not in self.consumed}

    def _make_client(self) -> httpx.AsyncClient:
        """Tests replace this to hand in an httpx.MockTransport."""
        return httpx.AsyncClient(timeout=self.timeout, headers=_headers(self.spec))

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:   # lazily, inside the event loop that will use it
            self._client = self._make_client()
        return self._client

    async def healthy(self) -> bool:
        """GET the health URL (default {url}/models, which every OpenAI-compatible server has; NeMo-Speech.cpp
        uses health = "/ready", 200 when loaded and 503 while loading). Authentication must succeed."""
        try:
            r = await self.client.get(self.health, timeout=min(5.0, self.timeout))
            return 200 <= r.status_code < 300
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# --------------------------------------------------------------------------- asr
class OpenAIAudioASR(_HttpProvider):
    """Options: language (forced language code; None = detect), response_format ("verbose_json" gives word
    timestamps when the server supports them; "json" otherwise), temperature. Any other option (for example
    NeMo-Speech.cpp's diarization = false) is sent as an extra multipart field."""

    consumed = frozenset({"language", "response_format", "temperature"})

    async def transcribe(self, pcm16: bytes, sample_rate: int, lang: str | None = None) -> Transcript:
        data: dict[str, Any] = {"response_format": str(self.spec.options.get("response_format", "verbose_json"))}
        if self.model:
            data["model"] = self.model
        language = lang or self.spec.options.get("language")
        if language:
            data["language"] = str(language)
        if "temperature" in self.spec.options:
            data["temperature"] = str(self.spec.options["temperature"])
        if data["response_format"] == "verbose_json":
            data["timestamp_granularities[]"] = "word"
        for k, v in self.extra_options().items():
            data[k] = str(v).lower() if isinstance(v, bool) else str(v)
        t0 = time.perf_counter()
        r = await self.client.post(f"{self.base}/audio/transcriptions", data=data,
                                   files={"file": ("audio.wav", pcm16_to_wav(pcm16, sample_rate), "audio/wav")})
        ms = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        return parse_transcription(r, lang or "", ms)


def parse_transcription(r: httpx.Response, fallback_lang: str, latency_ms: float) -> Transcript:
    ctype = r.headers.get("content-type", "")
    if "json" not in ctype:
        return Transcript(text=r.text.strip(), lang=fallback_lang or "en", latency_ms=latency_ms)
    body = r.json()
    words = [Word(text=str(w.get("word", "")).strip(), start=float(w.get("start", 0.0)), end=float(w.get("end", 0.0)))
             for w in body.get("words") or [] if isinstance(w, dict)]
    conf = 1.0
    segs = body.get("segments") or []
    if segs and all(isinstance(s, dict) and "no_speech_prob" in s for s in segs):
        conf = max(0.0, min(1.0, 1.0 - sum(float(s["no_speech_prob"]) for s in segs) / len(segs)))
    lang = str(body.get("language") or fallback_lang or "en")
    return Transcript(text=str(body.get("text", "")).strip(), lang=_short_lang(lang), conf=conf, words=words,
                      latency_ms=latency_ms)


_LANG_NAMES = {"english": "en", "chinese": "zh", "mandarin": "zh", "japanese": "ja", "spanish": "es",
               "french": "fr", "german": "de", "korean": "ko"}


def _short_lang(lang: str) -> str:
    low = lang.lower()
    return _LANG_NAMES.get(low, low[:2] if len(low) > 2 and low.isalpha() else low)


# --------------------------------------------------------------------------- tts
class OpenAISpeechTTS(_HttpProvider):
    """Options: voice (default voice id), speed, response_format (default "wav"); other options are sent as
    extra JSON fields."""

    consumed = frozenset({"voice", "speed", "response_format"})

    async def synthesize(self, text: str, voice: str | None = None) -> bytes:
        body: dict[str, Any] = {"input": text, "response_format": str(self.spec.options.get("response_format", "wav"))}
        if self.model:
            body["model"] = self.model
        v = voice or self.spec.options.get("voice")
        if v:
            body["voice"] = str(v)
        if "speed" in self.spec.options:
            body["speed"] = float(self.spec.options["speed"])
        body.update(self.extra_options())
        r = await self.client.post(f"{self.base}/audio/speech", json=body)
        r.raise_for_status()
        return r.content


# --------------------------------------------------------------------------- embeddings
class OpenAIEmbedder(_HttpProvider):
    """Options: dimensions (ask the server for a truncated vector), batch (max texts per request, default 64)."""

    consumed = frozenset({"dimensions", "batch"})

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        batch = int(self.spec.options.get("batch", 64))
        out: list[list[float]] = []
        for i in range(0, len(texts), batch):
            body: dict[str, Any] = {"input": texts[i:i + batch]}
            if self.model:
                body["model"] = self.model
            if "dimensions" in self.spec.options:
                body["dimensions"] = int(self.spec.options["dimensions"])
            r = await self.client.post(f"{self.base}/embeddings", json=body)
            r.raise_for_status()
            rows = sorted(r.json()["data"], key=lambda d: int(d.get("index", 0)))
            out.extend([float(x) for x in row["embedding"]] for row in rows)
        return out


# --------------------------------------------------------------------------- strands models (llm / vlm)
# Options every chat model takes: temperature (default 0), max_tokens, plus for "openai": api_key, timeout,
# extra_body (dict sent as-is, e.g. {chat_template_kwargs = {enable_thinking = false}} for Qwen3 on vLLM /
# llama.cpp), stream (default true; llama.cpp and vLLM both stream tool calls). A standard Chat Completions
# parameter (top_p, tool_choice, seed, ...) is sent as itself; any other key (reasoning, chat_template_kwargs,
# ...) goes into extra_body, because the OpenAI SDK rejects unknown keyword arguments.
_CHAT_CONSUMED = _COMMON_OPTIONS | {"temperature", "max_tokens", "extra_body", "stream", "name"}
CHAT_COMPLETIONS_PARAMS = frozenset({
    "top_p", "frequency_penalty", "presence_penalty", "stop", "seed", "n", "logit_bias", "user", "response_format",
    "tool_choice", "parallel_tool_calls", "reasoning_effort", "max_completion_tokens", "stream_options", "logprobs",
    "top_logprobs", "metadata", "store", "service_tier", "prediction", "modalities", "audio", "web_search_options",
    "prompt_cache_key", "safety_identifier", "verbosity"})


def chat_params(spec: RoleSpec, max_tokens: int | None = None) -> dict[str, Any]:
    o = spec.options
    params: dict[str, Any] = {"temperature": float(o.get("temperature", 0.0))}
    mt = max_tokens if max_tokens is not None else o.get("max_tokens")
    if mt is not None:
        params["max_tokens"] = int(mt)
    extra: dict[str, Any] = dict(o["extra_body"]) if o.get("extra_body") else {}
    for k, v in o.items():
        if k in _CHAT_CONSUMED:
            continue
        if k in CHAT_COMPLETIONS_PARAMS:
            params[k] = v
        else:
            extra[k] = v
    if extra:
        params["extra_body"] = extra
    return params


def strands_openai_model(spec: RoleSpec, max_tokens: int | None = None) -> Any:
    """strands OpenAIModel against any OpenAI-compatible server. Tool specs go out as Chat Completions
    `tools` (function name/description/JSON-schema parameters) and tool calls come back from
    `delta.tool_calls`, which is what vLLM (`--tool-call-parser`) and llama.cpp (`--jinja`) produce."""
    from strands.models.openai import OpenAIModel

    class CompatibleOpenAIModel(OpenAIModel):
        def format_request(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            request = super().format_request(*args, **kwargs)
            if not request.get("tools"):
                request.pop("tools", None)
                if request.get("tool_choice") in (None, "auto", "none"):
                    request.pop("tool_choice", None)
            return request

    if not spec.model:
        raise ValueError(f"role {spec.role!r} provider 'openai' needs a model id")
    client_args: dict[str, Any] = {"base_url": _base(spec), "api_key": api_key_for(spec),
                                   "timeout": timeout_of(spec), "max_retries": 1}
    return CompatibleOpenAIModel(client_args=client_args, model_id=spec.model, params=chat_params(spec, max_tokens),
                                 stream=bool(spec.options.get("stream", True)))


def strands_bedrock_model(spec: RoleSpec, max_tokens: int | None = None) -> Any:
    """strands BedrockModel (dev only; local_only profiles refuse it). Options: region, profile (AWS profile;
    omit for the default credential chain), temperature, max_tokens."""
    import boto3
    from strands.models import BedrockModel

    if not spec.model:
        raise ValueError(f"role {spec.role!r} provider 'bedrock' needs a model id")
    o = spec.options
    session = boto3.Session(profile_name=o.get("profile") or None, region_name=o.get("region") or None)
    mt = max_tokens if max_tokens is not None else o.get("max_tokens")
    kwargs: dict[str, Any] = {"temperature": float(o.get("temperature", 0.0))}
    if mt is not None:
        kwargs["max_tokens"] = int(mt)
    return BedrockModel(model_id=spec.model, boto_session=session, **kwargs)


def scripted_model(spec: RoleSpec, max_tokens: int | None = None) -> Any:
    """unspoken.core.fakes.ScriptedModel. Options: tool_calls = [["lights_on", {room = "living"}], ...]
    (as TOML arrays, or a JSON string), reply, fail (error text -> raises RuntimeError on every call)."""
    from ..core.fakes import ScriptedModel

    raw = spec.options.get("tool_calls") or []
    if isinstance(raw, str):
        raw = json.loads(raw)
    calls = [(str(c[0]), dict(c[1]) if len(c) > 1 and c[1] else {}) for c in raw]
    fail = spec.options.get("fail")
    return ScriptedModel(tool_calls=calls, reply=str(spec.options.get("reply", "Done.")),
                         fail=RuntimeError(str(fail)) if fail else None)
