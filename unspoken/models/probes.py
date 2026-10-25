"""Health probes for `python -m unspoken.models check`: per enabled role, healthy() (when the provider has
it) plus ONE tiny real request, timed. Nothing here is cached or kept; a probe must be cheap enough to
run on the day in front of the judges.

  decider*  one noul question                     llm/vlm  a 1-token chat completion (no tools)
  embed     one short string                      asr      0.5 s of silence
  tts       the word "ok"                         pii      one short sentence
  vad/diarization/faces/asd  build the object (that is the expensive part: model files, torch) and
                             open one session / stream
"""
from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any

from ..contracts import RoleHealth, RoleSpec
from .audio import silence

PROBE_TIMEOUT_S = 120.0
PROBE_STATE = "The lamp in the meeting room is on."
PROBE_QUESTION = {"probe": {"type": "noul", "instructions": "Is the lamp on?"}}
PROBE_CHAT = [{"role": "user", "content": [{"text": "Reply with the single word: ok"}]}]


def target_of(spec: RoleSpec, obj: Any = None) -> str:
    if spec.url:
        return spec.url
    if spec.provider == "bedrock" and spec.model:
        return f"bedrock:{spec.model}"
    for key in ("model_path", "path"):
        if key in spec.options:
            return str(spec.options[key])
    if spec.model and spec.provider != "fake":        # a fake decider's "model" is only its name
        return spec.model
    return "in-process"


def _detail(obj: Any) -> str:
    """A provider may describe itself (e.g. "silero v5, cpu") through a `detail` attribute or describe()."""
    d = getattr(obj, "describe", None)
    if callable(d):
        try:
            return str(d())
        except Exception:  # cosmetic only
            return ""
    return str(getattr(obj, "detail", "") or "")


async def _healthy(obj: Any) -> bool | None:
    """healthy() may be sync (vad/diarizer/faces/asd in unspoken.sensors) or async (ASR, HTTP providers)."""
    fn = getattr(obj, "healthy", None)
    if fn is None:
        return None
    res = fn()
    if inspect.isawaitable(res):
        res = await res
    return bool(res)


async def probe_decider(obj: Any) -> str:
    answers = await obj.ask(PROBE_STATE, PROBE_QUESTION)
    a = answers["probe"]
    return f"P(true)={a.noul:.2f}" + (f", model {a.model}" if a.model else "")


async def probe_chat(model: Any) -> str:
    """Consume one streamed completion; report the stop reason and the usage the server sent."""
    stop, usage, text = "", {}, []
    async for ev in model.stream(PROBE_CHAT, None, None):
        if "messageStop" in ev:
            stop = str(ev["messageStop"].get("stopReason", ""))
        if "metadata" in ev:
            usage = ev["metadata"].get("usage") or {}
        delta = ev.get("contentBlockDelta", {}).get("delta", {})
        if "text" in delta:
            text.append(delta["text"])
    reply = "".join(text).strip().replace("\n", " ")
    bits = [f"stop={stop}" if stop else "", f"tokens={usage.get('inputTokens', '?')}+{usage.get('outputTokens', '?')}" if usage else "",
            f"reply={reply[:30]!r}" if reply else ""]
    return ", ".join(b for b in bits if b)


async def probe_embed(obj: Any) -> str:
    vecs = await obj.embed(["ok"])
    if not vecs or not vecs[0]:
        raise RuntimeError("embed returned no vector")
    return f"dim={len(vecs[0])}"


async def probe_asr(obj: Any) -> str:
    t = await obj.transcribe(silence(0.5), 16000)
    return f"text={t.text[:30]!r}, lang={t.lang}"


async def probe_tts(obj: Any) -> str:
    wav = await obj.synthesize("ok")
    if not wav:
        raise RuntimeError("tts returned no audio")
    kind = "wav" if wav[:4] == b"RIFF" else "bytes"
    return f"{len(wav)} {kind}"


async def probe_pii(obj: Any) -> str:
    spans = await obj.find("Mail me at ok@example.com.")
    return f"{len(spans)} span(s)"


async def probe_session(obj: Any) -> str:
    if hasattr(obj, "stream"):
        obj.stream(16000).feed(silence(0.03))
    elif hasattr(obj, "session"):
        obj.session()
    return _detail(obj) or "in-process"


PROBES = {"decider": probe_decider, "decider_big": probe_decider, "llm": probe_chat, "vlm": probe_chat,
          "embed": probe_embed, "asr": probe_asr, "tts": probe_tts, "pii": probe_pii,
          "vad": probe_session, "diarization": probe_session, "faces": probe_session, "asd": probe_session}


async def check_role(spec: RoleSpec, get: Any, probe_model: Any, timeout: float = PROBE_TIMEOUT_S) -> RoleHealth:
    """get(): the registry's (cached) object for the role; probe_model(): an uncached 1-token chat model for
    llm/vlm so the probe never spends the role's configured max_tokens."""
    label = spec.label or spec.provider
    if spec.provider == "none":
        return RoleHealth(spec.role, spec.provider, label, "-", True, None, "disabled")
    t0 = time.perf_counter()
    try:
        obj = get()
        built_ms = (time.perf_counter() - t0) * 1000
        async with asyncio.timeout(timeout):
            healthy = await _healthy(obj)
            if healthy is False:
                return RoleHealth(spec.role, spec.provider, label, target_of(spec, obj), False,
                                  round((time.perf_counter() - t0) * 1000, 1), "healthy() is false (server down?)")
            t1 = time.perf_counter()
            probe = PROBES[spec.role]
            detail = await probe(probe_model() if spec.role in ("llm", "vlm") else obj)
            ms = (time.perf_counter() - t1) * 1000
        if spec.role in ("vad", "diarization", "faces", "asd"):
            ms = built_ms + ms      # loading the model is the cost that matters for in-process roles
        return RoleHealth(spec.role, spec.provider, label, target_of(spec, obj), True, round(ms, 1), detail)
    except TimeoutError:
        return RoleHealth(spec.role, spec.provider, label, target_of(spec), False,
                          round((time.perf_counter() - t0) * 1000, 1), f"timed out after {timeout:.0f} s")
    except Exception as exc:
        return RoleHealth(spec.role, spec.provider, label, target_of(spec), False,
                          round((time.perf_counter() - t0) * 1000, 1), f"{type(exc).__name__}: {str(exc)[:200]}")
