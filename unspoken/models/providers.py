"""Provider table: which class serves a (role, provider) pair, and how a RoleSpec becomes an object.

    build(spec)  -> the object for spec.role (raises RoleDisabled for provider "none")

Built-in providers (see docs/MODELS.md for the options of each):

  role(s)               providers
  decider, decider_big  strands-decider (unspoken.core.decider.HttpDecider), fake (FakeDecider)
  llm, vlm              openai (strands OpenAIModel), bedrock (strands BedrockModel; dev only), scripted
  asr                   openai-audio, fake
  tts                   openai-speech, fake
  embed                 openai, fake
  pii                   regex, fake
  vad, diarization,     fake
  faces, asd
  any role              python:<module>:<Factory>   Factory(spec) -> object implementing the role's protocol
  any role              none                        role disabled (Registry.enabled(role) is False)

Everything heavy (openai, strands, boto3, torch in a python: factory) is imported inside the builder that
needs it, so `load_registry()` and `check` on a profile that never uses a provider never import it.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable

from ..contracts import ROLES, RoleDisabled, RoleSpec

Builder = Callable[[RoleSpec], Any]
PYTHON_PREFIX = "python:"
DISABLED = "none"
_ANY = "*"

CHAT_ROLES = ("llm", "vlm")
DECIDER_ROLES = ("decider", "decider_big")


# --------------------------------------------------------------------------- builders
def _decider(spec: RoleSpec) -> Any:
    from ..core.decider import HttpDecider

    if not spec.url:
        raise ValueError(f"role {spec.role!r} provider 'strands-decider' needs a url")
    # the client's name ("decider-2b" / "decider-12b"): options.name, else the block's model, else the role
    d = HttpDecider(str(spec.options.get("name") or spec.model or spec.role), spec.url,
                    timeout=float(spec.options.get("timeout", spec.options.get("timeout_s", 60.0))),
                    max_concurrency=int(spec.options.get("max_concurrency", 1)))
    d.label = spec.label or spec.role   # type: ignore[attr-defined]  # the UI's X-ray reads it
    return d


def _fake_decider(spec: RoleSpec) -> Any:
    from ..core.decider import FakeDecider

    script = [tuple(s) for s in spec.options.get("script", [])]
    d = FakeDecider(str(spec.options.get("name") or spec.model or "fake"), script=script or None,
                    latency_ms=float(spec.options.get("latency_ms", 5.0)))
    d.label = spec.label or "fake decider"   # type: ignore[attr-defined]
    return d


def _openai_chat(spec: RoleSpec) -> Any:
    from .openai_compat import strands_openai_model
    return strands_openai_model(spec)


def _bedrock_chat(spec: RoleSpec) -> Any:
    from .openai_compat import strands_bedrock_model
    return strands_bedrock_model(spec)


def _scripted_chat(spec: RoleSpec) -> Any:
    from .openai_compat import scripted_model
    return scripted_model(spec)


def _openai_audio(spec: RoleSpec) -> Any:
    from .openai_compat import OpenAIAudioASR
    return OpenAIAudioASR(spec)


def _openai_speech(spec: RoleSpec) -> Any:
    from .openai_compat import OpenAISpeechTTS
    return OpenAISpeechTTS(spec)


def _openai_embed(spec: RoleSpec) -> Any:
    from .openai_compat import OpenAIEmbedder
    return OpenAIEmbedder(spec)


def _regex_pii(spec: RoleSpec) -> Any:
    from .pii import RegexPII
    return RegexPII(spec)


def _fake(cls_name: str) -> Builder:
    def build(spec: RoleSpec) -> Any:
        from . import fakes
        return getattr(fakes, cls_name)(spec)
    return build


BUILDERS: dict[tuple[str, str], Builder] = {
    ("decider", "strands-decider"): _decider, ("decider_big", "strands-decider"): _decider,
    ("decider", "fake"): _fake_decider, ("decider_big", "fake"): _fake_decider,
    ("llm", "openai"): _openai_chat, ("vlm", "openai"): _openai_chat,
    ("llm", "bedrock"): _bedrock_chat, ("vlm", "bedrock"): _bedrock_chat,
    ("llm", "scripted"): _scripted_chat, ("vlm", "scripted"): _scripted_chat,
    ("llm", "fake"): _scripted_chat, ("vlm", "fake"): _scripted_chat,     # "fake" is an alias of "scripted" here
    ("asr", "openai-audio"): _openai_audio, ("asr", "fake"): _fake("FakeASR"),
    ("tts", "openai-speech"): _openai_speech, ("tts", "fake"): _fake("FakeTTS"),
    ("embed", "openai"): _openai_embed, ("embed", "fake"): _fake("FakeEmbedder"),
    ("pii", "regex"): _regex_pii, ("pii", "fake"): _fake("FakePII"),
    ("vad", "fake"): _fake("FakeVAD"),
    ("diarization", "fake"): _fake("FakeDiarizer"),
    ("faces", "fake"): _fake("FakeFaces"),
    ("asd", "fake"): _fake("FakeASD"),
}


def providers_for(role: str) -> list[str]:
    """Provider names a role accepts (for error messages and docs)."""
    names = [p for (r, p) in BUILDERS if r == role]
    return sorted(set(names)) + ["python:<module>:<Factory>", DISABLED]


# --------------------------------------------------------------------------- python: factories
def load_python_factory(provider: str) -> Callable[[RoleSpec], Any]:
    """'python:unspoken.sensors.vad:SileroVAD' -> the SileroVAD callable (class or function)."""
    target = provider[len(PYTHON_PREFIX):]
    module_name, sep, attr = target.rpartition(":")
    if not sep or not module_name or not attr:
        raise ValueError(f"provider {provider!r}: expected 'python:<module>:<Factory>'")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(f"provider {provider!r}: cannot import {module_name!r}: {exc}") from exc
    try:
        factory = getattr(module, attr)
    except AttributeError:
        raise AttributeError(f"provider {provider!r}: {module_name!r} has no attribute {attr!r}") from None
    if not callable(factory):
        raise TypeError(f"provider {provider!r}: {attr!r} is not callable")
    return factory


# --------------------------------------------------------------------------- entry point
def build(spec: RoleSpec) -> Any:
    if spec.role not in ROLES:
        raise ValueError(f"unknown role {spec.role!r}; roles are {list(ROLES)}")
    if spec.provider == DISABLED:
        raise RoleDisabled(f"role {spec.role!r} is disabled (provider \"none\") in this profile")
    if spec.provider.startswith(PYTHON_PREFIX):
        return load_python_factory(spec.provider)(spec)
    builder = BUILDERS.get((spec.role, spec.provider))
    if builder is None:
        raise ValueError(f"role {spec.role!r} has no provider {spec.provider!r}; choose one of {providers_for(spec.role)}")
    return builder(spec)
