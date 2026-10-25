"""A minimal Registry over the in-process sensor providers, for `python -m unspoken.sensors.replay` and
tests when the models workstream's `unspoken.models.load_registry` is not on the path yet. It resolves the
same `python:module:Factory` provider strings the real registry will, so the TOML blocks below are exactly
what config/models.toml needs for the dev profile:

    [profiles.dev.vad]          provider = "python:unspoken.sensors.vad:SileroVAD"           label = "Silero VAD v5 (ONNX, CPU)"
    [profiles.dev.asr]          provider = "python:unspoken.sensors.asr_local:FasterWhisperASR"
                                model = "Systran/faster-whisper-base.en"                       label = "faster-whisper base.en (CPU int8)"
    [profiles.dev.asr.options]  device = "cpu"   compute_type = "int8"   language = "en"   threads = 4
    [profiles.dev.diarization]  provider = "python:unspoken.sensors.diar:EmbeddingDiarizer"  label = "WeSpeaker CAM++ (ONNX, CPU)"
    [profiles.dev.diarization.options]  threshold = 0.55   max_voices = 8
    [profiles.dev.faces]        provider = "python:unspoken.sensors.faces:YuNetFaces"          label = "YuNet 2023mar (OpenCV)"
    [profiles.dev.asd]          provider = "python:unspoken.sensors.asd:MotionASD"             label = "MotionASD (heuristic)"
"""
from __future__ import annotations

import importlib
from typing import Any

from ..contracts import RoleDisabled, RoleHealth, RoleSpec

DEV_ROLES: dict[str, RoleSpec] = {
    "vad": RoleSpec("vad", "python:unspoken.sensors.vad:SileroVAD", label="Silero VAD v5 (ONNX, CPU)"),
    "asr": RoleSpec("asr", "python:unspoken.sensors.asr_local:FasterWhisperASR", label="faster-whisper base.en (CPU int8)",
                    model="Systran/faster-whisper-base.en",
                    options={"device": "cpu", "compute_type": "int8", "language": "en", "threads": 4}),
    "diarization": RoleSpec("diarization", "python:unspoken.sensors.diar:EmbeddingDiarizer",
                            label="WeSpeaker CAM++ (ONNX, CPU)", options={"threshold": 0.55, "max_voices": 8}),
    "faces": RoleSpec("faces", "python:unspoken.sensors.faces:YuNetFaces", label="YuNet 2023mar (OpenCV)"),
    "asd": RoleSpec("asd", "python:unspoken.sensors.asd:MotionASD", label="MotionASD (heuristic)"),
}


def build_provider(spec: RoleSpec) -> Any:
    """Instantiate `python:module:Factory` with the spec (what the real registry does for in-process roles)."""
    if not spec.provider.startswith("python:"):
        raise ValueError(f"{spec.role}: SensorsRegistry only builds python: providers, got {spec.provider!r}")
    _, mod, factory = spec.provider.split(":", 2)
    return getattr(importlib.import_module(mod), factory)(spec)


class SensorsRegistry:
    profile = "sensors-local"
    local_only = True

    def __init__(self, specs: dict[str, RoleSpec] | None = None, overrides: dict[str, Any] | None = None,
                 disabled: set[str] | None = None) -> None:
        self._specs = {k: RoleSpec(v.role, v.provider, v.label, v.url, v.model, dict(v.options))
                       for k, v in (specs or DEV_ROLES).items()}
        for key, value in (overrides or {}).items():     # "asr.model", "diarization.options.threshold"
            role, _, rest = key.partition(".")
            if role not in self._specs:
                continue
            if rest.startswith("options."):
                self._specs[role].options[rest[len("options."):]] = value
            elif rest in ("model", "url", "label", "provider"):
                setattr(self._specs[role], rest, value)
        for role in disabled or set():
            if role in self._specs:
                self._specs[role].provider = "none"
        self._built: dict[str, Any] = {}

    def spec(self, role: str) -> RoleSpec:
        if role not in self._specs:
            return RoleSpec(role, "none")
        return self._specs[role]

    def enabled(self, role: str) -> bool:
        return self.spec(role).provider != "none"

    def label(self, role: str) -> str:
        return self.spec(role).label or self.spec(role).provider

    def _build(self, role: str) -> Any:
        if not self.enabled(role):
            raise RoleDisabled(role)
        if role not in self._built:
            self._built[role] = build_provider(self._specs[role])
        return self._built[role]

    def vad(self):
        return self._build("vad")

    def asr(self):
        return self._build("asr")

    def diarizer(self):
        return self._build("diarization")

    def faces(self):
        return self._build("faces")

    def asd(self):
        return self._build("asd")

    def decider(self, role: str = "decider"):
        raise RoleDisabled(role)

    def strands_model(self, role: str = "llm"):
        raise RoleDisabled(role)

    def tts(self):
        raise RoleDisabled("tts")

    def embedder(self):
        raise RoleDisabled("embed")

    def pii(self):
        raise RoleDisabled("pii")

    async def check(self, roles: list[str] | None = None) -> list[RoleHealth]:
        import asyncio
        import time

        out: list[RoleHealth] = []
        for role in roles or list(self._specs):
            spec = self.spec(role)
            if not self.enabled(role):
                out.append(RoleHealth(role, spec.provider, spec.label, "in-process", False, detail="disabled"))
                continue
            t0 = time.perf_counter()
            try:
                obj = self._build(role)
                h = getattr(obj, "healthy", None)
                ok = (await h()) if asyncio.iscoroutinefunction(h) else (bool(h()) if callable(h) else True)
                out.append(RoleHealth(role, spec.provider, spec.label, "in-process", ok,
                                      latency_ms=round((time.perf_counter() - t0) * 1000, 1)))
            except Exception as exc:
                out.append(RoleHealth(role, spec.provider, spec.label, "in-process", False,
                                      detail=f"{type(exc).__name__}: {exc}"))
        return out

    async def aclose(self) -> None:
        self._built.clear()


__all__ = ["SensorsRegistry", "DEV_ROLES", "build_provider"]
