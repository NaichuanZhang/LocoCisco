"""ModelRegistry: the one object code asks for models by ROLE (implements unspoken.contracts.Registry).

    reg = load_registry("gb10")
    reg.decider()                 # unspoken.core.decider.Decider for role "decider"
    reg.strands_model("llm")      # strands Model (OpenAIModel on vLLM / llama.cpp; BedrockModel in dev)
    reg.asr(), reg.vad(), reg.diarizer(), reg.faces(), reg.asd(), reg.tts(), reg.embedder(), reg.pii()
    await reg.check()             # one RoleHealth per role
    await reg.aclose()            # closes every HTTP client the built providers opened

Objects are built lazily on first use and cached per role, so two agents asking for "llm" share one
model object (and one connection pool). A local_only profile is validated when the registry is created:
every enabled role must point at this box or its LAN, or load_registry raises RemoteNotAllowed.
"""
from __future__ import annotations

from typing import Any

from ..contracts import (ASR, ROLES, TTS, VAD, ActiveSpeaker, Diarizer, Embedder, FaceTracker, PIIDetector,
                         Role, RoleDisabled, RoleHealth, RoleSpec)
from .config import ResolvedProfile
from .netpolicy import assert_local
from .probes import PROBE_TIMEOUT_S, check_role
from .providers import DISABLED, build

PROTOCOLS: dict[str, Any] = {"asr": ASR, "vad": VAD, "diarization": Diarizer, "faces": FaceTracker,
                             "asd": ActiveSpeaker, "tts": TTS, "embed": Embedder, "pii": PIIDetector}


class ModelRegistry:
    def __init__(self, resolved: ResolvedProfile) -> None:
        self.profile: str = resolved.profile
        self.local_only: bool = resolved.local_only
        self.path: str = resolved.path
        self.applied: list[str] = list(resolved.applied)
        self._specs: dict[str, RoleSpec] = dict(resolved.specs)
        self._built: dict[str, Any] = {}
        if self.local_only:
            for spec in self._specs.values():
                assert_local(spec, self.profile)

    # ------------------------------------------------------------------ introspection
    def spec(self, role: Role) -> RoleSpec:
        try:
            return self._specs[role]
        except KeyError:
            raise ValueError(f"unknown role {role!r}; roles are {list(ROLES)}") from None

    def enabled(self, role: Role) -> bool:
        return self.spec(role).provider != DISABLED

    def label(self, role: Role) -> str:
        """Human name for the UI: the TOML label, else "<model> (<provider>)" or just the provider."""
        s = self.spec(role)
        if s.label:
            return s.label
        if s.provider == DISABLED:
            return "off"
        return f"{s.model} ({s.provider})" if s.model else s.provider

    def specs(self) -> dict[str, RoleSpec]:
        return dict(self._specs)

    # ------------------------------------------------------------------ building
    def get(self, role: Role) -> Any:
        """The cached object for a role (any role; the typed getters below check the protocol)."""
        if role not in self._built:
            spec = self.spec(role)
            if self.local_only:
                assert_local(spec, self.profile)
            self._built[role] = build(spec)
        return self._built[role]

    def _typed(self, role: Role, proto: Any, what: str) -> Any:
        obj = self.get(role)
        if not isinstance(obj, proto):
            raise TypeError(f"role {role!r} provider {self.spec(role).provider!r} built {type(obj).__name__}, "
                            f"which does not implement {what}")
        return obj

    def decider(self, role: Role = "decider") -> Any:
        if role not in ("decider", "decider_big"):
            raise ValueError(f"decider() takes role 'decider' or 'decider_big', not {role!r}")
        obj = self.get(role)
        if not callable(getattr(obj, "ask", None)):
            raise TypeError(f"role {role!r} provider {self.spec(role).provider!r} built {type(obj).__name__}, "
                            "which has no ask(): not a Decider")
        return obj

    def strands_model(self, role: Role = "llm") -> Any:
        if role not in ("llm", "vlm"):
            raise ValueError(f"strands_model() takes role 'llm' or 'vlm', not {role!r}")
        obj = self.get(role)
        if not callable(getattr(obj, "stream", None)) or not callable(getattr(obj, "get_config", None)):
            raise TypeError(f"role {role!r} provider {self.spec(role).provider!r} built {type(obj).__name__}, "
                            "which is not a strands Model (no stream()/get_config())")
        return obj

    def asr(self) -> ASR:
        return self._typed("asr", ASR, "ASR (transcribe, healthy)")

    def vad(self) -> VAD:
        return self._typed("vad", VAD, "VAD (stream)")

    def diarizer(self) -> Diarizer:
        return self._typed("diarization", Diarizer, "Diarizer (session)")

    def faces(self) -> FaceTracker:
        return self._typed("faces", FaceTracker, "FaceTracker (session)")

    def asd(self) -> ActiveSpeaker:
        return self._typed("asd", ActiveSpeaker, "ActiveSpeaker (session)")

    def tts(self) -> TTS:
        return self._typed("tts", TTS, "TTS (synthesize)")

    def embedder(self) -> Embedder:
        return self._typed("embed", Embedder, "Embedder (embed)")

    def pii(self) -> PIIDetector:
        return self._typed("pii", PIIDetector, "PIIDetector (find)")

    # ------------------------------------------------------------------ health
    def _probe_model(self, role: Role) -> Any:
        """An UNCACHED 1-token chat model for the llm/vlm probe (same server, same model id)."""
        spec = self.spec(role)
        if spec.provider == "openai":
            from .openai_compat import strands_openai_model
            return strands_openai_model(spec, max_tokens=1)
        if spec.provider == "bedrock":
            from .openai_compat import strands_bedrock_model
            return strands_bedrock_model(spec, max_tokens=1)
        return self.get(role)

    async def check(self, roles: list[Role] | None = None, timeout: float = PROBE_TIMEOUT_S) -> list[RoleHealth]:
        """One RoleHealth per role (disabled roles report ok with detail "disabled"). Probes run one after
        another so a slow import or server in one role cannot inflate another role's latency."""
        wanted = list(roles) if roles else list(ROLES)
        for r in wanted:
            self.spec(r)                                 # unknown role -> clear error before any network
        out: list[RoleHealth] = []
        for r in wanted:
            out.append(await check_role(self.spec(r), lambda r=r: self.get(r), lambda r=r: self._probe_model(r), timeout))
        return out

    async def aclose(self) -> None:
        for obj in list(self._built.values()):
            fn = getattr(obj, "aclose", None)
            if callable(fn):
                try:
                    await fn()
                except Exception:   # closing is best effort
                    pass
        self._built.clear()

    def __repr__(self) -> str:
        on = [r for r in ROLES if self.enabled(r)]
        return f"ModelRegistry(profile={self.profile!r}, local_only={self.local_only}, enabled={on})"


__all__ = ["ModelRegistry", "RoleDisabled"]
