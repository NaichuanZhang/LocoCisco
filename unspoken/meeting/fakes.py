"""Fakes and fallbacks for the meeting brain: devices, notifier, embedder, and a small Registry.

Used by tests (profile "fake"), by sim mode when the devices / bridge / models packages are not present
yet, and by the eval harness. The real implementations come from the other workstreams through the
fixed interfaces in docs/PLAN.md; the runtime prefers them whenever they import.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Callable

from ..contracts import MessageRef, RoleDisabled, RoleHealth, RoleSpec
from ..core.decider import Decider, DeciderAnswer, FakeDecider, HttpDecider

SCENE_COLORS = {"talk": "#ffd9a0", "slides": "#6a5acd", "break": "#7fd1ae", "privacy": "#ffb347",
                "off_record": "#ff3b30", "bright": "#ffffff", "dim": "#55506a", "green": "#3cb371"}
SCENE_BRIGHTNESS = {"talk": 70, "slides": 25, "break": 50, "privacy": 40, "off_record": 60, "bright": 100, "dim": 20}


class FakeLight:
    kind = "light"

    def __init__(self, device_id: str = "lamp") -> None:
        self.device_id = device_id
        self._state: dict[str, Any] = {"on": True, "scene": "talk", "brightness": 70, "color": SCENE_COLORS["talk"]}
        self.commands: list[tuple[str, dict[str, Any]]] = []

    async def apply(self, command: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        args = dict(args or {})
        self.commands.append((command, args))
        if command == "on":
            self._state["on"] = True
        elif command == "off":
            self._state["on"] = False
        elif command == "scene":
            scene = str(args.get("scene") or "talk")
            self._state.update(on=True, scene=scene, brightness=SCENE_BRIGHTNESS.get(scene, 70),
                               color=SCENE_COLORS.get(scene, SCENE_COLORS["talk"]))
        elif command == "set":
            self._state["on"] = True
            if "brightness" in args:
                self._state["brightness"] = int(args["brightness"])
            if "color" in args:
                self._state["color"] = str(args["color"])
            self._state["scene"] = str(args.get("scene") or "custom")
        else:
            raise ValueError(f"light does not know {command!r}")
        return dict(self._state)

    async def state(self) -> dict[str, Any]:
        return dict(self._state)

    async def healthy(self) -> bool:
        return True


class FakePlug:
    kind = "plug"

    def __init__(self, device_id: str = "fan") -> None:
        self.device_id = device_id
        self._state: dict[str, Any] = {"on": True}
        self.commands: list[tuple[str, dict[str, Any]]] = []

    async def apply(self, command: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        self.commands.append((command, dict(args or {})))
        if command not in ("on", "off"):
            raise ValueError(f"plug does not know {command!r}")
        self._state["on"] = command == "on"
        return dict(self._state)

    async def state(self) -> dict[str, Any]:
        return dict(self._state)

    async def healthy(self) -> bool:
        return True


def fake_devices(config: dict[str, Any] | None = None) -> dict[str, Any]:
    dev = (config or {}).get("devices") or {}
    lamp, fan = str(dev.get("lamp", "lamp")), str(dev.get("fan", "fan"))
    return {lamp: FakeLight(lamp), fan: FakePlug(fan)}


@dataclass
class FakeNotifier:
    """Records every post and DM; `echo=True` also prints them (the console transport)."""
    transport: str = "console"
    echo: bool = False
    posts: list[dict[str, Any]] = field(default_factory=list)
    dms: list[dict[str, Any]] = field(default_factory=list)
    fail: Exception | None = None
    _n: int = 0

    async def post(self, channel: str, text: str, *, thread: str | None = None, kind: str = "note",
                   meta: dict[str, Any] | None = None) -> MessageRef:
        if self.fail is not None:
            raise self.fail
        self._n += 1
        ref = MessageRef(channel=channel, id=f"m{self._n}", thread=thread)
        self.posts.append({"channel": channel, "text": text, "thread": thread, "kind": kind, "meta": meta or {}, "id": ref.id})
        if self.echo:
            print(f"[#{channel}] {text}")
        return ref

    async def dm(self, user: str, text: str, *, meta: dict[str, Any] | None = None) -> MessageRef:
        if self.fail is not None:
            raise self.fail
        self._n += 1
        self.dms.append({"user": user, "text": text, "meta": meta or {}, "id": f"m{self._n}"})
        if self.echo:
            print(f"[dm {user}] {text}")
        return MessageRef(channel=f"dm:{user}", id=f"m{self._n}")

    async def healthy(self) -> bool:
        return self.fail is None


class MeteredNotifier:
    """Wraps any Notifier: counts the UTF-8 bytes of every text that leaves and reports them to on_egress
    (dest, kind, nbytes). Audio and video never pass here, so their counters stay at 0 by construction."""

    def __init__(self, inner: Any, on_egress: Callable[[str, str, int], Any] | None = None) -> None:
        self.inner = inner
        self.on_egress = on_egress
        self.transport = getattr(inner, "transport", "console")
        self.total_bytes = 0

    def _count(self, dest: str, text: str) -> None:
        n = len(text.encode("utf-8"))
        self.total_bytes += n
        if self.on_egress is not None:
            self.on_egress(dest, "text", n)

    async def post(self, channel: str, text: str, **kw: Any) -> MessageRef:
        ref = await self.inner.post(channel, text, **kw)
        self._count(self.transport, text)
        return ref

    async def dm(self, user: str, text: str, **kw: Any) -> MessageRef:
        ref = await self.inner.dm(user, text, **kw)
        self._count(self.transport, text)
        return ref

    async def healthy(self) -> bool:
        return bool(await self.inner.healthy())


class FakeEmbedder:
    """Deterministic bag-of-hashed-words vectors: good enough to test the cosine path offline."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in t.lower().split():
                h = int(hashlib.md5(w.strip(".,;:!?()\"'").encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


DEV_URLS = {"decider": "http://127.0.0.1:8765", "decider_big": "http://127.0.0.1:8766"}
DEV_LLM_URL = "http://127.0.0.1:8010/v1"

_KW = {
    "task": ("by friday", "by monday", "by tuesday", "by wednesday", "by thursday", "by tomorrow", "by eod", "by end of",
             "can you ", "could you ", "please ", "i'll ", "i will ", "i can take", "send the", "send me", "draft the",
             "follow up", "action item", "take care of", "take that", "set up", "ship the", "update the", "write the",
             "next week", "麻烦", "你来", "我来", "周三前", "周五前", "明天前", "发给"),
    "decision": ("let's go with", "let's decide", "we keep the", "we agreed", "we decided", "decision is", "we'll go with", "agreed:", "final answer",
                 "let's lock", "that's settled", "so we're going with", "就定", "决定", "那就这样", "we are going with"),
    "question": ("what did we decide", "what's our", "what is our", "what's the policy", "refund policy", "when is the",
                 "what did the", "remind me what", "do we know", "can someone look up", "what does the doc", "我们上次"),
    "room": ("freezing", "cold in here", "chilly", "too hot", "stuffy", "warm in here", "too dark", "can't see", "brighter",
             "dim the", "too bright", "lights up", "turn the fan", "fan on", "fan off", "the slides", "show the agenda",
             "put the agenda", "太冷", "太热", "太暗", "太亮", "灯", "开风扇", "关风扇"),
    "sensitive": ("salary", "salaries", "her pay", "his pay", "their pay", "paid ", "comp ", "compensation", "layoff",
                  "lay off", "let go", "let two", "fire ", "firing", "performance review", "pip ", "hr complaint",
                  "harass", "medical", "sick leave", "diagnos", "lawsuit", "legal exposure", "sue ", "裁员", "工资", "开除"),
    "small_talk": ("hi, i'm", "hi i'm", "i'm ", "hello", "good morning", "haha", " ha ", "lol", "joking", "kidding", "coffee",
                   "我是", "哈哈"),
    "off_topic": ("weekend", "lunch", "game last night", "traffic", "the weather", "my kid", "vacation", "周末", "午饭"),
}
_ROOM_KW = {
    "warmer": ("freezing", "cold", "chilly", "fan off", "turn the fan off", "太冷", "关风扇", "冷"),
    "cooler": ("hot", "stuffy", "warm in here", "fan on", "turn the fan on", "太热", "开风扇", "热"),
    "brighter": ("too dark", "can't see", "brighter", "lights up", "turn the lights on", "太暗", "亮一点"),
    "dimmer": ("dim the", "too bright", "lower the lights", "glare", "太亮", "暗一点"),
    "slides": ("slides", "agenda", "the timer", "task list"),
}


class KeywordDecider:
    """Deterministic keyword answers to the meeting questions (profile "fake", and the eval's baseline).
    It answers like a wire decider: kind by keyword table, owner by the name said in the sentence (or the
    speaker for "I'll ..."), room by keyword, sensitive noul by keyword. Confidence 0.92 on a hit."""

    name = "keyword"

    def __init__(self, hit: float = 0.92, miss: float = 0.6, latency_ms: float = 1.0) -> None:
        self.hit, self.miss, self.latency_ms = hit, miss, latency_ms
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def healthy(self) -> bool:
        return True

    async def ask(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, DeciderAnswer]:
        self.calls.append((state, questions))
        text = state.rsplit(" said: ", 1)[-1].strip('"').lower() if " said: " in state else state.lower()
        speaker = state.rsplit(" said: ", 1)[0].rsplit(". ", 1)[-1].strip() if " said: " in state else ""
        out: dict[str, DeciderAnswer] = {}
        kind_options = list(questions["kind"]["criteria"]) if "kind" in questions else []
        kind_pick = self._kind(text, kind_options) if kind_options else None
        for key, q in questions.items():
            if q["type"] == "noul":
                if key == "act":      # the v3 gate: a to-do, a decision, a question for the room, a room request
                    p = 0.9 if kind_pick in ("Task", "Decision", "Question for the room", "Room request") else 0.1
                else:                 # the sensitive hint
                    p = 0.9 if kind_pick == "Sensitive" else 0.08
                out[key] = DeciderAnswer("noul", noul=p, confidence=abs(2 * p - 1), probabilities={"true": p, "false": 1 - p},
                                         latency_ms=self.latency_ms, model=self.name)
                continue
            options = list(q["criteria"])
            if key == "kind":
                pick = kind_pick or options[-1]
            elif key == "owner":
                pick = self._owner(text, speaker, options)
            elif key == "room":
                pick = next((r for r, kws in _ROOM_KW.items() if any(k in text for k in kws) and r in options), "none")
            else:
                pick = options[0]
            conf = self.hit if pick not in (options[-1], "none", "No one named") or key == "kind" and pick != "Nothing" else self.miss
            rest = (1 - conf) / max(1, len(options) - 1)
            out[key] = DeciderAnswer("choice", choice=pick, confidence=conf,
                                     probabilities={o: (conf if o == pick else rest) for o in options},
                                     latency_ms=self.latency_ms, model=self.name)
        return out

    def _kind(self, text: str, options: list[str]) -> str:
        label = {"task": "Task", "decision": "Decision", "question": "Question for the room", "room": "Room request",
                 "sensitive": "Sensitive", "small_talk": "Small talk or joke", "off_topic": "Off topic", "none": "Nothing"}
        # order matters: a joke about firing is small talk; sensitive beats task; room beats task ("can you dim the lights")
        nothing = label["none"] if label["none"] in options else ("Nothing to do" if "Nothing to do" in options else options[-1])
        if any(k in text for k in _KW["small_talk"]) and not any(k in text for k in _KW["task"][:12]):
            return label["small_talk"] if label["small_talk"] in options else nothing
        for kind in ("sensitive", "room", "question", "decision", "task", "off_topic"):
            if any(k in text for k in _KW[kind]):
                return label[kind] if label[kind] in options else nothing
        return nothing

    @staticmethod
    def _owner(text: str, speaker: str, options: list[str]) -> str:
        for name in options:
            if name != "No one named" and name.lower() in text and name != speaker:
                return name
        if speaker in options and any(text.startswith(k) or f" {k}" in text for k in ("i'll", "i will", "i can", "let me", "我来")):
            return speaker
        return "No one named" if "No one named" in options else options[-1]


class LocalRegistry:
    """A small Registry (unspoken.contracts.Registry) for when unspoken.models is not importable.

    profile "fake": FakeDecider for decider / decider_big, every other role disabled (tests, sim with no models).
    profile "dev":  the CPU decider servers on :8765 / :8766, llm disabled.
    profile "dev-local": dev + the llm role on a llama.cpp OpenAI server at 127.0.0.1:8010 (needs `openai`).
    `overrides` accepts "llm.url", "llm.model", "decider.url", "decider_big.url".
    """

    local_only = True

    def __init__(self, profile: str = "fake", *, deciders: dict[str, Decider] | None = None,
                 overrides: dict[str, str] | None = None, embedder: Any = None, llm_model: Any = None) -> None:
        self.profile = profile
        ov = dict(overrides or {})
        self._specs: dict[str, RoleSpec] = {}
        self._deciders: dict[str, Decider] = dict(deciders or {})
        self._embedder = embedder
        self._llm_model = llm_model
        if profile == "fake":
            self._deciders.setdefault("decider", KeywordDecider())
            self._deciders.setdefault("decider_big", KeywordDecider())
            self._specs["decider"] = RoleSpec("decider", "fake", "keyword decider (fake)")
            self._specs["decider_big"] = RoleSpec("decider_big", "fake", "keyword decider (fake, big)")
            self._specs["llm"] = RoleSpec("llm", "fake" if llm_model is not None else "none", "scripted LLM")
        else:
            for role, url in DEV_URLS.items():
                u = ov.get(f"{role}.url", url)
                self._specs[role] = RoleSpec(role, "strands-decider", f"Strands Decider {'2B' if role == 'decider' else '12B'} (CPU)", url=u)
                self._deciders.setdefault(role, HttpDecider(f"decider-{'2b' if role == 'decider' else '12b'}", u))
            if profile == "dev-local" or "llm.url" in ov or llm_model is not None:
                self._specs["llm"] = RoleSpec("llm", "openai", ov.get("llm.label", "local LLM (llama.cpp)"),
                                              url=ov.get("llm.url", DEV_LLM_URL), model=ov.get("llm.model", "local"))
            else:
                self._specs["llm"] = RoleSpec("llm", "none", "local LLM (off)")
        self._specs["embed"] = RoleSpec("embed", "fake" if embedder is not None else "none", "fake embedder")
        for role in ("vlm", "asr", "vad", "diarization", "faces", "asd", "tts", "pii"):
            self._specs[role] = RoleSpec(role, "none", role)

    # ---- Registry protocol
    def spec(self, role: str) -> RoleSpec:
        return self._specs[role]

    def enabled(self, role: str) -> bool:
        return role in self._specs and self._specs[role].provider != "none"

    def label(self, role: str) -> str:
        return self._specs[role].label if role in self._specs else role

    def decider(self, role: str = "decider") -> Decider:
        if not self.enabled(role):
            raise RoleDisabled(role)
        return self._deciders[role]

    def strands_model(self, role: str = "llm") -> Any:
        if not self.enabled(role):
            raise RoleDisabled(role)
        if self._llm_model is not None:
            return self._llm_model
        spec = self._specs[role]
        from strands.models.openai import OpenAIModel   # needs the `openai` extra (models workstream pins it)
        self._llm_model = OpenAIModel(client_args={"base_url": spec.url, "api_key": "local"}, model_id=spec.model,
                                      params={"max_tokens": 300, "temperature": 0.0})
        return self._llm_model

    def embedder(self) -> Any:
        if not self.enabled("embed"):
            raise RoleDisabled("embed")
        return self._embedder

    def _disabled(self, role: str) -> Any:
        raise RoleDisabled(role)

    def asr(self): return self._disabled("asr")
    def vad(self): return self._disabled("vad")
    def diarizer(self): return self._disabled("diarization")
    def faces(self): return self._disabled("faces")
    def asd(self): return self._disabled("asd")
    def tts(self): return self._disabled("tts")
    def pii(self): return self._disabled("pii")

    async def check(self, roles: list[str] | None = None) -> list[RoleHealth]:
        out: list[RoleHealth] = []
        for role in roles or list(self._specs):
            spec = self._specs[role]
            target = spec.url or "in-process"
            if not self.enabled(role):
                out.append(RoleHealth(role, spec.provider, spec.label, target, False, detail="disabled"))
                continue
            ok, detail = True, ""
            if role in self._deciders:
                try:
                    ok = bool(await self._deciders[role].healthy())
                except Exception as exc:
                    ok, detail = False, str(exc)
            out.append(RoleHealth(role, spec.provider, spec.label, target, ok, detail=detail))
        return out

    async def aclose(self) -> None:
        for d in self._deciders.values():
            if hasattr(d, "aclose"):
                await d.aclose()


def keyword_decider(spec: Any = None) -> KeywordDecider:
    """Registry factory (`provider = "python:unspoken.meeting.fakes:keyword_decider"`) for the `sim` profile."""
    opts = dict(getattr(spec, "options", None) or {})
    d = KeywordDecider(**{k: float(v) for k, v in opts.items() if k in ("hit", "miss", "latency_ms")})
    d.label = getattr(spec, "label", "") or "keyword decider (fake)"
    return d
