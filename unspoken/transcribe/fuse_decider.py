"""Method `decider`: the hard utterances go to the Strands Decider (2B, then 12B when still unsure).

An utterance is "hard" when its speaker confidence (the margin of the audio-visual decision, fuse.py) is below
`Settings.decider_threshold`. For each one, lowest confidence first and at most `decider_max_calls`, the decider
gets a short text state (the line before, the line, the line after, the voice's history and the audio / video
evidence numbers) and ONE multiple-choice question "who said this line?" whose options are the people the
audio-visual stage considered. The 2B answers first; below `decider_act` confidence the 12B is asked (same
cascade idea as unspoken/core and meeting/cascade.py). The chosen person becomes the speaker (source "decider",
conf = the decider's confidence). `Transcript.meta["decider"]` reports calls, latency and changes.

The client is injected so unit tests use unspoken.core.decider.FakeDecider; the CLI uses DeciderCascade over
HttpDecider against the shared loopback servers (http://127.0.0.1:8765 / 8766).
"""
from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from .contracts import Transcript, Utterance
from .fuse import Settings


@dataclass
class DeciderCall:
    index: int
    asked: list[str]
    answer: str | None
    confidence: float
    model: str
    latency_ms: float
    changed: bool
    escalated: bool = False


class DeciderCascade:
    """Synchronous facade: ask the small decider, escalate to the big one when unsure.
    One private event loop for the whole run: the httpx AsyncClients stay bound to a live loop."""

    def __init__(self, small_url: str, big_url: str | None = None, act: float = 0.6, timeout: float = 120.0) -> None:
        from ..core.decider import HttpDecider
        self.small = HttpDecider("decider-2b", small_url, timeout=timeout)
        self.big = HttpDecider("decider-12b", big_url, timeout=timeout) if big_url else None
        self.act = act
        self.loop = asyncio.new_event_loop()

    def ask(self, state: str, question: dict[str, Any]) -> tuple[str | None, float, str, float, bool]:
        """-> (choice, confidence, model, latency_ms, escalated)."""
        return self.loop.run_until_complete(self._ask(state, question))

    async def _ask(self, state: str, question: dict[str, Any]) -> tuple[str | None, float, str, float, bool]:
        ans = (await self.small.ask(state, {"who": question}))["who"]
        total, escalated = ans.latency_ms, False
        if ans.confidence < self.act and self.big is not None:
            big = (await self.big.ask(state, {"who": question}))["who"]
            total += big.latency_ms
            escalated = True
            if big.confidence >= ans.confidence:
                ans = big
        return ans.choice, float(ans.confidence), ans.model, total, escalated

    async def aclose(self) -> None:
        await self.small.aclose()
        if self.big is not None:
            await self.big.aclose()


class FakeCascade:
    """Wraps a unspoken.core.decider.FakeDecider (or any Decider) without escalation, for tests."""

    def __init__(self, decider: Any) -> None:
        self.decider = decider
        self.loop = asyncio.new_event_loop()

    def ask(self, state: str, question: dict[str, Any]) -> tuple[str | None, float, str, float, bool]:
        ans = self.loop.run_until_complete(self.decider.ask(state, {"who": question}))["who"]
        return ans.choice, float(ans.confidence), ans.model, ans.latency_ms, False


# ----------------------------------------------------------------------------------------------- the question
def _label(tr: Transcript, key: str) -> str:
    info = tr.speakers.get(key)
    if info is not None and info.label:
        return info.label
    return f"{key[7:].title()} person" if key.startswith("person-") else f"Off-camera {key.split('-', 1)[-1]}"


def _clip(text: str, n: int = 30) -> str:
    w = text.split()
    return " ".join(w[-n:]) if len(w) > n else text


def build_state(tr: Transcript, i: int, evidence: dict[str, Any], history: dict[str, dict[str, int]]) -> str:
    utts = tr.utterances
    u = utts[i]
    before = f'{_label(tr, utts[i - 1].speaker)}: "{_clip(utts[i - 1].text)}"' if i > 0 else "(start of meeting)"
    after = f'{_label(tr, utts[i + 1].speaker)}: "{_clip(utts[i + 1].text)}"' if i + 1 < len(utts) else "(end of meeting)"
    voice = evidence.get("voice", "?")
    hist = history.get(voice, {})
    n_hist = sum(hist.values())
    if n_hist:
        top = max(hist, key=lambda k: hist[k])
        voice_line = (f"The voice of this line ({voice}) was attributed to {_label(tr, top)} in {hist[top]} of its "
                      f"{n_hist} other lines.")
    else:
        voice_line = f"The voice of this line ({voice}) has no other lines."
    audio = evidence.get("audio") or {}
    video = evidence.get("video") or {}
    aud = ", ".join(f"{_label(tr, k)} {v:.2f}" for k, v in sorted(audio.items(), key=lambda kv: -kv[1]))
    vid = ", ".join(f"{_label(tr, k)} {v:.2f}" for k, v in sorted(video.items(), key=lambda kv: -kv[1]))
    return (f"Meeting transcript, deciding who said one line.\n"
            f"Line before: {before}\n"
            f'Line in question (at {u.start:.1f}s, {u.end - u.start:.1f}s long): "{u.text}"\n'
            f"Line after: {after}\n"
            f"{voice_line}\n"
            f"Voice evidence (probability per person): {aud or 'none'}.\n"
            f"Camera evidence (probability the person's face is the one talking): {vid or 'none'}.")


def build_question(tr: Transcript, candidates: list[str]) -> dict[str, Any]:
    desc = {}
    for key in candidates:
        info = tr.speakers.get(key)
        if info is not None and info.on_camera:
            desc[_label(tr, key)] = f"the person sitting on the {info.seat} of the camera"
        else:
            desc[_label(tr, key)] = "a voice heard in the room but not on camera"
    return {"type": "choice", "instructions": "Who said the line in question? Use the surrounding lines, the voice "
                                             "history and the evidence numbers.", "criteria": desc}


def ask_decider(tr: Transcript, cascade: Any, s: Settings) -> Transcript:
    """Mutates and returns tr: low-confidence utterances re-attributed by the decider."""
    t0 = time.perf_counter()
    evidence = tr.meta.get("evidence") or [{} for _ in tr.utterances]
    calls: list[DeciderCall] = []
    report: dict[str, Any] = {"threshold": s.decider_threshold, "asked": 0, "changed": 0, "escalated": 0,
                              "skipped_no_client": cascade is None}
    if cascade is None:
        tr.meta["decider"] = report
        return tr
    order = sorted((i for i, u in enumerate(tr.utterances) if u.conf < s.decider_threshold),
                   key=lambda i: tr.utterances[i].conf)[:s.decider_max_calls]
    report["candidates"] = len([u for u in tr.utterances if u.conf < s.decider_threshold])
    history: dict[str, dict[str, int]] = {}
    for u, e in zip(tr.utterances, evidence):
        history.setdefault(e.get("voice", "?"), {}).setdefault(u.speaker, 0)
        history[e.get("voice", "?")][u.speaker] += 1
    label_to_key = {_label(tr, k): k for k in tr.speakers}
    for i in order:
        u: Utterance = tr.utterances[i]
        e = evidence[i] if i < len(evidence) else {}
        scores = e.get("scores") or {}
        cands = [k for k in scores if k in tr.speakers] or [k for k in tr.speakers]
        if u.speaker not in cands:
            cands.append(u.speaker)
        if len(cands) < 2:
            continue
        hist = {v: dict(h) for v, h in history.items()}             # the voice's other lines, not this one
        own = hist.setdefault(e.get("voice", "?"), {})
        own[u.speaker] = max(own.get(u.speaker, 1) - 1, 0)
        state = build_state(tr, i, e, hist)
        question = build_question(tr, cands)
        try:
            choice, conf, model, ms, escalated = cascade.ask(state, question)
        except Exception as exc:  # the decider is optional: keep the audio-visual answer
            report.setdefault("errors", []).append(f"{i}: {type(exc).__name__}: {exc}")
            continue
        key = label_to_key.get(choice or "", None)
        changed = key is not None and key != u.speaker
        calls.append(DeciderCall(i, cands, key, conf, model, ms, changed, escalated))
        if key is not None:
            for w in u.words:
                w.speaker = key
            u.speaker, u.conf, u.source = key, round(conf, 3), "decider"
    report["asked"] = len(calls)
    report["changed"] = sum(c.changed for c in calls)
    report["escalated"] = sum(c.escalated for c in calls)
    if calls:
        lat = [c.latency_ms for c in calls]
        report["latency_ms"] = {"mean": round(statistics.mean(lat), 1), "median": round(statistics.median(lat), 1),
                                "max": round(max(lat), 1)}
        report["by_model"] = {m: sum(1 for c in calls if c.model == m) for m in sorted({c.model for c in calls})}
    report["wall_s"] = round(time.perf_counter() - t0, 2)
    report["calls"] = [{"i": c.index, "answer": c.answer, "conf": round(c.confidence, 3), "model": c.model,
                        "ms": round(c.latency_ms, 1), "changed": c.changed} for c in calls]
    tr.meta["decider"] = report
    tr.meta.setdefault("stages", []).append("decider")
    return tr
