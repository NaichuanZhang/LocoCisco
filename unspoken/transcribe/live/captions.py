"""Caption records of the live transcriber, their JSONL log, and the final Transcript built from them.

A caption is one emission about one VAD segment: `{t_emit, start, end, speaker, text, final}` (plus the fields
below). A segment produces zero or more partials (final=False, text so far, speaker = best guess), exactly one
first final (final=True, revision 0) and, when the voice<->seat binding later changes its mind, revisions
(final=True, revision n>0, same start/end/text, new speaker). Every time is seconds of media (= seconds since
the capture started, the same clock as the video frames); t_emit is when the caption left the engine on that
clock, so a replay at 1x gives real latencies and a demo can burn captions in at the moment they appeared.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..contracts import SpeakerInfo, Transcript, Utterance, Word

UNKNOWN_VOICE = "voice-?"


@dataclass
class Caption:
    seg: str                    # segment id (stable across partials, final and revisions of one segment)
    t_emit: float               # media seconds when this caption was emitted
    start: float                # segment start (media seconds); partials: start of the open segment
    end: float                  # segment end; partials: end of the audio the text covers so far
    speaker: str                # "person-left" | "person-right" | "voice-B" | "voice-?"
    text: str
    final: bool
    revision: int = 0           # 0 = first emission of this kind; revisions of a final count up
    voice: str | None = None    # diarization label behind the speaker choice (None = not labelled)
    conf: float = 0.0           # confidence in the SPEAKER choice
    source: str = "fused"       # what decided the speaker: "audio" | "video" | "fused" | "none"
    words: list[dict[str, Any]] = field(default_factory=list)   # final only: [{start, end, text, conf}]
    latency: float | None = None    # t_emit - end for finals (speech end -> caption), None for partials

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Caption":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__ if k in d})


class CaptionLog:
    """Append-only JSONL writer (one caption per line) that also keeps the captions in memory."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.captions: list[Caption] = []
        self._fh = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "w", encoding="utf-8")

    def append(self, cap: Caption) -> None:
        self.captions.append(cap)
        if self._fh is not None:
            self._fh.write(json.dumps(cap.to_dict(), ensure_ascii=False) + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def read_captions(path: str | Path) -> list[Caption]:
    return [Caption.from_dict(json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]


def latest_finals(captions: Iterable[Caption]) -> dict[str, Caption]:
    """segment id -> its latest final (highest revision, then latest t_emit)."""
    out: dict[str, Caption] = {}
    for c in captions:
        if not c.final:
            continue
        cur = out.get(c.seg)
        if cur is None or (c.revision, c.t_emit) > (cur.revision, cur.t_emit):
            out[c.seg] = c
    return out


def speaker_label(key: str, seat: str | None) -> str:
    if seat:
        return f"{seat.title()} person"
    return f"Off-camera {key.split('-', 1)[-1]}" if key.startswith("voice-") else key


def build_transcript(captions: Iterable[Caption], session_id: str, duration: float,
                     meta: dict[str, Any] | None = None,
                     tracks_by_seat: dict[str, str] | None = None) -> Transcript:
    """The final transcript: one utterance per segment from its latest final caption, in time order."""
    finals = sorted(latest_finals(captions).values(), key=lambda c: (c.start, c.end))
    speakers: dict[str, SpeakerInfo] = {}
    voices: dict[str, Counter] = {}
    utts: list[Utterance] = []
    for c in finals:
        seat = c.speaker.split("-", 1)[1] if c.speaker.startswith("person-") else None
        if c.speaker not in speakers:
            speakers[c.speaker] = SpeakerInfo(speaker_label(c.speaker, seat), seat, bool(seat),
                                              None, (tracks_by_seat or {}).get(seat or ""))
        if c.voice and c.voice != UNKNOWN_VOICE:
            voices.setdefault(c.speaker, Counter())[c.voice] += 1
        words = [Word(float(w["start"]), float(w["end"]), str(w["text"]), float(w.get("conf", 1.0)), c.speaker)
                 for w in c.words]
        utts.append(Utterance(c.start, c.end, c.speaker, c.text, words, c.conf, c.source))
    for key, cnt in voices.items():
        speakers[key].voice = cnt.most_common(1)[0][0]
    return Transcript(session_id, duration, speakers, utts, dict(meta or {}))


__all__ = ["Caption", "CaptionLog", "read_captions", "latest_finals", "build_transcript", "speaker_label",
           "UNKNOWN_VOICE"]
