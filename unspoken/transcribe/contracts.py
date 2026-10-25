"""Contracts for the per-person transcriber (offline and live). Integrator-owned: do not edit in a workstream.

Pipeline (each stage reads/writes one JSON file, so stages can be swapped and scored alone):

  video.mp4 --(audio track)--> asr.json   AsrResult: segments with word timings
            --(audio track)--> diar.json  DiarResult: voice turns (voice-A, voice-B, ...), audio only
            --(video track)--> asd.json   AsdResult: per-face speaking probability over time, seat per face
  asr + diar + asd --(fuse)--> transcript.json  Transcript: utterances with a person per utterance

Speaker ids in a Transcript: "person-<seat>" for someone on camera (seat = "left" / "right" / "s1"...), and
"voice-<X>" for a voice never tied to a face (off camera). `speakers[id].seat` is what eval seat_acc reads.
Every time is seconds from the start of the input video. All JSON is UTF-8, written with `to_json`.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Word:
    start: float
    end: float
    text: str
    conf: float = 1.0
    speaker: str | None = None          # set by fusion (word-level attribution); None before


@dataclass
class AsrSegment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    conf: float = 1.0


@dataclass
class AsrResult:
    model: str                           # human label, e.g. "faster-whisper small.en (CPU int8)"
    segments: list[AsrSegment]
    timing: dict[str, float] = field(default_factory=dict)   # wall_s, audio_s, rtf, ...


@dataclass
class VoiceTurn:
    start: float
    end: float
    voice: str                           # "voice-A", "voice-B", ... (session-only labels)
    conf: float = 1.0


@dataclass
class DiarResult:
    model: str
    turns: list[VoiceTurn]
    timing: dict[str, float] = field(default_factory=dict)


@dataclass
class FaceInfo:
    track_id: str
    seat: str | None                     # "left" / "right" / seat zone id; None if unassigned
    box: tuple[float, float, float, float]   # mean box x, y, w, h in pixels of the input video
    frames: int = 0                      # frames the face was seen in


@dataclass
class AsdFrame:
    t: float
    scores: dict[str, float]             # track_id -> speaking probability 0..1 (faces visible at t only)


@dataclass
class AsdResult:
    model: str
    fps: float                           # rate of `frames`
    faces: list[FaceInfo]
    frames: list[AsdFrame]
    timing: dict[str, float] = field(default_factory=dict)


@dataclass
class SpeakerInfo:
    label: str                           # what a human sees: "Left person", "Priya", "Off-camera 1"
    seat: str | None = None
    on_camera: bool = False
    voice: str | None = None             # diarization voice bound to this person, if any
    track_id: str | None = None


@dataclass
class Utterance:
    start: float
    end: float
    speaker: str                         # key into Transcript.speakers
    text: str
    words: list[Word] = field(default_factory=list)
    conf: float = 1.0                    # confidence in the SPEAKER choice
    source: str = "fused"                # what decided the speaker: "audio" | "video" | "fused" | "rules" | "decider" | "llm"


@dataclass
class Transcript:
    session_id: str
    duration: float
    speakers: dict[str, SpeakerInfo]
    utterances: list[Utterance]
    meta: dict[str, Any] = field(default_factory=dict)   # models, timings, method, settings


def to_json(obj: Any, path: str | Path | None = None) -> str:
    s = json.dumps(asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj, indent=1, ensure_ascii=False)
    if path is not None:
        Path(path).write_text(s)
    return s


def _words(ws: list[dict]) -> list[Word]:
    return [Word(**w) for w in ws or []]


def load_asr(path: str | Path) -> AsrResult:
    d = json.loads(Path(path).read_text())
    return AsrResult(d["model"], [AsrSegment(s["start"], s["end"], s["text"], _words(s.get("words")),
                                             s.get("conf", 1.0)) for s in d["segments"]], d.get("timing", {}))


def load_diar(path: str | Path) -> DiarResult:
    d = json.loads(Path(path).read_text())
    return DiarResult(d["model"], [VoiceTurn(**t) for t in d["turns"]], d.get("timing", {}))


def load_asd(path: str | Path) -> AsdResult:
    d = json.loads(Path(path).read_text())
    return AsdResult(d["model"], d["fps"], [FaceInfo(f["track_id"], f.get("seat"), tuple(f["box"]), f.get("frames", 0))
                                           for f in d["faces"]],
                     [AsdFrame(f["t"], f["scores"]) for f in d["frames"]], d.get("timing", {}))


def load_transcript(path: str | Path) -> Transcript:
    d = json.loads(Path(path).read_text())
    return Transcript(d["session_id"], d["duration"], {k: SpeakerInfo(**v) for k, v in d["speakers"].items()},
                      [Utterance(u["start"], u["end"], u["speaker"], u["text"], _words(u.get("words")),
                                 u.get("conf", 1.0), u.get("source", "fused")) for u in d["utterances"]],
                      d.get("meta", {}))
