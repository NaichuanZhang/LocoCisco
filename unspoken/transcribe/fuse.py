"""Fuse ASR words, diarization voices and ASD faces into per-person utterances (offline post-processing).

    python -m unspoken.transcribe.fuse --asr asr.json --diar diar.json --asd asd.json --out transcript.json \
        --method av [--set merge_gap=0.3 ...] [--decider-url http://127.0.0.1:8765] [--llm-url http://127.0.0.1:8790]

Methods are cumulative ablations (each adds one stage):
  align    1. every ASR word (or segment, when the ASR has no word timings; or an interpolated word when
              `interp_words`) goes to the diarization turn it overlaps most; utterances split at voice changes,
              one-word islands inside another voice's run are attached to it, short same-voice gaps merged.
              Speakers are the diarization voices (voice-A, ...): no seats, so seat_acc is 0 by construction.
  bind     2. voices <-> seats on whole-session co-occurrence. Evidence E[voice, seat] = rank-based two-sample
              z (Mann-Whitney) of the seat's ASD score during the voice's turns against the same seat during
              the OTHER voices' turns: "is this face more active when this voice speaks than when the others
              do?" The Hungarian assignment maximises the total, and a pair is bound only when the voice prefers
              that seat by `bind_min_margin` z over its next seat and the evidence is positive. Unbound voices
              stay off camera (voice-X).
  av       3. per utterance: audio evidence (alignment confidence x binding confidence for the voice's seat) and
              video evidence (softmax of the seats' z during the utterance, temperature `av_tau`) are mixed; the
              video weight is self-calibrated on the session (how often the per-utterance video pick agrees with
              a confidently bound voice, rescaled above chance) so a noisy ASD cannot override a good voice. The
              winner is the speaker, the margin is the confidence.
  decider  4. av, then utterances below `decider_threshold` are put to the Strands Decider 2B (then 12B when
              still unsure) as a multiple-choice "who said this?" (fuse_decider.py).
  llm      5. av, then DiarizationLM-style text-only correction of speaker boundaries by a local llama-server
              (fuse_llm.py).
  all      av + llm + decider.
Every stage records its evidence in `Transcript.meta` so a transcript explains itself.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import (AsdResult, AsrResult, DiarResult, SpeakerInfo, Transcript, Utterance, VoiceTurn, Word,
                        load_asd, load_asr, load_diar, to_json)

METHODS = ("align", "bind", "av", "decider", "llm", "all")


@dataclass
class Settings:
    # 1. alignment
    interp_words: bool = True        # ASR segment without word timings -> words with character-proportional times
    nearest_slack: float = 0.5       # a word outside every turn joins the nearest turn; beyond this it is a guess
    island_max_words: int = 1        # a run of <= this many words of voice v inside voice w's run joins w (0 = off)
    island_max_dur: float = 0.6      # ... when it lasts at most this long (s)
    island_max_gap: float = 0.4      # ... and both neighbours are within this gap (s)
    merge_gap: float = 0.3           # consecutive same-speaker utterances closer than this merge (s)
    # 2. binding
    bind_min_margin: float = 1.0     # min z margin of the chosen seat over the voice's next seat
    bind_min_z: float = 0.0          # a seat must be at least this active (z) during the voice to be bound...
    bind_major_share: float = 0.05   # ...unless the room is "full": no more major voices (>= this share of the
                                     # speech time) than seats, then everyone is on camera and the assignment is
                                     # accepted on margin alone (the inactive face eliminates itself)
    bind_min_z_minor: float = 3.0    # a minor voice (below the share) takes a free seat only above this z
    bind_min_frames: int = 25        # min ASD frames of the voice with that seat visible (5 s at 5 Hz)
    # 3. per-utterance audio-visual decision
    av_tau: float = 2.0              # softmax temperature over the seats' z during the utterance
    av_min_frames: int = 3           # fewer ASD frames than this in an utterance -> no video evidence
    av_video_min_margin: float = 0.6  # the camera's top-two softmax margin must reach this to count at all
    av_unbound_conf: float = 0.6     # audio confidence multiplier for a voice bound to no seat
    av_video_weight: float = -1.0    # fixed video weight 0..1; < 0 = self-calibrate on the session
    av_calib_conf: float = 0.7       # utterances with at least this audio confidence calibrate the video weight
    av_calib_prior: float = 5.0      # pseudo-count shrinking the calibrated weight toward 0 when few utterances
    # 4. decider
    decider_threshold: float = 0.35  # utterances with a lower speaker confidence are asked
    decider_max_calls: int = 60      # lowest confidence first
    decider_act: float = 0.6         # 2B answers below this go to the 12B
    # 5. llm
    llm_max_calls: int = 400
    llm_window_words: int = 12       # words shown on each side of a boundary

    @classmethod
    def parse(cls, pairs: list[str]) -> "Settings":
        s = cls()
        kinds = {f.name: f.type for f in fields(cls)}
        for p in pairs:
            k, _, v = p.partition("=")
            if k not in kinds:
                raise SystemExit(f"unknown setting {k!r}; known: {', '.join(kinds)}")
            kind = kinds[k]
            setattr(s, k, (v.lower() in ("1", "true", "yes")) if kind == "bool" else int(v) if kind == "int" else float(v))
        return s


# ----------------------------------------------------------------------------------------------- 1. alignment
@dataclass
class Unit:
    """One attributable piece of text: a real ASR word, an interpolated word, or a whole segment."""
    start: float
    end: float
    text: str
    seg: int                          # index of the ASR segment it came from
    word: Word | None = None          # the real word (None when the timing is interpolated / segment-level)
    voice: str = ""
    conf: float = 0.0                 # confidence of the voice assignment


_TOKEN = re.compile(r"\S+")


def units_of(asr: AsrResult, s: Settings) -> list[Unit]:
    out: list[Unit] = []
    for i, seg in enumerate(asr.segments):
        if seg.words:
            for w in seg.words:
                if w.text.strip():
                    out.append(Unit(w.start, max(w.end, w.start + 0.02), w.text.strip(), i, w))
            continue
        toks = _TOKEN.findall(seg.text)
        if not toks:
            continue
        if not s.interp_words or len(toks) == 1:
            out.append(Unit(seg.start, max(seg.end, seg.start + 0.02), " ".join(toks), i))
            continue
        # character-proportional times inside the segment (a 12-letter word takes longer than "a")
        weights = np.array([len(t) + 1 for t in toks], dtype=float)
        edges = np.concatenate([[0.0], np.cumsum(weights) / weights.sum()])
        dur = max(seg.end - seg.start, 0.02 * len(toks))
        for k, t in enumerate(toks):
            out.append(Unit(seg.start + edges[k] * dur, seg.start + edges[k + 1] * dur, t, i))
    out.sort(key=lambda u: (u.start, u.end))
    return out


def assign_units(units: list[Unit], turns: list[VoiceTurn], s: Settings) -> None:
    """Sets unit.voice / unit.conf: the turn with the most overlap wins; the margin over the runner-up is the
    confidence; a unit touching no turn joins the nearest one (low confidence)."""
    if not turns:
        for u in units:
            u.voice, u.conf = "voice-?", 0.0
        return
    ts = sorted(turns, key=lambda t: (t.start, t.end))
    starts = [t.start for t in ts]
    max_end = np.maximum.accumulate(np.array([t.end for t in ts]))       # lets the backward scan stop safely
    for u in units:
        dur = max(u.end - u.start, 0.02)
        best: tuple[float, VoiceTurn | None] = (0.0, None)
        second = 0.0
        j = bisect_left(starts, u.end) - 1
        while j >= 0 and max_end[j] > u.start:
            t = ts[j]
            ov = min(u.end, t.end) - max(u.start, t.start)
            if ov > best[0]:
                second, best = best[0], (ov, t)
            elif ov > second:
                second = ov
            j -= 1
        if best[1] is not None:
            t = best[1]
            margin = (best[0] - second) / dur
            u.voice = t.voice
            u.conf = float(np.clip(margin, 0.05, 1.0)) * (0.5 + 0.5 * float(np.clip(t.conf, 0.0, 1.0)))
            continue
        mid = (u.start + u.end) / 2
        k = min(range(len(ts)), key=lambda i: min(abs(ts[i].start - mid), abs(ts[i].end - mid)))
        d = min(abs(ts[k].start - mid), abs(ts[k].end - mid))
        u.voice, u.conf = ts[k].voice, 0.3 if d <= s.nearest_slack else 0.1


@dataclass
class Group:
    units: list[Unit]
    voice: str
    source: str = "audio"

    @property
    def start(self) -> float:
        return self.units[0].start

    @property
    def end(self) -> float:
        return max(u.end for u in self.units)

    @property
    def text(self) -> str:
        return " ".join(u.text for u in self.units)

    @property
    def conf(self) -> float:
        w = np.array([max(u.end - u.start, 0.02) for u in self.units])
        return float(np.average([u.conf for u in self.units], weights=w))


def group_units(units: list[Unit], s: Settings) -> list[Group]:
    """Consecutive units of one voice inside one ASR segment form a group (a segment is the unit the per-utterance
    audio-visual decision can re-attribute); islands are attached; same-speaker groups merge again later."""
    groups: list[Group] = []
    for u in units:
        if (groups and groups[-1].voice == u.voice and groups[-1].units[-1].seg == u.seg
                and (u.start - groups[-1].end) <= max(s.merge_gap, 0.0)):
            groups[-1].units.append(u)
        else:
            groups.append(Group([u], u.voice))
    if s.island_max_words > 0:
        changed = True
        while changed:                               # attaching an island can create a new merge; iterate to rest
            changed = False
            for i in range(1, len(groups) - 1):
                g, a, b = groups[i], groups[i - 1], groups[i + 1]
                if (a.voice == b.voice != g.voice and len(g.units) <= s.island_max_words
                        and g.end - g.start <= s.island_max_dur
                        and g.start - a.end <= s.island_max_gap and b.start - g.end <= s.island_max_gap):
                    for u in g.units:
                        u.voice, u.conf = a.voice, min(u.conf, 0.4)
                    merged = Group(a.units + g.units + b.units, a.voice, "rules")
                    groups[i - 1:i + 2] = [merged]
                    changed = True
                    break
    return groups


# ----------------------------------------------------------------------------------------------- 2. seat evidence
class SeatEvidence:
    """Per-seat ASD score series over the speech frames, rank-transformed, queried by time intervals.

    Ranks (over the frames inside diarization turns, per seat) remove a face's own bias: a face that always
    moves a little scores high everywhere, but it is not *more* active during one voice than another. The
    two-sample statistic `z(seat, inside, outside)` is the Mann-Whitney z of the seat's ranks inside a set of
    intervals against the seat's ranks in the other speech frames: positive = this face is unusually active then.
    """

    def __init__(self, asd: AsdResult, speech: list[tuple[float, float]]) -> None:
        from scipy.stats import rankdata

        self.seat_of = {f.track_id: f.seat for f in asd.faces if f.seat}
        self.seats = sorted(set(self.seat_of.values()))
        self.track_of_seat: dict[str, str] = {}
        for seat in self.seats:
            tracks = [f for f in asd.faces if f.seat == seat]
            self.track_of_seat[seat] = max(tracks, key=lambda f: f.frames).track_id
        t = np.array([f.t for f in asd.frames], dtype=float)
        order = np.argsort(t, kind="stable")
        self.t = t[order]
        n, si = len(self.t), {s: i for i, s in enumerate(self.seats)}
        self.score = np.full((len(self.seats), n), np.nan)
        for col, j in enumerate(order):
            for tid, p in asd.frames[j].scores.items():
                seat = self.seat_of.get(tid)
                if seat is None:
                    continue
                cur = self.score[si[seat], col]
                self.score[si[seat], col] = p if np.isnan(cur) else max(cur, p)
        self.speech = self._mask(speech) if speech else np.ones(n, dtype=bool)
        # ranks per seat over the speech frames where the face is visible; nan elsewhere
        self.rank = np.full_like(self.score, np.nan)
        self.n_speech = np.zeros(len(self.seats), dtype=int)
        for i in range(len(self.seats)):
            ok = self.speech & ~np.isnan(self.score[i])
            self.n_speech[i] = int(ok.sum())
            if self.n_speech[i]:
                self.rank[i, ok] = rankdata(self.score[i, ok])
        self._tie = np.zeros(len(self.seats))             # tie correction term per seat
        for i in range(len(self.seats)):
            vals = self.score[i, self.speech & ~np.isnan(self.score[i])]
            if len(vals):
                _, counts = np.unique(vals, return_counts=True)
                self._tie[i] = float((counts ** 3 - counts).sum())

    def _idx(self, intervals: list[tuple[float, float]]) -> np.ndarray:
        parts = [np.arange(bisect_left(self.t, a), bisect_right(self.t, b)) for a, b in intervals]
        return np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=int)

    def _mask(self, intervals: list[tuple[float, float]]) -> np.ndarray:
        m = np.zeros(len(self.t), dtype=bool)
        m[self._idx(intervals)] = True
        return m

    def z_inside(self, intervals: list[tuple[float, float]]) -> dict[str, tuple[float, int]]:
        """seat -> (Mann-Whitney z of the seat inside the intervals vs the other speech frames, frames inside)."""
        inside = self._mask(intervals) & self.speech
        out = {}
        for i, seat in enumerate(self.seats):
            r = self.rank[i]
            a = r[inside & ~np.isnan(r)]
            n1, n = len(a), self.n_speech[i]
            n2 = n - n1
            if n1 == 0 or n2 == 0:
                out[seat] = (0.0, int(n1))
                continue
            u = float(a.sum()) - n1 * (n1 + 1) / 2.0                 # U statistic of the inside sample
            mu = n1 * n2 / 2.0
            var = n1 * n2 / 12.0 * ((n + 1) - self._tie[i] / (n * (n - 1)))
            out[seat] = ((u - mu) / math.sqrt(var) if var > 0 else 0.0, int(n1))
        return out


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class Binding:
    seat_of_voice: dict[str, str] = field(default_factory=dict)
    conf: dict[str, float] = field(default_factory=dict)            # voice -> binding confidence
    off_conf: dict[str, float] = field(default_factory=dict)        # unbound voice -> confidence it is off camera
    table: dict[str, dict[str, list[float]]] = field(default_factory=dict)   # voice -> seat -> [z, frames]
    note: str = ""


def bind_voices(turns: list[VoiceTurn], ev: SeatEvidence, s: Settings) -> Binding:
    """Hungarian assignment of voices to seats on the whole-session two-sample z, accepted on margin."""
    from scipy.optimize import linear_sum_assignment

    voices = sorted({t.voice for t in turns})
    b = Binding()
    if not voices or not ev.seats:
        b.note = "no voices or no seated faces"
        return b
    speech = {v: 0.0 for v in voices}
    for t in turns:
        speech[t.voice] += max(t.end - t.start, 0.0)
    total = sum(speech.values()) or 1.0
    major = [v for v in voices if speech[v] / total >= s.bind_major_share]
    full_room = 0 < len(major) <= len(ev.seats)
    b.note = (f"{len(major)} major voices for {len(ev.seats)} seats: everyone on camera, assignment accepted on margin"
              if full_room else f"{len(major)} major voices for {len(ev.seats)} seats: someone is off camera, "
              f"a seat needs positive evidence (z >= {s.bind_min_z})")
    Z = np.full((len(voices), len(ev.seats)), -1e3)
    for i, v in enumerate(voices):
        st = ev.z_inside([(t.start, t.end) for t in turns if t.voice == v])
        b.table[v] = {seat: [round(float(z), 3), n] for seat, (z, n) in st.items()}
        for j, seat in enumerate(ev.seats):
            z, n = st[seat]
            if n >= s.bind_min_frames:
                Z[i, j] = z

    def accept(i: int, j: int, strict: bool) -> bool:
        z = Z[i, j]
        if z <= -1e2:
            return False
        others = [Z[i, k] for k in range(len(ev.seats)) if k != j and Z[i, k] > -1e2]
        margin = z - max(others) if others else z
        if margin < s.bind_min_margin:
            return False
        if strict and z < s.bind_min_z_minor:
            return False
        if z <= s.bind_min_z and not (full_room and voices[i] in major):
            return False
        b.seat_of_voice[voices[i]] = ev.seats[j]
        b.conf[voices[i]] = round(_logistic(margin - s.bind_min_margin), 3)
        return True

    # tier 1: the major voices share the seats (Hungarian on the whole-session evidence)
    rows = [i for i, v in enumerate(voices) if v in major]
    if rows:
        ri, cj = linear_sum_assignment(-Z[rows])
        for r, j in zip(ri, cj):
            accept(rows[r], j, strict=False)
    # tier 2: a minor voice (a few seconds of speech, noisy evidence) takes a free seat only on strong evidence
    free = [j for j, seat in enumerate(ev.seats) if seat not in b.seat_of_voice.values()]
    minors = sorted((i for i, v in enumerate(voices) if v not in major), key=lambda i: -Z[i].max())
    for i in minors:
        if not free:
            break
        j = max(free, key=lambda k: Z[i, k])
        if accept(i, j, strict=True):
            free.remove(j)
    for i, v in enumerate(voices):
        if v in b.seat_of_voice:
            continue
        seen = [Z[i, k] for k in range(len(ev.seats)) if Z[i, k] > -1e2]
        # no face was ever unusually active while this voice spoke: the more negative the best seat's z, the surer
        b.off_conf[v] = round(_logistic(-max(seen)) if seen else 0.5, 3)
    return b


# ----------------------------------------------------------------------------------------------- 3. av decision
@dataclass
class Decision:
    speaker: str                 # "person-<seat>" or the voice id
    conf: float
    source: str
    scores: dict[str, float]     # candidate -> fused score
    audio: dict[str, float]
    video: dict[str, float]


def video_evidence(g: Group, ev: SeatEvidence | None, cands: list[str], s: Settings) -> dict[str, float] | None:
    """Softmax over the seats' z during the utterance; None when there is not enough video or the camera is not
    sure itself (top-two margin below `av_video_min_margin`): a weak video vote must not dilute a good voice."""
    if ev is None:
        return None
    st = ev.z_inside([(g.start, g.end)])
    zs = {c: 0.0 for c in cands}
    have = False
    for seat, (z, n) in st.items():
        key = f"person-{seat}"
        if key in zs and n >= s.av_min_frames:
            zs[key], have = z, True
    if not have:
        return None
    ex = {c: math.exp(min(v / s.av_tau, 50.0)) for c, v in zs.items()}
    tot = sum(ex.values())
    V = {c: e / tot for c, e in ex.items()}
    top = sorted(V.values(), reverse=True)
    if len(top) > 1 and top[0] - top[1] < s.av_video_min_margin:
        return None
    return V


def audio_evidence(g: Group, binding: Binding, cands: list[str], s: Settings) -> tuple[dict[str, float], str, float]:
    seat = binding.seat_of_voice.get(g.voice)
    pick = f"person-{seat}" if seat else g.voice
    if seat:
        c = g.conf * binding.conf.get(g.voice, 0.0)
    else:                                   # off camera: the session-wide "never with a face" evidence counts
        c = g.conf * (s.av_unbound_conf + (1 - s.av_unbound_conf) * binding.off_conf.get(g.voice, 0.5))
    c = float(np.clip(c, 0.0, 1.0))
    k = len(cands)
    if k == 1:
        return {cands[0]: 1.0}, pick, c
    p = 1.0 / k + c * (1.0 - 1.0 / k)              # c = 0 -> uniform (no opinion), c = 1 -> certain
    return {x: (p if x == pick else (1 - p) / (k - 1)) for x in cands}, pick, c


def candidates(g: Group, ev: SeatEvidence | None, binding: Binding) -> list[str]:
    cands = [f"person-{x}" for x in (ev.seats if ev else [])]
    if g.voice not in binding.seat_of_voice or not cands:
        cands.append(g.voice)
    return cands


def calibrate_video(groups: list[Group], ev: SeatEvidence | None, binding: Binding, s: Settings) -> dict[str, Any]:
    """How much to trust the per-utterance video: agreement of its pick with confidently bound voices, above chance."""
    if s.av_video_weight >= 0:
        return {"weight": float(np.clip(s.av_video_weight, 0.0, 1.0)), "how": "fixed"}
    if ev is None or len(ev.seats) < 2 or not binding.seat_of_voice:
        return {"weight": 0.0, "how": "no video / fewer than two seats / nothing bound"}
    agree = n = 0
    for g in groups:
        seat = binding.seat_of_voice.get(g.voice)
        if not seat or g.conf * binding.conf.get(g.voice, 0.0) < s.av_calib_conf:
            continue
        cands = [f"person-{x}" for x in ev.seats]
        V = video_evidence(g, ev, cands, s)
        if V is None:
            continue
        n += 1
        agree += int(max(cands, key=lambda c: V[c]) == f"person-{seat}")
    if n < 3:
        return {"weight": 0.0, "how": f"only {n} calibration utterances", "agree": agree, "n": n}
    chance = 1.0 / len(ev.seats)
    p = agree / n
    w = max(0.0, (p - chance) / (1.0 - chance)) * n / (n + s.av_calib_prior)       # shrunk toward 0 for small n
    return {"weight": round(w, 3), "how": "agreement with confidently bound voices", "agree": agree, "n": n,
            "agreement": round(p, 3), "chance": round(chance, 3)}


def av_decide(g: Group, ev: SeatEvidence | None, binding: Binding, s: Settings, video_w: float) -> Decision:
    cands = candidates(g, ev, binding)
    A, audio_pick, _ = audio_evidence(g, binding, cands, s)
    if g.voice not in binding.seat_of_voice:
        video_w *= 1.0 - binding.off_conf.get(g.voice, 0.5)        # a surely off-camera voice is hard to pull on camera
    V = video_evidence(g, ev, cands, s) if video_w > 0 else None
    if V is None:
        V = {c: 1.0 / len(cands) for c in cands}
        w = 0.0
    else:
        w = video_w
    S = {c: (1 - w) * A[c] + w * V[c] for c in cands}
    ranked = sorted(cands, key=lambda c: -S[c])
    best = ranked[0]
    margin = S[best] - (S[ranked[1]] if len(ranked) > 1 else 0.0)
    source = "audio" if w == 0.0 else ("video" if best != audio_pick else "fused")
    return Decision(best, round(float(np.clip(margin, 0.0, 1.0)), 3), source, {c: round(x, 3) for c, x in S.items()},
                    {c: round(x, 3) for c, x in A.items()}, {c: round(x, 3) for c, x in V.items()})


# ----------------------------------------------------------------------------------------------- transcript
def _label(key: str) -> str:
    if key.startswith("person-"):
        return f"{key[7:].title()} person"
    return f"Off-camera {key.split('-', 1)[-1]}"


def build_transcript(groups: list[Group], speaker_of: dict[int, tuple[str, float, str]], binding: Binding,
                     ev: SeatEvidence | None, session: str, duration: float, meta: dict[str, Any],
                     s: Settings, evidence_of: dict[int, dict[str, Any]] | None = None) -> Transcript:
    """speaker_of: group index -> (speaker key, conf, source). Adjacent same-speaker groups merge again.
    evidence_of: group index -> evidence dict; meta["evidence"] gets one entry per utterance (the first group's)."""
    utts: list[Utterance] = []
    evidence: list[dict[str, Any]] = []
    for i, g in enumerate(groups):
        key, conf, source = speaker_of[i]
        words = [Word(u.word.start, u.word.end, u.word.text, u.word.conf, key) for u in g.units if u.word]
        text = " ".join(u.text for u in g.units)
        if utts and utts[-1].speaker == key and g.start - utts[-1].end <= s.merge_gap:
            last = utts[-1]
            n_old, n_new = max(len(last.text.split()), 1), max(len(text.split()), 1)
            last.end, last.text = max(last.end, round(g.end, 3)), f"{last.text} {text}"
            last.words += words
            last.conf = round((last.conf * n_old + conf * n_new) / (n_old + n_new), 3)
            if last.source != source:
                last.source = "fused"
            evidence[-1]["merged"] = evidence[-1].get("merged", 1) + 1
            continue
        utts.append(Utterance(round(g.start, 3), round(g.end, 3), key, text, words, round(conf, 3), source))
        evidence.append({"voice": g.voice, **(evidence_of or {}).get(i, {})})
    meta["evidence"] = evidence
    speakers: dict[str, SpeakerInfo] = {}
    voice_of_seat = {seat: v for v, seat in binding.seat_of_voice.items()}
    for u in utts:
        if u.speaker in speakers:
            continue
        if u.speaker.startswith("person-"):
            seat = u.speaker[7:]
            speakers[u.speaker] = SpeakerInfo(_label(u.speaker), seat, True, voice_of_seat.get(seat),
                                              ev.track_of_seat.get(seat) if ev else None)
        else:
            speakers[u.speaker] = SpeakerInfo(_label(u.speaker), None, False, u.speaker, None)
    return Transcript(session, duration, speakers, utts, meta)


def fuse(asr: AsrResult, diar: DiarResult, asd: AsdResult | None, method: str, s: Settings | None = None,
         session: str = "", duration: float | None = None, decider: Any = None, llm: Any = None) -> Transcript:
    """Runs the named method; `decider` / `llm` are injected clients (see fuse_decider.py / fuse_llm.py)."""
    s = s or Settings()
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    t0 = time.perf_counter()
    units = units_of(asr, s)
    assign_units(units, diar.turns, s)
    groups = group_units(units, s)
    end = max([seg.end for seg in asr.segments] + [t.end for t in diar.turns] + [0.0])
    duration = duration if duration is not None else round(end, 3)
    meta: dict[str, Any] = {"method": method, "settings": asdict(s), "stages": ["align"],
                            "models": {"asr": asr.model, "diar": diar.model, "asd": asd.model if asd else None},
                            "word_timings": any(seg.words for seg in asr.segments)}
    binding, ev = Binding(note="method align: no vision"), None
    if method != "align" and asd is not None and asd.faces:
        ev = SeatEvidence(asd, [(t.start, t.end) for t in diar.turns])
        binding = bind_voices(diar.turns, ev, s)
        meta["stages"].append("bind")
        meta["binding"] = {"seat_of_voice": binding.seat_of_voice, "conf": binding.conf, "off_conf": binding.off_conf,
                           "table": binding.table, "seats": ev.seats, "note": binding.note}
    elif method != "align":
        binding.note = "no asd or no seated faces: nobody bound"
        meta["binding"] = {"note": binding.note}
    speaker_of: dict[int, tuple[str, float, str]] = {}
    evidence_of: dict[int, dict[str, Any]] = {}
    if method in ("align", "bind"):
        for i, g in enumerate(groups):
            seat = binding.seat_of_voice.get(g.voice)
            key = f"person-{seat}" if seat else g.voice
            speaker_of[i] = (key, g.conf * (binding.conf.get(g.voice, 1.0) if seat else 1.0), g.source)
    else:
        meta["stages"].append("av")
        calib = calibrate_video(groups, ev, binding, s)
        counts = {"audio": 0, "video": 0, "fused": 0}
        for i, g in enumerate(groups):
            d = av_decide(g, ev, binding, s, float(calib["weight"]))
            counts[d.source] += 1
            speaker_of[i] = (d.speaker, d.conf, d.source if g.source != "rules" else "rules")
            evidence_of[i] = {"audio": d.audio, "video": d.video, "scores": d.scores, "align_conf": round(g.conf, 3)}
        meta["av"] = {"video": calib, "decided_by": counts}
    tr = build_transcript(groups, speaker_of, binding, ev, session, duration, meta, s, evidence_of)
    if method in ("llm", "all"):
        from .fuse_llm import correct_boundaries
        tr = correct_boundaries(tr, llm, s)
    if method in ("decider", "all"):
        from .fuse_decider import ask_decider
        tr = ask_decider(tr, decider, s)
    tr.meta["timing"] = {"fuse_wall_s": round(time.perf_counter() - t0, 3)}
    tr.meta["utterances"] = len(tr.utterances)
    return tr


# ----------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asr", required=True, type=Path)
    ap.add_argument("--diar", required=True, type=Path)
    ap.add_argument("--asd", type=Path, help="optional; without it nobody is bound to a seat")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--method", default="av", choices=METHODS)
    ap.add_argument("--session", default="", help="session id written to the transcript")
    ap.add_argument("--duration", type=float, help="video duration (s); default: end of the last segment")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a Settings field")
    ap.add_argument("--decider-url", default="http://127.0.0.1:8765", help="Strands Decider 2B (method decider)")
    ap.add_argument("--decider-big-url", default="http://127.0.0.1:8766", help="Strands Decider 12B; '' = none")
    ap.add_argument("--llm-url", default="http://127.0.0.1:8790", help="llama-server for method llm")
    ap.add_argument("--render", action="store_true", help="also write .txt / .srt / .md next to --out")
    a = ap.parse_args(argv)
    s = Settings.parse(a.set)
    asr, diar = load_asr(a.asr), load_diar(a.diar)
    asd = load_asd(a.asd) if a.asd else None
    decider = llm = None
    if a.method in ("decider", "all"):
        from .fuse_decider import DeciderCascade
        decider = DeciderCascade(a.decider_url, a.decider_big_url or None)
    if a.method in ("llm", "all"):
        from .fuse_llm import LlamaClient
        llm = LlamaClient(a.llm_url)
    tr = fuse(asr, diar, asd, a.method, s, a.session or a.asr.parent.name, a.duration, decider=decider, llm=llm)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    to_json(tr, a.out)
    if a.render:
        from .render import render_files
        render_files(tr, a.out.with_suffix(""))
    print(json.dumps({"out": str(a.out), "utterances": len(tr.utterances), "speakers": list(tr.speakers),
                      "binding": tr.meta.get("binding", {}).get("seat_of_voice"), **tr.meta.get("timing", {})}))


if __name__ == "__main__":
    main()
