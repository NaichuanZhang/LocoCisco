"""A vision-language model as the judge of who is talking (workstream tx-vlm).

For every unit of time (a baseline ASR segment, or a short window inside one) the judge is shown a few frames
(see vlm_frames.py) and asked, with a GBNF grammar that only admits `left | right | neither | both`, who is
speaking. The answer's probability is read from the model's token log-probabilities at the answer position
(pre-grammar softmax over the vocabulary, renormalised over the four options), so asd.json carries a real
P(left speaking) / P(right speaking) per unit instead of a self-reported confidence.

Outputs per run (testdata/ami-runs/vlm/<method>/<case>/):
  units.json         every unit: times, answer, probabilities, latency, prompt tokens
  asd.json           AsdResult: faces seat-left / seat-right, one AsdFrame per unit
  transcript.json    Transcript: baseline text, speaker chosen by the VLM (`--bind` says how the segment's
                     voice label is used when the VLM is unsure); transcript.<bind>.json for the other binds
  scores*.json       eval.transcribe.score output

Server: llama.cpp `llama-server --mmproj ...` on 127.0.0.1 (vlm_server.py starts one). Client is stdlib only.

Usage:
  python -m unspoken.transcribe.vlm_judge --case testdata/ami/IS1009a-duo-AC-dev \
      --out testdata/ami-runs/vlm/qwen3vl2b-faces-k3/IS1009a-duo-AC-dev --server http://127.0.0.1:8795 \
      --view faces --frames 3 --prompt cues --unit segments --bind voice
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from .contracts import (AsdFrame, AsdResult, AsrResult, DiarResult, FaceInfo, SpeakerInfo, Transcript, Utterance,
                        load_asr, load_diar, to_json)
from .vlm_frames import SEATS, UnitPictures, UnitSampler, VideoFrames, jpeg_data_url, windows, yunet_finder

OPTIONS = ("left", "right", "neither", "both")
GRAMMAR_WORD = 'root ::= "left" | "right" | "neither" | "both"'
# JSON variant: the model must state each mouth before it names the speaker (a grammar-shaped chain of thought)
GRAMMAR_JSON = ('root ::= "{\\"left_mouth\\": " st ", \\"right_mouth\\": " st ", \\"speaker\\": \\"" ans "\\"}"\n'
                'st ::= "\\"closed\\"" | "\\"open\\"" | "\\"moving\\"" | "\\"hidden\\""\n'
                'ans ::= "left" | "right" | "neither" | "both"')
JSON_ANSWER_PREFIX = '"speaker": "'

SYSTEM = ("You judge who is talking in a meeting from camera frames. Two people can be in view: the LEFT person "
          "and the RIGHT person. Other people may be talking from outside the camera view.")

PROMPTS: dict[str, str] = {
    # plain question
    "plain": ("{layout} Who is speaking during this moment? Answer with exactly one word: left, right, both, or "
              "neither."),
    # what to look for
    "cues": ("{layout} Decide who is speaking during this moment. A speaking person's mouth is open or changes "
             "shape from one moment to the next, teeth or tongue may show, and the head often moves while "
             "talking. A listening person's lips stay closed and still, and they often look at the speaker. If "
             "neither visible person moves their lips, someone off camera is speaking: answer neither. Answer "
             "with exactly one word: left, right, both, or neither."),
    # same cues, JSON answer that first states each mouth
    "json": ("{layout} First describe the LEFT person's mouth and the RIGHT person's mouth across the moments "
             "(closed, open, moving, or hidden), then name who is speaking: left, right, both, or neither (neither "
             "= an off-camera person is talking). Answer as JSON."),
}

LAYOUTS = {
    "faces": ("The picture is a grid: each row is one moment in time (top row first, {k} moments), the left "
              "column shows the LEFT person's face and the right column shows the RIGHT person's face."),
    "faces-diff": ("The picture is a grid: each row is one moment in time (top row first, {k} moments). Columns: "
                   "the LEFT person's face, the LEFT person's motion (bright = moved since the previous instant), "
                   "the RIGHT person's face, the RIGHT person's motion."),
    "full": ("You see {k} frames of the same scene in time order. The LEFT person is in the left half of each "
             "frame and the RIGHT person in the right half."),
}


def build_prompt(prompt: str, view: str, k: int) -> str:
    return PROMPTS[prompt].format(layout=LAYOUTS[view].format(k=k))


def grammar_for(prompt: str) -> str:
    return GRAMMAR_JSON if prompt == "json" else GRAMMAR_WORD


@dataclass
class Judgement:
    choice: str                               # left | right | neither | both (grammar-constrained answer)
    probs: dict[str, float]                   # P(option) from token logprobs, sums to 1 over OPTIONS
    latency_s: float = 0.0
    prompt_tokens: int = 0
    text: str = ""                            # raw completion
    error: str | None = None

    @property
    def seat_scores(self) -> dict[str, float]:
        """P(left speaking), P(right speaking): 'both' counts for both seats."""
        return {s: min(1.0, self.probs.get(s, 0.0) + self.probs.get("both", 0.0)) for s in SEATS}


class Judge(Protocol):
    name: str

    def judge(self, pics: UnitPictures, prompt: str, grammar: str) -> Judgement: ...


def _option_of(token: str) -> str | None:
    tok = token.strip().strip('"').lower()
    if not tok:
        return None
    for o in OPTIONS:
        if tok.startswith(o) or (o.startswith(tok) and (len(tok) >= 2 or tok == o[0])):
            return o
    return None


def probs_from_logprobs(content: list[dict[str, Any]], answer_offset: int = 0) -> dict[str, float] | None:
    """Option probabilities from OpenAI-style `logprobs.content` at the token where the answer starts
    (`answer_offset` = characters of grammar-forced text before it; 0 for the bare-word grammar)."""
    pos, target = 0, None
    for i, c in enumerate(content):
        if pos == answer_offset:
            target = content[i]
            break
        pos += len(c.get("token", ""))
    if target is None:
        return None
    mass: dict[str, float] = {o: 0.0 for o in OPTIONS}
    cands = list(target.get("top_logprobs") or [])
    if not any(c.get("token") == target.get("token") for c in cands):
        cands.append({"token": target.get("token", ""), "logprob": target.get("logprob", 0.0)})
    for c in cands:
        o = _option_of(str(c.get("token", "")))
        if o is not None:
            mass[o] += math.exp(float(c.get("logprob", -1e9)))
    total = sum(mass.values())
    if total <= 0:
        return None
    return {o: v / total for o, v in mass.items()}


def parse_answer(text: str, prompt: str) -> str:
    t = text.strip()
    if prompt == "json":
        try:
            t = str(json.loads(t).get("speaker", "")).strip()
        except json.JSONDecodeError:
            i = t.rfind(JSON_ANSWER_PREFIX)
            t = t[i + len(JSON_ANSWER_PREFIX):].split('"')[0] if i >= 0 else t
    return _option_of(t) or "neither"


class LlamaServerJudge:
    """OpenAI-compatible chat completions on a llama.cpp server with --mmproj. stdlib HTTP, one request per unit."""

    def __init__(self, url: str = "http://127.0.0.1:8795", *, name: str | None = None, top_logprobs: int = 10,
                 timeout_s: float = 600.0, jpeg_quality: int = 85, retries: int = 1, constraint: str = "grammar",
                 model: str | None = None) -> None:
        """constraint: "grammar" (llama.cpp GBNF), "guided_choice" (vLLM: the four words; the JSON prompt is not
        supported there), or "none" (prompt only, answer parsed from the text). `model` is sent in the request
        (vLLM needs it; llama-server ignores it)."""
        if constraint not in ("grammar", "guided_choice", "none"):
            raise ValueError(f"unknown constraint {constraint!r}")
        self.url = url.rstrip("/")
        self.top_logprobs, self.timeout_s, self.jpeg_quality, self.retries = top_logprobs, timeout_s, jpeg_quality, retries
        self.constraint, self.model = constraint, model
        self.name = name or self._model_name() or "llama-server"

    def _get(self, path: str) -> Any:
        with urllib.request.urlopen(self.url + path, timeout=10) as r:
            return json.loads(r.read().decode())

    def healthy(self) -> bool:
        try:
            return self._get("/health").get("status") == "ok"
        except Exception:
            return False

    def _model_name(self) -> str | None:
        try:
            d = self._get("/v1/models")
            return Path(d["data"][0]["id"]).name
        except Exception:
            return None

    def request_body(self, images: list[str], prompt: str, grammar: str) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": u}} for u in images]
        content.append({"type": "text", "text": prompt})
        body: dict[str, Any] = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
                                "max_tokens": 48 if "left_mouth" in grammar else 4, "temperature": 0.0,
                                "logprobs": True, "top_logprobs": self.top_logprobs, "stream": False}
        if self.model:
            body["model"] = self.model
        if self.constraint == "grammar":
            body["grammar"] = grammar
            body["cache_prompt"] = True
        elif self.constraint == "guided_choice":
            if grammar != GRAMMAR_WORD:
                raise ValueError("guided_choice supports the one-word answer only (use --prompt plain|cues)")
            body["guided_choice"] = list(OPTIONS)
        return body

    def post(self, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return json.loads(r.read().decode())

    def judge(self, pics: UnitPictures, prompt: str, grammar: str) -> Judgement:
        body = self.request_body([jpeg_data_url(im, self.jpeg_quality) for im in pics.images], prompt, grammar)
        t0 = time.perf_counter()
        last_err = None
        for _ in range(self.retries + 1):
            try:
                resp = self.post(body)
                return parse_response(resp, prompt_kind="json" if grammar == GRAMMAR_JSON else "word",
                                      latency_s=time.perf_counter() - t0)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, KeyError, ValueError) as e:
                last_err = f"{type(e).__name__}: {e}"
        return Judgement("neither", {o: 0.25 for o in OPTIONS}, time.perf_counter() - t0, 0, "", last_err)


def parse_response(resp: dict[str, Any], prompt_kind: str = "word", latency_s: float = 0.0) -> Judgement:
    ch = resp["choices"][0]
    text = ch["message"].get("content") or ""
    choice = parse_answer(text, "json" if prompt_kind == "json" else "plain")
    content = ((ch.get("logprobs") or {}).get("content")) or []
    offset = 0
    if prompt_kind == "json":
        i = text.find(JSON_ANSWER_PREFIX)
        offset = i + len(JSON_ANSWER_PREFIX) if i >= 0 else -1
    probs = probs_from_logprobs(content, offset) if content and offset >= 0 else None
    if probs is None:
        probs = {o: (1.0 if o == choice else 0.0) for o in OPTIONS}
    return Judgement(choice, probs, latency_s, int((resp.get("usage") or {}).get("prompt_tokens", 0)), text)


class FakeJudge:
    """Tests / dry runs: answers from a callback on the pictures (default: brighter column = speaking)."""

    name = "fake"

    def __init__(self, fn: Callable[[UnitPictures], str] | None = None, latency_s: float = 0.0) -> None:
        self.fn, self.latency_s, self.calls = fn, latency_s, 0

    def judge(self, pics: UnitPictures, prompt: str, grammar: str) -> Judgement:
        self.calls += 1
        if self.fn is not None:
            c = self.fn(pics)
        else:
            img = pics.images[0]
            h, w = img.shape[:2]
            l, r = float(img[h // 4:, : w // 2].mean()), float(img[h // 4:, w // 2:].mean())
            c = "neither" if abs(l - r) < 2 else ("left" if l > r else "right")
        p = {o: 0.05 for o in OPTIONS}
        p[c] = 0.85
        return Judgement(c, p, self.latency_s, 100, c)


# ---------------------------------------------------------------- units, transcript, asd

@dataclass
class UnitResult:
    segment: int
    start: float
    end: float
    choice: str
    probs: dict[str, float]
    latency_s: float
    prompt_tokens: int = 0
    frame_times: list[float] = field(default_factory=list)
    boxes: list[dict[str, list[float]]] = field(default_factory=list)
    error: str | None = None


def units_for(asr: AsrResult, unit: str, window: float) -> list[tuple[int, float, float]]:
    out = []
    for i, s in enumerate(asr.segments):
        if unit == "windows":
            out += [(i, a, b) for a, b in windows(s.start, s.end, window)]
        else:
            out.append((i, s.start, s.end))
    return out


def judge_units(units: list[tuple[int, float, float]], sampler: UnitSampler, judge: Judge, prompt: str, view: str,
                k: int, log: Callable[[str], None] | None = None) -> list[UnitResult]:
    text, grammar = build_prompt(prompt, view, k), grammar_for(prompt)
    out: list[UnitResult] = []
    for n, (seg, a, b) in enumerate(units):
        pics = sampler.sample(a, b)
        if pics is None:
            out.append(UnitResult(seg, a, b, "neither", {o: 0.25 for o in OPTIONS}, 0.0, error="no frames"))
            continue
        j = judge.judge(pics, text, grammar)
        out.append(UnitResult(seg, a, b, j.choice, j.probs, j.latency_s, j.prompt_tokens, pics.times,
                              [{s: [round(v, 4) for v in bx[s]] for s in SEATS} for bx in pics.boxes], j.error))
        if log and (n % 10 == 0 or n == len(units) - 1):
            log(f"  unit {n + 1}/{len(units)} t={a:.1f}-{b:.1f} -> {j.choice} {j.probs} {j.latency_s:.2f}s"
                + (f" ERR {j.error}" if j.error else ""))
    return out


def segment_probs(units: list[UnitResult], n_segments: int) -> list[dict[str, float]]:
    """Per segment: duration-weighted mean of the unit probabilities (zeros = no unit judged)."""
    acc = [defaultdict(float) for _ in range(n_segments)]
    wsum = [0.0] * n_segments
    for u in units:
        w = max(u.end - u.start, 1e-3)
        for o, p in u.probs.items():
            acc[u.segment][o] += w * p
        wsum[u.segment] += w
    return [{o: (a[o] / wsum[i] if wsum[i] else 0.0) for o in OPTIONS} for i, a in enumerate(acc)]


def seat_vector(p: dict[str, float]) -> dict[str, float]:
    """{left, right, neither} with 'both' folded into both seats, renormalised."""
    v = {"left": p.get("left", 0) + p.get("both", 0), "right": p.get("right", 0) + p.get("both", 0),
         "neither": p.get("neither", 0)}
    t = sum(v.values()) or 1.0
    return {k: x / t for k, x in v.items()}


def voice_for(seg_start: float, seg_end: float, diar: DiarResult | None) -> str:
    if diar is None:
        return "voice-?"
    best, overlap = None, 0.0
    for t in diar.turns:
        o = min(seg_end, t.end) - max(seg_start, t.start)
        if o > overlap:
            best, overlap = t.voice, o
    return best or "voice-?"


def decide(seg_p: list[dict[str, float]], voices: list[str], durations: list[float], bind: str,
           min_conf: float = 0.6, fuse_w: float = 0.5) -> list[tuple[str, float, str]]:
    """Per segment (choice in left/right/neither, confidence, source). bind:
      none        the VLM alone
      voice       the VLM; when its best option is below `min_conf`, the voice label's seat affinity decides
      fuse        fuse_w * VLM + (1 - fuse_w) * the voice label's affinity (duration-weighted mean over its segments)
      voice-seat  the voice label's affinity alone: audio decides WHICH voice, the VLM only decides which SEAT
                  that voice sits in (the baseline's vote binding with VLM votes instead of MotionASD)
    """
    vecs = [seat_vector(p) for p in seg_p]
    aff: dict[str, dict[str, float]] = {}
    tot: dict[str, float] = defaultdict(float)
    for v, vec, d in zip(voices, vecs, durations):
        a = aff.setdefault(v, {"left": 0.0, "right": 0.0, "neither": 0.0})
        for k in a:
            a[k] += d * vec[k]
        tot[v] += d
    for v, a in aff.items():
        for k in a:
            a[k] /= tot[v] or 1.0
    out = []
    for v, vec in zip(voices, vecs):
        best = max(vec, key=vec.get)
        if bind == "none":
            out.append((best, vec[best], "video"))
        elif bind == "voice":
            if vec[best] >= min_conf:
                out.append((best, vec[best], "video"))
            else:
                a = aff[v]
                b = max(a, key=a.get)
                out.append((b, a[b], "fused"))
        elif bind == "fuse":
            a = aff[v]
            f = {k: fuse_w * vec[k] + (1 - fuse_w) * a[k] for k in vec}
            b = max(f, key=f.get)
            out.append((b, f[b], "fused"))
        elif bind == "voice-seat":
            a = aff[v]
            b = max(a, key=a.get)
            out.append((b, a[b], "fused"))
        else:
            raise ValueError(f"unknown bind {bind!r}")
    return out


def build_transcript(session: str, duration: float, asr: AsrResult, diar: DiarResult | None,
                     units: list[UnitResult], bind: str, min_conf: float, meta: dict[str, Any]) -> Transcript:
    seg_p = segment_probs(units, len(asr.segments))
    voices = [voice_for(s.start, s.end, diar) for s in asr.segments]
    decisions = decide(seg_p, voices, [max(s.end - s.start, 1e-3) for s in asr.segments], bind, min_conf)
    speakers: dict[str, SpeakerInfo] = {}
    utts = []
    for s, v, (choice, conf, source) in zip(asr.segments, voices, decisions):
        if choice in SEATS:
            key = f"person-{choice}"
            speakers.setdefault(key, SpeakerInfo(f"{choice.title()} person", choice, True, None, f"seat-{choice}"))
        else:
            key = v
            speakers.setdefault(key, SpeakerInfo(f"Off-camera {v.split('-')[-1]}", None, False, v, None))
        utts.append(Utterance(s.start, s.end, key, s.text, list(s.words), round(conf, 4), source))
    return Transcript(session, duration, speakers, utts, {**meta, "bind": bind, "min_conf": min_conf})


def build_asd(units: list[UnitResult], sampler: UnitSampler, width: int, height: int, model: str,
              timing: dict[str, float]) -> AsdResult:
    faces = []
    for s in SEATS:
        mb = sampler.boxes.mean_box(s) or sampler.boxes.box(s)
        faces.append(FaceInfo(f"seat-{s}", s, (round(mb[0] * width, 1), round(mb[1] * height, 1),
                                               round(mb[2] * width, 1), round(mb[3] * height, 1)),
                              sampler.boxes.seen.get(s, 0)))
    frames = []
    for u in units:
        ss = Judgement(u.choice, u.probs).seat_scores
        frames.append(AsdFrame(round((u.start + u.end) / 2, 3), {f"seat-{s}": round(ss[s], 4) for s in SEATS}))
    mean_len = statistics.mean([u.end - u.start for u in units]) if units else 1.0
    return AsdResult(model, round(1.0 / max(mean_len, 1e-3), 3), faces, frames, timing)


# ---------------------------------------------------------------- run

def run(case: Path, out: Path, judge: Judge, *, asr_path: Path, diar_path: Path | None, view: str = "faces",
        k: int = 3, prompt: str = "cues", unit: str = "segments", window: float = 1.5, binds: tuple[str, ...] = ("voice",),
        min_conf: float = 0.6, tile: int = 224, zoom: float = 2.2, max_side: int = 704, faces: str = "yunet",
        limit: int | None = None, threads: int | None = None, log: Callable[[str], None] | None = print,
        source: Any | None = None) -> dict[str, Any]:
    meta = json.loads((case / "meta.json").read_text())
    asr = load_asr(asr_path)
    diar = load_diar(diar_path) if diar_path and Path(diar_path).exists() else None
    src = source or VideoFrames(case / "video.mp4")
    finder = yunet_finder() if faces == "yunet" else None
    sampler = UnitSampler(src, view=view, k=k, tile=tile, zoom=zoom, max_side=max_side, finder=finder)
    units = units_for(asr, unit, window)
    if limit:
        units = units[:limit]
    load1 = os.getloadavg()[0]
    t0 = time.perf_counter()
    results = judge_units(units, sampler, judge, prompt, view, k, log)
    wall = time.perf_counter() - t0
    lat = [r.latency_s for r in results if r.latency_s > 0]
    audio_s = float(meta.get("duration") or src.duration)
    speech_s = sum(b - a for _, a, b in units)
    timing = {"wall_s": round(wall, 2), "audio_s": audio_s, "rtf": round(wall / max(audio_s, 1e-3), 4),
              "speech_s": round(speech_s, 2), "rtf_speech": round(wall / max(speech_s, 1e-3), 4),
              "units": len(results), "frames_read": sampler.frames_read,
              "frames_per_s": round(sampler.frames_read / max(wall, 1e-3), 3),
              "unit_latency_mean_s": round(statistics.mean(lat), 3) if lat else 0.0,
              "unit_latency_median_s": round(statistics.median(lat), 3) if lat else 0.0,
              "prompt_tokens_mean": round(statistics.mean([r.prompt_tokens for r in results]), 1) if results else 0,
              "threads": threads or 0, "load1_start": round(load1, 2), "load1_end": round(os.getloadavg()[0], 2),
              "errors": sum(1 for r in results if r.error)}
    settings = {"view": view, "frames": k, "prompt": prompt, "unit": unit, "window": window, "tile": tile,
                "zoom": zoom, "max_side": max_side, "faces": faces, "limit": limit}
    model = f"VLM judge {judge.name} ({view}, k={k}, {prompt}, {unit})"
    out.mkdir(parents=True, exist_ok=True)
    (out / "units.json").write_text(json.dumps({"model": judge.name, "settings": settings, "timing": timing,
                                                "prompt_text": build_prompt(prompt, view, k),
                                                "units": [asdict(r) for r in results]}, indent=1))
    to_json(build_asd(results, sampler, src.width, src.height, model, timing), out / "asd.json")
    tmeta = {"method": f"vlm-judge: baseline ASR text, speaker by {judge.name}", "models": {"asr": asr.model,
             "diar": diar.model if diar else None, "vlm": judge.name}, "settings": settings, "timing": timing}
    scores: dict[str, Any] = {}
    from eval.transcribe.score import score_transcript  # eval package is on the path in the repo
    for i, bind in enumerate(binds):
        tr = build_transcript(meta["case"], audio_s, asr, diar, results, bind, min_conf, tmeta)
        suffix = "" if i == 0 else f".{bind}"
        to_json(tr, out / f"transcript{suffix}.json")
        sc = score_transcript(case, json.loads(to_json(tr)))
        sc.update({"bind": bind, "rtf": timing["rtf"], "threads": timing["threads"], "model": judge.name,
                   **{k_: v for k_, v in settings.items() if k_ in ("view", "frames", "prompt", "unit")}})
        (out / f"scores{suffix}.json").write_text(json.dumps(sc, indent=1))
        scores[bind] = sc
        if log:
            log(f"{meta['case']} bind={bind}: seat_acc={sc['seat_acc']:.3f} sa_acc={sc['sa_acc']:.3f} "
                f"cpwer={sc['cpwer']:.3f} der={sc['der']:.3f} rtf={timing['rtf']:.3f} "
                f"lat={timing['unit_latency_mean_s']:.2f}s/unit")
    return {"timing": timing, "scores": scores, "out": str(out)}


def rescore(case: Path, out: Path, *, asr_path: Path, diar_path: Path | None, binds: tuple[str, ...], min_conf: float,
            log: Callable[[str], None] | None = print) -> dict[str, Any]:
    """Rebuild transcript*.json / scores*.json from an existing units.json (no model calls)."""
    from eval.transcribe.score import score_transcript

    meta = json.loads((case / "meta.json").read_text())
    u = json.loads((out / "units.json").read_text())
    results = [UnitResult(**{k: v for k, v in r.items() if k in UnitResult.__dataclass_fields__}) for r in u["units"]]
    asr = load_asr(asr_path)
    diar = load_diar(diar_path) if diar_path and Path(diar_path).exists() else None
    audio_s = float(meta.get("duration") or 0.0)
    tmeta = {"method": f"vlm-judge: baseline ASR text, speaker by {u['model']}", "models": {"asr": asr.model,
             "diar": diar.model if diar else None, "vlm": u["model"]}, "settings": u["settings"], "timing": u["timing"]}
    scores: dict[str, Any] = {}
    for bind in binds:
        tr = build_transcript(meta["case"], audio_s, asr, diar, results, bind, min_conf, tmeta)
        existing = (out / "scores.json").exists() and json.loads((out / "scores.json").read_text()).get("bind")
        suffix = "" if bind == existing else f".{bind}"
        to_json(tr, out / f"transcript{suffix}.json")
        sc = score_transcript(case, json.loads(to_json(tr)))
        sc.update({"bind": bind, "rtf": u["timing"]["rtf"], "threads": u["timing"]["threads"], "model": u["model"],
                   **{k_: v for k_, v in u["settings"].items() if k_ in ("view", "frames", "prompt", "unit")}})
        (out / f"scores{suffix}.json").write_text(json.dumps(sc, indent=1))
        scores[bind] = sc
        if log:
            log(f"{meta['case']} bind={bind}: seat_acc={sc['seat_acc']:.3f} sa_acc={sc['sa_acc']:.3f} "
                f"cpwer={sc['cpwer']:.3f} der={sc['der']:.3f} (rescored)")
    return scores


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True, type=Path, help="testdata/ami/<case> (video.mp4 + meta.json)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--asr", type=Path, help="asr.json (default: testdata/ami-baseline/<case>/asr.json)")
    ap.add_argument("--diar", type=Path, help="diar.json (default: testdata/ami-baseline/<case>/diar.json)")
    ap.add_argument("--server", default="http://127.0.0.1:8795")
    ap.add_argument("--model-name", default=None, help="label for the judge (default: the server's model file)")
    ap.add_argument("--constraint", default="grammar", choices=["grammar", "guided_choice", "none"],
                    help="grammar = llama.cpp GBNF; guided_choice = vLLM; none = prompt only")
    ap.add_argument("--model-id", default=None, help="model id to send in requests (vLLM needs it)")
    ap.add_argument("--view", default="faces", choices=["full", "faces", "faces-diff"])
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--prompt", default="cues", choices=sorted(PROMPTS))
    ap.add_argument("--unit", default="segments", choices=["segments", "windows"])
    ap.add_argument("--window", type=float, default=1.5)
    ap.add_argument("--bind", default="voice,none,fuse,voice-seat", help="comma list; the first one is transcript.json")
    ap.add_argument("--min-conf", type=float, default=0.6)
    ap.add_argument("--tile", type=int, default=224)
    ap.add_argument("--zoom", type=float, default=2.2)
    ap.add_argument("--max-side", type=int, default=704)
    ap.add_argument("--faces", default="yunet", choices=["yunet", "none"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N units (smoke test)")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "0") or 0),
                    help="recorded in timings only; the server owns its threads")
    ap.add_argument("--rescore", action="store_true", help="rebuild transcripts/scores from OUT/units.json, no model")
    ap.add_argument("--fake", nargs="?", const="auto", default=None, metavar="ANSWER",
                    help="FakeJudge instead of a server: 'auto' (brighter column) or a constant answer, e.g. "
                         "`--fake left` = the trivial floor 'the left person always talks'")
    a = ap.parse_args(argv)
    base = a.case.parent.parent / "ami-baseline" / a.case.name
    asr_path = a.asr or base / "asr.json"
    diar_path = a.diar or base / "diar.json"
    if a.rescore:
        rescore(a.case, a.out, asr_path=asr_path, diar_path=diar_path, binds=tuple(a.bind.split(",")),
                min_conf=a.min_conf)
        return
    if a.fake:
        judge: Judge = FakeJudge() if a.fake == "auto" else FakeJudge(lambda pics, c=a.fake: c)
        judge.name = f"fake-{a.fake}"
    else:
        judge = LlamaServerJudge(a.server, name=a.model_name, constraint=a.constraint, model=a.model_id)
        if not judge.healthy():
            sys.exit(f"no healthy llama-server at {a.server} (start one: python -m unspoken.transcribe.vlm_server)")
    run(a.case, a.out, judge, asr_path=asr_path, diar_path=diar_path, view=a.view, k=a.frames, prompt=a.prompt,
        unit=a.unit, window=a.window, binds=tuple(a.bind.split(",")), min_conf=a.min_conf, tile=a.tile, zoom=a.zoom,
        max_side=a.max_side, faces=a.faces, limit=a.limit, threads=a.threads)


if __name__ == "__main__":
    main()
