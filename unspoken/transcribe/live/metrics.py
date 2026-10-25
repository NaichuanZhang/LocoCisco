"""Latency and throughput numbers of one live run (latency.json).

  final_latency     speech end -> first final caption: t_emit - segment end, where the segment end is the
                    VAD's (last speech window + pad_ms of trailing silence). The VAD needs min_silence_ms of
                    silence to close a segment, so this includes that wait: it is what a viewer experiences.
  partial_latency   how stale a partial is when shown: t_emit - end of the audio its text covers
  first_partial     segment start -> its first partial (how long before anything appears)
  revision_delay    first final -> a later speaker revision of the same segment
  rtf               wall seconds / audio seconds of the whole run (1.0 = real time, by construction at 1x)
  cpu_rtf           process CPU seconds / audio seconds (cores busy on average, all threads)
  busy_rtf          model time (final ASR + partial ASR + speaker + faces + ASD) / audio seconds
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from .captions import Caption


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    k = (len(ys) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(ys) - 1)
    return round(ys[lo] + (ys[hi] - ys[lo]) * (k - lo), 3)


def dist(xs: list[float]) -> dict[str, Any]:
    return {"n": len(xs), "p50": pct(xs, 0.5), "p95": pct(xs, 0.95), "max": round(max(xs), 3) if xs else None,
            "mean": round(sum(xs) / len(xs), 3) if xs else None}


def latency_report(captions: Iterable[Caption], stats: dict[str, Any], *, threads: dict[str, Any] | None = None,
                   load: tuple[float, float, float] | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    caps = list(captions)
    first_final: dict[str, Caption] = {}
    first_partial: dict[str, Caption] = {}
    revisions: list[float] = []
    partial_lat: list[float] = []
    for c in caps:
        if c.final:
            if c.seg not in first_final:
                first_final[c.seg] = c
            elif c.revision > 0:
                revisions.append(c.t_emit - first_final[c.seg].t_emit)
        else:
            partial_lat.append(c.t_emit - c.end)
            first_partial.setdefault(c.seg, c)
    final_lat = [c.t_emit - c.end for c in first_final.values()]
    first_partial_delay = [c.t_emit - c.start for c in first_partial.values()]
    audio_s = float(stats.get("audio_s") or 0.0)
    wall_s = float(stats.get("wall_s") or 0.0)
    cpu_s = float(stats.get("cpu_s") or 0.0)
    busy_ms = 0.0
    for key, n in (("asr_ms_avg", "finals"), ("partial_asr_ms_avg", "partials"), ("diar_ms_avg", "finals"),
                   ("faces_ms_avg", "frames_detected"), ("asd_ms_avg", "frames_detected")):
        if stats.get(key) is not None:
            busy_ms += float(stats[key]) * float(stats.get(n) or 0)
    if load is None:
        try:
            load = os.getloadavg()
        except OSError:
            load = None
    out: dict[str, Any] = {
        "final_latency_s": dist(final_lat), "partial_latency_s": dist(partial_lat),
        "first_partial_delay_s": dist(first_partial_delay), "revision_delay_s": dist(revisions),
        "segments_with_partials": len(first_partial), "finals": len(first_final),
        "partials": sum(1 for c in caps if not c.final), "revisions": len(revisions),
        "rtf": round(wall_s / audio_s, 3) if audio_s else None,
        "cpu_rtf": round(cpu_s / audio_s, 3) if audio_s else None,
        "busy_rtf": round(busy_ms / 1000.0 / audio_s, 3) if audio_s else None,
        "cpu_cores_avg": round(cpu_s / wall_s, 2) if wall_s else None,
        "frames": {k: stats.get(k) for k in ("frames", "frames_detected", "frames_stale", "frames_dropped")},
        "model_ms": {k: stats.get(k) for k in ("asr_ms_avg", "partial_asr_ms_avg", "diar_ms_avg", "faces_ms_avg", "asd_ms_avg")},
        "threads": threads or {}, "load_avg": list(load) if load else None, "stats": stats,
    }
    out.update(extra or {})
    return out


def write_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False))


__all__ = ["latency_report", "dist", "pct", "write_json"]
