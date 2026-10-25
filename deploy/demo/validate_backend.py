#!/usr/bin/env python3
"""Offline, backend-only validation using installed gb10 models and local AMI files.

Run in the existing image (does not start/restart a service or contact its browser API):
  docker exec lococisco-demo python /app/deploy/demo/validate_backend.py

Default: two complete 1x media runs, a 60 s reset/replay, then separate controlled
text cases. Reports go outside Git, to /runtime/validation. No full real transcript,
image, PCM, embedding, credentials, environment dump, or request headers are saved.
Optional bounded public action excerpts aid review; they are disabled by default.
The socket gate covers Python networking in THIS process, not the host or native
subprocesses. Only local-file ffprobe/FFmpeg subprocesses are used by replay.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import functools
import hashlib
import importlib.metadata
import inspect
import ipaddress
import json
import logging
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = ("decider", "llm", "asr", "vad", "diarization", "faces", "asd")
SOURCE_FILES = (
    "deploy/demo/validate_backend.py", "config/models.toml", "config/meeting.toml",
    "unspoken/meeting/demo.py", "unspoken/meeting/replay_media.py",
    "unspoken/meeting/questions.py", "unspoken/meeting/cascade.py",
    "unspoken/meeting/llm_agent.py", "unspoken/meeting/fusion.py",
    "unspoken/meeting/reflex.py", "unspoken/meeting/policy.py",
    "unspoken/sensors/pipeline.py", "unspoken/transcribe/asd_live.py",
    "unspoken/transcribe/asd_lightasd.py",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text_shape(text: str) -> dict:
    return {"characters": len(text), "sha256": hashlib.sha256(text.encode()).hexdigest()}


def distribution(values) -> dict:
    xs = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not xs:
        return {"count": 0}
    return {"count": len(xs), "min": round(xs[0], 5), "max": round(xs[-1], 5),
            "mean": round(sum(xs) / len(xs), 5),
            "p50": round(xs[math.ceil(len(xs) * .5) - 1], 5),
            "p95": round(xs[math.ceil(len(xs) * .95) - 1], 5)}


def error_category(text: str) -> str:
    """Categorize diagnostics without retaining prompt-bearing exception/log text."""
    if re.search(r"\b422\b", text):
        return "HTTP 422 / invalid model request"
    if re.search(r"\b(?:401|403)\b", text):
        return "HTTP authentication/authorization failure"
    if "timeout" in text.lower() or "timed out" in text.lower():
        return "timeout"
    if "decider failed" in text.lower():
        return "decider failure"
    if "uncertain" in text.lower() or "identity" in text.lower():
        return "identity evidence warning"
    return "message omitted (may contain meeting/model content)"


class LoopbackGate:
    """Deny non-loopback TCP/UDP and DNS before imports/provider initialization."""

    def __init__(self):
        self.phase = "self_test"
        self.allowed = collections.Counter()
        self.denied = collections.Counter()
        self.originals = []

    def check(self, address, operation: str, family=None):
        if family == socket.AF_UNIX:
            return
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, bytes):
            host = host.decode("ascii", errors="replace")
        host = str(host)
        local = host.rstrip(".").lower() == "localhost"
        try:
            ip = ipaddress.ip_address(host.split("%")[0])
            local |= ip.is_loopback or bool(getattr(ip, "ipv4_mapped", None) and ip.ipv4_mapped.is_loopback)
        except ValueError:
            pass
        if not local:
            # Do not log arbitrary external host strings (they could contain data).
            self.denied[(self.phase, operation)] += 1
            raise PermissionError("Backend validator permits loopback networking only")
        port = address[1] if isinstance(address, tuple) and len(address) > 1 else None
        self.allowed[(self.phase, operation, host, port)] += 1

    def patch(self, obj, name, value):
        self.originals.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def __enter__(self):
        for name in ("connect", "connect_ex"):
            original = getattr(socket.socket, name)

            def connect(sock, address, _original=original, _name=name):
                self.check(address, _name, sock.family)
                return _original(sock, address)
            self.patch(socket.socket, name, connect)
        create = socket.create_connection

        def create_connection(address, *args, **kwargs):
            self.check(address, "create_connection")
            return create(address, *args, **kwargs)
        self.patch(socket, "create_connection", create_connection)
        gai = socket.getaddrinfo

        def getaddrinfo(host, port, *args, **kwargs):
            self.check((host, port), "getaddrinfo")
            # Resolve even localhost without a DNS query.
            if str(host).rstrip(".").lower() == "localhost":
                family = args[0] if args else kwargs.get("family", 0)
                host = "::1" if family == socket.AF_INET6 else "127.0.0.1"
            return gai(host, port, *args, **kwargs)
        self.patch(socket, "getaddrinfo", getaddrinfo)
        for name in ("gethostbyname", "gethostbyname_ex"):
            original = getattr(socket, name)

            def lookup(host, _original=original, _name=name):
                self.check((host, None), _name)
                if str(host).rstrip(".").lower() == "localhost":
                    host = "127.0.0.1"
                return _original(host)
            self.patch(socket, name, lookup)

        def reverse(host):
            self.check((host, None), "gethostbyaddr")
            return ("localhost", [], [str(host)])
        self.patch(socket, "gethostbyaddr", reverse)
        sendto = socket.socket.sendto

        def udp(sock, data, *args):
            self.check(args[-1], "sendto", sock.family)
            return sendto(sock, data, *args)
        self.patch(socket.socket, "sendto", udp)
        if hasattr(socket.socket, "sendmsg"):
            sendmsg = socket.socket.sendmsg

            def msg(sock, buffers, ancdata=(), flags=0, address=None):
                if address is not None:
                    self.check(address, "sendmsg", sock.family)
                    return sendmsg(sock, buffers, ancdata, flags, address)
                return sendmsg(sock, buffers, ancdata, flags)
            self.patch(socket.socket, "sendmsg", msg)
        self.self_test()
        self.phase = "initialization"
        return self

    def self_test(self):
        attempts = [
            lambda: socket.create_connection(("203.0.113.1", 9), timeout=.01),
            lambda: socket.create_connection(("192.168.1.1", 9), timeout=.01),
            lambda: socket.create_connection(("::ffff:203.0.113.1", 9), timeout=.01),
            lambda: socket.getaddrinfo("not-local.invalid", 443),
        ]
        with socket.socket() as tcp, socket.socket(type=socket.SOCK_DGRAM) as udp:
            attempts += [lambda: tcp.connect(("203.0.113.1", 9)),
                         lambda: tcp.connect_ex(("203.0.113.1", 9)),
                         lambda: udp.sendto(b"validator self-test", ("203.0.113.1", 9))]
            for attempt in attempts:
                try:
                    attempt()
                except PermissionError:
                    continue
                raise AssertionError("Network gate self-test did not reject a non-loopback destination")

    def report(self):
        return {
            "scope": "this Python process only; no host firewall or service changes",
            "installed_before_registry": True,
            "self_test_denials": sum(n for (phase, _), n in self.denied.items() if phase == "self_test"),
            "unexpected_denials": sum(n for (phase, _), n in self.denied.items() if phase != "self_test"),
            "allowed": [{"phase": p, "operation": op, "host": host, "port": port, "count": n}
                        for (p, op, host, port), n in sorted(self.allowed.items(), key=str)],
            "denied": [{"phase": p, "operation": op, "count": n} for (p, op), n in self.denied.items()],
        }

    def __exit__(self, *exc):
        for obj, name, original in reversed(self.originals):
            setattr(obj, name, original)


class Phase:
    def __init__(self, name: str):
        self.name = name
        self.calls = collections.defaultdict(lambda: {"success": 0, "errors": 0, "cancelled": 0, "ms": []})
        self.values = collections.defaultdict(list)
        self.counts = collections.Counter()
        self.warnings = collections.Counter()
        self.track_ids = set()
        self.face_examples = []
        self.speaking_examples = []
        self.utterance_observed_at = {}
        self.action_proposals = []
        self.verdicts = collections.defaultdict(list)

    def summary(self):
        return {
            "name": self.name,
            "calls": {k: {**{f: v[f] for f in ("success", "errors", "cancelled")},
                          "wall_ms": distribution(v["ms"])} for k, v in sorted(self.calls.items())},
            "values": {k: distribution(v) for k, v in sorted(self.values.items())},
            "counts": dict(sorted(self.counts.items())),
            "distinct_face_tracks": len(self.track_ids), "face_box_examples": self.face_examples,
            "scored_face_examples": self.speaking_examples,
            "warnings": [{"channel": ch, "source": src, "level": level, "category": cat, "count": n}
                         for (ch, src, level, cat), n in self.warnings.items()],
        }


class Instrumentation(logging.Handler):
    """Observers call the original methods and return their exact, unchanged outputs."""

    def __init__(self, gate):
        super().__init__(logging.WARNING)
        self.gate = gate
        self.phase = Phase("initialization")
        self.patches = []

    def begin(self, name):
        self.phase = Phase(name)
        self.gate.phase = name
        return self.phase

    def emit(self, record):
        self.phase.warnings[("python", record.name, record.levelname, error_category(record.getMessage()))] += 1

    def wrap(self, obj, name, label, observe=None):
        original = getattr(obj, name)
        had_own = name in vars(obj)

        def finish(phase, result, args):
            phase.calls[label]["success"] += 1
            if observe:
                observe(phase, result, args)

        @functools.wraps(original)
        def sync(*args, **kwargs):
            phase, t0 = self.phase, time.perf_counter()
            try:
                result = original(*args, **kwargs)
                finish(phase, result, args)
                return result
            except Exception:
                phase.calls[label]["errors"] += 1
                raise
            finally:
                phase.calls[label]["ms"].append((time.perf_counter() - t0) * 1000)

        @functools.wraps(original)
        async def async_call(*args, **kwargs):
            phase, t0 = self.phase, time.perf_counter()
            try:
                result = await original(*args, **kwargs)
                finish(phase, result, args)
                return result
            except asyncio.CancelledError:
                phase.calls[label]["cancelled"] += 1
                raise
            except Exception:
                phase.calls[label]["errors"] += 1
                raise
            finally:
                phase.calls[label]["ms"].append((time.perf_counter() - t0) * 1000)

        @functools.wraps(original)
        async def stream(*args, **kwargs):
            phase, t0 = self.phase, time.perf_counter()
            try:
                async for event in original(*args, **kwargs):
                    yield event
                phase.calls[label]["success"] += 1
            except asyncio.CancelledError:
                phase.calls[label]["cancelled"] += 1
                raise
            except Exception:
                phase.calls[label]["errors"] += 1
                raise
            finally:
                phase.calls[label]["ms"].append((time.perf_counter() - t0) * 1000)

        replacement = stream if inspect.isasyncgenfunction(original) else (
            async_call if inspect.iscoroutinefunction(original) else sync)
        self.patches.append((obj, name, original, had_own))
        setattr(obj, name, replacement)

    def install(self, registry):
        from unspoken.sensors.vad import SileroStream
        from unspoken.sensors.diar import EmbeddingDiarizer
        from unspoken.sensors.faces import YuNetSession
        from unspoken.transcribe.diar import OnlineWindowSession
        from unspoken.transcribe.asd_live import LightASDLiveSession
        from unspoken.transcribe.asd_lightasd import LightASDScorer

        def segments(p, result, args):
            p.counts["vad_segments_returned"] += len(result)
            p.values["vad_segment_seconds"].extend(s.end - s.start for s in result)

        def faces(p, result, args):
            p.values["faces_per_update"].append(len(result))
            for f in result:
                p.track_ids.add(f.track_id)
                if len(p.face_examples) < 5:
                    p.face_examples.append({"track_id": f.track_id, "bbox": list(f.bbox), "seat": f.seat})

        def scores(p, result, args):
            # Only scalars; never retain the returned array or model input crops/features.
            p.values["asd_window_probability_min"].append(float(result.min()))
            p.values["asd_window_probability_max"].append(float(result.max()))
            p.values["asd_window_probability_mean"].append(float(result.mean()))
            p.counts["asd_window_output_frames"] += len(result)

        def scorer_created(p, result, args):
            self.wrap(args[0].model, "forward", "asd.torch_forward")

        def tracks_scored(p, result, args):
            p.values["asd_track_speaking"].extend(float(f.speaking) for f in result)
            for f in result:
                if f.speaking > 0 and len(p.speaking_examples) < 5:
                    p.speaking_examples.append({"track_id": f.track_id, "bbox": list(f.bbox),
                                               "seat": f.seat, "speaking": float(f.speaking)})

        self.wrap(SileroStream, "feed", "vad.feed", segments)
        self.wrap(SileroStream, "flush", "vad.flush", segments)
        self.wrap(SileroStream, "_probability", "vad.onnx_probability",
                  lambda p, r, a: p.values["vad_probability"].append(float(r)))
        self.wrap(registry.asr(), "transcribe", "asr.transcribe",
                  lambda p, r, a: p.counts.update({"asr_nonempty": int(bool(r.text.strip()))}))
        self.wrap(OnlineWindowSession, "label", "diarization.label",
                  lambda p, r, a: p.values["diarization_confidence"].append(r.conf))
        self.wrap(EmbeddingDiarizer, "embed", "diarization.embedding")
        self.wrap(YuNetSession, "update", "faces.update", faces)
        self.wrap(YuNetSession, "detect", "faces.yunet_detect")
        self.wrap(LightASDLiveSession, "update", "asd.session_update", tracks_scored)
        self.wrap(LightASDLiveSession, "_score", "asd.session_score_attempt")
        self.wrap(LightASDScorer, "__init__", "asd.load", scorer_created)
        self.wrap(LightASDScorer, "score", "asd.window_score", scores)
        self.wrap(registry.decider(), "ask", "decider.ask")
        self.wrap(type(registry.strands_model("llm")), "stream", "llm.stream")

    async def on_event(self, event):
        from unspoken.meeting.contracts import MT
        p, d = self.phase, event.data
        p.counts["event:" + event.topic] += 1
        if event.topic == MT.SEGMENT:
            p.values["segment_voice_confidence"].append(float(d.get("voice_conf") or 0))
        elif event.topic == MT.UTTERANCE:
            sp = d.get("speaker") or {}
            p.utterance_observed_at[d.get("utterance_id")] = time.perf_counter()
            p.counts["origin:" + str(d.get("source"))] += 1
            p.counts["identity:" + ("known" if sp.get("name") else "unknown")] += 1
            p.values["utterance_speaker_confidence"].append(float(sp.get("conf") or 0))
        elif event.topic == MT.PROPOSAL:
            for key in ("mode", "action", "route", "confidence_source"):
                p.counts["proposal_" + key + ":" + str(d.get(key))] += 1
            p.values["proposal_latency_ms"].append(float(d.get("latency_ms") or 0))
            p.values["proposal_confidence"].append(float(d.get("confidence") or 0))
            if d.get("action") != "none":
                p.action_proposals.append({"utterance_id": d.get("utterance_id"), "proposal": proposal_summary(d)})
        elif event.topic == MT.VERDICT:
            p.counts["verdict:" + str(d.get("verdict")) + ":" + str(d.get("action"))] += 1
            p.verdicts[d.get("utterance_id")].append({k: d.get(k) for k in ("verdict", "action", "risk")})
            observed = p.utterance_observed_at.get(d.get("utterance_id"))
            if observed is not None:
                p.values["observed_utterance_to_verdict_ms"].append((time.perf_counter() - observed) * 1000)
        elif event.topic == MT.ITEM:
            p.counts["item:" + str(d.get("kind")) + ":" + str(d.get("status"))] += 1
        elif event.topic == MT.LOG and d.get("level") in {"warn", "warning", "error"}:
            p.warnings[("bus", str(d.get("agent")), str(d.get("level")), error_category(str(d.get("text"))))] += 1

    def restore(self):
        for obj, name, original, had_own in reversed(self.patches):
            if had_own:
                setattr(obj, name, original)
            else:
                delattr(obj, name)


def invariant(report, name, ok):
    report.setdefault("checks", []).append({"name": name, "ok": bool(ok)})
    return bool(ok)


def state_invariants(runtime, result):
    egress = dict(runtime.state.egress)
    result.update(egress=egress, local_previews=runtime.snapshot()["previews"])
    invariant(result, "zero_actual_external_egress",
              all(egress[k] == 0 for k in ("bytes", "total_bytes", "audio_bytes_out", "video_bytes_out")))
    invariant(result, "all_items_are_local_previews",
              all(i.get("preview") and i.get("external_delivery") is False and i.get("message") is None
                  for i in runtime.state.items.values()))
    invariant(result, "all_cards_are_local_previews",
              all(c.get("preview") and c.get("external_delivery") is False and c.get("card") != "posted"
                  for c in runtime.state.screen.get("cards") or []))
    result["bus_errors"] = [{"subscriber": name, "exception_type": type(exc).__name__,
                             "category": error_category(str(exc))} for name, _, exc in runtime.bus.errors]
    invariant(result, "no_bus_errors", not runtime.bus.errors)


def phase_invariants(result, phase):
    invariant(result, "no_provider_errors", not any(c["errors"] for c in phase.calls.values()))
    invariant(result, "no_warning_or_error_logs", not phase.warnings)


async def reset_checked(runtime, controller):
    handle = runtime._sensors
    await controller.control("restart")
    result = {"generation": controller.generation, "privacy_epoch": controller.privacy_epoch}
    for name, ok in {
        "identities_cleared": not runtime.state.people,
        "transcript_cleared": not runtime.state.transcript,
        "history_and_provenance_cleared": not runtime.bus.history and not runtime.bus._causes,
        "items_cards_cleared": not runtime.state.items and not runtime.state.screen["cards"],
        "previews_cleared": not runtime.notifier.posts and not runtime.notifier.dms
                            and runtime.snapshot()["previews"]["text_bytes"] == 0,
        "approvals_cleared": not runtime.agent("actuator").pending,
        "policy_context_cleared": not runtime.agent("policy")._utterances,
        "fusion_context_cleared": not runtime.agent("fusion").frames and not runtime.agent("fusion").pending_chat,
        "sensor_sessions_released": handle is None or not handle.sessions,
    }.items():
        invariant(result, name, ok)
    return result


async def media_phase(runtime, controller, instrumentation, path, *, name, limit=None, include_examples=False):
    await controller.load(path)
    phase = instrumentation.begin(name)
    duration = controller.media.duration
    target = min(duration, limit) if limit is not None else duration
    bounded = target < duration
    started = time.perf_counter()
    await controller.control("play")
    handle = runtime._sensors
    next_progress = 30
    while controller.state not in {"ended", "error"}:
        position = controller.clock.now()
        if bounded and position >= target:
            await controller.control("pause")
            # Stop only this validator's sources; do not report cancellation as EOF.
            if controller._monitor:
                controller._monitor.cancel()
                await asyncio.gather(controller._monitor, return_exceptions=True)
            await runtime.stop_sensors()
            await runtime.bus.drain(180)
            break
        if time.perf_counter() - started > target + 180:
            raise TimeoutError("Bounded backend media validation exceeded its deadline")
        if position >= next_progress:
            print(json.dumps({"progress": name, "media_seconds": round(position, 1),
                              "segments": handle.stats.segments_published}), flush=True)
            next_progress += 30
        await asyncio.sleep(.2)
    await runtime.bus.drain(180)
    result = {
        "input_kind": "genuine_local_video_no_injected_utterances_or_perception",
        "media": {"file": path.name, "sha256": digest(path), "duration_seconds": duration},
        "playback_rate": 1.0, "requested_seconds": target, "bounded": bounded,
        "complete": not bounded and controller.state == "ended",
        "state": controller.state, "wall_seconds": round(time.perf_counter() - started, 3),
        "audio_media_seconds": round(handle.audio_media_time, 3),
        "video_media_seconds": round(handle.video_media_time, 3),
        "sensor_stats": handle.stats.summary(), "measurements": phase.summary(),
        "retained_public_utterances": len(runtime.state.transcript),
        "final_items_by_kind": dict(collections.Counter(i["kind"] for i in runtime.state.items.values())),
        "final_identities": {"known": sum(bool(p.name) for p in runtime.state.people.values()),
                             "unknown": sum(not p.name for p in runtime.state.people.values())},
        "operator_note": "ReplayController supplies Demo operator as meeting owner; this is not an inferred participant.",
    }
    source = path.with_suffix(".source.json")
    if source.is_file():
        result["media"]["source_manifest"] = json.loads(source.read_text())
    retained = {u["utterance_id"]: u for u in runtime.state.transcript}
    result["public_action_examples"] = []
    for entry in phase.action_proposals:
        utterance = retained.get(entry["utterance_id"])
        if utterance is None:  # A later privacy wipe must also exclude diagnostic examples.
            continue
        text = str(utterance.get("text") or "")
        example = {"proposal": entry["proposal"], "source": {
            "start": utterance.get("start"), "end": utterance.get("end"), "origin": utterance.get("source"),
            "speaker_known": bool((utterance.get("speaker") or {}).get("name")), **text_shape(text)},
            "final_verdicts_for_utterance": phase.verdicts[entry["utterance_id"]]}
        if include_examples:
            example.update(public_excerpt=text[:160], excerpt_truncated=len(text) > 160)
        result["public_action_examples"].append(example)
        if len(result["public_action_examples"]) >= 6:
            break
    invariant(result, "source_completed_or_explicitly_bounded", bounded or result["complete"])
    invariant(result, "no_sensor_errors", handle.stats.errors == 0)
    for label in ("vad.onnx_probability", "asr.transcribe", "diarization.label", "diarization.embedding",
                  "faces.yunet_detect", "asd.session_update", "asd.window_score", "asd.torch_forward", "decider.ask"):
        invariant(result, "real_calls:" + label, phase.calls[label]["success"] > 0)
    invariant(result, "real_face_boxes_and_tracks_returned", bool(phase.track_ids) and bool(phase.face_examples))
    invariant(result, "real_asd_probabilities_returned", bool(phase.values["asd_window_probability_max"]))
    state_invariants(runtime, result)
    phase_invariants(result, phase)
    # Select only an actual retained on-record ASR utterance; never assemble or clean words.
    candidate = next((dict(u) for u in runtime.state.transcript
                      if re.search(r"\b(?:decided|agreed|decision)\b", u.get("text", ""), re.I)), None)
    if candidate:
        candidate["_fixture"] = path.name
    return result, candidate


def proposal_summary(data):
    # No requests, model replies, tool text, or generated meeting content in reports.
    keep = ("by", "mode", "action", "route", "model", "confidence", "confidence_source",
            "confidence_calibrated", "confidence_action", "latency_ms", "owner",
            "owner_confidence", "owner_evidence", "safety", "tokens")
    out = {k: data[k] for k in keep if k in data}
    out["args"] = {k: v for k, v in (data.get("args") or {}).items() if k in {"owner", "due", "scene", "on", "card"}}
    out["tool_names"] = [c.get("name") for c in data.get("tool_calls") or []]
    if "safety" in out:
        # Safety may include source/generated text in proposed_args.
        out["safety"] = {k: v for k, v in out["safety"].items() if k in {"reason", "outcome", "proposed_action"}}
    return out


async def controlled_phase(runtime, controller, instrumentation, candidate):
    from unspoken.core.events import Event
    from unspoken.meeting.base import Person
    from unspoken.meeting.contracts import MT
    from unspoken.meeting.dates import parse_due
    results = []
    cases = [
        ("known_self_commitment", "I'll send the design notes by Friday", "Alex", ["Alex"], "person"),
        ("unknown_with_roster", "I'll send the design notes by Friday", None, ["Alex"], "unknown"),
        ("empty_roster", "I'll send the design notes by Friday", None, [], "unknown"),
        ("negation", "I won't send the design notes by Friday", "Alex", ["Alex"], "person"),
        # Exactly the prior observed false-positive context, not a forced LLM route.
        ("negation_context_regression", "I won't send the design notes by Friday", "Alex", ["Alex"], "person"),
        ("tentative", "Maybe we should send the design notes by Friday.", "Alex", ["Alex"], "person"),
        ("irrelevant", "The weather is nice today.", "Alex", ["Alex"], "person"),
        ("media_origin", "I'll send the design notes by Friday", "Alex", ["Alex"], "media"),
    ]
    if candidate:
        cases.append(("real_asr_derived_decision", candidate["text"], None, [], "unknown"))
    for name, text, speaker_name, roster, origin in cases:
        reset = await reset_checked(runtime, controller)
        phase = instrumentation.begin("controlled:" + name)
        # Synthetic metadata is explicit, not a manufactured face/voice observation.
        # Most cases use neutral context; one reproduces the previously failing prompt.
        title = "CONTROLLED integration test — NOT video transcription" if name == "negation_context_regression" else "Meeting"
        await runtime.bus.publish(Event(MT.STATUS, {"state": "started", "title": title, "owner": None, "agenda": []},
                                        source="backend-validator.controlled-test", ts=0))
        await runtime.bus.drain()
        for person in roster:
            runtime.state.people["controlled:" + person] = Person("controlled:" + person, name=person, guest=False)
        data = {"utterance_id": "controlled:" + name, "start": 1, "end": 2, "text": text,
                "speaker": {"name": speaker_name, "guest": speaker_name is None, "voice": None,
                            "seat": None, "conf": 1.0 if speaker_name else 0.0},
                "source": origin, "context": runtime.state.context_line(),
                "validation": {"phase": "controlled_text_only", "synthetic_participant_metadata": True,
                               "text_origin": "retained_public_ASR" if name == "real_asr_derived_decision" else "synthetic"}}
        started = time.perf_counter()
        await runtime.bus.publish(Event(MT.UTTERANCE, data, source="backend-validator.controlled-test", ts=2))
        await runtime.bus.drain(180)
        proposals = [e.data for e in runtime.bus.find(MT.PROPOSAL) if e.data.get("utterance_id") == data["utterance_id"]]
        verdicts = [e.data for e in runtime.bus.find(MT.VERDICT) if e.data.get("utterance_id") == data["utterance_id"]]
        items = list(runtime.state.items.values())
        result = {"case": name, "input_kind": data["validation"], "roster": roster, "speaker": speaker_name,
                  "model_context": data["context"],
                  "origin": origin, "wall_ms": round((time.perf_counter() - started) * 1000, 1),
                  "proposals": [proposal_summary(p) for p in proposals],
                  "verdicts": [{k: v.get(k) for k in ("verdict", "action", "risk", "rule")} for v in verdicts],
                  "items": [{k: i.get(k) for k in ("kind", "owner", "due", "status", "preview", "external_delivery")}
                            for i in items], "measurements": phase.summary(), "reset_before": reset}
        if name == "real_asr_derived_decision":
            result["input"] = {**text_shape(text), "fixture": candidate["_fixture"],
                               "start": candidate.get("start"), "end": candidate.get("end"),
                               "unchanged_retained_public_ASR": True}
        else:
            result["synthetic_text"] = text
        invariant(result, "real_decider_request_succeeded", phase.calls["decider.ask"]["success"] > 0)
        invariant(result, "model_proposal_returned", bool(proposals))
        if name in {"known_self_commitment", "unknown_with_roster", "empty_roster"}:
            expected_owner = "Alex" if name == "known_self_commitment" else None
            invariant(result, "explicit_commitment_created_grounded_task",
                      bool(items) and all(i["kind"] == "task" and i.get("owner") == expected_owner
                                          and i.get("due") == parse_due(text) for i in items))
        if name == "empty_roster":
            invariant(result, "empty_roster_keeps_real_decider_route",
                      any(p.get("route") == "local-decider" for p in proposals))
        if name.startswith("negation") or name in {"irrelevant", "media_origin"}:
            invariant(result, "no_item_or_notification", not items and not runtime.notifier.posts and not runtime.notifier.dms)
        if name == "irrelevant":
            invariant(result, "irrelevant_none", any(p.get("mode") == "act" and p.get("action") == "none" for p in proposals))
        if name == "tentative":
            invariant(result, "tentative_never_auto_acts", not any(v.get("verdict") == "act" and v.get("action") != "none" for v in verdicts)
                      and all(i.get("status") == "pending_approval" for i in items))
        if name == "media_origin":
            invariant(result, "media_origin_policy_denial", any(v.get("verdict") == "deny" for v in verdicts))
        for p in proposals:
            if p.get("by") == "llm":
                invariant(result, "qwen_score_labeled_uncalibrated",
                          p.get("confidence_source") == "adapter_constant" and p.get("confidence_calibrated") is False)
        # No positive-action assertion for the genuine ASR-derived candidate.
        state_invariants(runtime, result)
        phase_invariants(result, phase)
        results.append(result)
        print(json.dumps({"controlled_case": name, "actions": [p["action"] for p in proposals],
                          "items": len(items), "checks_ok": all(c["ok"] for c in result["checks"])}), flush=True)
    return {"input_kind": "separate_controlled_text_only_not_video_perception",
            "cases": results, "asr_candidate_available": candidate is not None,
            "negation_qwen_path_exercised": any(p.get("by") == "llm" for r in results if r["case"].startswith("negation") for p in r["proposals"]),
            "note": "No forced escalation, threshold changes, synthetic perception, or positive ASR-action requirement."}


def manifest():
    result = {"python": sys.version.split()[0], "source_sha256": {p: digest(ROOT / p) for p in SOURCE_FILES}}
    try:
        result["git_head"] = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                                     text=True, stderr=subprocess.DEVNULL).strip()
        result["dirty_files"] = subprocess.check_output(["git", "-C", str(ROOT), "diff", "--name-only"],
                                                        text=True, stderr=subprocess.DEVNULL).splitlines()
    except (OSError, subprocess.CalledProcessError):
        result["git_head"] = "unavailable; use source hashes"
    result["packages"] = {}
    for package in ("torch", "onnxruntime", "strands-agents", "openai", "httpx", "opencv-python-headless", "av"):
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def failures(value, path="report"):
    out = []
    if isinstance(value, dict):
        if set(value) >= {"name", "ok"} and value["ok"] is False:
            out.append(path + ":" + value["name"])
        for key, item in value.items():
            out.extend(failures(item, path + "." + key))
    elif isinstance(value, list):
        for n, item in enumerate(value):
            out.extend(failures(item, f"{path}[{n}]"))
    return out


async def validate(args, report, gate, save):
    # No application/provider imports occur before the network gate is installed.
    sys.path.insert(0, str(ROOT))
    from unspoken.models import load_registry
    from unspoken.meeting.demo import DemoRuntime, ReplayController
    from unspoken.meeting.config import load_config
    from unspoken.meeting.fakes import FakeNotifier
    from unspoken.models.probes import probe_chat

    registry = load_registry(profile="gb10", overrides={
        "llm.options.api_key_file": str(args.api_key_file), "llm.options.stream": "false",
        "tts.provider": "none", "vlm.provider": "none", "decider_big.provider": "none",
    })
    if not registry.local_only:
        raise AssertionError("gb10 registry must be local-only")
    report["providers"] = {}
    for role in REQUIRED:
        spec = registry.spec(role)
        if spec.provider in {"none", "fake", "scripted", "keyword"} or not registry.enabled(role):
            raise AssertionError("A required role is not a real provider: " + role)
        report["providers"][role] = {"provider": spec.provider, "model": spec.model, "label": spec.label, "url": spec.url}
    report["disabled_roles"] = {r: not registry.enabled(r) for r in ("tts", "vlm", "decider_big")}
    recorder = Instrumentation(gate)
    root_logger = logging.getLogger()
    old_handlers = root_logger.handlers[:]
    root_logger.handlers[:] = [recorder]  # prevent raw model/prompt exception text leaking to console
    runtime = controller = None
    try:
        recorder.install(registry)
        with tempfile.TemporaryDirectory(prefix="lococisco-backend-validator-") as tmp:
            documents = Path(tmp) / "documents"
            documents.mkdir()
            cfg = load_config(args.meeting_config)
            cfg["meeting"].update(title="LocoCisco meeting replay", owner=None, agenda=[], docs_dir=str(documents))
            runtime = DemoRuntime(profile="gb10", registry=registry, config=cfg, notifier=FakeNotifier(echo=False), echo=False)
            controller = ReplayController(runtime, Path(tmp) / "uploads")
            runtime.bus.subscribe("*", recorder.on_event, name="backend-validation-aggregates")
            warm = recorder.begin("warmup")
            started = time.perf_counter()
            await runtime.start(ticking=False)
            report["readiness"] = [{**{k: v for k, v in h.items() if k != "detail"},
                                     "detail": h["detail"] if h["ok"] else error_category(h.get("detail", ""))}
                                    for h in runtime.models_health]
            # The normal one-token readiness probe is supplemented with a real full reply.
            report["qwen_direct_probe"] = await probe_chat(registry.strands_model("llm"))
            report["warmup"] = {"wall_ms": round((time.perf_counter() - started) * 1000, 1), "measurements": warm.summary()}
            invariant(report["warmup"], "all_seven_required_roles_ready", not controller.unavailable())
            invariant(report["warmup"], "qwen_answered_ok", "reply='ok'" in report["qwen_direct_probe"].lower())
            invariant(report["warmup"], "empty_documents_and_agenda", not runtime.docs.chunks and not runtime.state.agenda)
            phase_invariants(report["warmup"], warm)
            report["configuration"] = {
                "profile": "gb10", "act_threshold": runtime.settings.act_threshold,
                "ask_threshold": runtime.settings.ask_threshold, "owner_threshold": cfg["meeting"]["owner_threshold"],
                "question_version": cfg["questions"]["version"], "escalate": runtime.settings.escalate,
                "llm_stream": registry.spec("llm").options["stream"], "docs_empty": True, "agenda": [],
                "offline_flags": {"HF_HUB_OFFLINE": True, "TRANSFORMERS_OFFLINE": True, "HF_HUB_DISABLE_TELEMETRY": True},
            }
            report["version"] = manifest()
            save()
            if failures(report):
                return
            candidate = None
            report["real_media"] = []
            report["resets"] = []
            for n, path in enumerate(args.media):
                result, found = await media_phase(runtime, controller, recorder, path,
                                                   name=f"real_media:{n}:{path.name}", limit=args.media_limit_seconds,
                                                   include_examples=args.include_public_examples)
                report["real_media"].append(result)
                # Prefer the design session's candidate over introductory highlights.
                candidate = found or candidate
                report["resets"].append(await reset_checked(runtime, controller))
                save()
                if result["state"] == "error":
                    return
            replay, _ = await media_phase(runtime, controller, recorder, args.media[0],
                                           name="reset_replay", limit=args.replay_seconds,
                                           include_examples=args.include_public_examples)
            report["reset_replay"] = replay
            report["resets"].append(await reset_checked(runtime, controller))
            save()
            report["controlled_text"] = await controlled_phase(runtime, controller, recorder, candidate)
            invariant(report["controlled_text"], "retained_real_asr_decision_candidate_available", candidate is not None)
            report["resets"].append(await reset_checked(runtime, controller))
            report["version_at_finish"] = manifest()
            report["source_files_changed_during_run"] = [
                p for p, h in report["version"]["source_sha256"].items()
                if h != report["version_at_finish"]["source_sha256"][p]]
            save()
    finally:
        if controller:
            await controller.close()
        if runtime:
            await runtime.close()
        else:
            await registry.aclose()
        recorder.restore()
        root_logger.handlers[:] = old_handlers


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runtime-dir", type=Path, default=Path("/runtime"))
    parser.add_argument("--media", action="append", type=Path, help="repeatable; defaults to both installed 180 s fixtures")
    parser.add_argument("--output", type=Path, help="new JSON path outside Git (never overwrites an existing report)")
    parser.add_argument("--api-key-file", type=Path, default=Path("/run/secrets/vllm-api-key"))
    parser.add_argument("--meeting-config", type=Path, help="optional existing settings file; no threshold overrides")
    parser.add_argument("--replay-seconds", type=float, default=60)
    parser.add_argument("--media-limit-seconds", type=float, help="diagnostic partial runs; explicitly NOT complete playback")
    parser.add_argument("--include-public-examples", action="store_true",
                        help="include at most six 160-character retained on-record action excerpts per media run")
    parser.add_argument("--self-test", action="store_true", help="test socket denial and observer passthrough only; no models/files")
    args = parser.parse_args(argv)
    if args.replay_seconds <= 0 or (args.media_limit_seconds is not None and args.media_limit_seconds <= 0):
        parser.error("durations must be positive")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key] = "1"
    if args.self_test:
        with LoopbackGate() as gate:
            observer = Instrumentation(gate)

            class Passthrough:
                def call(self, value):
                    return value
            obj, value = Passthrough(), object()
            observer.wrap(obj, "call", "test")
            assert obj.call(value) is value
            assert observer.phase.calls["test"]["success"] == 1
            observer.restore()
            assert obj.call(value) is value
            assert distribution([1, 2, 3])["p95"] == 3
            print(json.dumps({"self_test": "passed", "network": gate.report()}))
        return 0
    args.media = [p.resolve() for p in (args.media or [
        args.runtime_dir / "media/ami-meeting-highlights.mp4",
        args.runtime_dir / "media/ami-design-decisions.mp4",
    ])]
    if not args.runtime_dir.is_dir() or any(not p.is_file() for p in args.media) or not args.api_key_file.is_file():
        parser.error("Existing runtime directory, local media, and credential file are required; nothing is downloaded")
    if args.meeting_config and not args.meeting_config.is_file():
        parser.error("--meeting-config must already exist")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or args.runtime_dir / "validation" / f"backend-{stamp}-{os.getpid()}.json").resolve()
    if output.is_relative_to(ROOT) or output.exists():
        parser.error("Report must be a NEW file outside the repository")
    # Inspect the immediate parent before creating the one report directory.
    if not output.parent.exists():
        if not output.parent.parent.is_dir():
            parser.error("Report directory's parent must already exist")
        output.parent.mkdir()
    report = {"schema_version": 1, "started_at": stamp, "purpose": "backend_only_no_browser_or_recording",
              "privacy": "No PCM, frames, embeddings, full real transcript, model prompts, keys, or headers are saved.",
              "timing_note": "Provider CPU wall times; Torch-forward times are dispatch wall time, not synchronized GPU timings. "
                             "ASD window-score times include output synchronization. Segment publication is not browser latency."}

    def save():
        # This process owns this newly-created path. No existing report is overwritten.
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    with output.open("x"):
        pass
    code = 1
    try:
        with LoopbackGate() as gate:
            try:
                asyncio.run(validate(args, report, gate, save))
            finally:
                report["network"] = gate.report()
                invariant(report, "no_unexpected_network_attempts", report["network"]["unexpected_denials"] == 0)
        report["failed_checks"] = failures(report)
        code = int(bool(report["failed_checks"]))
    except (Exception, KeyboardInterrupt) as exc:
        # Exception messages/tracebacks can quote credentials or model content.
        report["fatal"] = {"type": type(exc).__name__, "category": error_category(str(exc))}
        report["failed_checks"] = failures(report)
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["exit_code"] = code
        save()
        print(json.dumps({"report": str(output), "exit_code": code,
                          "failed_checks": report.get("failed_checks", []),
                          "fatal_type": (report.get("fatal") or {}).get("type")}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
