"""python -m unspoken.transcribe.live: play a recording like a camera + mic (or use the real ones), emit captions as
people talk, write transcript.json (and captions.jsonl, latency.json, faces.jsonl next to it) when it ends.

  python -m unspoken.transcribe.live --video CASE/video.mp4 [--audio CASE/audio.wav] [--speed 1.0] \
      --out OUT/transcript.json [--web 127.0.0.1:8790] [--config testdata/ami-seats/sensors.toml]
  python -m unspoken.transcribe.live --camera 0 --mic Jabra --duration 600 --out OUT/transcript.json --web 127.0.0.1:8790

--fast runs as fast as the models go (no pacing; latencies are then meaningless, accuracy is not).
Models are the in-process sensors providers (unspoken.sensors.local_registry); --set overrides a knob, e.g.
--set asr.model=Systran/faster-whisper-small.en or --set diarization.options.threshold=0.5.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "6")       # the dev box budget; before numpy / onnxruntime load

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from ...sensors.config import load_sensors_config, seat_zones  # noqa: E402
from ..contracts import to_json  # noqa: E402
from .binding import SeatBinder  # noqa: E402
from .captions import CaptionLog  # noqa: E402
from .engine import DEFAULT_SEATS, LiveSettings, LiveTranscriber, thread_budget  # noqa: E402
from .metrics import latency_report, write_json  # noqa: E402

log = logging.getLogger("unspoken.transcribe.live")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m unspoken.transcribe.live", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("source")
    src.add_argument("--video", help="video file (its audio track is used unless --audio is given)")
    src.add_argument("--audio", help="WAV with the same audio (16 kHz mono preferred)")
    src.add_argument("--camera", help="live camera: index, /dev/videoN or a name substring")
    src.add_argument("--mic", default="", help="live microphone name substring (default device when empty)")
    src.add_argument("--speed", type=float, default=1.0, help="file playback speed (1.0 = real time)")
    src.add_argument("--fast", action="store_true", help="no pacing: as fast as the models go")
    src.add_argument("--duration", type=float, default=None, help="stop after this many seconds (live sources)")
    out = ap.add_argument_group("output")
    out.add_argument("--out", required=True, help="transcript.json path")
    out.add_argument("--outdir", default=None, help="captions.jsonl / latency.json / faces.jsonl dir (default: --out's dir)")
    out.add_argument("--web", default=None, metavar="HOST:PORT", help="live viewer, loopback only, e.g. 127.0.0.1:8790")
    out.add_argument("--session-id", default=None)
    out.add_argument("--print", action="store_true", help="print captions to stderr as they are emitted")
    knobs = ap.add_argument_group("engine")
    knobs.add_argument("--config", default=None, help="sensors.toml (seat zones, video size)")
    knobs.add_argument("--profile", default=None,
                       help="config/models.toml profile via unspoken.models (gb10, dev-sensors); default: the "
                            "in-process sensors providers (unspoken.sensors.local_registry)")
    knobs.add_argument("--set", action="append", default=[], metavar="ROLE.KEY=VALUE", help="registry override")
    knobs.add_argument("--no-partials", action="store_true")
    knobs.add_argument("--partial-every", type=float, default=1.0, help="seconds of new audio per partial")
    knobs.add_argument("--asd-hz", type=float, default=10.0)
    knobs.add_argument("--view-fps", type=float, default=12.5, help="frames taken from the video per second")
    knobs.add_argument("--max-segment-s", type=float, default=8.0, help="VAD: cut a long utterance here")
    knobs.add_argument("--min-silence-ms", type=int, default=400, help="VAD: pause that closes a segment")
    knobs.add_argument("--asr-threads", type=int, default=4)
    knobs.add_argument("--diar-threshold", type=float, default=None, help="speaker model: new-voice threshold")
    knobs.add_argument("--merge-sim", type=float, default=0.30,
                       help="a second voice joins an occupied seat only if its centroid cosine to the seat's voice is "
                            ">= this; > 1 means one voice per seat (no merging), 0 means always merge")
    knobs.add_argument("--no-diar", action="store_true")
    knobs.add_argument("--no-video", action="store_true", help="audio only (no faces / ASD)")
    knobs.add_argument("--timeout", type=float, default=7200.0)
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def make_registry(args: argparse.Namespace) -> Any:
    overrides: dict[str, Any] = {"vad.options.max_segment_s": args.max_segment_s,
                                 "vad.options.min_silence_ms": args.min_silence_ms,
                                 "asr.options.threads": args.asr_threads}
    if args.diar_threshold is not None:
        overrides["diarization.options.threshold"] = args.diar_threshold
    for kv in args.set:
        k, _, v = kv.partition("=")
        overrides[k] = _coerce(v)
    disabled = set()
    if args.no_diar:
        disabled.add("diarization")
    if args.no_video:
        disabled |= {"faces", "asd"}
    if args.profile:
        from ...models import load_registry          # the real registry: GB10 speech server etc.

        for role in disabled:
            overrides[f"{role}.provider"] = "none"
        return load_registry(args.profile, overrides={k: str(v) for k, v in overrides.items()})
    from ...sensors.local_registry import SensorsRegistry

    return SensorsRegistry(overrides=overrides, disabled=disabled)


def _coerce(v: str) -> Any:
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


def _role(registry: Any, role: str, getter: str) -> Any | None:
    try:
        if not registry.enabled(role):
            return None
        return getattr(registry, getter)()
    except Exception as exc:
        log.warning("%s unavailable: %s", role, exc)
        return None


async def run(args: argparse.Namespace, registry: Any = None) -> int:
    if not args.video and args.camera is None and not args.audio:
        print("need --video (or --audio), or --camera/--mic", file=sys.stderr)
        return 2
    config = load_sensors_config(args.config)
    zones = seat_zones(config) or dict(DEFAULT_SEATS)
    width, height = int(config["video"].get("width", 0) or 0), int(config["video"].get("height", 0) or 0)
    registry = registry or make_registry(args)
    try:
        import cv2
        cv2.setNumThreads(1)        # decode/resize on one thread: the budget is 6 for the whole process
    except Exception:
        pass
    out_path = Path(args.out)
    outdir = Path(args.outdir) if args.outdir else out_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    settings = LiveSettings(speed=args.speed, realtime=not args.fast, partials=not args.no_partials,
                            partial_every_s=args.partial_every, asd_hz=args.asd_hz, seats=zones,
                            speaking_threshold=float(config["speakers"].get("speaking_threshold", 0.5)))
    base = time.monotonic()
    clock = lambda: time.monotonic() - base  # noqa: E731

    viewer = None
    if args.web:
        from .web import LiveViewer

        host, _, port = args.web.rpartition(":")
        viewer = LiveViewer(host or "127.0.0.1", int(port or 8790), speaking_threshold=settings.speaking_threshold).start()
        print(f"viewer: {viewer.url}", file=sys.stderr)

    def on_caption(cap) -> None:
        if viewer is not None:
            viewer.on_caption(cap)
        if args.print:
            kind = ("FINAL" + (f" r{cap.revision}" if cap.revision else "")) if cap.final else "partial"
            lat = f" +{cap.latency:.2f}s" if cap.latency is not None else ""
            print(f"[{cap.t_emit:7.2f}] {kind:9s} {cap.start:7.2f}-{cap.end:7.2f} {cap.speaker:13s}{lat} {cap.text}",
                  file=sys.stderr)

    engine = LiveTranscriber(vad=_role(registry, "vad", "vad"), asr=_role(registry, "asr", "asr"),
                             diar=_role(registry, "diarization", "diarizer"),
                             faces=None if args.no_video else _role(registry, "faces", "faces"),
                             asd=None if args.no_video else _role(registry, "asd", "asd"),
                             settings=settings, clock=clock, log=CaptionLog(outdir / "captions.jsonl"),
                             faces_log=outdir / "faces.jsonl", on_caption=on_caption,
                             on_frame=viewer.on_frame if viewer else None, binder=SeatBinder(merge_sim=args.merge_sim))
    if engine.vad is None or engine.asr is None:
        print("vad and asr are required", file=sys.stderr)
        return 2
    print(f"models: {engine.models}", file=sys.stderr)
    await engine.warm()

    # -------------------------------------------------------------- sources
    from ...sensors.audio import MicSource, WavFileSource
    from ...sensors.video import CameraSource, Mp4FileSource

    video_src = audio_src = None
    if args.camera is not None or (not args.video and not args.audio):
        audio_src = MicSource(clock=clock, device=args.mic)
        if args.camera is not None and engine.faces is not None:
            video_src = CameraSource(clock=clock, device=args.camera, fps=args.view_fps, width=width or 640,
                                     height=height or 360)
    else:
        has_video = bool(args.video) and engine.faces is not None
        # fast replay of two files: audio at most 1 s ahead of video, video never ahead of audio, and either
        # side runs free once the other has finished (else the longer file waits forever)
        audio_pace = (lambda: float("inf") if engine.video_done.is_set() else engine.video_media_time + 1.0) \
            if has_video else None
        video_pace = lambda: float("inf") if engine.audio_done.is_set() else engine.audio_media_time  # noqa: E731
        audio_src = WavFileSource(args.audio or args.video, clock=clock, realtime=not args.fast, speed=args.speed,
                                  t0=0.0, pace=audio_pace if args.fast else None)
        if has_video:
            video_src = Mp4FileSource(args.video, clock=clock, realtime=not args.fast, fps=args.view_fps,
                                      width=width, height=height, t0=0.0, speed=args.speed,
                                      pace=video_pace if args.fast else None)
    if args.duration:
        async def stopper() -> None:
            await asyncio.sleep(args.duration / max(args.speed, 0.1))
            for s in (audio_src, video_src):
                if s is not None:
                    await s.aclose()
        asyncio.get_running_loop().create_task(stopper())

    load0 = os.getloadavg()
    t0 = time.perf_counter()
    try:
        await asyncio.wait_for(engine.run(audio_src, video_src), args.timeout)
    except asyncio.TimeoutError:
        print("timeout: the run did not finish", file=sys.stderr)
    finally:
        if viewer is not None:
            viewer.stop()
    load1 = os.getloadavg()
    session = args.session_id or Path(args.video or args.audio or "live").resolve().parent.name
    duration = engine.audio_media_time
    tr = engine.transcript(session, duration, {"source": {"video": args.video, "audio": args.audio,
                                                           "speed": args.speed, "realtime": not args.fast}})
    to_json(tr, out_path)
    stats = engine.stats.summary()
    report = latency_report(engine.log.captions, stats,
                            threads={**thread_budget(), "asr_threads": args.asr_threads, "vad": 1, "speaker": 1,
                                     "faces": 1}, load=load1,
                            extra={"load_avg_start": list(load0), "models": engine.models, "speed": args.speed,
                                   "realtime": not args.fast, "wall_s": round(time.perf_counter() - t0, 2),
                                   "bindings": dict(engine.binder.bound), "session_id": session})
    write_json(report, outdir / "latency.json")
    print(json.dumps({"finals": stats["finals"], "partials": stats["partials"], "revisions": stats["revisions"],
                      "final_latency_p50": report["final_latency_s"]["p50"],
                      "final_latency_p95": report["final_latency_s"]["p95"], "rtf": report["rtf"],
                      "cpu_rtf": report["cpu_rtf"], "frames": report["frames"], "bindings": report["bindings"],
                      "out": str(out_path)}), file=sys.stderr)
    aclose = getattr(registry, "aclose", None)
    if aclose:
        await aclose()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
