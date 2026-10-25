"""Real-time per-person transcriber: `python -m unspoken.transcribe.live` (see __main__.py).

engine.py    LiveTranscriber: VAD -> partial/final ASR, online voice labels, ASD, revisable voice->seat binding
binding.py   SeatBinder: duration-weighted ASD evidence per voice; several voices may merge into one person
captions.py  Caption records, captions.jsonl, the final Transcript (unspoken.transcribe.contracts)
metrics.py   latency.json: caption latency p50/p95, partial staleness, RTF, CPU, frames
web.py       the loopback-only live viewer (MJPEG + SSE)
render.py    burn logged captions into a demo mp4 at their real t_emit; contact sheet
bench.py     run the AMI cases at 1x, score them, print the results table
"""
from .binding import SeatBinder
from .captions import Caption, CaptionLog, build_transcript, read_captions
from .engine import LiveSettings, LiveStats, LiveTranscriber
from .metrics import latency_report

__all__ = ["LiveTranscriber", "LiveSettings", "LiveStats", "SeatBinder", "Caption", "CaptionLog",
           "build_transcript", "read_captions", "latency_report"]
