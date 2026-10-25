"""Mux an already-downloaded AMI room view and microphone track into a local demo excerpt."""
from __future__ import annotations

import argparse
import json
import wave
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


def prepare(raw_dir: Path, output: Path, start: float = 50, seconds: float = 180,
            ranges: list[tuple[float, float]] | None = None, meeting: str = "IS1009a") -> None:
    ranges = ranges or [(start, start + seconds)]
    if any(begin < 0 or end <= begin for begin, end in ranges) or sum(end - begin for begin, end in ranges) > 600:
        raise ValueError("Choose positive excerpts totaling at most ten minutes.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not meeting.isalnum():
        raise ValueError("Use an AMI meeting identifier.")
    video_path = raw_dir / f"{meeting}.L.avi"
    audio_path = raw_dir / f"{meeting}.Array1-01.wav"
    with wave.open(str(audio_path), "rb") as source:
        rate = source.getframerate()
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError("Expected the AMI mono PCM16 microphone track.")
        audio_parts = []
        for begin, end in ranges:
            source.setpos(int(begin * rate))
            audio_parts.append(np.frombuffer(source.readframes(int((end - begin) * rate)), dtype="<i2"))
        pcm = np.concatenate(audio_parts)
    peak = max(1, int(np.max(np.abs(pcm.astype(np.int32)))))
    gain = min(4.0, 0.95 * 32767 / peak)
    pcm = (pcm.astype(np.float32) * gain).astype("<i2")
    with av.open(str(video_path)) as source, av.open(str(output), "w", options={"movflags": "+faststart"}) as target:
        original = source.streams.video[0]
        fps = original.average_rate or Fraction(25)
        video = target.add_stream("libx264", rate=fps)
        video.width, video.height = original.codec_context.width, original.codec_context.height
        video.pix_fmt = "yuv420p"
        video.options = {"crf": "20", "preset": "fast"}
        audio = target.add_stream("aac", rate=rate)
        audio.layout = "mono"
        audio.bit_rate = 64000
        cursor = 0.0
        for begin, end in ranges:
            source.seek(int(begin / original.time_base), stream=original)
            for frame in source.decode(original):
                position = float(frame.pts * frame.time_base) - begin
                if position < 0:
                    continue
                if position >= end - begin:
                    break
                frame = frame.reformat(format="yuv420p")
                frame.pts = round((cursor + position) * fps)
                frame.time_base = 1 / fps
                for packet in video.encode(frame):
                    target.mux(packet)
            cursor += end - begin
        for packet in video.encode():
            target.mux(packet)
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=rate)
        for offset in range(0, len(pcm), 1024):
            frame = av.AudioFrame.from_ndarray(pcm[offset:offset + 1024].reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate, frame.pts, frame.time_base = rate, offset, Fraction(1, rate)
            for converted in resampler.resample(frame):
                for packet in audio.encode(converted):
                    target.mux(packet)
        for converted in resampler.resample(None):
            for packet in audio.encode(converted):
                target.mux(packet)
        for packet in audio.encode():
            target.mux(packet)
    metadata = {
        "title": f"AMI Meeting Corpus — {meeting}", "license": "CC BY 4.0",
        "source": f"https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{meeting}/",
        "license_url": "https://groups.inf.ed.ac.uk/ami/corpus/license.shtml",
        "modifications": "L room-view video and Array1-01 microphone muxed; edited excerpts "
                         + ", ".join(f"{begin:g}–{end:g}s" for begin, end in ranges)
                         + "; transcoded H.264/AAC, audio level normalized.",
        "source_ranges_seconds": ranges, "audio_gain": round(gain, 4),
        "attribution": f"AMI Meeting Corpus · {meeting} · CC BY 4.0 · edited excerpts, audio/video mux and level normalization",
    }
    output.with_suffix(".source.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=float, default=50)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--ranges", nargs="+", metavar="START:END")
    parser.add_argument("--meeting", default="IS1009a")
    args = parser.parse_args()
    ranges = [tuple(map(float, item.split(":"))) for item in args.ranges] if args.ranges else None
    prepare(args.raw_dir, args.output, args.start, args.seconds, ranges, args.meeting)
    print(f"Prepared {args.output.name}; attribution saved beside the local video.")
