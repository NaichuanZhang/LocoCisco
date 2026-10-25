"""Baseline stage outputs from the existing live sensors pipeline (unspoken.sensors.replay JSON lines).

The live pipeline is: Silero VAD -> faster-whisper base.en per VAD segment, CAM++ online voice label per
segment, YuNet faces + MotionASD at 5 Hz. This converts its bus events into the transcribe contracts so every
later method is compared against "what the room brain does today":

  asr.json         one AsrSegment per audio.segment (no word timings: the live event does not carry them)
  diar.json        one VoiceTurn per audio.segment (the segment's voice label)
  asd.json         every video.speakers event as an AsdFrame; faces summarised per track (seat = majority)
  transcript.json  naive fusion: speaker = the segment's voice, bound to the seat that spoke most during it

Usage:
  python -m unspoken.sensors.replay --audio CASE/audio.wav --video CASE/video.mp4 --fast --local \
      --config testdata/ami-seats/sensors.toml > replay.jsonl
  python -m eval.transcribe.baseline_from_replay replay.jsonl --case CASE --out OUTDIR
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from unspoken.transcribe.contracts import (AsdFrame, AsdResult, AsrResult, AsrSegment, DiarResult, FaceInfo,
                                           SpeakerInfo, Transcript, Utterance, VoiceTurn, to_json)


def convert(lines: list[dict], session: str, duration: float) -> tuple[AsrResult, DiarResult, AsdResult, Transcript]:
    segs = [e["data"] for e in lines if e["topic"] == "audio.segment"]
    spk = [e["data"] for e in lines if e["topic"] == "video.speakers"]
    asr = AsrResult("faster-whisper base.en (CPU int8), live VAD segments",
                    [AsrSegment(s["start"], s["end"], s["text"], [], s.get("asr_conf", 1.0)) for s in segs])
    diar = DiarResult("WeSpeaker CAM++ online clustering, per VAD segment",
                      [VoiceTurn(s["start"], s["end"], s.get("voice") or "voice-?", s.get("voice_conf", 0.0))
                       for s in segs])
    seats: dict[str, Counter] = defaultdict(Counter)
    boxes: dict[str, list] = defaultdict(list)
    frames = []
    for f in spk:
        sc = {}
        for face in f["faces"]:
            tid = face["track_id"]
            seats[tid][face.get("seat")] += 1
            boxes[tid].append(face["bbox"])
            sc[tid] = face["speaking"]
        frames.append(AsdFrame(f["t"], sc))
    faces = [FaceInfo(t, seats[t].most_common(1)[0][0],
                      tuple(round(sum(b[i] for b in bs) / len(bs), 4) for i in range(4)), len(bs))
             for t, bs in boxes.items()]
    asd = AsdResult("YuNet + MotionASD (heuristic), 5 Hz; boxes in 0..1 of the frame", 5.0, faces, frames)
    seat_of = {f.track_id: f.seat for f in faces}
    # voice -> seat by co-occurrence: sum of ASD probability per seat over the voice's segments
    votes: dict[str, Counter] = defaultdict(Counter)
    for s in segs:
        for fr in frames:
            if s["start"] <= fr.t <= s["end"]:
                for tid, p in fr.scores.items():
                    if seat_of.get(tid):
                        votes[s.get("voice") or "voice-?"][seat_of[tid]] += p
    taken: dict[str, str] = {}
    for voice, c in sorted(votes.items(), key=lambda kv: -sum(kv[1].values())):
        for seat, _ in c.most_common():
            if seat not in taken.values():
                taken[voice] = seat
                break
    speakers, utts = {}, []
    for s in segs:
        v = s.get("voice") or "voice-?"
        seat = taken.get(v)
        key = f"person-{seat}" if seat else v
        speakers.setdefault(key, SpeakerInfo(f"{seat.title()} person" if seat else f"Off-camera {v[-1]}",
                                             seat, bool(seat), v, None))
        utts.append(Utterance(s["start"], s["end"], key, s["text"], [], s.get("voice_conf", 0.0), "audio"))
    tr = Transcript(session, duration, speakers, utts,
                    {"method": "baseline: live sensors pipeline, speaker = segment voice, voice->seat by ASD votes",
                     "models": {"asr": asr.model, "diar": diar.model, "asd": asd.model}})
    return asr, diar, asd, tr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("replay", type=Path)
    ap.add_argument("--case", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    meta = json.loads((a.case / "meta.json").read_text())
    lines = [json.loads(x) for x in a.replay.read_text().splitlines() if x.strip()]
    a.out.mkdir(parents=True, exist_ok=True)
    for name, obj in zip(("asr", "diar", "asd", "transcript"), convert(lines, meta["case"], meta["duration"])):
        to_json(obj, a.out / f"{name}.json")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
