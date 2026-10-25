"""Build per-person transcription test cases from the AMI Meeting Corpus (CC BY 4.0).

AMI: https://groups.inf.ed.ac.uk/ami/corpus/ (licence: https://groups.inf.ed.ac.uk/ami/corpus/license.shtml).
Room-view cameras L and R each show two participants side by side, which is our "camera on a table" case.

Two kinds of case:
  duo   two close-up cameras side by side (person on the left half, person on the right half; faces ~100 px,
        like a 1080p webcam) with a clean two-person mix: each person's headset, gated to that person's own
        reference speech (+-0.3 s, 50 ms fades) so the other two participants' bleed is removed. This is
        "a video of two people talking".
  room  one wide room-view camera (L or R: two people side by side, faces ~30 px) with the far-field table mic
        (Array1-01): all four voices, two of them off camera. This is the hard, realistic case.

For each case this writes, under OUT/<case>/ :
  video.mp4        the camera view with an audio track (the transcriber's only input)
  audio.wav        the same audio, 16 kHz mono PCM (for audio-only experiments)
  ref.words.json   every reference word: speaker, seat, on_camera, start, end, text
  ref.seglst.json  reference segments (meeteval SegLST shape: session_id, speaker, start_time, end_time, words)
  ref.rttm         reference speaker turns (for DER)
  meta.json        speakers -> seat/on_camera/role, durations, sources, licence

Cases: <meeting>-duo-<left><right>, <meeting>-<view>-room, and a dev excerpt of each (`-dev`, 180 s from
DEV_START) for fast iteration.

Usage: python eval/transcribe/ami_prepare.py --out DIR [--meeting IS1009a] [--raw DIR]
Needs ffmpeg on PATH. Downloads ~430 MB per meeting into --raw (default OUT/raw).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import urllib.request
import wave
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

MIRROR = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
ANNOTATIONS = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
NITE = "{http://nite.sourceforge.net/}"

# Who sits where in each room-view camera, read by eye from the close-up cameras (Closeup<n> = agent).
SEATS = {
    "IS1009a": {"L": {"left": "A", "right": "C"}, "R": {"left": "D", "right": "B"}},
}
# Close-up camera number per agent (meetings.xml `camera="Closeup<n>"`).
GATE_PAD, GATE_FADE = 0.3, 0.05
DEV_START, DEV_LEN = 50.0, 180.0


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print("download", url)
        urllib.request.urlretrieve(url, dest)
    return dest


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000 and w.getsampwidth() == 2, path
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
        return x.reshape(-1, w.getnchannels()).mean(axis=1)


def write_wav(path: Path, x: np.ndarray) -> None:
    peak = float(np.max(np.abs(x))) or 1.0
    y = np.clip(x * (0.9 * 32767 / peak), -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes(y.tobytes())


def channels(ann: Path, meeting: str) -> dict[str, dict]:
    root = ET.parse(ann / "corpusResources" / "meetings.xml").getroot()
    m = next(e for e in root.iter("meeting") if e.get("observation") == meeting)
    return {s.get("nxt_agent"): {"channel": int(s.get("channel")), "role": s.get("role"),
                                 "global_name": s.get("global_name")} for s in m.iter("speaker")}


def words(ann: Path, meeting: str, agent: str) -> list[dict]:
    root = ET.parse(ann / "words" / f"{meeting}.{agent}.words.xml").getroot()
    out = []
    for w in root.iter("w"):
        if w.get("punc") == "true" or w.get("starttime") is None or not (w.text or "").strip():
            continue
        out.append({"id": w.get(f"{NITE}id"), "start": float(w.get("starttime")),
                    "end": float(w.get("endtime")), "text": w.text.strip()})
    return out


def segments(ann: Path, meeting: str, agent: str, ws: list[dict]) -> list[dict]:
    root = ET.parse(ann / "segments" / f"{meeting}.{agent}.segments.xml").getroot()
    segs = []
    for s in root.iter("segment"):
        child = s.find(f"{NITE}child")
        m = re.search(r"id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?", child.get("href") if child is not None else "")
        if not m:
            continue
        a, b = m.group(1), m.group(2) or m.group(1)
        # punctuation ids are dropped from `ws`; take every kept word whose numeric id lies in [a, b]
        lo, hi = int(a.rsplit("words", 1)[1]), int(b.rsplit("words", 1)[1])
        seg_words = [w for w in ws if lo <= int(w["id"].rsplit("words", 1)[1]) <= hi]
        if seg_words:
            segs.append({"start": seg_words[0]["start"], "end": max(w["end"] for w in seg_words),
                         "words": seg_words})
    return segs


def gate(x: np.ndarray, segs: list[dict]) -> np.ndarray:
    """Keep x only inside the speaker's own reference segments (+-GATE_PAD), with GATE_FADE ramps."""
    g = np.zeros(len(x), dtype=np.float32)
    for s in segs:
        g[max(0, int((s["start"] - GATE_PAD) * 16000)):int((s["end"] + GATE_PAD) * 16000)] = 1.0
    k = int(GATE_FADE * 16000)
    ramp = np.convolve(g, np.ones(k, dtype=np.float32) / k, mode="same")
    return x * ramp


def clip(items: list[dict], t0: float, t1: float) -> list[dict]:
    out = []
    for it in items:
        if it["start"] >= t0 and it["end"] <= t1:
            out.append({**it, "start": round(it["start"] - t0, 3), "end": round(it["end"] - t0, 3)})
    return out


def build_case(out: Path, case: str, video: Path | list[Path], audio: np.ndarray, speakers: dict[str, dict],
               refs: dict[str, tuple[list[dict], list[dict]]], t0: float, t1: float, sources: dict) -> None:
    d = out / case
    d.mkdir(parents=True, exist_ok=True)
    write_wav(d / "audio.wav", audio[int(t0 * 16000):int(t1 * 16000)])
    vids = video if isinstance(video, list) else [video]
    vin = [a for v in vids for a in ("-ss", f"{t0}", "-t", f"{t1 - t0}", "-i", str(v))]
    vmap = (["-filter_complex", f"{''.join(f'[{i}:v]' for i in range(len(vids)))}hstack=inputs={len(vids)}[v]",
             "-map", "[v]"] if len(vids) > 1 else ["-map", "0:v:0"])
    subprocess.run(["ffmpeg", "-v", "error", "-y", *vin, "-i", str(d / "audio.wav"), *vmap,
                    "-map", f"{len(vids)}:a:0", "-c:v", "mpeg4",
                    "-q:v", "2", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
                    "-shortest", str(d / "video.mp4")], check=True)
    all_words, seglst, rttm = [], [], []
    for agent, (ws, segs) in refs.items():
        sp = speakers[agent]
        for w in clip(ws, t0, t1):
            all_words.append({"speaker": agent, "seat": sp["seat"], "on_camera": sp["on_camera"],
                              "start": w["start"], "end": w["end"], "text": w["text"]})
        for s in segs:
            if s["start"] >= t0 and s["end"] <= t1:
                st, en = round(s["start"] - t0, 3), round(s["end"] - t0, 3)
                seglst.append({"session_id": case, "speaker": agent, "start_time": st, "end_time": en,
                               "words": " ".join(w["text"] for w in s["words"])})
                rttm.append(f"SPEAKER {case} 1 {st:.3f} {en - st:.3f} <NA> <NA> {agent} <NA> <NA>")
    all_words.sort(key=lambda w: (w["start"], w["end"]))
    seglst.sort(key=lambda s: s["start_time"])
    (d / "ref.words.json").write_text(json.dumps(all_words, indent=1))
    (d / "ref.seglst.json").write_text(json.dumps(seglst, indent=1))
    (d / "ref.rttm").write_text("\n".join(sorted(rttm, key=lambda r: float(r.split()[3]))) + "\n")
    meta = {"case": case, "kind": sources.get("kind"), "start_offset_in_meeting": t0, "duration": round(t1 - t0, 3),
            "speakers": {a: speakers[a] for a in refs},
            "word_counts": {a: sum(1 for w in all_words if w["speaker"] == a) for a in refs},
            "sources": sources,
            "licence": "AMI Meeting Corpus, CC BY 4.0, https://groups.inf.ed.ac.uk/ami/corpus/license.shtml"}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    print(case, meta["duration"], "s", meta["word_counts"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--meeting", default="IS1009a", choices=sorted(SEATS))
    ap.add_argument("--raw", type=Path)
    a = ap.parse_args()
    mtg, raw = a.meeting, (a.raw or a.out / "raw")
    zp = fetch(ANNOTATIONS, raw / "ami_public_manual_1.6.2.zip")
    ann = raw / "ann"
    if not (ann / "words" / f"{mtg}.A.words.xml").exists():
        with zipfile.ZipFile(zp) as z:
            z.extractall(ann, [n for n in z.namelist() if n.startswith((f"words/{mtg}.", f"segments/{mtg}.",
                                                                        "corpusResources/"))])
    ch = channels(ann, mtg)
    refs = {}
    for agent in ch:
        ws = words(ann, mtg, agent)
        refs[agent] = (ws, segments(ann, mtg, agent, ws))
    heads = {ag: read_wav(fetch(f"{MIRROR}/{mtg}/audio/{mtg}.Headset-{c['channel']}.wav",
                                raw / f"{mtg}.Headset-{c['channel']}.wav")) for ag, c in ch.items()}
    array = read_wav(fetch(f"{MIRROR}/{mtg}/audio/{mtg}.Array1-01.wav", raw / f"{mtg}.Array1-01.wav"))
    for view, seats in SEATS[mtg].items():
        seat_of = {ag: s for s, ag in seats.items()}
        speakers = {ag: {"seat": seat_of.get(ag), "on_camera": ag in seat_of, **ch[ag]} for ag in ch}
        on = [seats["left"], seats["right"]]
        # duo: close-ups of the two side by side, clean gated two-person mix
        closeups = [fetch(f"{MIRROR}/{mtg}/video/{mtg}.Closeup{ch[ag]['channel'] + 1}.avi",
                          raw / f"{mtg}.Closeup{ch[ag]['channel'] + 1}.avi") for ag in on]
        n = min(len(heads[ag]) for ag in on)
        duo = sum(gate(heads[ag][:n], refs[ag][1]) for ag in on)
        name = f"{mtg}-duo-{''.join(on)}"
        src = {"kind": "duo",
               "video": [f"{MIRROR}/{mtg}/video/{p.name}" for p in closeups],
               "audio": [f"{MIRROR}/{mtg}/audio/{mtg}.Headset-{ch[ag]['channel']}.wav" for ag in on],
               "audio_processing": f"each headset gated to its own speaker's reference segments +-{GATE_PAD}s",
               "annotations": ANNOTATIONS}
        sub = {ag: refs[ag] for ag in on}
        dur = min(_duration(p) for p in closeups)
        end = min(dur, n / 16000)
        build_case(a.out, name, closeups, duo, speakers, sub, 0.0, end, src)
        build_case(a.out, f"{name}-dev", closeups, duo, speakers, sub, DEV_START, DEV_START + DEV_LEN, src)
        # room: wide view, far-field mic, everyone
        video = fetch(f"{MIRROR}/{mtg}/video/{mtg}.{view}.avi", raw / f"{mtg}.{view}.avi")
        src = {"kind": "room", "video": f"{MIRROR}/{mtg}/video/{mtg}.{view}.avi",
               "audio": [f"{MIRROR}/{mtg}/audio/{mtg}.Array1-01.wav"], "annotations": ANNOTATIONS}
        end = min(_duration(video), len(array) / 16000)
        build_case(a.out, f"{mtg}-{view}-room", video, array, speakers, refs, 0.0, end, src)
        build_case(a.out, f"{mtg}-{view}-room-dev", video, array, speakers, refs, DEV_START, DEV_START + DEV_LEN, src)


def _duration(path: Path) -> float:
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                 str(path)], capture_output=True, text=True, check=True).stdout)


if __name__ == "__main__":
    main()
