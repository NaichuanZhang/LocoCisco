"""Build the 3-voice meeting fixture (meeting3.wav + meeting3.json) with Piper TTS on the CPU.

Dev-only: the fixture is committed, so tests never need Piper. Re-run only to change the script.

    .venv/bin/python -m pip install piper-tts==1.8.0
    .venv/bin/python -m piper.download_voices --data-dir "$VOICES" en_US-lessac-medium en_US-ryan-medium en_GB-alba-medium
    .venv/bin/python tests/sensors/fixtures/make_meeting_wav.py --voices "$VOICES"

Voices (Piper, https://github.com/OHF-Voice/piper1-gpl; voice files from https://huggingface.co/rhasspy/piper-voices):
  en_US-lessac-medium -> "Priya"   en_US-ryan-medium -> "Sam"   en_GB-alba-medium -> "Lee"
The JSON lists every line with its speaker and the [start, end] seconds it occupies in the WAV, so tests can
score VAD segmentation, ASR word accuracy and diarization purity against ground truth.
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SR = 16000
GAP_S = 0.9          # silence between lines; longer than the VAD's min silence so each line is one segment
LEAD_S = 0.5
SPEAKERS = {"Priya": "en_US-lessac-medium", "Sam": "en_US-ryan-medium", "Lee": "en_GB-alba-medium"}
LINES: list[tuple[str, str]] = [
    ("Priya", "Hi, I'm Priya. Let's get started with the pricing sync."),
    ("Sam", "Hi, I'm Sam. I have the numbers from last quarter."),
    ("Lee", "Hi, I'm Lee. I'm joining from the design team."),
    ("Priya", "Sam, send the deck by Friday."),
    ("Sam", "Sure, I will send the deck by Friday."),
    ("Lee", "Should we keep the launch date in March?"),
    ("Priya", "Let's decide. We keep the March launch."),
    ("Sam", "One question. Who owns the pricing page?"),
    ("Lee", "I can own the pricing page."),
    ("Priya", "Great. Lee owns the pricing page."),
    ("Sam", "Can we turn the lamp down a bit?"),
    ("Priya", "Off the record for a minute."),
]


def synth(voice, text: str) -> np.ndarray:
    """One line as float32 mono at 16 kHz."""
    from piper import SynthesisConfig

    chunks = list(voice.synthesize(text, SynthesisConfig(normalize_audio=True, volume=0.9)))
    pcm = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
    rate = chunks[0].sample_rate
    if rate != SR:
        # Linear interpolation is fine for a TTS fixture (TTS output has little energy above 7 kHz).
        n = int(round(len(pcm) * SR / rate))
        pcm = np.interp(np.linspace(0, len(pcm) - 1, n), np.arange(len(pcm)), pcm).astype(np.float32)
    # trim leading/trailing near-silence so the ground-truth times are tight
    idx = np.nonzero(np.abs(pcm) > 0.01)[0]
    if len(idx):
        pcm = pcm[max(0, idx[0] - 400): idx[-1] + 400]
    return pcm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", required=True, help="directory holding <voice>.onnx(.json) files")
    ap.add_argument("--out", default=str(HERE / "meeting3"), help="output stem (.wav and .json are added)")
    args = ap.parse_args()
    from piper import PiperVoice

    voices = {name: PiperVoice.load(str(Path(args.voices) / f"{v}.onnx")) for name, v in SPEAKERS.items()}
    parts: list[np.ndarray] = [np.zeros(int(LEAD_S * SR), np.float32)]
    script: list[dict] = []
    t = LEAD_S
    for who, text in LINES:
        pcm = synth(voices[who], text)
        dur = len(pcm) / SR
        script.append({"speaker": who, "voice": SPEAKERS[who], "text": text,
                       "start": round(t, 3), "end": round(t + dur, 3)})
        parts += [pcm, np.zeros(int(GAP_S * SR), np.float32)]
        t += dur + GAP_S
    audio = np.concatenate(parts)
    pcm16 = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(args.out + ".wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm16.tobytes())
    meta = {"sample_rate": SR, "duration": round(len(audio) / SR, 3), "speakers": SPEAKERS,
            "tts": "piper-tts 1.8.0 (https://github.com/OHF-Voice/piper1-gpl), voices from rhasspy/piper-voices",
            "lines": script}
    Path(args.out + ".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {args.out}.wav ({len(pcm16) * 2 / 1e6:.2f} MB, {meta['duration']} s) and {args.out}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
