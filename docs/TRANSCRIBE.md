# Per-person transcriber: real test and the stack it picked

Video in, "who said what, when" out. Measured on real meetings against human transcripts, 2026-10-24/25.
Reports per stage: `docs/reports/tx-{asr,diar,asd,vlm,fuse,live}.md`.

## Run it

```bash
# offline: one video -> transcript.json / .txt / .srt / .md (+ scores.json with --ref)
python -m unspoken.transcribe --video meeting.mp4 --out out/ [--profile cpu|gb10] [--ref CASE_DIR]
# live: captions as people talk, viewer on loopback
python -m unspoken.transcribe.live --video meeting.mp4 --profile dev-best --view-fps 25 --out out/transcript.json --web 127.0.0.1:8798
python -m unspoken.transcribe.live --camera 0 --mic Jabra --profile gb10 --view-fps 25 --out out/transcript.json   # the room
```

## Test data

AMI Meeting Corpus, meeting IS1009a (CC BY 4.0), built by `eval/transcribe/ami_prepare.py`:

| Case | Video | Audio | People |
|---|---|---|---|
| duo-AC, duo-DB | two close-up cameras side by side | each person's headset, gated to their own speech | 2, on camera |
| L-room, R-room | one wide room camera (faces ~25 px) | far-field table mic | 4, 2 on camera |

Each ~830 s, plus 180 s `-dev` excerpts used for all tuning. Scorer: `eval/transcribe/score.py`
(WER, cpWER = speaker-attributed WER, DER, seat accuracy = words given to the right face, no mapping help).

## The stack

| Stage | Winner | Runs on |
|---|---|---|
| Words | Parakeet TDT 0.6B v3 (fp32 ONNX here; NeMo-Speech.cpp Q8 on the GB10) | CPU here, GPU on GB10 |
| Voices | WeSpeaker CAM++ on 1.5 s windows + spectral clustering (offline) / causal labels (live) | CPU |
| Who is talking on camera | YuNet faces + Light-ASD (AVA weights, MIT) | CPU here, GPU on GB10 |
| Fuse | `bind`: words to voice turns, voices to seats on whole-meeting evidence | CPU, < 1 s |

Not picked, on the numbers: Whisper large-v3-turbo (same WER, 3-10x the CPU), Parakeet int8 export (drops whole
sentences), Sortformer and pyannote clustering (higher DER), MotionASD (at chance), the Strands Decider and a
local LLM as speaker fixers (changed 0-1 lines), a small VLM judge (good on big faces, blind at 25 px, never
says "nobody on camera"; on the GB10 Qwen3.6 can be a tie-breaker).

## Results (full cases, run once)

| Case | cpWER before -> after | Seat accuracy before -> after |
|---|---|---|
| duo-AC | 0.321 -> 0.184 | 0.857 -> 0.871 |
| duo-DB | 0.247 -> 0.156 | 0.033 -> 0.892 |
| L-room | 0.497 -> 0.287 | 0.808 -> 0.819 |
| R-room | 0.488 -> 0.287 | 0.517 -> 0.697 |

"Before" = the live sensors pipeline as it was (`testdata/ami-baseline`). The cpWER floor is overlapping
speech: 15 % of duo-AC words and 30 % of room words overlap another speaker, which one ASR stream cannot write.
R-room is the hardest: the two people on camera say 15 % of the words; the loudest voice is off camera.

One-command pipeline on the dev excerpts: duo-AC-dev cpWER 0.076, seat 0.980; L-room-dev cpWER 0.111, seat 0.952.

Live, real time, duo-AC-dev (`--profile dev-best`, this x86 CPU): cpWER 0.072, seat 0.972, final caption
0.51 s (p50) / 0.77 s (p95) after the VAD segment end (speech end + 0.2 s pad), RTF 1.0, ~3.6 cores. Demo: `unspoken.transcribe.live.render`
burns the captions in at the moment they were emitted.

## On the GB10

`--profile gb10` uses the speech server (:8770) for words and Light-ASD on CUDA (torch 2.14.1+cu130 from the
USB, `requirements/gpu/transcribe-asd-torch.txt`). Voices and faces stay on the CPU. First check on the box:
run the ASR scorer against the speech server on one `-dev` case; the Q8 GGUF must not drop windows the way the
int8 ONNX export did.
