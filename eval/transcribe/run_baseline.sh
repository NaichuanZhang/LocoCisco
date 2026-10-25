#!/usr/bin/env bash
# Baseline = today's live sensors pipeline on every AMI case; writes testdata/ami-baseline/<case>/{asr,diar,asd,transcript}.json
# and scores.json. Usage: eval/transcribe/run_baseline.sh [case ...]   (default: every case under testdata/ami)
set -euo pipefail
cd "$(dirname "$0")/../.."
T=../testdata/ami; B=../testdata/ami-baseline; CFG=../testdata/ami-seats/sensors.toml
cases=("$@"); [ ${#cases[@]} -gt 0 ] || cases=($(cd $T && ls -d IS* ))
run() {
  c=$1; mkdir -p $B/$c
  .venv/bin/python -m unspoken.sensors.replay --audio $T/$c/audio.wav --video $T/$c/video.mp4 --fast --local \
     --config $CFG --timeout 3600 > $B/$c/replay.jsonl 2> $B/$c/replay.err
  .venv/bin/python -m eval.transcribe.baseline_from_replay $B/$c/replay.jsonl --case $T/$c --out $B/$c > /dev/null
  .venv/bin/python -m eval.transcribe.score --ref $T/$c --hyp $B/$c/transcript.json --json 2>/dev/null > $B/$c/scores.json
}
for c in "${cases[@]}"; do run $c & done; wait
for c in "${cases[@]}"; do echo "$c $(cat $B/$c/scores.json)"; done
