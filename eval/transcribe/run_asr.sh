#!/bin/bash
# Run one ASR method on AMI cases and score it: WER (eval.transcribe.score --asr) and word-timing error
# (eval.transcribe.asr_timing). Outputs testdata/ami-runs/asr/<method>/<case>/{asr.json,scores.json,timing.json}.
#
#   eval/transcribe/run_asr.sh METHOD [CASE ...]            # default: the four -dev cases
#   THREADS=6 OPTS="--opt beam_size=1" eval/transcribe/run_asr.sh vad-fw-base.en IS1009a-duo-AC
#   TESTDATA=/path/to/testdata (default: the session's testdata dir next to the repo or the worktrees dir)
set -euo pipefail
cd "$(dirname "$0")/../.."
METHOD="${1:?method (python -m unspoken.transcribe.asr --list)}"; shift || true
if [ -z "${TESTDATA:-}" ]; then                     # <session>/testdata: next to the repo, or next to unspoken-wt/
  TESTDATA="$(cd .. && pwd)/testdata"; [ -d "$TESTDATA" ] || TESTDATA="$(cd ../.. && pwd)/testdata"
fi
THREADS="${THREADS:-6}"
PY="${PY:-.venv/bin/python}"
CASES=("$@")
[ ${#CASES[@]} -eq 0 ] && CASES=(IS1009a-duo-AC-dev IS1009a-duo-DB-dev IS1009a-L-room-dev IS1009a-R-room-dev)
export OMP_NUM_THREADS="$THREADS"
for CASE in "${CASES[@]}"; do
  OUT="$TESTDATA/ami-runs/asr/$METHOD/$CASE"
  mkdir -p "$OUT"
  echo "[$(date +%H:%M:%S)] $METHOD $CASE (threads $THREADS, $(uptime | sed 's/.*load average: //'))"
  # shellcheck disable=SC2086
  $PY -m unspoken.transcribe.asr --in "$TESTDATA/ami/$CASE/audio.wav" --out "$OUT/asr.json" --method "$METHOD" \
      --threads "$THREADS" ${OPTS:-} 2> >(grep -v Warning >&2)
  $PY -m eval.transcribe.score --ref "$TESTDATA/ami/$CASE" --asr "$OUT/asr.json" --json > "$OUT/scores.json" 2>/dev/null
  $PY -m eval.transcribe.asr_timing --ref "$TESTDATA/ami/$CASE" --asr "$OUT/asr.json" --json > "$OUT/timing.json"
  $PY - "$OUT" <<'EOF'
import json, sys
d = sys.argv[1]
s, t, a = (json.load(open(f"{d}/{n}.json")) for n in ("scores", "timing", "asr"))
tm = a["timing"]
print(f"  wer {s['wer']:.3f}  start|err| median {t['start']['median_abs']:.3f} (within 0.25 s {t['start']['within_0.25']:.1%}, "
      f"matched {t['match_rate']:.1%})  rtf {tm['rtf']:.3f}  load1 {tm['load1_start']}->{tm['load1_end']}")
EOF
done
