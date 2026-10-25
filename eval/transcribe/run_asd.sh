#!/usr/bin/env bash
# ASD runs for the per-person transcriber: writes testdata/ami-runs/asd/<method>/<case>/{asd.json,scores.json}.
#   eval/transcribe/run_asd.sh METHOD [--durations 1,2,3 ...] -- CASE [CASE ...]
# METHOD is motion | avcorr | lightasd | lightasd-talkset (python -m unspoken.transcribe.asd --method). Extra
# arguments before `--` go to the asd CLI. Cases run in parallel (each process capped at 6 threads); the
# output directory name is METHOD plus a suffix from UNSPOKEN_RUN_SUFFIX if set (e.g. lightasd-d1).
set -euo pipefail
cd "$(dirname "$0")/../.."
T=../../testdata/ami; R=../../testdata/ami-runs/asd
method=$1; shift
extra=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do extra+=("$1"); shift; done
[ $# -gt 0 ] && shift
cases=("$@"); [ ${#cases[@]} -gt 0 ] || cases=($(cd $T && ls -d IS*))
name="$method${UNSPOKEN_RUN_SUFFIX:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
export UNSPOKEN_MODELS_DIR=${UNSPOKEN_MODELS_DIR:-../../dev-models/asd}
run() {
  c=$1; d=$R/$name/$c; mkdir -p $d
  .venv/bin/python -m unspoken.transcribe.asd --in $T/$c/video.mp4 --out $d/asd.json --method $method --quiet \
      "${extra[@]}" 2> $d/run.err
  .venv/bin/python -m eval.transcribe.score_asd --ref $T/$c --asd $d/asd.json --json > $d/scores.json
}
for c in "${cases[@]}"; do run $c & done; wait
for c in "${cases[@]}"; do
  echo "$name $c $(.venv/bin/python -c "import json,sys; r=json.load(open('$R/$name/$c/scores.json')); t=r['timing']; print({k: (round(v,3) if isinstance(v,float) else v) for k,v in r.items() if k in ('map','acc','f1','which_seat_acc','talk_f1','hyp_faces')}, 'rtf', t.get('rtf'), 'threads', t.get('threads'), 'load', t.get('load1_start'), t.get('load1_end'))")"
done
