#!/usr/bin/env bash
# Several trainings and re-scorings on one machine, one after another. Data must be built.
# Usage: scripts/run_jobs.sh <job> [<job> ...]
#   name@seed   train and score one version from run_mixed.sh, e.g. token@2
#   score@run   score a finished run in runs/<run> again, e.g. score@plain_s0
# Every job is checked before the first one starts. A failed job is written to runs/FAILED_JOBS
# and the next job still runs; at the end runs/DONE marks success, runs/STOPPED any failure.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=4 SPIDORY_DATASETS=story,maths,qa,mixed,explain,compose,wordmath
PY=${PY:-/venv/main/bin/python}
CHECK=0
[ "${1:-}" = --check ] && { CHECK=1; shift; }  # check the job list only, before the data build
[ $# -gt 0 ] || { echo "no jobs given" > runs/STOPPED; exit 1; }
for JOB in "$@"; do
  case "$JOB" in
    score@*)
      [[ "${JOB#score@}" =~ ^[A-Za-z0-9_]+$ ]] || { echo "bad run in job $JOB" > runs/STOPPED; exit 1; } ;;
    ?*@*)
      NAME=${JOB%@*}; SEED=${JOB#*@}
      [[ "$SEED" =~ ^[0-9]+$ ]] || { echo "bad seed in job $JOB" > runs/STOPPED; exit 1; }
      grep -qF -- "  \"$NAME:" scripts/run_mixed.sh || { echo "unknown version in job $JOB" > runs/STOPPED; exit 1; } ;;
    *) echo "bad job $JOB" > runs/STOPPED; exit 1 ;;
  esac
done
[ $CHECK -eq 1 ] && exit 0
failed=0
for JOB in "$@"; do
  [ -f runs/LIMIT_REACHED ] && { echo "$(date -u +%FT%TZ) time limit, not starting $JOB"; exit 1; }
  echo "$(date -u +%FT%TZ) start $JOB"
  rm -f runs/JOB_FAILED
  case "$JOB" in
    score@*)
      RUN=${JOB#score@}
      if [ ! -f "runs/$RUN/last.pt" ]; then
        echo "missing runs/$RUN/last.pt" > runs/JOB_FAILED; false
      else
        $PY -m spidory.evaluate --single "$RUN" --out "runs/rescore_$RUN.json" > "runs/rescore_$RUN.txt" 2>&1
        code=$?
        # A checkpoint uploaded from the laptop (marked UPLOADED) is a copy: removed once scored,
        # so it is not copied back. A run trained here is never removed.
        [ $code -eq 0 ] && [ -f "runs/$RUN/UPLOADED" ] && rm -rf "runs/$RUN"
        [ $code -eq 0 ]
      fi ;;
    *)
      NO_DONE=1 DROP_STEPS=1 STOP_FILE=JOB_FAILED scripts/run_mixed.sh "${JOB#*@}" "${JOB%@*}" ;;
  esac
  if [ $? -eq 0 ]; then
    echo "$(date -u +%FT%TZ) done $JOB"
  else
    failed=1
    echo "$JOB $(cat runs/JOB_FAILED 2>/dev/null)" >> runs/FAILED_JOBS
    rm -f runs/JOB_FAILED
    echo "$(date -u +%FT%TZ) FAILED $JOB"
  fi
done
if [ $failed -eq 1 ]; then
  { echo "failed jobs:"; cat runs/FAILED_JOBS; } > runs/STOPPED
else
  touch runs/DONE
fi
