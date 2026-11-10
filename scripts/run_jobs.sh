#!/usr/bin/env bash
# Several trainings and re-scorings on one machine, one after another. Data must be built.
# Usage: scripts/run_jobs.sh <job> [<job> ...]
#   name@seed   train and score one version from run_mixed.sh, e.g. token@2
#   score@run   score a finished run in runs/<run> again, e.g. score@plain_s0
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=4 SPIDORY_DATASETS=story,maths,qa,mixed,explain,compose,wordmath
PY=${PY:-/venv/main/bin/python}
fail() { echo "$1" > runs/STOPPED; exit 1; }
for JOB in "$@"; do
  echo "$(date -u +%FT%TZ) start $JOB"
  case "$JOB" in
    score@*)
      RUN=${JOB#score@}
      [ -f "runs/$RUN/last.pt" ] || fail "missing runs/$RUN/last.pt"
      $PY -m spidory.evaluate --single "$RUN" --out "runs/rescore_$RUN.json" > "runs/rescore_$RUN.txt" 2>&1 \
        || fail "$RUN scoring failed" ;;
    *@*)
      NO_DONE=1 scripts/run_mixed.sh "${JOB#*@}" "${JOB%@*}" || fail "$JOB failed" ;;
    *) fail "bad job $JOB" ;;
  esac
  echo "$(date -u +%FT%TZ) done $JOB"
done
touch runs/DONE
