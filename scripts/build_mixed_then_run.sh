#!/usr/bin/env bash
# On a GPU machine with data v3 in data/: grow to v4, clean to v5, add the mixed-topic and explained
# maths sets, then run the models given. Each step is skipped when its marker file exists.
# Usage: scripts/build_mixed_then_run.sh <seed> <name,name,...>
#    or: scripts/build_mixed_then_run.sh jobs <job> [<job> ...]   (see run_jobs.sh)
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true
PY=${PY:-/venv/main/bin/python}
mkdir -p runs
stop() { echo "$1" > runs/STOPPED; exit 1; }
# A job list is checked before the long data build, so a typo costs nothing.
if [ "${1:-}" = jobs ]; then
  shift; scripts/run_jobs.sh --check "$@" || exit 1; set -- jobs "$@"
fi
[ -f data/V4_DONE ] || $PY -m spidory.grow_v4 --root data > runs/grow_v4.log 2>&1 || stop "data growth failed"
[ -f data/V5_DONE ] || $PY -m spidory.clean_v5 --root data > runs/clean_v5.log 2>&1 || stop "data cleaning failed"
if [ ! -f data/MIXED_DONE ]; then
  $PY -m spidory.mixed_data --root data --templates templates > runs/mixed_data.log 2>&1 || stop "mixed data failed"
  cp data/mixed_stats.json data/MIXED_DONE
fi
# A fingerprint of the built data, so machines and seeds can be checked to share it.
(cd data && sha256sum *exam*.json *stats*.json *.test.bin *.val.bin) > runs/data_sha.txt 2>/dev/null
if [ "${1:-}" = jobs ]; then
  shift; scripts/run_jobs.sh "$@" > runs/run_jobs.log 2>&1
else
  scripts/run_mixed.sh "${1:-0}" "${2:-all}" > runs/run_mixed.log 2>&1
fi
