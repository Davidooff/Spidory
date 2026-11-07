#!/usr/bin/env bash
# On a GPU machine: grow the data to v4 if needed, then run the between-block experiment.
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true
[ -f data/V4_DONE ] || /venv/main/bin/python -m spidory.grow_v4 --root data > runs/grow_v4.log 2>&1 || { echo "data growth failed" > runs/STOPPED; exit 1; }
scripts/run_between.sh "${1:-0}" > runs/run_between.log 2>&1
