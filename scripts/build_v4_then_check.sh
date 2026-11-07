#!/usr/bin/env bash
# On a GPU machine: grow the data to v4, then measure every training phase.
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true
PY=/venv/main/bin/python
[ -f data/V4_DONE ] || $PY -m spidory.grow_v4 --root data > runs/grow_v4.log 2>&1
$PY scripts/check_gpu.py > runs/check_gpu.log 2>&1
echo DONE >> runs/check_gpu.log
