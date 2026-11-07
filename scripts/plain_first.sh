#!/usr/bin/env bash
# Plain warm-up and its early check. run_all.sh later skips this finished work.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 PYTHONPATH=.
PY=/venv/main/bin/python
$PY -m spidory.train --name plain_warm_s0 --plain --stage1 30000 --stage2 30000 --total 84000 --batch-size 64 --seq-len 256 --seed 0
if $PY -m spidory.evaluate --loop-check plain_warm_s0; then touch runs/PLAIN_PASSED; else echo "early check failed" > runs/STOPPED; fi
