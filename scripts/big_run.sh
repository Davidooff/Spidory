#!/usr/bin/env bash
# The 100M experts model: all stages, rounds, run neuron and evaluation. Safe to run again.
# Usage: scripts/big_run.sh <stage1> <stage2> <round_steps> <passes> <batch>
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=4
PY=${PY:-/venv/main/bin/python}
S1=$1; S2=$2; ROUND=$3; PASSES=$4; BATCH=$5
TOTAL=$((S1 + S2 + ROUND * 4 * PASSES))
$PY -m spidory.train --name big_warm_s0 --d-model 672 --expert-dim 252 --stage1 $S1 --stage2 $S2 --total $TOTAL --batch-size $BATCH --seed 0 || { echo "train failed" > runs/STOPPED; exit 1; }
$PY -m spidory.rounds --name big_s0 --warm big_warm_s0 --round-steps $ROUND --passes $PASSES --seed 0 || { echo "rounds failed" > runs/STOPPED; exit 1; }
$PY -m spidory.run_head --name big_s0 --seed 0 || { echo "run head failed" > runs/STOPPED; exit 1; }
$PY -m spidory.evaluate --single big_s0 --out runs/results_big.json | tee runs/results_big.txt
touch runs/DONE
