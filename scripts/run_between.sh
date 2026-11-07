#!/usr/bin/env bash
# Between-block architecture, shared and per gap, with the user's run neuron. Safe to run again.
# Same data, seed and schedule as v3, so v3 seed 0 is the baseline.
# Usage: scripts/run_between.sh [seed]
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=4
PY=${PY:-/venv/main/bin/python}
S1=24000; S2=24000; CYCLE=6000; ROUND=2400; PASSES=2; BATCH=64; SEED=${1:-0}; LAYERS=6
TOTAL=$((S1 + S2 + CYCLE + ROUND * 4 * PASSES))
COMMON="--stage1 $S1 --stage2 $S2 --cycle $CYCLE --total $TOTAL --batch-size $BATCH --seed $SEED --arch ngpt --lr 6e-3 --warmup 1000 --between-layers $LAYERS"
R="--round-steps $ROUND --passes $PASSES --seed $SEED"
fail() { echo "$1" > runs/STOPPED; exit 1; }

for MODE in shared per_gap; do
  $PY -m spidory.train --name between_${MODE}_warm_s$SEED --between-mode $MODE $COMMON || fail "$MODE training failed"
  $PY -m spidory.rounds --name neuron_between_${MODE}_s$SEED --warm between_${MODE}_warm_s$SEED --run-mode joint --neuron net --neuron-train sampled $R || fail "$MODE rounds failed"
  $PY -m spidory.evaluate --single neuron_between_${MODE}_s$SEED --out runs/results_between_${MODE}.json | tee runs/results_between_${MODE}.txt
done
cp runs/results_between_per_gap.json runs/results_between.json
touch runs/DONE
