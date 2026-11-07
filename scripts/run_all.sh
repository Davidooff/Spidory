#!/usr/bin/env bash
# Runs Plain, Constant and Variable for one seed. Safe to run again: finished work is skipped.
# Usage: scripts/run_all.sh [stage1] [stage2] [round_steps] [passes] [seq_len] [seed]
# Set BATCH to change the batch size.
set -euo pipefail
cd "$(dirname "$0")/.."
S1=${1:-2100}; S2=${2:-900}; ROUND=${3:-300}; PASSES=${4:-2}; SEQ=${5:-256}; SEED=${6:-0}
BATCH=${BATCH:-16}
TOTAL=$((S1 + S2 + ROUND * 4 * PASSES))
PY=${PY:-.venv/bin/python}
T="s${SEED}"
COMMON="--stage1 $S1 --stage2 $S2 --total $TOTAL --batch-size $BATCH --seq-len $SEQ --seed $SEED"
ROUNDS="--round-steps $ROUND --passes $PASSES --seed $SEED"

# Plain goes first: if loops do not help, nothing else is worth paying for.
$PY -m spidory.train --name plain_warm_$T --plain $COMMON
if ! $PY -m spidory.evaluate --loop-check plain_warm_$T; then
  echo "EARLY STOP: loops did not beat one run on held-out data." >&2
  exit 3
fi
$PY -m spidory.train --name moe_warm_$T $COMMON
$PY -m spidory.rounds --name plain_$T --warm plain_warm_$T --control-only $ROUNDS
$PY -m spidory.rounds --name constant_$T --warm moe_warm_$T --control-only $ROUNDS
$PY -m spidory.rounds --name variable_$T --warm moe_warm_$T $ROUNDS
$PY -m spidory.run_head --name variable_$T --seed $SEED
$PY -m spidory.evaluate --seeds $SEED --out runs/results_$T.json | tee runs/results_$T.txt
