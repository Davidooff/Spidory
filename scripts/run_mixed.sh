#!/usr/bin/env bash
# Run signal and per-token routing, on data with mixed topics and explained maths. Safe to run again.
# Usage: scripts/run_mixed.sh <seed> <name,name,...>   (names from VERSIONS below; default all)
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=4
# Five datasets, each a fifth of every batch: story, maths, qa, mixed topics, explained maths.
export SPIDORY_DATASETS=story,maths,qa,mixed,explain
PY=${PY:-/venv/main/bin/python}
# 2 passes over 5 datasets: 10 rounds of 1,920 steps keep stage 3 at 19,200 steps, as in v3.
S1=24000; S2=24000; CYCLE=6000; ROUND=1920; PASSES=2; BATCH=64; SEED=${1:-0}
TOTAL=$((S1 + S2 + CYCLE + ROUND * 5 * PASSES))
COMMON="--stage1 $S1 --stage2 $S2 --cycle $CYCLE --total $TOTAL --batch-size $BATCH --seed $SEED --arch ngpt --lr 6e-3 --warmup 1000"
R="--round-steps $ROUND --passes $PASSES --seed $SEED"
NEURON="--run-mode joint --neuron net --neuron-train sampled"
fail() { echo "$1" > runs/STOPPED; exit 1; }

# name : training flags : rounds flags
VERSIONS=(
  "plain:--plain:--control-only"
  "plain_signal:--plain --run-signal:--control-only"
  "plain_signal_nonorm:--plain --run-signal --no-boundary-norm:--control-only"
  "perloop:--route-per-loop:$NEURON"
  "token:--route-per-token --route-per-loop:$NEURON"
  "token_signal:--route-per-token --route-per-loop --run-signal:$NEURON"
  "between_signal:--between-layers 6 --between-mode shared --run-signal:$NEURON"
)
WANTED=",${2:-all},"
for ENTRY in "${VERSIONS[@]}"; do
  NAME=${ENTRY%%:*}; REST=${ENTRY#*:}; TRAIN=${REST%%:*}; ROUNDS=${REST#*:}
  [ "$WANTED" = ",all," ] || [[ "$WANTED" == *",$NAME,"* ]] || continue
  $PY -m spidory.train --name ${NAME}_warm_s$SEED $TRAIN $COMMON || fail "$NAME training failed"
  $PY -m spidory.rounds --name ${NAME}_s$SEED --warm ${NAME}_warm_s$SEED $ROUNDS $R || fail "$NAME rounds failed"
  $PY -m spidory.evaluate --single ${NAME}_s$SEED --out runs/results_${NAME}.json > runs/results_${NAME}.txt 2>&1 \
    || fail "$NAME evaluation failed"
done
touch runs/DONE
