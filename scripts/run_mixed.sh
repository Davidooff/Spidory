#!/usr/bin/env bash
# Run signal and per-token routing, on data with mixed topics and explained maths. Safe to run again.
# Usage: scripts/run_mixed.sh <seed> <name,name,...>   (names from VERSIONS below; default all)
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=4
# Seven datasets, each a seventh of every batch: story, maths, qa, mixed topics, maths explained in
# words, writing a problem when asked, and public word problems (Orca-Math and GSM8K train).
export SPIDORY_DATASETS=story,maths,qa,mixed,explain,compose,wordmath
PY=${PY:-/venv/main/bin/python}
# 2 passes over 7 datasets: 14 rounds of 1,370 steps keep stage 3 near v3's 19,200 steps.
S1=24000; S2=24000; CYCLE=6000; ROUND=1370; PASSES=2; BATCH=64; SEED=${1:-0}
TOTAL=$((S1 + S2 + CYCLE + ROUND * 7 * PASSES))
COMMON="--stage1 $S1 --stage2 $S2 --cycle $CYCLE --total $TOTAL --batch-size $BATCH --seed $SEED --arch ngpt --lr 6e-3 --warmup 1000"
R="--round-steps $ROUND --passes $PASSES --seed $SEED"
NEURON="--run-mode joint --neuron net --neuron-train sampled"
# Under run_jobs.sh a failure goes to another file: STOPPED would end the rental at once.
fail() { echo "$1" > "runs/${STOP_FILE:-STOPPED}"; exit 1; }

# name : training flags : rounds flags
VERSIONS=(
  "plain:--plain:--control-only"
  "plain_signal:--plain --run-signal:--control-only"
  "plain_signal_nonorm:--plain --run-signal --no-boundary-norm:--control-only"
  "perloop:--route-per-loop:$NEURON"
  "token:--route-per-token --route-per-loop:$NEURON"
  "token_signal:--route-per-token --route-per-loop --run-signal:$NEURON"
  "token_signal_once:--route-per-token --run-signal:$NEURON"  # experts chosen per token once, kept in every run
  "between_signal:--between-layers 6 --between-mode shared --run-signal:$NEURON"
  "noloop:--plain --max-runs 1:--control-only"  # no loop: the 10-layer centre runs once
  "noloop_deep:--plain --max-runs 1 --n-core 30:--control-only"  # no loop, compute of Plain at k = 3
)
WANTED=",${2:-all},"
matched=0
for ENTRY in "${VERSIONS[@]}"; do
  NAME=${ENTRY%%:*}; REST=${ENTRY#*:}; TRAIN=${REST%%:*}; ROUNDS=${REST#*:}
  [ "$WANTED" = ",all," ] || [[ "$WANTED" == *",$NAME,"* ]] || continue
  matched=1
  $PY -m spidory.train --name ${NAME}_warm_s$SEED $TRAIN $COMMON || fail "$NAME training failed"
  $PY -m spidory.rounds --name ${NAME}_s$SEED --warm ${NAME}_warm_s$SEED $ROUNDS $R || fail "$NAME rounds failed"
  $PY -m spidory.evaluate --single ${NAME}_s$SEED --out runs/results_${NAME}_s$SEED.json > runs/results_${NAME}_s$SEED.txt 2>&1 \
    || fail "$NAME evaluation failed"
done
[ $matched -eq 1 ] || fail "no version matches $2"
[ -n "${NO_DONE:-}" ] || touch runs/DONE  # run_jobs.sh marks the end itself
