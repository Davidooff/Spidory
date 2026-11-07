#!/usr/bin/env bash
# Two seeds of the full experiment, then the combined comparison. Safe to run again.
set -uo pipefail
cd "$(dirname "$0")/.."
export PY=${PY:-/venv/main/bin/python} OMP_NUM_THREADS=4 PYTHONPATH=. BATCH=64
STAGE1=${STAGE1:?}; STAGE2=${STAGE2:?}; ROUND=${ROUND:?}; PASSES=${PASSES:-2}
for SEED in ${SEEDS:-0 1}; do
  scripts/run_all.sh $STAGE1 $STAGE2 $ROUND $PASSES 256 $SEED
  CODE=$?
  if [ $CODE -ne 0 ]; then echo "seed $SEED exit $CODE" > runs/STOPPED; exit $CODE; fi
done
$PY -m spidory.evaluate --seeds ${SEEDS:-0 1} --out runs/results.json | tee runs/results.txt
touch runs/DONE
