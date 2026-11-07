#!/usr/bin/env bash
# Waits for Plain's early check, then continues with the full experiment.
cd "$(dirname "$0")/.."
until [ -f runs/PLAIN_PASSED ] || [ -f runs/STOPPED ]; do sleep 20; done
[ -f runs/PLAIN_PASSED ] && STAGE1=30000 STAGE2=30000 ROUND=3000 PASSES=2 SEEDS="0 1" scripts/long_run.sh >> runs/long_run.log 2>&1
