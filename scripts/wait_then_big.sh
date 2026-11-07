#!/usr/bin/env bash
# Waits for data growth to finish, then trains the 100M model.
cd "$(dirname "$0")/.."
while pgrep -f '[g]row_data' >/dev/null; do sleep 20; done
scripts/big_run.sh 46000 46000 4500 2 64 >> runs/big_run.log 2>&1
