#!/usr/bin/env bash
# Starts training and its time limit fully detached from this login.
cd /workspace/Spidory
setsid nohup scripts/watchdog.sh "$1" > /dev/null 2>&1 < /dev/null &
setsid nohup scripts/plain_first.sh > runs/plain_first.log 2>&1 < /dev/null &
disown -a
exit 0
