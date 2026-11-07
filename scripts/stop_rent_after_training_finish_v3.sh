#!/usr/bin/env bash
# Waits for training to end, copies everything, verifies it, then ends the rental.
# It only ends the rental when every remote file matches its local copy by checksum.
# Usage: scripts/stop_rent_after_training_finish.sh <instance_id> [--check]
#   --check: copy and verify now, and never end the rental.
set -uo pipefail
cd "$(dirname "$0")/.."
INSTANCE=${1:?instance id required}
LOCAL=${LOCAL:-runs/remote_v3}
LOG=${LOG:-runs/stop_rent_v3.log}
CHECK_ONLY=${2:-}
read -r HOST PORT < "${HOST_FILE:-/tmp/claude-1000/-home-david-projects-Spidory/e0700a11-f54c-4d9e-afbb-b9c196a6def7/scratchpad/vast_host_v3}"
SSH="timeout 60 ssh -q -n -p $PORT -o ConnectTimeout=20 -o ServerAliveInterval=15 root@$HOST"
RSH="ssh -q -p $PORT -o ConnectTimeout=20 -o ServerAliveInterval=15"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
stop_keep() { say "KEEPING RENTAL: $1"; exit "$2"; }

mkdir -p "$LOCAL"
if [ "$CHECK_ONLY" != "--check" ]; then
  say "waiting for training to end"
  failures=0; unknown=0
  for _ in $(seq 1 360); do  # 30 hours at most
    state=$($SSH 'cd /workspace/Spidory/runs
      if pgrep -f "[r]un_v3.sh|[b]uild_v4_then_check.sh|[s]pidory\.(train|rounds|evaluate|clean_v5)" >/dev/null; then echo RUNNING
      elif [ -f DONE ]; then echo DONE
      elif [ -f STOPPED ]; then echo STOPPED
      elif [ -f LIMIT_REACHED ]; then echo LIMIT
      else echo UNKNOWN; fi' 2>/dev/null | tail -1)
    case "$state" in
      RUNNING) failures=0; unknown=0 ;;
      DONE|STOPPED|LIMIT) break ;;
      UNKNOWN) unknown=$((unknown + 1))  # nothing running and no marker: crashed or rebooted
         [ "$unknown" -ge 2 ] && { state=ENDED_UNEXPECTEDLY; break; } ;;
      *) failures=$((failures + 1))
         [ "$failures" -ge 6 ] && stop_keep "machine unreachable for 30 minutes" 5 ;;
    esac
    sleep 300
  done
  [ "${state:-}" = DONE ] || [ "${state:-}" = STOPPED ] || [ "${state:-}" = LIMIT ] || [ "${state:-}" = ENDED_UNEXPECTEDLY ] || stop_keep "training never ended" 6
  say "training ended: $state"
  if [ "$state" = DONE ]; then
    $SSH 'cd /workspace/Spidory && OMP_NUM_THREADS=4 PYTHONPATH=. timeout 50 /venv/main/bin/python scripts/measure_speed.py' \
      || say "speed measurement failed, continuing"
  fi
fi

for attempt in 1 2 3 4 5; do
  rsync -az --partial --timeout=120 -e "$RSH" "root@$HOST:/workspace/Spidory/runs/" "$LOCAL/" && break
  say "copy attempt $attempt failed"
  [ "$attempt" = 5 ] && stop_keep "copy failed" 7
done

differences=$(rsync -anc --timeout=300 --itemize-changes -e "$RSH" "root@$HOST:/workspace/Spidory/runs/" "$LOCAL/" 2>&1)
code=$?
[ "$code" -eq 0 ] || stop_keep "verify failed with code $code" 8
remote_files=
for attempt in 1 2 3 4 5 6; do
  remote_files=$($SSH 'find /workspace/Spidory/runs -type f | wc -l' 2>/dev/null | tail -1)
  [ -n "$remote_files" ] && break
  sleep 10
done
local_files=$(find "$LOCAL" -type f | wc -l)
say "remote files: $remote_files, local files: $local_files"
if [ "$CHECK_ONLY" = "--check" ]; then
  say "check only, changed files: $(printf '%s' "$differences" | grep -c '^[>c]')"
  exit 0
fi
[ -z "$(printf '%s' "$differences" | grep '^[>c]')" ] || stop_keep "local copy differs: $differences" 9
[ -n "$remote_files" ] && [ "$local_files" -ge "$remote_files" ] || stop_keep "file count mismatch" 10
if [ -f "$LOCAL/DONE" ]; then
  [ -s "$LOCAL/results_v3.json" ] || say "WARNING: results_v3.json is missing; checkpoints are copied, so it can be evaluated later"
fi

label=$(vastai show instance "$INSTANCE" --raw 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('label'))")
[ "$label" = spidory ] || stop_keep "instance label is $label" 12
vastai destroy instance "$INSTANCE" -y --raw 2>&1 | tail -1 | tee -a "$LOG"
sleep 20
if vastai show instances --raw 2>/dev/null | python3 -c "import sys,json; sys.exit(0 if any(i['id']==$INSTANCE for i in json.load(sys.stdin)) else 1)"; then
  stop_keep "destroy did not complete, charges may continue" 13
fi
say "rental ended, all data verified in $LOCAL"
