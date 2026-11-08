#!/usr/bin/env bash
# Time limit: stop training. The laptop script then copies the data and ends the rental.
sleep "$1"
[ -f runs/DONE ] && exit 0
pkill -f '[r]un_mixed.sh'; pkill -f '[b]uild_mixed_then_run.sh'; pkill -f '[r]un_v3.sh'; pkill -f '[r]un_between.sh'; pkill -f '[b]uild_then_between.sh'; pkill -f '[w]ait_then_big.sh'; pkill -f '[b]ig_run.sh'; pkill -f '[s]cripts/long_run.sh'; pkill -f '[s]cripts/run_all.sh'; pkill -f '[s]pidory\.'
touch runs/LIMIT_REACHED
