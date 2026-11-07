#!/usr/bin/env bash
# v3: the run neuron trained jointly, in three ways, against Plain and Constant. Safe to run again.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=4
PY=${PY:-/venv/main/bin/python}
S1=24000; S2=24000; ROUND=2400; PASSES=2; BATCH=64; SEED=0
TOTAL=$((S1 + S2 + ROUND * 4 * PASSES))
# nGPT blocks, SwiGLU and RoPE. Learning rate from a short laptop comparison (see TRAINING_VERSIONS.md).
COMMON="--stage1 $S1 --stage2 $S2 --total $TOTAL --batch-size $BATCH --seed $SEED --arch ngpt --lr 6e-3 --warmup 1000"
R="--round-steps $ROUND --passes $PASSES --seed $SEED"
fail() { echo "$1" > runs/STOPPED; exit 1; }

$PY -m spidory.train --name plain_warm_s0 --plain $COMMON || fail "plain training failed"
$PY -m spidory.evaluate --loop-check plain_warm_s0 || { echo "early check failed" > runs/STOPPED; exit 3; }
$PY -m spidory.train --name moe_warm_s0 $COMMON || fail "expert training failed"
$PY -m spidory.rounds --name plain_s0 --warm plain_warm_s0 --control-only $R || fail "plain rounds failed"
$PY -m spidory.rounds --name constant_s0 --warm moe_warm_s0 --control-only $R || fail "constant rounds failed"
$PY -m spidory.rounds --name neuron_net_sampled_s0 --warm moe_warm_s0 --run-mode joint --neuron net --neuron-train sampled $R || fail "neuron net sampled failed"
$PY -m spidory.rounds --name neuron_one_sampled_s0 --warm moe_warm_s0 --run-mode joint --neuron linear --neuron-train sampled $R || fail "neuron one sampled failed"
$PY -m spidory.rounds --name neuron_net_average_s0 --warm moe_warm_s0 --run-mode joint --neuron net --neuron-train average $R || fail "neuron net average failed"
$PY -m spidory.rounds --name neuron_net_mix_s0 --warm moe_warm_s0 --run-mode joint --neuron net --neuron-train mix $R || fail "neuron net mix failed"
$PY -m spidory.train --name moe2_warm_s0 --active-experts 2 $COMMON || fail "two-expert training failed"
$PY -m spidory.rounds --name neuron_net_sampled_top2_s0 --warm moe2_warm_s0 --run-mode joint --neuron net --neuron-train sampled $R || fail "two-expert neuron failed"
$PY -m spidory.evaluate --seeds $SEED --names plain constant neuron_net_sampled neuron_one_sampled neuron_net_average neuron_net_mix neuron_net_sampled_top2 --out runs/results_v3.json | tee runs/results_v3.txt
touch runs/DONE
