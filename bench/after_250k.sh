#!/usr/bin/env bash
# After the 250K engine run exits: the P6 sweeps, then PLAN 14.9 from that run's logs.
# Started in the background so the GPU is never shared by two measurements.
set -u
cd "$(dirname "$0")/.."
LOG=/var/tmp/scratch/kvtest/after-250k.log
: > "$LOG"
while pgrep -f "c/inkling -f bench/prompt-250k" > /dev/null; do sleep 20; done
echo "[after] the 250K run exited $(date -Is)" >> "$LOG"
echo "[after] kernel sweep" >> "$LOG"
python3 bench/sweep_kernels.py --quick --out bench/results-speed.json >> "$LOG" 2>&1
echo "[after] engine sweep" >> "$LOG"
python3 bench/sweep_engine.py \
  --snap /run/media/nazar/Gaaaymes/models/Inkling-2-Mini-GGUF/iq3xxs.colibri \
  --out bench/results-engine.json >> "$LOG" 2>&1
echo "[after] PLAN 14.9" >> "$LOG"
python3 bench/fill_plan_250k.py /var/tmp/scratch/kvtest/250k-new.err >> "$LOG" 2>&1
echo "[after] done $(date -Is)" >> "$LOG"
