#!/usr/bin/env bash
# The gate every commit on the long-context branch passes: the attention kernels against the CPU
# reference, then the needle on the CUDA path with the served KV type. The typed-mirror bug in
# PLAN 15 passed every kernel test and every speed number and still produced garbage, because no
# test asked whether a *long prompt* was attended to. The needle is that test; keep it in the loop
# for every change that touches the chain, the mirror, the queue or the tier.
#
#   ./bench/gate.sh              # q4, 1K/2K/4K/8K
#   KV=f16 ./bench/gate.sh
#
# The teacher-forced oracle (PLAN section 10) is the other half and is run by hand: a greedy run
# is not bit-reproducible on this box, so it needs OMP_NUM_THREADS=1 and COLI_CUDA=0.
set -euo pipefail
cd "$(dirname "$0")/.."
KV=${KV:-q4}
SIZES=${SIZES:-1,2,4,8}
echo "== cuda-inkc-check (the kernels against the CPU reference)"
make -C c cuda-inkc-check CUDA=1 CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
echo "== needle, INK2_KV=$KV, sizes $SIZES (the end-to-end read of a long prompt)"
python3 bench/needle_check.py --kv "$KV" --sizes "$SIZES"
