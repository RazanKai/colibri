# Goal prompt for the next session — Inkling-2-Mini 256K, Colibri fork

Paste the block below as the first message of a new chat.

---

**Goal.** Continue the Inkling-2-Mini long-context work in `/home/nazar/src/colibri`, branch
`kv-attention-longctx` (fork of JustVugg/colibri). The target is **262,144 tokens** — the model is
256K-native; 1M is YaRN-only and is out of scope. Speed *and* correctness both count: the typed-KV
path silently lost the context until recently, so every speed claim must be accompanied by an
end-to-end retrieval check.

**Box and files.**
- RTX 5070 12 GB (sm_120), i5-10400F (6C/12T, AVX2), 23 GB RAM. CUDA 13.4 at `/opt/cuda`.
- Snapshot `/run/media/nazar/Gaaaymes/models/Inkling-2-Mini-GGUF/iq3xxs.colibri` (UD-IQ3_XXS;
  24 layers × 128 experts, 8 global attention layers, layers 6–17 run three loop passes).
- Build: `make -C c inkling CUDA=1 CUDA_HOME=/opt/cuda`. Tests: `make -C c cuda-inkc-check`.
  Kernel bench: `make -C c cuda-inkc-bench`.
- Run: `SNAP=<snap> CTX_MAX=262144 INK2_KV=q4 COLI_CUDA=1 OMP_NUM_THREADS=6 ./c/inkling -f
  bench/prompt-250k.txt -n 24`.

**Where things stand (all verified on this tree unless noted).**
- 250K, 250,156 tokens: prefill **1598.5 s** (was 3651.5, 2.28×), device side **804.7 s** (was
  2706.7, 3.36×), decode **12.18 tok/s** (was 8.44, 1.44×), RSS 11.1 GB.
- 64K: prefill 265.6 s (chunk 1024) / 236.2 s (chunk 2048); decode 16.8–17.2 tok/s.
- **The 250K needle passes**: the engine answers `7X4Q-ALPHA-93` at 250,156 tokens. 1K/2K/4K/8K
  also pass (`bench/needle_check.py --kv q4 --sizes 1,2,4,8`).
- Attention kernels: prompt path 1.03 → **5.0 TFLOP/s**; decode step 8.03 → 4.15 ms.
- `CCG_RA_TIMING=1` splits the engine's `attn` phase: 64K is 48.1 s kernel / 22.6 s
  projections+norms+dconv+gate; scale to 250K ≈ 547 s kernel, 258 s other.
- The expert phase is the other half: 612 s at 250K (host step, serialized with the device).

**The bug that mattered (do not regress it).** With `INK2_KV=f16/q8/q4` the CUDA chain quantized
rows into the host cache but never marked them for the device mirror (`ink2cc_kv_down` now marks
`dlo/dhi`). Everything after the first 1K chunk attended to zeros. It passed every kernel test and
every speed number. `bench/gate.sh` (parity + needle) is the gate for any change touching the
chain, the mirror, the queue or the tier — run it, and keep the needle in the loop.

**Immediate work, in order.**

1. **Record/confirm the 250K needle on the current tree** — commit, `budget X GiB = N experts`
   line, free VRAM at launch, verbatim answer. This is the headline number.
2. **Whole-layer residency A/B** (PLAN §15.2–15.3). `COLI_VK_TIER_WHOLE=n` already exists in
   `c/vk_tier.c` (n hottest layers whole, default 0). The replay estimate
   (`bench/tier_policy_estimate.py`) says per-expert ranking completes **0 of 24 layers**, while
   n=12 completes all twelve loop layers carrying **77%** of activations; but within a loop layer
   the top 40 experts already carry 72–87%, so per-expert residency covers **~90%**. The A/B is
   whether the in-frame path beats a ~13-point coverage loss:
   `0.77·c_f + 0.23·c_h < 0.90·c_p + 0.10·c_h`.
   Cells at 64K: n=0 twice (noise floor), n=12 (clean test), n=6/13/17; **one 250K n=12 cell
   regardless of the 64K result**. Log per cell: budget + experts, free VRAM at launch, evictions,
   uploads, cold experts, "N of 24 layers in the frame", per-layer streaming, phases, prefill,
   decode tok/s separately, RSS, needle+oracle for n>0. Reject a cell whose budget drifts >0.1 GiB
   or free VRAM >0.2 GB. The budget follows free VRAM, not `CUDA_RESERVE_GB`/`CUDA_EXPERT_GB`.
3. **DSpark gate** (the official drafter is a *separate* 402 MB `dflash` model,
   `dspark-Inkling-2-Mini-Q8_0.gguf`, loaded by llama.cpp with `-md`). This engine's `INK2_MTP` is
   a different mechanism (in-snapshot `<snap>/mtp/mtp.safetensors`, absent here). Price the
   drafter before building anything: llama.cpp on the same prompt **with and without `-md`**, 8K
   and 64K (plus 128K if the 64K delta is large), same `ncmoe` and q4 KV. Log prompt/decode tok/s,
   `draft_n`/`draft_n_accepted` and **accepted tokens per verify step**. The without-`-md` number
   is the fair comparison for our 16.55 tok/s at 64K.
4. **Then, with results**: either the chunk wavefront (hide the 612 s host step behind the device;
   per-(chunk,layer) KV-upload ordering is the risk, gate with `bench/gate.sh`) or the tensor-core
   prompt kernel (only with a vectorized q4 unpack in the same change, or dequant becomes the
   limiter; ~2–2.5× on a 547 s kernel). Re-profile first and pick one.

**Known dead ends — don't repeat them.** Prompt chunk 4096 is a ~14× expert-step cliff; 2048 is
good at 64K (1.12×) and catastrophic at 250K (the tier's step becomes 16384 activated rows).
`COLI_CUDA_RELATTN_ROWS=16` gives only 1.05× and silently falls back to scalar at a head group of
8. `CUDA_RESERVE_GB` and `CUDA_EXPERT_GB` do not pin the tier budget. `COLI_CUDA_CHAIN=0`,
`INK2_KV=f32` and the CPU path all work but are slow; they are the isolation ladder, not settings.

**Rules.** Measure, don't derive — tag every number `measured` / `derived` / `estimated`. One GPU
process at a time on the 12 GB card (concurrent runs OOM and have already muddied one session).
Behaviour changes go behind env flags with the f32 path untouched. Keep `PLAN-colibri.md` §14–§15
and `docs/ENVIRONMENT.md` current, and commit each verified step.
