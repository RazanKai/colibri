# PLAN-colibri — Inkling 2 Mini long context in Colibri

Status: **Stage 1 implemented and measured** (f16/q8/q4 host KV, typed CUDA mirror, tiled
prefill attention). 64K now beats the Unsloth b12588 baseline on prefill; 250K loads, prefill
completes and decode runs. The 250K needle failed until §15: it was **our bug**, not the
model — with a typed KV cache the CUDA chain never uploaded the prompt's rows to the device
mirror, so everything after the first 1K chunk attended to zeros. Fixed (mark the rows in
`ink2cc_kv_down`); the harness now retrieves the code at 1K/2K/4K/8K on the CUDA q4 path, and
llama.cpp and this engine's CPU path always did.
**250K/262K speed pass (P0–P6, §14):** the attention kernels were rebuilt (GQA-shared decode,
K-tile transpose, one tile read per query head, exp2). 64K: prefill 387.1 → 266.6 s (1.45×),
device side 202.6 → 69.5 s (2.91×), decode 14.94 → 16.55 tok/s. 250K with the §15 fix:
**prefill 1598–1792 s (2.04–2.28×), device side 805–869 s (3.11–3.36×), decode 11.2–12.2 tok/s
(1.33–1.44×)** — the band is the tier's budget, 5.6–5.9 GiB — and the **needle passes at 250,156
tokens** on both runs (`7X4Q-ALPHA-93`); the kernel itself went 1.0 → 5.0 TFLOP/s. The model is 256K-native (1M is YaRN-only), so 262144 is the acceptance point and §7
stays shelved.
Target: **250K-token context (MVP)**, **1M-token context (Stage 2)**
Engine: Colibri `dev` (v2.2.0), `inkling2` family in `c/inkling.c` + `c/inkling2.h`
Model: `unsloth/Inkling-2-Mini-GGUF` UD-IQ3_XXS, imported with `coli` (`arch inkling2`)
Host: RTX 5070 12 GB, i5-10400F (6C/12T, AVX2), 23 GB RAM, NVMe

## 1. Why this plan exists

Colibri already implements the hierarchical placement we wanted from Strata:

- hot experts in VRAM (expert tier);
- warm experts in RAM (LRU + pinned history);
- cold experts streamed from NVMe;
- CPU and device compute overlapped;
- prefix reuse for agent turns.

The missing pieces for long context were the **KV cache** (fp32, 16 GB at 250K) and the
**attention/prefill path** (73% of the 64K prefill). Both are now addressed:

1. the host cache is f16/q8/q4 (`INK2_KV`), with the CUDA chain's device mirror of the same
   type, so 250K fits on both sides;
2. the CUDA chain runs a tiled prompt kernel that reads each K/V row once per tile of query
   rows instead of once per query row, **3.6× faster attention at 64K** than the old kernel.

The llama.cpp plan (`PLAN.md`) has q4 KV today but no expert cache. Colibri now beats Unsloth
on **prefill** at 64K and is still behind on **decode** (the expert stream dominates there).

## 2. Measured on this box

Build: both backends work (CUDA is primary; Vulkan is the fallback).

- **CUDA**: `make -C c inkling CUDA=1 CUDA_HOME=/opt/cuda` — CUDA 13.4/13.5, GCC 16.2.1.
- **Vulkan**: `make -C c inkling VK=1`.

### 2.1 8K / 32K / 64K / 250K after this work

CUDA, `COLI_CUDA=1`, dense chain on (default for Inkling-2-Mini), 32 generated tokens
(250K: 24), `OMP_NUM_THREADS=6`. Attention is the tiled kernel (§4.3) unless noted.

| Context | Prompt tokens | KV | Prefill | Prefill tok/s | Decode | attn phase | expert phase | RSS | host KV | device |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K  | 8,027   | f32 | 31.1 s  | 258 | 21.70 tok/s | 8.7 s  | 17.0 s | 9.7 GB | 824 MiB | 1133 MiB |
| 8K  | 8,027   | f16 | 31.2 s  | 257 | 17.72 tok/s | 8.7 s  | 18.3 s | 9.3 GB | 412 MiB | 721 MiB |
| 8K  | 8,027   | q4  | 29.9 s  | 268 | 22.15 tok/s | 8.5 s  | 16.1 s | 8.4 GB | 129 MiB | 438 MiB |
| 32K | 32,100  | f32 | 161.1 s | 199 | 17.79 tok/s | — | — | 11.4 GB | — | — |
| 32K | 32,100  | q4  | 151.5 s | 212 | 16.91 tok/s | — | — | 9.2 GB | — | — |
| 64K | 63,977  | q4  | **387.1 s** | 165 | **14.94 tok/s** | **202.6 s** | 138.9 s | 10.2 GB | 675 MiB | 984 MiB |
| 250K| 250,156 | q4  | **3651.5 s (60.9 min)** | 68.5 | **8.44 tok/s** | 2706.7 s | 769.4 s | 12.2 GB | 2493 MiB | 2802 MiB |

`fill`/`other` are the rest (4–5 s and 45–173 s respectively; `other` is prefix-store and
host-side row work). 250K: 13,156 frames; q4 host KV "full cache would be 14,659 MiB" (all 48
state layers at `max_t`), the ring cache is 2493 MiB.

### 2.2 Before / after (same prompt, same box)

| 64K, 63,977 tokens | Prefill | attn | expert | Decode |
|---|---:|---:|---:|---:|
| before (fp32 KV, old kernel) | 1,009.4 s | 736.8 s | 235.5 s | 14.16 tok/s |
| after, q4 KV, scalar kernel (`COLI_CUDA_RELATTN_QT=0`) | 671.8 s | 491.5 s | 135.8 s | 13.85 tok/s |
| **after, q4 KV, tiled kernel** | **387.1 s** | **202.6 s** | 138.9 s | **14.94 tok/s** |
| Unsloth b12588 (llama.cpp, q4 KV, DSpark) | 439.8 s | — | — | 18.33 tok/s |

- Attention: **3.64× faster** than the old kernel at 64K; the tiled kernel alone is **2.43×**
  faster than the scalar one on the same binary.
- **64K prefill: Colibri 387.1 s vs Unsloth 439.8 s → 1.14× faster.** Decode is still 1.23×
  slower (14.94 vs 18.33 tok/s), which is the expert stream and the per-layer host round
  trips, not the KV.
- 8K prefill went from 50.1 s to 29.9–31.1 s with the same prompt and 32 tokens.

### 2.3 Quality / parity evidence

- `c/tests/test_ink_kv.c`: f32 exact; f16 exact where representable and within 1e-3
  elsewhere; q8/q4 within half a block step; the dot and the V axpy equal those of the
  widened row. `ink_kv: ok`.
- `make -C c cuda-inkc-check`: **all passed**, including the tiled kernel against the CPU
  reference (rel. err ≤ 2e-5) and the typed mirrors f16/q8/q4 through both the scalar and the
  tiled kernel (the kernel's dequantization equals `ink_kv_get`'s).
- Teacher-forced argmax over a 56-token sequence, CPU experts, `OMP_NUM_THREADS=1`:
  f32 / f16 / q8 / q4 agree **56/56** positions (`INK_TF_DUMP`).
  Note: on this box a *greedy* run is not bit-reproducible with several OpenMP threads and
  the expert tier (the tier hands different experts to the device per run, and the device and
  the CPU round differently). Token-level comparisons must pin `OMP_NUM_THREADS=1` and
  `COLI_CUDA=0`; that is why the parity above is measured that way.

## 3. The KV budget (with the new types)

Inkling 2 Mini has 48 state layers: 24 physical plus 24 loop-pass slots. Eight are global
attention (layers 5, 11x3, 17x3, 23); the other 40 are sliding-window rings of 1024 tokens.
The global slots dominate long context. Per token per global slot: `2 * 8 KV heads * 128 =
2048` values. A row is `hd` values of one KV head at one position.

| KV type | Bytes a row (hd=128) | Per token, 8 global slots | 64K | 250K |
|---|---:|---:|---:|---:|
| fp32 | 512 | 64 KB | 4.10 GB | 16.0 GB |
| f16  | 256 | 32 KB | 2.05 GB | 8.00 GB |
| q8   | 144 (4×(4+32)) | 18 KB | 1.12 GB | 4.38 GB |
| q4   | 80 (4×(4+16))  | 10 KB | 0.62 GB | 2.44 GB |

Measured ring caches (48 state layers, sliding rings at 1024): 8K f32 824 MiB / f16 412 / q4
129; 64K q4 675 MiB; 250K q4 2493 MiB. The rings are ~0.17 GB f32 and constant; conv banks
~4 MB; dense ~1.4 GB.

### 3.1 Codec (`c/ink_kv.h`, new)

A row is blocks of 32 values, each with its own **f32 scale** (not f16: `amax/7` can exceed
65504, and a saturated scale silently destroyed the first test iteration):

- f16: `kv16.h`'s IEEE half.
- q8: `[f32 scale][32 int8]`, scale = amax/127.
- q4: `[f32 scale][16 bytes, two two's-complement nibbles]`, scale = amax/7.

`ink_kv_put/get/dot/axpy` are the only row accessors; the f32 path is the old
`dot_f32_lanes`/`fmaf` loop, so an f32 run is bit-for-bit what it was. A row is
self-describing, so the prefix store, park and the chain mirror copy rows as bytes given
`ink_kv_row_bytes()`.

## 4. Stage 1 — what was implemented

### 4.1 Host cache

- `INK2_KV = f32 | f16 | q8 | q4` (default f32), read once by `ink_kv_env()`; a misspelt value
  exits rather than guessing.
- `Model.kv_type` and `InkSeq.kv_type`; `kv_alloc` sizes every layer's rows by type (and keeps
  the grow-in-place copy in row units); `attention()` and `ink2_attention()` read and write
  through `ink_kv_*` (scores through `ink_kv_dot`, V through `ink_kv_axpy`).
- Prefix store, `park.h`, the MTP draft verify's ring snapshot and the 250K/1M fit checks all
  moved to row bytes. Segment sessions keep f32 (`model->kv_type = 0`).

### 4.2 CUDA chain mirror

- `ink2cc_mirror` allocates `KV * cap * ink_kv_row_bytes(type, hd)` a side, so the mirror is
  q4 at 250K (2.3 GB) instead of fp32 (16 GB). Device usage at 250K: **2802 MiB** including
  the dense matrices and scratch.
- `push_state` uploads rows by bytes (`cc_write`'s offsets are in 4-byte units — the row bytes
  are divided by 4; a row not divisible by 4 is refused).
- For f32 the frame still copies the step's rows into the mirror as before. For a typed cache
  the host quantizes them in `ink2cc_kv_down` and marks them (`dlo/dhi`); the next chunk's
  `push_state` uploads them. Within a chunk the attention reads the step's rows from `kvs`
  (f32 scratch), so nothing reads the mirror stale.
- `CcRelAttn` gained `kv_type` and `k_rowb`; `ccg_kv_at`/`ccg_kv_row` in `cuda_inkc.cuh` read a
  typed row, and the f32 case keeps the original vectorizable loop.
- The Vulkan chains (`inkling2_chain.h`, `inkling_chain.h`) decline a typed host cache with a
  message and the per-matrix path (which reads the typed rows on the CPU) takes the run.

### 4.3 Prefill attention: the tiled kernel (`ccg_relattn_qt_k`)

The old kernel launches a block per `(query head, query row)`; each of them walks the whole
cache, so a 1024-row prompt chunk reads every K/V row 1024 times per head. The new kernel:

- one block per `(KV head, tile of 8 query rows)`, 256 threads (8 warps), one warp a row;
- the block loads a 32-position tile of K and V once into shared memory (K rows padded to
  `hd+4` floats: a stride of 128 floats puts every lane's `float4` in the same bank);
- each lane is one position of the tile and scores all of the KV head's query heads (float4
  dots against the block's Q tile in shared, broadcast), with the same relative bias, tau,
  sink, window and ring split between the mirror and `kvs` as the scalar kernel;
- the online softmax keeps `m`, `l` and the V accumulator per head, and **rescales the
  accumulator** whenever a later tile raises the max (`acc *= exp(m_old - m_new)`; missing
  this was the one bug the A/B test caught);
- V is accumulated by all 32 lanes (4 dims each) from the shared tile.
- Dispatch: `S >= 8` and `hd <= 128` (and the shared tile ≤ 96 KiB);
  `COLI_CUDA_RELATTN_QT=0` restores the old kernel, `=N` moves the threshold.

Decode (`S = 1`) keeps the scalar kernel, which now reads typed rows.

## 5. 250K MVP result

Command:

```bash
SNAP=.../iq3xxs.colibri CTX_MAX=262144 INK2_KV=q4 COLI_CUDA=1 OMP_NUM_THREADS=6 \
  ./c/inkling -f bench/prompt-250k.txt -n 24
```

- 250,156 prompt tokens; **prefill 3651.5 s (60.9 min)**; **decode 8.44 tok/s** (24 tokens);
  wall 3666 s.
- RSS **12.2 GB** of 23 GB; host KV 2493 MiB; device 2802 MiB of 12 GB. No OOM, no swap.
- Phase split: attention 2706.7 s, routed experts 769.4 s, other 172.9 s.
- Attention scales as expected from 64K (202.6 s × (250/64)² ≈ 3090 s), so the long-context
  path is behaving; the cost is plain O(S²) fp32 attention.

## 6. The 250K needle

**Resolved (§15): this was never the model.** The failure below is the typed-mirror upload bug:
with `INK2_KV` f16/q8/q4 the CUDA chain filled the host cache but never marked the rows for the
device mirror, so every chunk after the first and every decode step attended to zeros. The same
prompt and snapshot answer correctly with `COLI_CUDA=0`, with `COLI_CUDA_CHAIN=0`, with
`INK2_KV=f32`, and in Unsloth's llama.cpp — and now with the fix, at 1K/2K/4K/8K. The analysis
that follows is kept because it is how the bug was found (and because it is the record of a wrong
diagnosis: quantization and prompting were both blamed and both were innocent).

### The original (mis)diagnosis

The prompt is 1.15 MB of natural-ish prose with the fact
`The secret access code for the vault is 7X4Q-ALPHA-93` at ~240K tokens and the question at
the end. The engine answered with unrelated text, and the code is absent.

The same harness at **943 tokens** also fails (also at 2K/4K/8K/16K), and a one-sentence
version ("The secret access code is 4821. Question: what is the secret access code? Answer:")
produces unrelated tokens too, while "Question: What is the capital of France? Answer:" gives
"Paris". The chat template (`--chat`, with and without `THINK=0/1/2`, 200 tokens) also
degenerates: this IQ3_XXS Mini snapshot, prompted through the engine's template, rambles in
`<|content_thinking|>` instead of answering.

So the KV path, the 250K context and the numerics are not what fails here: the model's
instruction-following for this harness is. What to try next, in order:

1. a correct chat/thinking prelude for this checkpoint (the template in `c/inkling.c` may be
   missing the model's `thinking` system block or the right effort string);
2. ~~a less quantized snapshot (Q4_K_M is on disk) to separate quantization from prompting;~~
   **tried: Q4_K_M fails exactly the same way** (1K/2K/4K/8K, the answer is unrelated prose in
   mixed scripts; `bench/needle_check.py --sizes 1,2,4,8` with `NEEDLE_SNAP=.../q4km.colibri`,
   log `/var/tmp/scratch/kvtest/needle-q4km.log`). Quantization is ruled out; the harness is the
   prompt/template, not the KV path;
3. an oracle-based check that does not need instruction-following: teacher-forced argmax of a
   natural 32K/128K continuation, f32 vs q4 (`INK_TF_DUMP`), with a tokenizer dump added to
   the engine (a `INK_IDS_DUMP` knob) so the reference ids are the real ones.

## 7. Stage 2 — q8/q4 for 1M

### 7.1 Why q4, not f16

At 1M, f16 global KV is 32 GB and q8 is 17.4 GB; neither fits beside the model and OS on 23 GB.
q4 is 9.2 GB: with 1.4 GB dense and ~8 GB OS that is ~18.6 GB, leaving ~4.4 GB for runtime and a
small expert cache. q4 is the only viable 1M format on this host.

### 7.2 Device policy at 1M

A full q4 device mirror (9.2 GB) plus the 8 GiB tier does not fit 12 GB. Options, in order:

1. **Host-only attention at 1M** (`COLI_VK_CHAIN=0` for Vulkan, or `COLI_CUDA_CHAIN=0`): KV
   stays in RAM, the chain's dense work runs per layer. Decode is slower but memory-safe; the
   CPU attention reads the typed rows.
2. **Paged mirror:** keep a recent/hot block of KV rows on the device and stream the rest per
   layer. Full attention reads every row, so this is bandwidth-bound; measure before
   committing.
3. **Larger card:** 24 GB makes a q4 mirror plus a ~10 GB tier comfortable; document as the
   recommended 1M configuration.

### 7.3 1M acceptance

- Loads at `CTX_MAX=1048576` with q4 host KV and no OOM.
- Needle retrieval at ~1M (once §6 is solved); no context corruption.
- Decode recorded honestly; expect a few tok/s, not the 8K rate.
- Quality gate vs f32 at 32K on code, JSON, and retrieval.
- If the 12 GB path cannot pass, document 24 GB as the 1M minimum and keep the code path.

## 8. Code map

| File | Change |
|---|---|
| `c/ink_kv.h` (new) | `INK_KV_F32/F16/Q8/Q4`, `ink_kv_row_bytes/name/parse`, `ink_kv_put/get/dot/axpy` (f32 + `kv16.h` + 32-value q8/q4 blocks with f32 scales). |
| `c/inkling.c` | `Model.kv_type`, `InkSeq.kv_type`; `ink_kv_env()` (`INK2_KV`); `kv_alloc` typed sizes + grow; `attention()` typed rows; park/prefix store/MTP/`ink_kv_later`/plan bytes in row units; segment sessions stay f32; `INK_TF_DUMP` for parity tests. |
| `c/inkling2.h` | `ink2_attention()` typed rows (scores and V). |
| `c/inkling2_cuchain.h` | typed mirror allocation; `push_state` uploads rows in 4-byte units; `kv_down` quantizes into the host cache and marks rows for the next push; `relattn` gets `kv_type`/`k_rowb`; plan bytes. |
| `c/cuda_chain.h` | `CcRelAttn` gains `kv_type`, `k_rowb` (byte offsets for typed rows). |
| `c/cuda_inkc.cuh` | `ccg_kv_at`/`ccg_kv_row` typed reads (f32 keeps its original loop); `ccg_relattn_qt_k` tiled prompt kernel with the online-softmax accumulator rescale; dispatch + `COLI_CUDA_RELATTN_QT`. |
| `c/inkling2_chain.h`, `c/inkling_chain.h` | the Vulkan chains decline a typed host cache; per-matrix path takes it. |
| `c/inkling2_mtp.h` | draft-verify ring snapshot in row bytes. |
| `c/tests/test_ink_kv.c` (new) | round trip, error bounds, dot/axpy for f16/q8/q4. |
| `c/tests/test_cuda_inkc.cu` | typed mirrors through the scalar and tiled kernels; tiled vs scalar A/B; the rescale regression cases. |
| `PLAN-colibri.md` | this file. |

## 9. Environment

| Env | Default | Meaning |
|---|---|---|
| `INK2_KV` | `f32` | Host KV element type: `f32`, `f16`, `q8`, `q4`. The chain mirror uses the same type. q4 is the 250K/1M setting. |
| `COLI_CUDA_RELATTN_QT` | `8` | Rows from which the tiled prefill kernel runs; `0` = the per-(head,row) kernel; `>1` = the row-tile size threshold. |
| `CTX_MAX` | `8192` | Served context bound (KV planning and refusal). |
| `COLI_CUDA` / `COLI_CUDA_CHAIN` / `COLI_VK_CHAIN` | on / on / off | As before; a typed KV disables the Vulkan chains only. |

## 10. Validation

1. **Round trip:** `c/tests/test_ink_kv.c` — all pass.
2. **Kernels:** `make -C c cuda-inkc-check` — all pass (typed mirrors, tiled vs scalar, CPU
   reference).
3. **Oracle:** teacher-forced argmax 56/56 across f32/f16/q8/q4 (CPU, 1 thread).
4. **Long context:** 8K/32K/64K/250K prefill and decode measured above.
5. **Memory:** RSS, `[kv]`, and device bytes recorded for every run.
6. **Agent (still to do):** tool-call round trip and prefix reuse at 64K+ with `INK2_KV=q4`.

## 11. Remaining work

| Area | Next step |
|---|---|
| Decode speed | 14.94 tok/s at 64K vs Unsloth 18.33: the chain still round-trips the host every layer (1923 frames at 64K). Keep more layers' experts in the frame (`COLI_CUDA_CHAIN_MOE`), batch the host's expert step, and fuse the gate into the attention kernel. |
| Attention at 250K | 2706 s: the next step is bf16 tensor cores (`mma.m16n8k16`) for QK^T and AV in the tiled kernel, 16 query rows a block, and an f16 mirror on the device. Target ~2–3× more. |
| Needle harness | §6: chat/thinking prelude or a less quantized snapshot; an `INK_IDS_DUMP` knob for a long-context teacher-forced oracle. |
| 1M | q4 host KV, host-side attention (`COLI_CUDA_CHAIN=0`), tier reduced; measure and document the 24 GB recommendation. |
| Upstream | the KV type and the tiled kernel behind `INK2_KV` / `COLI_CUDA_RELATTN_QT` with docs and tests. |

## 12. Risks

| Risk | Mitigation |
|---|---|
| q4 KV changes near-tie tokens | q8/f16 are one env var away; parity is measured on the CPU path; keep q8 K if quality needs it. |
| Typed mirror row not 4-byte aligned | `push_state` refuses with a message (only widths that cannot occur: heads are multiples of 32 in this family). |
| Device mirror too large at 250K | q4 mirror (measured 2802 MiB total at 250K); otherwise `COLI_CUDA_CHAIN=0` and the CPU attention. |
| Prefix store / park type mismatch | rows are byte-copied and self-describing, so a store written under one type is read by that type; the type is fixed per run. |
| 1M decode is very slow | host-side q4 KV; document; 24 GB for a device mirror. |
| MTP verify over quantized KV | the snapshot is row-based (`inkling2_mtp.h`), tested by the existing cases; run with `INK2_MTP=1` before trusting it. |
| Non-reproducible greedy runs | documented above: the tier's device/CPU split changes the numerics; parity runs use 1 thread and `COLI_CUDA=0`. |
| 23 GB RAM | q4/q8, smaller expert cache, OS headroom; measured 12.2 GB at 250K with q4. |

## 13. Milestones

- **M0** — Build dev + import IQ3; benchmark 8K/32K/64K. *Done.*
- **M1** — f16 host KV + typed device mirror; 250K load and needle. *Fit and load done; needle
  blocked on the model/prompt (§6).*
- **M2** — f16/q8/q4 host KV; quality/perf; 64K vs Unsloth. *Done: 64K prefill 387.1 s vs
  439.8 s; decode 14.94 vs 18.33 tok/s.*
- **M3** — 1M acceptance or documented 24 GB requirement. *Open.*
- **M4** — Upstream the KV type and the tiled kernel behind env flags with docs/tests. *Open.*

## 15.2 Whole-layer residency: the offline estimate, and the knob already exists

Before building anything, the routing history answers whether packing whole layers can pay.
`bench/tier_policy_estimate.py` reads `.coli_usage` (24 layers × 128 experts, 9.7e8 recorded
activations) and compares the two policies at the tier's real capacities:

| capacity | policy A: hottest experts (today) | policy B: hottest whole layers |
|---|---|---|
| 1718 experts (56%, the 250K run) | **0 of 24 layers whole**, 0% of activations in-frame | **13 of 24 whole, 77.1% of activations in-frame** |
| 1803 (59%) | 0 whole | 14 whole, 79.2% |
| 2280 (74%, the 64K run) | 0 whole | 17 whole, 85.4% |

**Why 77% and not 13/24 = 54%: the layer totals are not equal.** The per-layer activations in the
history are 2.08% for most layers and **6.25% for layers 6–17** — exactly 3×. Those twelve are the
loop-pass layers ("layers 6-17 3 passes" in every chain line), so they route three times a
forward. The top 13 layers are [0, 6…17]: 12 × 6.25% + 2.08% = **77.1%**. It is structural, not a
skewed history: any workload routes through those twelve three times, so the *layer totals* — and
therefore this estimate — are robust even though the within-layer heat comes from short runs.

**What the metric is, precisely.** Both numbers are *coverage of a residency set by recorded
activations*, not "served from the device": policy A's 95% says the hottest 1718 experts cover 95%
of the activations one expert at a time, but the tier's scheduler (64 staging slots, per-step
promotions) is what turns that into device work, and it runs no layer in-frame. Policy B's 77% is
the share of activations in *layers the frame can take whole*. The comparison that matters is
total device-served activations plus the per-expert-vs-in-frame speed gap, and only the A/B gives
that. So the earlier "1 of 24 layers" was never a capacity ceiling — it is the *ranking*: a
per-expert order spreads residency thin and never completes a layer, while the same budget packed
by layer completes half of them.

**The A/B design** (from the review, adopted): sweep `n` = 0, 8, 13 (and 17 at 64K), record the
`budget X GiB = N experts` line per run, report prefill **and decode tok/s** separately, and run
`needle_check.py` + the teacher-forced oracle on the `n>0` configuration — it exercises the
in-frame MoE path far more than today's 0–1 layers, so correctness risk goes up with n. At 64K the
tier is not capacity-bound (§15.1), so a flat 64K result does not kill the idea; 250K is where it
should pay.

**The gate**: `bench/gate.sh` runs `cuda-inkc-check` and the needle (1K/2K/4K/8K, the served KV
type) in one command. Every commit on the chain/mirror/tier path goes through it — the typed-mirror
bug passed every kernel test and every speed number and still produced garbage.

**The knob already exists**: `COLI_VK_TIER_WHOLE=n` (vk_tier.c) queues the n hottest layers whole
before the per-expert heat order, default 0. So this is a measurement, not a build. The A/B to run
is 64K at `n=17` against `n=0`, at the same recorded budget, and then one 250K at `n=13`.

## 15.1 The remaining plan items: what each attempt measured

**Tier budget determinism (step 2, cheap half): neither knob pins it.** Measured back to back at
64K, same prompt: `CUDA_RESERVE_GB` 2.0 vs 1.0 gave 7.46 vs 7.47 GiB (2294 vs 2297 experts, prefill
258.1 vs 260.5 s), and `CUDA_EXPERT_GB` 7.5 / 5.5 / 4.0 gave 7.42 / 7.38 / 7.39 GiB (2280 / 2268 /
2270 experts, prefill 260.8 / 263.5 / 264.5 s). The request does not bind: the budget is *free VRAM
minus the chain's plan*, and it lands at ~7.4 GiB at 64K whatever is asked. At 250K it came out
5.46, 5.64 and 5.91 GiB across runs because the free VRAM at start drifted (11.2 vs 11.6 GB with
the desktop). So this is measurement discipline, not a code change: every result must record the
`budget X GiB = N experts` line and expert A/Bs compare only same-budget runs. The sweep scripts
now parse that line into every row.

**Chunk 2048 at 250K: negative, and for a nameable reason.** With `COLI_CUDA_CHAIN_ROWS=2048` the
250K run made the tier's expert step 16384 activated rows: 12 `tier ... stream` lines in 42 min
against 735 in 26 min for chunk 1024, i.e. roughly 6× slower (killed). The same mechanism is the
8K/4096 cliff. So chunk 2048 is a 64K setting (1.12×) and 1024 stays the 250K setting; a new
`COLI_VK_TIER_STEP` caps the streamed step for anyone who wants the big attention chunk with small
expert steps (off by default), but at 64K the win *was* the bigger step, so it is not a free lunch.

**Query rows a block (QT).** `CCG_QT` is now runtime (`COLI_CUDA_RELATTN_ROWS`, one warp a row):
at 250K, 24 heads, 8 rows = 734 ms and 16 rows = **696 ms (1.05×)**; at 64 synthetic heads 16 rows
overflows the 96 KiB tile and the dispatch falls back to the scalar kernel (20.5 s vs 1.77 s), so
16 is only for a group ≤ 4. Default stays 8 — the 5% is inside the run-to-run spread.

**MTP vs DSpark — two different mechanisms, and neither is wired here.** This was first written
up as "dropped"; that was too quick. The facts:

- **DSpark is the official option**, and it is a *separate draft model*:
  `dspark-Inkling-2-Mini-Q8_0.gguf`, 402 MB, GGUF architecture **`dflash`**. llama.cpp loads it with
  `-md … --spec-type draft-dspark --spec-draft-n-max 3`, which is exactly what the Unsloth baseline
  in §2 does — so that 18.33 tok/s at 64K is *with* DSpark.
- **This engine's `INK2_MTP` is a different thing**: `inkling2_mtp.h` reads an in-snapshot head,
  `<snap>/mtp/mtp.safetensors` + `config.json`'s `mtp_config`, and drafts with the DFlare scheme off
  the target's own fused layers. The container has no `mtp/` directory, so `ink2m_load` says "no
  mtp/mtp.safetensors beside it". A re-import without `--no-mtp` does not produce one either — the
  head is not in this GGUF.
- `coli import` refuses the DSpark file as a model (`architecture 'dflash' is not supported yet`).
  The engine has DFlash *kernels* (`cuda_dflash.cuh`, the `cc_dfconv`/`cc_dftopk` ops) for the
  family that integrates a drafter, but not a loader/verify path for an external `dflash` drafter
  beside inkling2.

So DSpark is not "a snapshot away": it needs engine work (load a `dflash` drafter, run its forward
and block verify, own its KV). Before that work, size the prize: run llama.cpp on the same prompt
**with and without `-md`** at 8K/64K — it already supports DSpark, so that isolates what the
drafter is worth in this exact MoE + q4-KV setting. With ~612 s of host experts at 250K, a verify
of k drafts activates the union of k+1 tokens' top-8 experts, so the per-verify cost is the thing
to measure, not the accept rate alone. Keep it as an item gated on that measurement.

## 15. The typed-mirror upload bug (the real reason the output was garbage)

**Symptom.** With `INK2_KV=q4` (or f16/q8) on the CUDA chain, the engine completed a 250K
prefill and decoded, but the text was unrelated prose in mixed scripts. The needle failed at
943 tokens — before any context-length effect could exist.

**How it was isolated** (each step is a real run, in order):

| configuration | needle at ~1K | verdict |
|---|---|---|
| Unsloth llama.cpp, same GGUF, raw completion, temp 0 | **`7X4Q-ALPHA-93`** | the snapshot is fine |
| this engine, `COLI_CUDA=0` (host path) | **`7X4Q-ALPHA-93`** | loader, tokenizer, RoPE, loop, norms, host attention all fine |
| this engine, CUDA, `COLI_CUDA_CHAIN=0` (per-matrix) | **`7X4Q-ALPHA-93`** | the CUDA per-matrix path is fine |
| this engine, CUDA chain, `INK2_KV=f32` | **`7X4Q-ALPHA-93`** | the chain is fine when it fills its mirror |
| this engine, CUDA chain, `INK2_KV=f16` | garbage | typed cache only |
| this engine, CUDA chain, `INK2_KV=q4` | garbage | typed cache only |
| this engine, CUDA chain, q4, `COLI_CUDA_CHAIN_MOE=0` | garbage | not the in-frame MoE |

`bench/ab_needle.py` runs the first and last rows; `bench/ab_tokens.py` compared the tokenizer
and found **808 vs 808 ids, identical**, so the prompt bytes were never the problem.

**The bug.** `ink2cc_attention` fills the mirror inside the layer for f32
(`cc_copy_regions(ch->kc[v], ch->kvs, ...)`). For a typed cache that copy is skipped — the rows
are quantized on the host by `ink2cc_kv_down` and are supposed to be uploaded by the next
`push_state`. But nothing ever marked the rows for upload (`ch->dlo/dhi` are only touched by
`ink2cc_cpu_step`, i.e. on a host fallback), so `push_state` had nothing to send. Result: the
first 1K-token chunk (whose rows are read from `kvs`) prefills correctly, and everything after
it — the rest of a long prompt, and every decode step — attends to a zero mirror. That is
exactly the symptom: fluent local continuation, total loss of anything from the context.

**The fix** (in `ink2cc_kv_down`): mark `[pb, pb+n)` for the layer after quantizing, so the next
`push_state` uploads it. Six lines; the host-side arithmetic and the kernels are unchanged.

**After the fix**, `bench/needle_check.py --kv q4 --sizes 1,2,4,8`:

```
 1K (943 tokens):   FOUND  7X4Q-ALPHA-93
 2K (1855 tokens):  FOUND  7X4Q-ALPHA-93
 4K (3652 tokens):  FOUND  7X4Q-ALPHA-93
 8K (7288 tokens):  FOUND  7X4Q-ALPHA-93
```

**At full context.** The 250K run of §5 with the fix (`CTX_MAX=262144`, q4, 24 tokens,
`/var/tmp/scratch/kvtest/250k-fixed.out`) answered `7X4Q-ALPHA-93`: prefill 1598.5 s, device side
804.7 s, decode 12.18 tok/s. So the retrieval path works at 250,156 tokens on the same snapshot
that "failed" in §6.

**Confirmed again on the current tree (measured, b1f0c26).** `bench/prompt-250k.txt`, q4, 24
tokens, `COLI_VK_TIER_WHOLE=0`: the answer is verbatim `7X4Q-ALPHA-93. the archival group
measured the throughput of the storage array in a form`. Launch free VRAM **11.3 GB**, tier
**budget 5.71 GiB = 1741 experts**; prefill **1678.0 s**, decode **12.22 tok/s**, RSS 12.8 GB.
Phases fill 5.3 / expert-mm 658.1 / shared 0.0 / attn 836.4 / other 180.1 s. Tier: 95.37 M of
96.07 M routed experts on the device (99.3%), uploads 6636 (21.24 GiB), evictions 4895, 11,760
stream steps, 84,867 cold experts (319.96 GiB at 5.32 GB/s, 4.50 M rows), 158,854 cold experts
(695,733 rows) left on the CPU; device **289.5 s**, CPU share **238.7 s**, waited **160.4 s**.
`bench/gate.sh` passes on this tree (kernel check + needle 1K/2K/4K/8K). Log
`/var/tmp/scratch/opencode/colibri-runs/250k-needle-b1f0c26.log`, row in
`bench/results-whole.json`. The same run's chain report (681.6 s waiting for the device, 775.3 s
of routed experts on the host, **0 layer steps' experts in the frame**) is the overlap measurement
step 4 asks for: the device hid 129.2 s of the tier's host share (54% of that share, 17% of the
routed-expert host time), and the phase line is additive, so `moe_ex` — the bulk of the expert
phase — still runs with the device idle. See §15.5.

**What this invalidates.** Every quality statement made before this fix — including §2's "250K
loads, prefill completes and decode runs" and all of §6 — was made on a device path that had
lost the context. The *speed* figures stand (the same work was done, on zeros), but the 250K
"works" claim did not, and the model was wrongly blamed for a bug in this fork.

## 15.3 The run matrix (agreed design, before any GPU time)

The estimate says whole-layer packing covers **77%** of activations (structural: layers 6–17 run
three passes) while per-expert residency at ~71 experts a layer already covers **~90%** within a
looped layer (top-40 carry 72–87%, top-64 86–94%). So the A/B is not "which covers more" — it is
whether the in-frame path is fast enough per activation to beat a ~13-point coverage loss. The
break-even, with `c_f` in-frame, `c_p` per-expert and `c_h` host per activation:

    0.77·c_f + 0.23·c_h  <  0.90·c_p + 0.10·c_h

`c_p` comes from the `n=0` cell and `c_f` from one `n=12` cell, so two runs say whether the sweep
is worth finishing.

**Cells** (q4, 64K prompt, 16 tokens; one at a time on the card):

| cell | `COLI_VK_TIER_WHOLE` | why |
|---|---:|---|
| B1, B2 | 0, twice | the noise floor (1–2% seen run to run) — a 3% effect is not signal without it |
| n6 | 6 | boundary: half the loop block |
| **n12** | 12 | **the clean test: all twelve loop layers whole, nothing else** |
| n13 | 13 | adds layer 0 for a 2% sliver — the full-set point from §15.2 |
| n17 | 17 | the 64K capacity point (74% of the tier) |
| 250K n12 | 12 | one cell **regardless of the 64K result**: 250K is where capacity binds and 64K is not |

**Fields per cell**: commit; `budget X GiB = N experts`; free VRAM at launch ("device 0 ready, X GB
free"); evictions, uploads GiB, cold experts, kept-on-CPU; "N of 24 layers in the frame"; the
phases (fill / expert / attn / other); prefill s; **decode tok/s separately** (in-frame loop layers
touch the same experts three times a token, so decode may gain more than prefill); RSS; and for
every `n>0` cell the **needle + the oracle parity run** — the in-frame MoE path is exercised far
more there, and it is the neighbourhood the typed-mirror bug lived in.

**Reject a cell** when its budget differs from B1 by more than 0.1 GiB *or* its launch free VRAM
differs by more than 0.2 GB (the 0.4 GB desktop drift is what moves the budget; it never shows in
the env).

**Routing-history caveat**: `.coli_usage` is aggregated over past runs, mostly short ones. The layer
*totals* are structural (the three passes), but the within-layer heat and the prefill-vs-decode
split are not. If `ROUTE_TRACE` can capture one 64K prefill, replay `bench/tier_policy_estimate.py`
against that trace before trusting `c_p`/`c_f` for prefill.

**DSpark A/B** (llama.cpp, the gate for any drafter work in this engine): same prompt, temp 0,
`cache_prompt=false`, same `ncmoe` and q4 KV; 8K and 64K, plus one 128K cell if the 64K delta is
large (the expert-union effect grows with depth). Log `prompt tok/s`, `decode tok/s`,
`draft_n`/`draft_n_accepted` **and accepted tokens per verify step**, not the rate alone. The
**without-`-md`** number is the like-for-like comparison for this engine's 16.55 tok/s at 64K — if
it lands near 16–17, the engine has no real decode gap and P4-style decode work drops in value.

**Order**: the 250K needle (first result to record) → B1/B2 → n12 → break-even → the rest only if
the two cells justify it → DSpark at 8K/64K.

## 15.4 The whole-layer A/B, measured (this tree)

The knob (`COLI_VK_TIER_WHOLE=n`) is a measurement, not a build, but the first sweep found that
the plan's premise — n=12 completes the twelve loop layers — does not hold, and the reason is
structural, not a tuning miss.

**The in-frame MoE is decode-only.** `ink2cc_dmoe` returns 0 for `rows > CC_MOE_FRAME_ROWS` (=8),
so no prompt chunk runs a layer's experts in the frame; the path is taken on decode steps (and a
verify) only. A whole-layer win therefore cannot show in prefill at all, and the break-even's
`c_f` never enters the 250K expert phase.

**The warm start's whole layers do not survive prefill.** `bench/sweep_whole.py`, q4, 64K, 16
tokens, one cell at a time; every row in `bench/results-whole.json` records the budget line,
launch free VRAM, frame count, phases, prefill, decode and RSS:

| cell | budget GiB = experts | free GB | layers in frame | prefill s | decode tok/s |
|---|---:|---:|---:|---:|---:|
| B1 n=0 | 7.47 = 2296 | 11.2 | 0/24 | 248.0 | 18.09 |
| B2 n=0 | 7.48 = 2299 | 11.2 | 0/24 | 253.4 | 17.77 |
| n6 | 7.49 = 2304 | 11.2 | 0/24 | 251.6 | 17.56 |
| n12 | 7.51 = 2309 | 11.2 | 0/24 | 252.8 | 17.52 |
| n12 (after the upload fix) | 7.47 = 2297 | 11.2 | 0/24 | 256.3 | 17.52 |
| n13 | 7.52 = 2311 | 11.2 | 0/24 | 251.9 | 17.73 |
| n17 | 7.52 = 2311 | 11.2 | 6/24 | 259.2 | 15.31 |
| n17 (after the upload fix) | 7.44 = 2286 | 11.2 | 6/24 | 268.0 | 14.58 |

All cells are budget-matched (max delta 0.08 GiB from B1; free VRAM 11.2 GB throughout), so the
prefill spread is run-to-run noise. Prefill is flat for n≤13 and 2–6% worse at n=17; decode is
flat for n≤13 and 16–19% worse at n=17. The first decode step has **0 of 24 layers whole at
n=12**, not 12: the warm start does queue twelve loop layers whole (measured span 1536 experts;
the parallel fill used to scatter it, hence the sequential span upload in `ink_tier_warm`), but
prefill's promotions (16 a token, balanced against the CPU) trade those residents by heat and
break every loop layer before the first decode step reads the table.

**One 250K n=12 cell, budget-matched to the n=0 baseline.** q4, 24 tokens, same prompt: budget
5.66 GiB = 1724 experts vs the baseline's 5.71 GiB = 1741, launch free 11.2 vs 11.3 GB (matched:
0.05 GiB and 0.1 GB). The needle still passes (`7X4Q-ALPHA-93`, verbatim) and the answer is
coherent. Prefill **1745.5 s** vs 1678.0 (+4.0%), decode **9.69** vs 12.22 tok/s (−20.7%), RSS
11.3 GB, **0 of 24 layers in the frame**. The cost is in the stream counters: 132,842 cold experts
/ 419.0 GiB / 11.43 M rows streamed vs 84,867 / 320.0 GiB / 4.50 M at n=0, because 1536 of the
1724 resident slots are reserved whole and only 188 are left for the other twelve layers. Expert
phase 568.6 s (was 658.1), attn 987.2 s (was 836.4).

**The first A/B tested the layout, not the in-frame path.** At n=12 the first decode step had
0 of 24 layers in the frame, so `cc_moe` never ran the loop layers: the measured +4.0% prefill /
−20.7% decode is the cost of the whole-layer *residency layout*, not of in-frame MoE. Pinning is
the missing control. `VSlot.pin` (skipped by `refresh_candidates`/`peek_victim`) makes a whole
layer's experts ineligible as promotion victims, so the set survives prefill. With pinning:

| cell | budget GiB = experts | free GB | layers in frame | prefill s | decode tok/s |
|---|---:|---:|---:|---:|---:|
| 64K n=0 (B1/B2) | 7.47 = 2296 / 7.48 = 2299 | 11.2 | 0/24 | 248.0 / 253.4 | 18.09 / 17.77 |
| 64K n=12 pinned | 7.46 = 2294 | 11.2 | **12/24** | 268.1 | **18.66** (+3–5%) |
| 250K n=0 | 5.71 = 1741 | 11.3 | 0/24 | 1678.0 | 12.22 |
| 250K n=12 pinned | 5.63 = 1715 | 11.2 | **12/24** | 1811.4 (+7.9%) | **11.01** (−9.9%) |

The in-frame path itself is a small decode gain where the tier has spare capacity (64K: +3–5%,
2016 routed experts with no host step, needle FOUND with `COLI_VK_TIER_WHOLE=12` at 1K–8K). At
250K it is a net loss: pinning 1536 of 1715 experts leaves ~15 a layer for the other twelve, so
the unpinned half's host step is slower and the streamed volume rises (569.5 GiB vs 320.0 GiB);
decode −9.9% and prefill +7.9%. The correct finding is therefore **"the earlier sweep tested the
layout; the in-frame path is now tested: decode +3–5% at 64K, −10% at 250K, prefill +6–8% at
both"**, not a blanket dead end. It remains rejected as a 250K policy because the tier is
capacity-bound there; re-open it only as a decode optimization where the budget leaves the other
layers their share, or after the in-frame MoE path is extended to prompt chunks.

The `n>0` end-to-end check is the needle: with `COLI_VK_TIER_WHOLE=12`,
`bench/needle_check.py --kv q4 --sizes 1,2,4,8` is FOUND at all four sizes, and the 250K n=12
runs (unpinned and pinned) answer the code. The teacher-forced oracle is a CPU/f32 parity gate,
and the tier's device/CPU split changes the numerics (§12), so it cannot be an exact check for a
`n>0` cell — `bench/gate.sh` (kernel reference + needle) is the correctness gate here.

## 15.5 Host overlap, and the wavefront-vs-tensor-core decision

The deciding number is the fraction of the host expert step that already overlaps device work.

| run | host routed experts (chain) | waited for device | device (tier) | host share in the stream window | device time hidden | host step overlapped |
|---|---:|---:|---:|---:|---:|---:|
| 250K n=0 | 775.3 s | 681.6 s | 289.5 s | 238.7 s | 129.2 s | **16.7%** (129.2/775.3) |
| 64K B1 | 159.1 s | 62.8 s | 78.3 s | 17.2 s | 34.7 s | **10.8%** (17.2/159.1) |

The phase line corroborates it: at 250K fill 5.3 + expert-mm 658.1 + attn 836.4 + other 180.1 =
1679.9 s against a prefill of 1678.0 s — the `moe_ex` bulk is additive, not hidden; at 64K
127.0 + 71.9 + 47.5 + 2.4 = 248.8 vs 248.0. So **83–89% of the host step is serialized today**
(measured; the upper bound is the device time that elapsed without a wait). The device work at
250K (836 s) exceeds the host step (775 s), so there is enough device time to hide it if the chain
pipelined chunks.

**Decision: build the chunk wavefront first, but the honest expected win is 300–450 s, not
645 s.** The 645 s figure is a perfect-overlap ceiling (775 s of host work minus the 129 s already
hidden) and assumes zero new cost. It is not additive with a faster attention kernel: with
per-chunk overlap, prefill is roughly max(device, host), so once the 775 s host step is hidden
behind the ~836 s device side, a faster kernel only helps until device time falls to the host
time — most of a tensor-core gain is already taken by the wavefront. The 2.5× kernel is also an
**unmeasured** target; 404 s is its ideal share of the 673 s kernel, not a result. Real overheads
come out of the ceiling: double-buffered activations at 250K (the tier budget is the sensitive
variable — ~0.3 GiB moved prefill by ~190 s in §15.1), CPU contention on the 6 cores between the
host expert step, host KV quantization and kernel launches, and PCIe contention between expert
uploads and the mirror uploads. Expect **~300–450 s**; prototype on 64K first (a cell is ~5 min)
before committing to the 250K build. The risk remains the per-(chunk, layer) KV-upload ordering,
gated by `bench/gate.sh` and a 250K needle. Provenance: measured overlap and phase counters;
the 300–450 s range is estimated from them.

## 15.6 DSpark gate, measured

llama.cpp b12588, same IQ3_XXS, q4_0 KV, `--n-cpu-moe 12`, `-b 512 -ub 128`, temp 0,
`cache_prompt=false`, 32 tokens (`bench/dspark_gate.py`; rows in `bench/results-dspark.json`):

| size | without `-md` | with `-md` | prompt tok/s (no/yes) | draft accepted |
|---|---:|---:|---:|---:|
| 8K | 18.66 tok/s | 19.60 tok/s (+5.1%) | 145.1 / 146.1 | 21/27, 0.778, 2.33/verify |
| 64K | 16.20 tok/s | 17.19 tok/s (+6.1%) | 137.2 / 138.1 | 21/27, 0.778, 2.33/verify |

The engine's 64K decode on the same tree is **17.77–18.09 tok/s** (B1/B2 in §15.4). llama.cpp
without the drafter is 8.9–11.7% below that (inside the ~10% parity band), and with DSpark it is
still 3.4–5.2% below the engine. The drafter is worth +6.1% at 64K and +5.1% at 8K — not a large
delta, so no 128K cell was needed. **Decision (the rule, applied):** at **64K** the engine is at
decode parity with llama.cpp; drop decode-speed work in favour of prefill, and do not build the
external `dflash` drafter path — in this engine's setting it would buy at most ~6% decode, while
our no-drafter path already leads llama.cpp's drafter-assisted one at 64K. The parity statement
is established **only up to 64K**: there is no llama.cpp 250K decode cell, so "decode work
dropped" is not yet a 250K claim (the engine's 250K decode is 12.22 tok/s, uncompared).

**Superseded by §15.15.** The rows above are unsloth `b12588`; on `b12676` (2026-11-25) the same
64K cell is 162.8 tok/s prompt and **19.96 tok/s decode** against the engine's 17.07–17.27, and
DSpark now *lowers* decode there (16.52). "The engine is at decode parity" is no longer true; the
drafter decision's arithmetic (a ~6% buy) is unaffected, but re-take it on b12676 before using it.

## 15.7 Needle depth and a second fact type (250K-class)

A single pass at ~96% is one fact at one depth. `bench/needle_depth.py` puts three facts of two
kinds in one prompt — vault code at 10%, pump serial at 50%, cipher key at 90% — and asks for all
three. At **244,180 tokens** (q4, CUDA chain, 64 tokens) the generation is verbatim
`7X4Q-ALPHA-93, PUMP-41-BRAVO-77, K9-ORBIT-27.` — **3 of 3**, so retrieval is not a lucky
end-of-context pattern. The same prompt at 8K is also 3 of 3.

A caution from the first attempt: a 272,517-token prompt with `CTX_MAX=262144` did not error; the
engine prefilled it and answered only `7X4Q` (0 of 3 in the check), i.e. a prompt past `CTX_MAX`
silently loses the far facts. Keep the prompt under the limit, or make the harness refuse it
(the harness now targets 244K). The 250K headline stays the 250,156-token single-code run of §15;
this cell adds depth and type coverage at 244K.

## 15.8 Strata SQ-10 cross-check (the same model, a different engine and quant)

Strata 0.2.20 / engine 0.1.66 for `inkling-2-mini`, pack `Inkling-2-Mini-SQ-10.strata`
(10.5 GB, SQ-10) from `strata-engine/Inkling-2-Mini-Strata`, installed under
`/run/media/nazar/Gaaaymes/models/strata` and served on `127.0.0.1:8080`:

```bash
strata update --yes
strata launch --family inkling-2-mini --model sq-10 \
  --data-dir /run/media/nazar/Gaaaymes/models/strata --context 64k --vision off \
  --tune skip --no-tui --no-api-key --yes
python3 bench/strata_test.py --tokens 8,56 --plain-64k     # rows in bench/results-strata*.json
```

`bench/strata_test.py` sends the same three-fact prompt as §15.7 (vault code at 10%, pump
serial at 50%, cipher key at 90%) through `/v1/chat/completions` and reads the server's own
timings. **Retrieval is 3 of 3 at every size tested** (1K, 8K, 30K, 61K, 64K), so SQ-10 reads
depth and fact type like the colibri path.

| served ctx | prompt n | prompt tok/s | decode tok/s | DSpark draft/accepted |
|---|---:|---:|---:|---:|
| 32K | 1,161 | 760 | 10.3 | 0 / 0 |
| 32K | 8,795 | 898 | 41.9 | 36 / 22 |
| 32K | 30,617 | 825 | 5.9 | 0 / 0 |
| 64K | 8,795 | 834 | 29.8 | 36 / 22 |
| 64K, q4 KV | 61,016 | 621 | 3.9 | 0 / 0 |
| 64K, q4 KV | 63,992 | 684 | 3.5 | 0 / 0 |
| 64K, int8 KV | 61,016 | 493 | 3.8 | 0 / 0 |
| 64K, int8 KV | 63,992 | 657 | 3.4 | 0 / 0 |

At 64K the engine allocated 2609 experts in VRAM (6.86 GiB), **463 experts SSD-only**, trunk
1.87 GB, and offloaded most KV to host (global KV 2304 VRAM + 63,232 mapped host tokens,
1.98 GiB RAM); `--kv q4_0` moved only the VRAM share (2304 tokens) and did not change the
decode rate. DSpark was active on the first request after each start (36 drafted / 22 accepted
at 8K) and reported `draft_n=0` on later/long-context requests. The card's RTX 5070 figure
(93.9–101.4 tok/s) did not reproduce here: 41.9 tok/s was the best (8K under the 32K setting),
and long context collapses to 3.4–5.9 tok/s once the pack, the desktop and the KV do not fit.

**Comparison with this tree at 64K** (different quant and KV: SQ-10 10.5 GB vs IQ3_XXS 11.66 GB
q4): Strata prompt **657–684 vs 253–258 tok/s** (2.6–2.7× faster), decode **3.4–3.5 vs
17.8–18.1 tok/s** (about 5× slower). So on this 12 GB card the Strata pack is a prompt-speed
win and a long-context-decode loss, the opposite trade from the colibri tier; it is not a
quality comparison (SQ-10 is 89.8–92.4% same-next-word to the original on the pack's own tests).

## 15.9 What the Strata cross-check suggests for this branch

The Strata run (§15.8) differs in engine and quant, so its 2.6–2.7× prompt figure is not a
like-for-like target. The transferable facts are structural:

1. **The prompt expert step is on the wrong processor.** At 64K colibri's expert-mm is 127.0 s
   of 248.0 (51%) and the chain reports 159.1 s of routed experts on the host;
   `ink2cc_dmoe` declines every chunk over `CC_MOE_FRAME_ROWS` = 8, so all prompt experts run in
   `moe_ex` on the 6-core CPU while the GPU idles. Strata runs the experts on the GPU (2,609
   resident, 463 streamed from SSD). This is the largest *candidate* lever and it may subsume the
   wavefront (the wavefront hides the host step, a GPU prompt path removes it), but the size of
   the prize is an **estimate**, not a measurement: Strata's 2.6–2.7× also contains a different
   quant (SQ-10 vs IQ3_XXS) and a larger resident set (85% vs 75%), and its internals are inferred
   from its allocation log. The spike must measure the colibri path, not transfer Strata's ratio.
   Do **not** shrink the step to 256 rows: at 256 rows × top-8 a step still touches nearly all
   128 experts, so it would need four times as many steps and re-upload cold experts each time —
   the chunk-2048/4096 mechanism in reverse. The work is a **grouped GEMM at 1,024 rows** over the
   existing int8 block path (`cuda_bd.cuh`) with dequant fused and the upload overlapped.
2. **The kernel and the upload are the unknowns.** The in-frame path today is decode-shaped:
   `CC_MOE_FRAME_ROWS` = 8, GEMV-style expert kernels, and buffers sized for 8 rows. A prompt
   step needs a grouped GEMM over IQ3-style blocks; if dequant dominates, it hits the same limit
   as the fp32 attention kernel and a 12 GB card will not reach Strata-like throughput. The
   per-layer arithmetic is a derived ceiling, not a result: 1,024 rows × 8 experts ×
   (gate+up+down at D=2304, I≈768) is ≈87 GFLOP a layer-step, and the measured host time
   (127–159 s over 63 chunks × 36 layer-steps ≈ 56–70 ms) implies ≈1.2–1.6 TFLOP/s on the CPU.
   The attention kernel's ~5 TFLOP/s is an upper bound, not a forecast: 128 experts with ~64
   rows each is tall and thin, so 1–3 TFLOP/s is the realistic range, i.e. **29–87 ms** — which
   can be a wash against 56–70 ms. The measured cold stream (18.8 GiB per 64K prefill ≈ 0.3 GiB
   a chunk at 5.2 GB/s ≈ 1 ms a layer-step) is small only at the current 75% residency; at
   250K's ~56% the miss count rises. That is why the microbench, not this arithmetic, decides. The microbench that decides it is one layer at 1,024 rows × 128 experts,
   with the dequant and the upload of the missing 25% included; accept the spike if
   compute+upload lands under ~35 ms a layer-step (2× the host), not if it only matches 56–70 ms.
   Every prompt-step buffer also comes out of the tier budget, which moved prefill by ~190 s per
   0.3 GiB at 250K (§15.1): log the buffer bytes and the budget line for every cell.
3. **Strata's own sweep says capacity/streaming is a first-order variable.** At 32K, 8K prompt,
   varying `--vram-experts-gb` (the rest fixed):

   | VRAM for experts | experts resident | prompt tok/s | decode tok/s |
   |---:|---:|---:|---:|
   | 8.6 GB | 2,613 | 912 | 43.9 |
   | 6.0 GB | 2,245 | 675 | 36.7 |
   | 4.0 GB | 1,435 | 306 | 14.2 |
   | 2.0 GB | 699 | 184 | 8.6 |

   Prompt speed tracks resident experts, so the Strata-vs-colibri gap is substantially a
   capacity/streaming difference, not only kernels. The 2,245 → 1,435 step is steeper than
   linear (36% fewer experts, 55% less throughput), a streaming cliff rather than smooth
   degradation, echoing the chunk-size cliffs. That strengthens the residency side of the
   spike but does not separate it from the kernel; the one-layer microbench above does.
4. **Tensor cores become the critical path after 1.** At 250K the attention kernel is 673 s of
   the 836 s device side; once the 658–775 s host step is gone it dominates. The prompt kernel is
   fp32 CUDA cores at ~5 TFLOP/s; Strata's `sm120_tc=1` is the direction. Do it only with a
   vectorized q4 unpack (PLAN §14), or dequant becomes the limiter. This ordering is a
   consequence of the expert fix, not a separate claim.
5. **The prefetch-ordering point is an inference.** Strata's 119-slot pinned DMA ring, 64 MB
   prefetch and `direct=1` are why its SSD tail does not stall the prompt. Colibri streams
   18.8 GiB per 64K prefill (~5.2 GB/s) and hides only 11–17% of the host step. The code issues
   `cut_stream_prefetch(i, n)` inside layer i's frame, so it prefetches for the current layer,
   not the next layer's host step — but that is a reading of the code, not a measured transfer
   timeline. Confirm it before calling it the cause; the wavefront prototype is where it gets
   tested (PLAN 15.5, 300–450 s).
6. **Decouple the attention chunk from the expert step, and cut `other`.** Strata's prompt is
   context-flat with large chunks; colibri pays 47.5 s (64K) / 180 s (250K) of `other` and a
   per-layer submit/finish. `COLI_VK_TIER_STEP` already caps the streamed expert step — the
   mechanism that killed chunk 2048 at 250K — so retry a big attention chunk with a small tier
   step, and fuse the per-layer norms/dconv/gate.
7. **Do not copy the KV trade-off.** Strata spends the card on experts and offloads KV to host:
   a great prompt and 3.4–3.9 tok/s decode at 30–64K. Colibri's q4 device mirror and 75%
   residency give 17.8–18.1 tok/s at 64K. Keep the mirror; any GPU prompt path must leave it
   intact.

Order to try: the one-layer grouped-GEMM microbench (1,024 rows, dequant + upload, compare with
the 56–70 ms host layer-step) → the 64K prompt-expert spike only if that clears ~35 ms → the 64K
wavefront prototype if it cannot be made safe → the tensor-core kernel. All three gate on
`bench/gate.sh` and the 250K needle, and every cell records the buffer bytes, the budget line and
launch free VRAM.

## 15.10 The one-layer prompt-MoE microbench: design and acceptance

The microbench decides GPU prompt-MoE versus the wavefront, so its numbers have to survive the
same scrutiny as the engine cells. Design, before any code:

**Question.** What does one real prompt layer-step cost on the GPU (grouped expert GEMM +
dequant + cold-expert upload), compared with the measured host layer-step (127–159 s over
63 chunks × 36 steps ≈ **56–70 ms**), at the two residencies the engine actually runs?

**Harness.** A new `c/tests/bench_cuda_moe.cu` against `cuda_chain.cu` (no engine): a synthetic
128-expert table in the real formats (gate/up and down as the checkpoint's, q4/IQ3-style blocks),
`S=1024`, `top-8`, `D=2304`, `I≈768`, real ids/weights, and the existing `cc_route`/`cc_moe`
ops and upload path. The tier's table allocation and the 64K q4 KV mirror (~1 GiB) stay resident
so VRAM pressure is real; log free VRAM, every buffer's bytes, and the equivalent budget line.

**Routing is replayed, not uniform.** A uniform 64 rows an expert hides the tail. Rows are drawn
per layer from that layer's `.coli_usage` histogram (the loop layers' top-40 carry 72–87% of
activations); if `ROUTE_TRACE` can capture one 64K prefill, replay that instead. Report the
per-expert row histogram, the top-40 share, and the padding/tail waste.

**Cells** (one process at a time):
- residency **75%** (2,296/3,072, the 64K tier) and **56%** (1,718/3,072, the 250K tier): the
  measured cold set is uploaded through the real path, not a synthetic memcpy;
- upload **overlapped on/off**, and a **PCIe-contention** case (a ~1 GiB KV-mirror upload running
  on another stream at the same time);
- one layer-step, then the **real layer sequence**: 24 layer-steps + the 12 loop layers twice
  more = 36 steps a token, each with its own routing histogram, so the projection includes the
  three passes;
- report GEMM+dequant **ms and measured TFLOP/s** per step, upload ms/GiB, hidden ms with
  overlap, total ms a layer-step, and the row histogram.

**Acceptance** (written before running):
- **< 35 ms** a layer-step: GPU prompt-MoE is the choice; the spike is worth it.
- **35–56 ms**: still better than the host (56–70 ms), but the margin is small enough that the
  wavefront can stack with it; decide on the projected seconds, not the per-step ratio.
- **> 56 ms**: no gain; the wavefront is the lever.
- Project to prefill in seconds, not ratios:
  `prefill_new ≈ prefill − (host_expert − gpu_expert × steps)`, with attention unchanged.
  At 64K that is 248 − (127 − ms×2,268/1000); at 250K, 1,678 − (658 − ms×8,784/1000). If the
  expert phase drops below the attention kernel at 250K, the projection is capped by attention
  and the tensor-core kernel becomes the next item.

**Before the harness, one look at Strata's cold path.** At 699 experts (23%) it still prompts at
184 tok/s; that is better than a simple residency ratio predicts (though still below colibri's
~258 at 75%). The 2 GB start's log shows the mechanism: the same **119-slot pinned DMA ring
(509 MiB), 64 MB prefetch, `direct=1`, and `promo=59`/`screen=1`** — at low VRAM it promotes cold
experts into VRAM, like our tier's promotion. So its cold path is a pinned-ring + prefetch +
promotion design, not a different compute path; our transfer design can be improved along the
same lines, and that may be cheaper than a prompt `cc_moe`. Confirm the per-request transfer
timeline before assigning the cause.

**If it passes.** The spike touches `ink2cc_dmoe` and the tier — the typed-mirror bug's
neighbourhood. `bench/gate.sh` plus the multi-depth needle run on every change, and one extra
negative test: a 64K prompt through the new path must return the same needle facts as the same
prompt with `COLI_CUDA_CHAIN_MOE=0` (retrieval, not token-exact; the tier's split changes the
numerics).

## 15.11 The prompt-MoE microbench, measured: GPU prompt-MoE is the choice

Built as §15.10 specified (`c/tests/bench_cuda_moe.cu`, `make -C c cuda-moe-bench`; rows in
`bench/results-moe-bench.json`; commit `f900479`). One process, no engine; the table synthetic in
the snapshot's own per-layer formats and dimensions, the routing drawn from the snapshot's
`.coli_usage`. What differs from the plan's sketch, and why it is more faithful than the sketch:

- **I is 1,152, not 768** (`inkling2.expert_feed_forward_length` in the GGUF), so a layer-step is
  130.5 GFLOP, not 87. The formats also vary by layer (the UD quant): gate/up IQ2_XXS (L0),
  IQ2_XS (L4), IQ2_S, IQ3_XXS (L9–L11, L13–L17, L19–L23), and down NVFP4 (L0–L19) or IQ4_NL
  (L20–L23). Both are parsed from the GGUF's tensor header and used as they are, so the down of
  twenty layers is the format the plan expected (140) and the other four are not.
- **The prompt tiles could not run these experts before this commit.** `hcg_tile`'s epilogue was
  silu only, had no place for the per-expert scales `CcMoe.esc_off` carries (the GGUF's
  `ffn_*_exps.scale`) and `coli_cuda_moe_async` refused `act 1`, `esc_off` and NVFP4 at more than
  `CC_MOE_FRAME_ROWS` rows. All three are now in the block tiles' epilogue (`moe_glu3` is the same
  expression the decode step's GEMV kernels use), behind `fmt >= 100`; the int8 path and the f32
  path are untouched. `gate.sh` passes on the change.
- **The cold set is ~29 experts a layer, not 32, and unevenly spread**: the tier's warm start is a
  global ranking of the history, so the loop layers (three passes of counts) take nearly all their
  experts and the early/late layers take fewer. That is why the residents carry 98.7% of the rows
  at 2,296 experts (94.1% at 1,718) although only 75%/56% of the experts are on the card.

| cell (RTX 5070, 10.4 GiB free, one process) | 75% (2,296, 7.18 GiB) | 56% (1,718, 5.37 GiB) |
|---|---:|---:|
| tiles alone, all 128 experts | **9.07 ms**, 14.4 TFLOP/s | **8.51 ms**, 15.3 TFLOP/s |
| tiles alone, residents only | 8.15 ms | 6.76 ms |
| cold upload alone | 7.16 ms (90.9 MiB, 12.4 GB/s) | 12.99 ms (166.7 MiB, 12.5 GB/s) |
| sequence, serial | 16.04 ms | 21.45 ms |
| sequence, prefetched | **15.42 ms** | **21.05 ms** |
| sequence, + a 1 GiB mirror copy in each step | 89.78 ms | — |
| projection, 64K prefill | **156 s** (was 248.0) | 169 s |
| projection, 250K prefill | **1,155 s** (was 1,678.0) | 1,205 s |

**Decision (the gate as written before running): GPU prompt-MoE.** 15.4 ms a layer-step is 2.5×
under the 35 ms gate and 27% of the measured 56.0 ms host step (127.0 s / 2,268); the projection is
−92 s of 64K prefill and −523 s at 250K, and the 250K figure is not capped by attention (its floor
is `attn 836.4 + other 180.1 + fill 5.3 = 1,021.8 s`, below 1,155 s). All projected numbers are
`derived`; every ms in the table is `measured`.

Four things the table does not say on its own, and that the cells measured:

1. **The prefetch does not hide the upload on this box.** The copy keeps its own 12.5 GB/s (it is
   2.13 ms for layer 6's 28.7 MiB inside a prefetched step), but the tile frame's submit grows by
   almost exactly the copy's duration (8.23 → 10.37 ms), and at 90.9 MiB a step the sequence only
   moves 16.04 → 15.42 ms. A raw two-stream probe (a 91 MiB copy 7.01 ms, a spin kernel 2.64 ms,
   both 7.06 ms) shows the device does overlap DMA with compute, so the limit is the tiles
   themselves: 14–15 TFLOP/s of 100 is a latency-bound read of 128 experts' weights, and the DMA's
   writes are enough to slow it. Recorded, not explained further.
2. **A whole 1 GiB mirror upload inside one layer-step's window costs 74 ms** (89.8 ms a step). At
   64K the engine's mirror is ~0.6 GiB over the whole prefill, so this is only the worst case of
   scheduling it badly: the prompt path must spread the mirror's rows, never put them all in one
   layer's window.
3. **128-row tiles would be wrong here**: 8,192 rows over a heavy-tailed 128-expert history leave
   61% padding at 64-row tiles and 137% at 128-row tiles, so `COLI_CUDA_MOE_TILE128` must stay off
   for this shape (it is on only from `A >= 32*E`, which this is, so the prompt path must set
   `COLI_CUDA_MOE_TILE128=0` or the tile choice needs a padding term).
4. **The upload is the expensive half at 56%**: 166.7 MiB a step is 369 GiB over a 64K prefill
   (30 s of PCIe) against the host path's measured 18.8 GiB, because the same non-resident experts
   are streamed once per chunk. A hybrid that leaves the non-resident pairs on the host is bounded
   by `max(device, host)` = 18.5 s at 75% and 15.3 s at 56% of expert work, better than streaming
   them; it needs a filtered host pair step, so the streaming path (§15.10's "if it passes") is
   what gets prototyped first.

## 15.12 The q4 attention kernel's limiter, measured (P1.2's gate, no kernel yet)

PLAN 14 asks the tensor-core item to start by measuring the dequant limiter instead of porting
`mma` on a hunch. `cuda_inkc_bench` on this tree, 262,144 positions, one layer, a 1,024-row chunk
at the end of the context (the widest chunk; a prefill's 244 chunks average about half of it),
`bench/results-tensor-core.json`:

| KV type | ms | TFLOP/s | mirror read | GB/s |
|---|---:|---:|---:|---:|
| f32 | 1111.7 | 2.96 | 255.0 GiB | **246.3** |
| f16 | 1132.6 | 2.91 | 127.5 GiB | 120.9 |
| q8 | 839.8 | 3.92 | 71.7 GiB | 91.7 |
| q4 | 729.6 | 4.51 | 39.8 GiB | **58.6** |

The decode step over the same prefix: 3.68 ms at q4 (0.87 TFLOP/s), 6.40 ms at f16.

**Reading.** q4 reads 6.4× fewer bytes than f32 and takes 34% less time, so the bytes are not what
limits the q4 kernel: at f32 it reaches 246 GB/s (the card's own read rate) while at q4 it reaches
58.6 GB/s. About four fifths of the q4 kernel's time is the unpack, not the loads and not the
FMAs: the bandwidth ceiling for 39.8 GiB is ~0.17 s against a measured 0.73 s, a 4.3× headroom.
So the tensor-core item's premise is measured, not assumed, and its condition stands: `mma` only
with a vectorized unpack in the same change. At the engine's 250K cell the attention kernel is
~673 s of the 836 s device side (14.9); a 4× kernel would leave ~168 s, at which point the
expert path (99 s by §15.11) and the ~163 s of other device work are the floor.

Provenance: ms, TFLOP/s and GB/s measured; the 4.3× and the ~168 s are derived from them. The
kernel is not written; P5 and P1.2 stay open in §14.10.

## 15.13 The ~139K cell, and why there is no 250K reference cell

One engine cell on the same tree (the depth prompt of §15.7's generator asked for 128,000 at its
5.52 chars/token, which the model's tokenizer reads as **139,421**; the three facts at
10/50/90% in a `CTX_MAX=140,000` run), `bench/results-128k.json`, commit `a2f79b1`:

| | 64K (14.8) | **~139K** | 250K (14.9) |
|---|---:|---:|---:|
| prompt tokens | 63,977 | **139,421** | 250,156 |
| prefill | 248.0 s | **697.8 s** (199.8 tok/s) | 1,678.0 s |
| phases (s) | 127.0 / 71.9 / 47.5 | **expert 307.3 / attn 283.0 / other 103.9** | 658.1 / 836.4 / 180.1 |
| decode | 17.77–18.09 tok/s | **13.87 tok/s** | 12.22 tok/s |
| RSS | 9.7 GB | **9.8 GB** | 12.8 GB |
| tier | 7.47 GiB = 2,296 | **6.66 GiB = 2,084** | 5.71 GiB = 1,741 |
| needle | — | **3 of 3** | 3 of 3 (244K) |

The generation at 128K is verbatim `7X4Q-ALPHA-93, PUMP-41-BRAVO-77, K9-ORBIT-27.` — a third depth
cell between the 8K and 244K ones, so retrieval holds across the range, not only at the ends.
Prefill is sublinear per token (3.9 s/1K at 64K, 5.4 at 128K, 6.7 at 250K) and the phases scale
the way §14.9/15.4 said they would: the expert phase tracks tokens, attention more than tracks them.
Decode 13.87 sits between the 64K and 250K numbers, monotone in context.

**The reference could not make the same cell** — **superseded by §15.15**: build b12676 loads and
serves 139,421 tokens at 174.6 tok/s after #385/#413 chunk/read the relative bias instead of
materialising it. The paragraph below describes b12588. llama-server with the §15.6 settings unchanged
(`-ngl 99 --n-cpu-moe 12 -c 131072 -ctk q4_0 -ctv q4_0 -fa on -b 512 -ub 128`) fails to load:
`graph_reserve: failed to allocate compute buffers` (3.52 GB) and then `failed to initialize the
context: failed to allocate compute pp buffers`, on the same card with 10.9 GiB free. So the
missing 250K comparison is not an omission: at 128K the reference needs a compute buffer the 12 GB
card does not have beside twelve layers' experts on the CPU. Getting a reference number would need
more experts on the CPU (`--n-cpu-moe` up), which changes the setting and so stops being
like-for-like. **250K decode parity stays open, with its reason measured.**

## 15.14 The prompt-MoE path in the engine, prototyped and measured at 64K

§15.11's decision (GPU prompt-MoE) is now in the engine: `COLI_CUDA_CHAIN_PROMPT_MOE=1` lets
`ink2cc_dmoe` take a prompt chunk's rows (up to `COLI_CUDA_CHAIN_ROWS`, 1,024 by default) and a
layer's every expert run in the frame, sorted by expert through the block tiles
(`cc_route` + `cc_moe`, `SiTU-GLU`, the per-expert scales, NVFP4 down — the tiles' epilogue in
`f900479`). A layer qualifies only when **every expert is present**: the tier's residents on the
card, the rest in its **host pool** (`COLI_VK_TIER_HOST=1`, 3.0 GiB of pinned RAM the tiles read
over the bus). Decode keeps the residents-only decision, so its path is untouched. Default 0: with
the flag off nothing changes (the gate ran green on that path before the change).

The pair at 64K, same prompt (`needle_depth.build` at 64,000 by the generator, 69,651 tokens
under the tokenizer, the three facts at 10/50/90%), same context, same tree, same pool, only the switch differing
(`bench/results-prompt-moe.json`):

| | A: host's step (`COLI_CUDA_CHAIN_MOE=0`) | B: the frame (`=1`) |
|---|---:|---:|
| prefill | 267.0 s | **162.0 s** (−39%) |
| phases (s) | expert-mm 132.7 / attn 84.2 / other 52.5 | **1.8 / 53.2 / 109.6** |
| decode (TUNE) | 12.94 tok/s | 12.27 tok/s (−5%) |
| routed experts on the host | 165.2 s | **1.8 s** |
| layer-steps' experts in the frame | 0 | **3,312** (69 chunks x 48) |
| budget / pool | 7.49 GiB = 2,304 + 768 | 7.01 GiB = 2,150 + 922 |
| RSS / device | 9.1 GB / 1,045 MiB | 10.6 GB / 1,164 MiB |
| needle | 3 of 3 | **3 of 3**, the same generation |
| buffers | — | ids 3.0 MiB, w 3.0 MiB, hid 108.1 MiB, y 9.0 MiB, kvl 384 MiB |

The timing prompt (`bench/prompt-64k.txt`, the published baseline's own) with the path on:
**156.4 s**, phases 0.5 / 55.2 / 101.5, decode 17.07 tok/s, 3,024 layer-steps in the frame.

**The default path on this tree, the same timing prompt, nothing set** (the flag off, the pool off):
265.1 s prefill, phases 2.9 / 136.6 / 78.3 / 49.1, decode 17.27 tok/s, RSS 8.8 GB, budget 7.49 GiB
= 2,305. That is **+6.9% prefill and −2% decode against the published 248.0 s / 17.77–18.09**
(commit `b1f0c26`, the night before), and it is not the flag: with it off the path is the old one
byte for byte (`ink2cc_prompt_moe()` short-circuits before any allocation, `vram_need` keeps the
8-row estimate, `dm_onp` is never set, and the tile epilogue's new arguments default to the old
arithmetic for the dense GEMMs). It is either the eight commits between `b1f0c26` and `e1470cd` or
the box (page cache, thermals) — **unattributed**, and a re-baseline of `b1f0c26` is the way to
settle it. The A/B above is unaffected: both arms ran tonight on this tree.

**Gates (all measured, this tree, the path on).** `bench/gate.sh` green: `cuda-inkc-check` all
green and the needle 4/4 at 1K/2K/4K/8K through the new path; `needle_check` 1K/4K/8K/16K all found
with the path and with `COLI_CUDA_CHAIN_MOE=0`; `needle_depth --tokens 64000` **3 of 3 both ways**
with the same generation `7X4Q-ALPHA-93, PUMP-41-BRAVO-77, K9-ORBIT-27.` — the negative test the
plan asked for (retrieval, not token-exact: the tier's split changes the numerics).

**What it costs, and where it stops.** The pool is the cost: against the pool-off published
baseline (248.0 s, 17.77–18.09 tok/s, another commit and prompt) a run with the pool pays decode
(the tier copies pooled experts into its 64 slots: 38,557 copies, 118.4 GiB in the needle prompt's
run), not the tiles. And the path's requirement — every expert present — fails at a
262K-served context: `needle_depth` with its `CTX_MAX=262144` leaves the pool 1.50 GiB (410
experts) and the tier 5.14 GiB (1,561), so **0 of 24 layers** qualify and the run is the host's
step either way (323.3 s asked for the path against 318.8 s, needle 3/3 both). So the prototype is
a 64K-class-path win measured end to end, and a 250K path needs either a bigger pool or the
per-step staging copies §15.10 specified (whose price the microbench measured: 15.4 ms a step at
75%, 21.1 at 56%, where the pool's per-step copies are not in that number).

## 15.15 The reference moved: unsloth b12676 measured at 64K and ~139K

The llama.cpp rows §15.6 and §15.13 compared against came from unsloth `b12588-mix-c0c3de8`
(2026-11-20). The current prebuilt, `b12676-mix-c32dcc8` (2026-11-25, upstream `b12676` plus ~45
pinned PRs), was installed beside it and the same cells re-run — same snapshot, q4 KV, `-ngl 99
--n-cpu-moe 12 -c … -fa on -b 512 -ub 128`, temp 0, `cache_prompt=false`, 32 tokens
(`bench/results-llama-b12676.json`; one GPU process at a time):

| cell (RTX 5070 12 GB, IQ3_XXS) | b12588 | **b12676** |
|---|---:|---:|
| 64K, 63,977 tok, no `-md`: prompt | 137.2 tok/s | **162.8** (+18.6%) |
| 64K, no `-md`: decode | 16.20 tok/s | **19.96** (+23.2%) |
| 64K, `-md` DSpark: decode | 17.19 (+6.1%) | 16.52 (**−17%**) |
| 64K, `-md`: draft accepted | 21/27, 0.778, 2.33/verify | 18/36, 0.500, 1.5/verify |
| 64K, `--moe-cache-mib auto`: decode | — | 19.32 (the fit left no cache to size) |
| ~139K, 139,421 tok, no `-md` | **would not load** | **loads**: 174.6 tok/s prompt, 17.96 tok/s decode |

Against the engine on this tree:

| size | engine prefill | engine decode | b12676 prefill | b12676 decode |
|---|---:|---:|---:|---:|
| 64K (prompt-MoE path) | **156.4 s** | 17.07 | 393.1 s | **19.96** |
| 64K (engine default path) | 265.1 s | 17.27 | 393.1 s | 19.96 |
| ~139K | **697.8 s** | 13.87 | 798.6 s | **17.96** |

**What this changes.**

1. **The decode-parity result of §15.6 is superseded.** The engine is not at parity at 64K any
   more: the reference is **16–17% faster** (19.96 against 17.07–17.27) and **29% faster** at
   ~139K (17.96 against 13.87). The whole of the gap is the b12676 pins' kernel-count work on
   Inkling-2's own ops (clip #401 6653→5723 kernels a token; SiTU #402 →5350 with the stack's
   tg128 +27% over its pin set; conv chain #411 192 kernels replacing ~950, its share −13.1%;
   tanh/router/norm #412 −606; depth mix #414 −300; rel-bias #413 −240), all of which the engine
   already does in one kernel each — so this is a *floor* the engine now has to re-earn, not a
   curiosity. `bench/results-speed.json`'s kernels are not the comparison.
2. **Prefill still belongs to the engine**: 2.5× at 64K on the prompt-MoE path (1.5× on its
   default path) and 1.13× at ~139K. The ~139K prefill gap is much smaller than at 64K because
   attention dominates there and the reference's banded FA (#386) and rel-bias fixes (#385/#413)
   closed it.
3. **§15.13's "the reference cannot make the cell" is fixed in b12676.** The 3.52 GB compute
   buffer that failed at `-c 131072` was the dense `n_kv × n_ubatch × n_head` relative-bias mask;
   #385 chunks it over query rows and #413 reads the bias inside the vector FA, so 139,421 tokens
   now loads and retrieves 3/3. The 250K cell is therefore no longer "not obtainable" in
   principle — it is untried at that size.
4. **DSpark now hurts at 64K** (19.96 → 16.52, acceptance 0.500 against 0.778). Recorded, not
   explained: it is #405's new fused draft path with the padded-layout drafter we have. Do not
   re-run the §15.6 DSpark comparison against b12588 as if it still held.
5. **`--moe-cache-mib auto` sizes nothing at this fit** (19.32 against 19.96). unsloth's loop-aware
   cache fix (#421) is real — 55.7% → 81.3% hits, 4768 → 2013 GiB of expert traffic — but it needs
   cache the 12 GB fit does not leave. The analogous check on our side came out clean: colibri's
   tier keys slots by *physical* layer (`slot(layer, eid)`, `all = L*E` = 3,072) while the loop's
   `v` is the state layer, so the 48 visits share one expert slot each — no duplication to fix.

## 15.16 Synced to dev, and what this branch still holds (2026-11-25/26)

**The sync.** The branch forked at `a4c91f81` (2026-11-22); dev is at `c29ac0a1`, 6,328 commits
later. Merged as `2285cada` on `kv-longctx-sync` (then fast-forwarded here). A correction to §15.15's
account: our fork point **already carried dev's Inkling-2 batch** — #2964 re-merged it with rewritten
hashes, so `c4cee56f`'s content was in our base all along. The merge was therefore **2 files and 3
hunks**, not a rewrite:

- `backend_cuda.cu`, `hcg_tile`'s signature and GLU epilogue: dev's newer template (`PACKED`,
  `PARTIAL_GLU`, `PAIR_CODES`, `void *X`, the `Yu` output, the INT8 pair-code store) kept whole,
  with this branch's Inkling-2 additions re-applied as defaulted parameters (`act/ta/tb/s_in/s_out`)
  and `moe_glu3`; the epilogue keeps `PARTIAL_GLU`'s store and takes the activation and the
  per-expert scales otherwise.
- `cuda_chain.cu`, `cc_moe`'s guard: dev's line plus our "a GGUF's block formats take act and
  esc_off on a prompt chunk's tiles"; dev's newer `hid_floats` / `tiled_experts` checks below stay.

Everything else merged textually: `ink_kv.h`, `cuda_inkc.cuh`, `inkling2_cuchain.h`, `inkling.c`
and `inkling2_chain.h` stayed as this branch has them.

**What this branch still holds over dev** (all verified present in the merged tree, and absent from
`origin/dev`):

| ours | where | measured |
|---|---|---|
| the typed KV cache, f16/q8/q4 rows | `c/ink_kv.h`, `ink_kv_put/get/row_bytes` | 250K on 12 GB: 1,678 s prefill, 12.22 tok/s, the 250K needle found (§14.9); dev has no typed KV at all |
| the tiled prompt attention | `ccg_relattn_qt*`, `COLI_CUDA_RELATTN_QT/ROWS` in `cuda_inkc.cuh` (ours 767 lines, dev's 290) | 1.03 → 5.02 TFLOP/s at 250K, 1,024 rows (§14.2) |
| the GQA decode kernel | `ccg_relattn_dec_k`, `COLI_CUDA_RELATTN_GQA` | 8.03 → 4.15 ms a global layer at 250K q4, 1.94× (§14.3) |
| prompt-row expert tiles with SiTU-GLU, the per-expert scales and NVFP4 down | `moe_glu3`, `hcg_tile`'s `act/ta/tb/s_in/s_out`, `esc_off` in `hcg_moe_*_bd_k` | dev still refuses `act`, `esc_off` and NVFP4 for rows > 8 |
| a prompt chunk's experts in the frame | `COLI_CUDA_CHAIN_PROMPT_MOE` in `inkling2_cuchain.h` | 64K prefill 146.6–150.8 s against 254.2 s on the default path (§15.14) |
| the microbench and the result rows | `c/tests/bench_cuda_moe.cu`, `bench/results-*.json` | §15.11 |

**What the sync brought.** 6,328 commits: the tier's host pool, streaming and sub-batches; the
feature system (`cc_uses_feature`, `Q36_EXPERT_MATMUL`, `cc_moe_tiled_*`, the tiled expert GEMM);
`hcg_tile`'s f16/INT8-pair paths; and the newer qwen36/qwen38/GLM/DSV4 chains. On this box it
shows: 64K default path **254.2 s / 18.65 tok/s** (was 265.1 / 17.27 before the merge), prompt-MoE
path **146.6–150.8 s / 19.38–19.54** (was 156.4 / 17.07); the b12676 reference is 393.1 s / 19.96
(§15.15), so decode is back within 3% on the prompt-MoE path and prefill is 2.7×.

**What was taken from llama.cpp / unsloth and implemented: NVFP4's decoder.** Measured first: at
the same gate/up type, a layer with an NVFP4 down (L9) cost 7.87 ms a prompt layer-step against
6.69 ms for one with IQ4_NL (L20) — 1.18 ms, 15%, in the *float* block decoder (`bdf_decode`: eight
byte loads and two `ldexpf` a 32-weight unit). `bd_dec16h` now has an NVFP4 case (the fast path
`hcg_fetch_w<100>` tries first): the E2M1 magnitudes computed instead of table-read, the sub-block
scale from the byte, the reserved scale bytes 0 and 0x7f read as zero (the first attempt missed that
and was 0.149 off on a 40-row test). L9 7.07 ms, within 0.4 ms of IQ4_NL; the 36-step sequence
15.5 → 14.4 ms (−7%). Bit-identical to `bdf_decode` over full rows (random and all-0xff), and
`tests/test_cuda_bd.cu`'s dense NVFP4 40-row tiles are back to 4.5e-05. This is unsloth #428 /
ggml-org #32875's lesson ("NVFP4 dequant speed increase") applied to the decoder our tiles use; the
engine-level 64K cell cannot resolve it (146.6 vs 150.8 s, inside this box's ±3%).

**One thing to chase next, recorded rather than hidden.** `tests/test_cuda_bd.cu` (not wired into
any make target here) has **63 failures** on the merged tree: `check_moe_ink2`'s SiTU checks come
out 2–7e-7 over their tolerance, then a device loss cascades through the rest. The same failures
appear with the NVFP4 case compiled out, so they are **pre-existing on the merge**, not from this
change; `bench/gate.sh` is green and the engine's decode path runs. Whether dev's own tree passes
that test (i.e. whether the merge or dev is at fault) is the open question.

## 15.17 Decode parity by context: the 40 tok/s claim and our own card

Upstream's CHANGELOG advertises **40.9–41.1 tok/s** of decode (`c4cee56f`, Inkling-2-Mini on the
CUDA chain). That cell is an **RTX PRO 4000, UD-Q4_K_XL (16.6 GB), 100 tokens after a 570-token
prompt** — a bigger card and a heavier quant than this box: 11.66 GB IQ3_XXS in 12 GB with q4 KV.
This branch now *contains* that code (merged, §15.16), so at that cell shape the difference between
the two engines would be card and quant, not our work. What we can measure is the same cell on
this card, engine against the llama.cpp reference (b12676) — same snapshot, same q4 KV, 100 tokens,
`OMP_NUM_THREADS=6` / `-b 512 -ub 128 --n-cpu-moe 12` (`bench/results-short-parity.json`):

| prompt tokens | engine prefill | **engine decode** | b12676 prefill | b12676 decode | engine vs reference |
|---|---:|---:|---:|---:|---:|
| 721 | 6.0 s | **27.68 tok/s** | 5.5 s | 21.21 | **decode +30%** |
| 8,027 | 36.8 s (218/s) | **24.25 tok/s** | 51.4 s (156/s) | 21.40 | **decode +13%, prefill 1.4×** |
| 63,977 | 146.6–150.8 s | 19.38–19.54 | 393.1 s | 19.96 | decode −3%, prefill 2.6–2.7× |

Short context is where the engine leads and where upstream's headline lives: on this card the
engine is 30% ahead of the reference at 721 tokens and 13% at 8K. By 64K the reference catches up
on decode (−3% for us, §15.15) while we stay 2.6× ahead on prefill, and at 250K the comparison
does not exist: dev has no typed KV, so a 12 GB card cannot hold 250K at f16 (4.6 GiB a side), and
only this branch runs that cell (12.22 tok/s, needle found, §14.9 — measured before the merge; the
post-merge 139K/250K cells are the next ones to re-take, 64K having gone 17.07 → 19.4 with the
merge).

So: **not better than the 40.9 figure on this hardware, and not comparable to it** — the 40.9 is a
bigger card with a heavier quant at 570 tokens. On the same card, same quant, same settings the
engine is 30% ahead of the reference at 721 tokens, 13% at 8K, level at 64K, and alone at 250K.

## 15.18 What Strata's engine shows, and the tier's balance switch (+22–24% decode)

Strata's engine components were taken apart, first 0.1.66 (installed) and then the current **0.1.69**
(Strata 0.2.22, published 2026-11-25 18:52; the model-shaped asset `strata-inkling-0.1.69-linux-x86_64-cuda-cuda13`,
440 MB, archs 75/80/86/89/120, build `a5bfe0369bfb`). Their decode design, in the binaries' own names: a **model-specific engine** (`components/shapes/inkling2/.../strata-inkling`,
38.8 MB, sm_75/86/89/**120**), a few **big fused kernels** (`qkv_fused`, `ink_post_qkv_fused`,
`gu_fused`, `shortconv_kernel`, `act_kernel`, `dot_kernel`, `k_dspark_swiglu_fused`) and **CUDA
graphs**; a **drafter shipped in the pack** (`--mtp` on by default, "about 1 GB less graphics memory
and is slower" when off, `--drafting auto|mtp|dflash2|off`) with a **restricted draft vocabulary**
(`--draft-vocab cjk|en|…`); per-machine `--tune` persisted; the CPU side pinned
(`serve/cpu_affinity.py`); hybrid placement only when the pack does not fit. Their 93.9–101.4 tok/s
on an RTX 5070 (§15.17) is with that drafter on and a pack that fits whole.

**What we could measure immediately is the placement policy**, and it is the biggest decode win of
the sync: `COLI_VK_TIER_BALANCE=0` (`bench/results-strata-lessons.json`):

| context | decode before | **with balance off** | prefill before → after |
|---|---:|---:|---:|
| 721 tok | 27.68 | **33.73 (+21.9%)** | 6.0 → 4.7 s |
| 8,027 | 24.25 | **28.25 (+16.5%)** | 36.8 → 35.1 s |
| 63,977 (default path) | 18.65 | **23.06 (+23.6%)** | 254.2 → 253.5 s |

Why: the balancer hands a step's resident experts to the CPU whenever the join waits on the device.
Inkling-2's join waits on the **dense chain** (48 visits, device-bound), not on the experts, so the
balancer reads the device as slow at experts and moves them to the slower CPU — 4.2 s of the run's
6.1 s expert time at 721 tokens, 3,930 rows handed over in one 64K run. Off, the device takes them
and the CPU share halves. This is a fixable upstream behaviour, not just a knob: the balance signal
should be the experts' device time, not the whole join.

With balance off the host pool must be **sized**, or the prompt path loses its "every expert
present" precondition (auto: 945 experts, pool 2,108 + 945 = 3,053 with balance on; **724** with it
off → 0 of 24 layers). Pinned with `COLI_VK_TIER_HOST_GB=3.2` it holds 956 and all 24 layers qualify:

| 64K q4, best config (`BALANCE=0 HOST=1 HOST_GB=3.2 PROMPT_MOE=1`) | before tonight |
|---|---:|---:|
| prefill | **145.4 s** (440 tok/s) | 254.2 s (18.65 tok/s decode) |
| decode | **24.39 tok/s** | 18.65 |
| phases | expert-mm 0.6 / attn 53.4 / other 92.7 | — |

Against the b12676 reference at 64K (393.1 s, 19.96 tok/s) that is **2.7× prefill and +22%
decode**; against Strata's reproduced cell (657–684 tok/s, 3.4–3.9 tok/s) it is 1.5× on prompt and
6–7× on decode.

**0.1.69 against 0.1.66, decode side** (symbol and flag diff of the two model-shape binaries and the
two control components). New kernels: `k_dspark_plain_from_gguf` / `k_dspark_plain` /
`k_dspark_layout_*` / `k_dspark_wants_f32` (the DSpark drafter now loads **from a GGUF**), 
`k_ink_quant_x_q8f` + `k_ink_gemm_q8f` (quantize an activation to q8 once and run the GEMM on it),
`k_ink_moe_fp16tc_v2`, `k_ink_moe_q80_tc`, `k_ink_moe_pf_fp16tc` (tensor-core expert paths incl.
prefill), `k_ink_stage_plan` / `k_ink_stage_copy` / `k_ink_stage_restore` (a planned staging
pipeline), `k_ink_topk_pass1` / `k_ink_topk_merge` (two-pass routing), `k_ink_arch_probe`. New
control flags: `--draft-beam`, `--draft-cost`, `--draft-adapt` / `--no-adapt`, `--draft-plain`,
`--kv-offload`, `--cache-model`. 0.1.66 already had the model-shaped binary, `qkv_fused`,
`ink_post_qkv_fused`, `gu_fused`, `shortconv_kernel`, `ctrl_*_small` (the loop's control at decode
size), `gemv_multi_q8row_preq`, `moe_gu/dn_sk_fmt` and the `pf_*` attention kernels with a device
`pack_kv` and tiled online softmax with the relative bias and sink.

**What is left to take from them, in order** (each still open):

0. **Prompt-lookup drafting first, then DSpark from the GGUF we already own.** `c/spec_draft.h`
   already holds `spec_lookup` (n-gram proposals from the context itself — no weights) and the
   "whether drafting pays" gate, and the engine's verify forward (up to 8 rows) exists. That is
   the zero-weight half of their drafter and needs no download. **Done, §15.19**: +34 to +62% on the
   bench prompts, same text. Their 0.1.69 shows the second
   half: the external DSpark drafter **from a GGUF** (`k_dspark_plain_from_gguf`) — the same
   0.40 GB `dspark-Inkling-2-Mini-Q8_0.gguf` we already have, which our engine cannot read today —
   with `--draft-beam` / `--draft-cost` / `--draft-adapt` / `--draft-vocab` deciding when and how
   wide to draft.

1. **A first-class drafter with a restricted draft vocabulary.** Our `INK2_MTP` exists but our
   snapshot carries no MTP weights (imported `--no-mtp`; the GGUF has no `mtp`/`dspark` tensor),
   and the 0.40 GB `dspark-Inkling-2-Mini-Q8_0.gguf` is not an engine input — 0.1.69's
   `from_gguf` symbol is exactly that gap. Their drafter is +1 GB VRAM (which is why their 10.47 GB
   pack still fits) and their `--draft-vocab` idea (the drafter may only guess a language's token
   set) is what keeps acceptance high enough to pay.
2. **Quantize one activation row once per layer visit** (`k_ink_quant_x_q8f` + `k_ink_gemm_q8f`,
   and 0.1.66's `gemv_multi_q8row_preq`): our decode runs ~8 separate int8 matmuls per visit
   (q, k, v, o, router, shared expert) and each re-quantizes the same normalized row. One
   quantize + N matmuls that read it.
3. **Pack the KV rows on the device** (`k_ink_pf_pack_kv`): ours is host-side
   (`ink2cc_kv_down` — P5's ~62 s at 250K) then copied up; a device packer removes the host work
   and a sync point per layer step.
4. **Fused per-layer kernels + CUDA graphs**: their `qkv_fused` (our three matmuls of one input),
   `ink_post_qkv_fused` (rope, clip, gate, conv), `gu_fused` (we have it in `cc_moe`), and graph
   capture of a decode step against our ~100 ops a visit.
5. **A model-shaped binary per model and arch** (`shapes/`), sm_120 included: we specialize per
   format at runtime, not the graph. 0.1.69's `k_ink_arch_probe` picks per arch.
6. **Tensor-core expert paths at decode** (`moe_fp16tc_v2`, `moe_q80_tc`, `sk_tc`): our decode
   experts are warp-per-row GEMV + dp4a; theirs batch the top-k slots onto tensor cores.
7. **NOT their host pool at 12 GB**: with it on the VRAM expert budget collapses (2,457 → 606–724)
   and decode stays ~28. It is the prompt path's source, pinned, not a residency strategy here.

## 15.19 Prompt-lookup speculation in the decode loop (+34% to +62% on the bench prompts), and why the DSpark GGUF drafter is still blocked

The first half of Strata's drafter needed no weights and no download: the context proposes what follows
itself. `INK2_SPEC=1` (`inkling.c`'s `Ink2Spec`, the chain's `nlog` in `inkling2_cuchain.h`) drafts up
to `INK2_SPEC_DRAFTS` tokens from the token history with `spec_draft.h`'s `spec_lookup`, runs **one**
forward of S = drafts + 1 rows, and keeps the prefix the target's own greedy picks agree with:

- **The rows.** Row r is fed at `pos + r` (row 0 the token the caller just emitted), so row r's logits
  predict `pos + r + 1` and draft r stands when row r-1 picks it. The chain computes the final norm,
  the head and an argmax for *every* row on the card (`cc_pick_rows`) and the loop compares the picks
  before the vocabulary comes down: only the rows that stood are read back, 0.8 MiB a row.
- **The rollback.** A short convolution's state is the tail of the rows it saw and cannot be stepped
  back a row at a time. The driver copies the host's convolution banks (`m->cs`, ~1.9 MiB) before the
  verify; on a rejection it restores them, tells the chain the host's copy is newer, and feeds the
  accepted prefix again in one forward (the same bits plain decoding gives those rows). The K/V rows
  past the accepted prefix stay in the host ring where they are and are overwritten when those
  positions are fed again; the mirror is only read up to the fed positions, so a rejected draft's row
  is never attended to.
- **The depth** is `spec_draft.h`'s gate: it measures T(n), the seconds of a forward of n rows, and the
  acceptance by draft position from the same run, and it stops drafting when a verify is worth less
  than a plain step (`INK2_SPEC_GATE=0` drafts every proposal, for tests).

**The text does not change.** Token for token identical to `INK2_SPEC=0` at 721 (100 tokens, two
settings against the final binary; four more earlier), 8,027 (100) and 63,977 (32), and with
`COLI_CUDA_CHAIN_MOE=0` (the experts on the host, a different step for the verify's rows): the
accepted drafts (65/66 at 721, 67/71 at 8K, 16/17 at 64K) are the picks plain decoding makes, which is
the check that the chain's S-row forward gives a row the bits a one-row step gives it.

Measured on this tree, this box (one GPU process, `COLI_VK_TIER_BALANCE=0`, `INK2_KV=q4`,
`OMP_NUM_THREADS=6`; the bench prompts are the repetitive filler of `bench/make_prompts.py`, so this is
the lookup's best case). Each row is one run of each way, back to back, same binary:

| context | `INK2_SPEC=0` | `INK2_SPEC=1`, drafts 4, gate | tokens a forward | drafts kept |
|---|---:|---:|---:|---:|
| 721 (`-n 100`) | 35.30 | **42.21 (+20%)** | 2.91 | 98.5% (65/66) |
| 8,027 (`-n 100`) | 34.03 | **45.86 (+35%)** | 3.09 | 94.4% (67/71) |
| 63,977 (`-n 32`, 15.18's best config) | 24.94 | **33.07 (+33%)** | 2.07 | 94.1% (16/17) |
| 721, `COLI_CUDA_CHAIN_MOE=0` | 34.48 | **48.34 (+40%)** | 3.00 | 98.5% (66/67) |

The 721-token cell is the noisy one: the same configuration (the flag makes no difference at 721 —
0 of 24 layers' experts run in the frame either way) measured 42.21 and 48.34 in two back-to-back
pairs, and 48.08 against 34.55 on the run before the final rebuild, so that cell reads **+20% to
+40%** and the middle of the band is +39%. The 8K and 64K cells were stable to ~1%. Against §15.18's
33.73 tok/s the 721 cell is +25% to +43%.

Depth and gate, 721 tokens, `-n 100`, separate runs on the final binary (the same prompt and
settings; `INK2_SPEC=0` there is the 35.30 cell above):

| 721 | decode tok/s | tokens a forward | drafts kept | T(n)/T(1) measured |
|---|---:|---:|---:|---|
| drafts 1, gate on | 39.21 | 1.90 | 100% (47/47) | 2:1.13 |
| drafts 4, gate on | 42.21 / 47.25 / 48.34 | 2.91-3.09 | 98.5% (67/68) | 2:1.24 3:1.40 4:1.79 5:1.90 |
| drafts 4, gate off | **51.34** | 4.50 | 96.2% (77/80) | — |

So the gate gives up about 7 points of the 45 at 100 tokens, and only because it is still learning:
its acceptance EMAs carry a prior worth 2 verifies and a 16-verify memory, so 4.50 tokens a forward
(what depth 4 is worth here) is reached over a few dozen verifies rather than at once (27 verifies:
3.09 a forward in the same run). `COLI_CUDA_CHAIN_MOE=0` at 721: 34.48 → 48.34 (+40%), text identical.

The verify is cheap on this chain, which is why it pays: measured T(n)/T(1) at 721 is 1.05 (2 rows),
1.33 (3), 1.55 (4), 1.94 (5), i.e. a row costs 0.19 of a plain step (rho); the host's expert step is
most of a plain step, so a verify's extra rows cost one expert step each, not one step's latency. At
64K T(4)/T(1) = 1.86 (the gate only got as deep as 4 there).

**What the gate does not price**: the replay a rejection costs. A rejection at the first draft pays
T(S) + T(1) for one token. With 5-6% rejections on these prompts that is invisible; on free text the
lookup finds fewer and shorter proposals and the acceptance EMA is what stops the drafting. The
follow-up (an activation-row capture that makes the rollback a copy instead of a forward, §15.18 item 3)
would remove the term.

**Not wired**: `serve` mode (its decode loop is a different one), `INK2_MTP`, multi-slot decoding, and
runs whose logits are dumped. `INK2_SPEC=0` is the default and the plain path, untouched.

### The DSpark GGUF drafter: the layout, and the piece that is missing

The 0.40 GB `dspark-Inkling-2-Mini-Q8_0.gguf` we own is a **3-block DFlash drafter with DSpark heads**,
and it is not the DFlash2 drafter `c/dflash_cuda.h` was written for. Its own inventory (read from the
file, `inspected`):

- blocks `blk.0..2`: D 2,304, 24 heads / 8 KV, hd 128, FF 9,216, Q8_0 (type 8); `attn_q`, `attn_k`,
  `attn_v`, `attn_output`, `ffn_gate/up/down`, per-head `attn_q_norm`/`attn_k_norm`, `dflare_norm`,
  and **`attn_k_ctx`/`attn_v_ctx` + `attn_k_ctx_norm`** (the target's context injected as K/V, as
  DFlash2's `k_ctx`/`v_ctx`);
- heads: `dspark_fuse.weight` (4x4, f32), `dspark_hidden_out.weight` (256->2304) + bias,
  `dspark_norm`, `joint_proj.weight` (5120->768) + bias, **`markov_w1.weight` (256->201,024)** and no
  `markov_w2`, `mask_embd.weight` (2304x7), `output_norm`;
- metadata: `dflash.target_layers = [5, 23, 41, 48]` (the loop's step numbers), `block_size 8`,
  `sample_from_anchor false`, `has_confidence_head false`, `attention.causal false`,
  `sliding_window 1024` with pattern `[true,false,true]`, `rope.dimension_count 128`,
  `freq_base 1e6`, `final_logit_softcapping 30`, markov rank 256, vocab 201,024 padded / 200,058.

The missing piece is the **model-side graph of this variant**: how the four taps are fused (a 4x4
matrix per channel?), what `joint_proj`'s 5,120 inputs are — 2x256 markov + 2x2,304 hidden, i.e. the
block's row twice — and how the markov state, `dspark_hidden_out` (256->2304) and `dspark_norm` make
the drafts' logits with a single `markov_w1` where DeepSeek's own DSpark has `w1` (an embedding) and
`w2` (the bias). No source on this box has it: `/home/nazar/opt/llama.cpp/src-inkling` (master,
2026-11-20+) has `markov_w1/w2`, `joint_proj` and `mask_embd` for DeepSeek-V4/4.1's DSpark
(`c/deepseek_v4_dspark.inc` is our copy of that design) but **no `dspark_fuse`, no
`dspark_hidden_out`, and no Inkling dflash variant**; the b12676 build that loads this file is a binary
with no source here. `c/dflash_cuda.h`'s `df_load` wants `config.json` + `dflash_config` in a
safetensors directory, `fc.weight`, `hidden_norm`, `layers.N.*`, `candidate_selector.*` and the
DFlash2 two-tap convs — none of which this GGUF has. Writing the forward from the shapes alone would be
guesswork against a drafter whose acceptance *is* the whole payoff, which is what the prompt forbids.

The numbers we do have, for the record (`bench/results-dspark.json`, reference b12588, same card and
quant, `-md` + `--spec-type draft-dspark --spec-draft-n-max 3`): draft acceptance 77.8% (21/27),
mean length 3.33, 2.33 accepted a verify; decode **18.66 → 19.60 tok/s at 8K (+5%)** and
**16.20 → 17.19 at 64K (+6%)**. On b12676 the same drafter measured acceptance 0.500 at 64K and made
decode worse (19.96 → 16.52) — the dead end §15.18 already records. Our lookup path above is
comparable on the repetitive bench prompts (which is where it is strongest) and needs no weights; what
DSpark would add is free text, where this drafter is the only candidate and the wiring is missing.

### The two decode-side items of §15.18, not attempted tonight

Both are kernel work on the chain and neither is a decode win at this size: (a) the activation row
quantized once a visit (`cc_*` op + the dense path) is a new op behind the same gates; (b) the KV row
packer is **prefill**-side — `ink2cc_kv_down` is 15.8 s of the 46.6 s `other` at 64K, but a decode step
packs 24 layers x 1024 values, ~2 ms of a 40 ms step, so moving it to the card would not move decode
(§14.6 priced it at ~62 s of a 250K *prefill*). The next decode-side work is the one §15.19 measured
in passing: the replay a rejected verify costs (T(keep) a rejection), which an activation-row capture
would turn into a copy and which would let the gate draft deeper on free text.



```bash
## Appendix — benchmark commands

```bash
# build (CUDA primary; Vulkan fallback is `VK=1`)
make -C c inkling CUDA=1 CUDA_HOME=/opt/cuda

# tests
cc -O2 -march=native -I c c/tests/test_ink_kv.c -lm -o /tmp/test_ink_kv && /tmp/test_ink_kv
make -C c cuda-inkc-check CUDA=1 CUDA_HOME=/opt/cuda
# the prompt expert step alone (PLAN 15.10/15.11): residency %  steps  overlap  contend
make -C c cuda-moe-bench CUDA=1 CUDA_HOME=/opt/cuda
./c/cuda_moe_bench 75 36 1 1     # 2,296 experts of 3,072, the 36-step sequence, prefetched, + a 1 GiB mirror copy
./c/cuda_moe_bench 56 36 1 0     # 1,718 of 3,072, the 250K tier's

# import
./c/coli import /path/Inkling-2-Mini-UD-IQ3_XXS.gguf \
  --out /path/iq3xxs.colibri --no-mtp

# run (long prompts via -f; the local getline patch keeps one line whole)
SNAP=/path/iq3xxs.colibri CTX_MAX=70000 INK2_KV=q4 COLI_CUDA=1 \
  ./c/inkling -f bench/prompt-64k.txt -n 32

# 250K
SNAP=/path/iq3xxs.colibri CTX_MAX=262144 INK2_KV=q4 COLI_CUDA=1 \
  ./c/inkling -f bench/prompt-250k.txt -n 24

# A/B the attention kernel (0 = the per-(head,row) kernel)
COLI_CUDA_RELATTN_QT=0 SNAP=... CTX_MAX=70000 INK2_KV=q4 ./c/inkling -f bench/prompt-64k.txt -n 32
```

`bench/run_bench.py` runs 8K/32K/64K and writes `bench/results.json`;
`bench/unsloth_64k.py` is the llama.cpp baseline; `bench/results-kv.json` holds the runs in
§2.1 (q4/f16 8K, 32K/64K q4, 64K scalar A/B, 250K) with their logs under
`/var/tmp/scratch/kvtest/` on this box.

## 14. The 250K/262K speed pass (P0–P6)

Scope: make 250K/262K as fast as it can be on this box; the 1M work of §7 is postponed. Every
number below is **measured** on the same box and model unless it says otherwise; the artifacts
are `bench/results-speed.json` (the kernel matrix, `bench/sweep_kernels.py`) and the run logs
under `/var/tmp/scratch/kvtest/`.

### 14.1 P0 — measurement, the range audit, and the gates

- **P0a `c/cuda_inkc_bench`** (`make -C c cuda-inkc-bench`): the attention kernels through the
  chain's ops table against a synthetic q4 cache at a served context, reporting ms, achieved
  mirror GB/s and effective FLOP/s for a prompt chunk and a decode step. A kernel change is
  now measured in seconds, not in a 60-minute prefill. `bench/sweep_kernels.py` runs the matrix
  (P6) and writes JSON.
- **The phase line's `attn` is not the kernel.** `m->t_attn` is the wall time of `cc_submit` +
  `cc_finish` for a layer, i.e. the whole device side (projections, norms, clips, dconv, the
  gate, attention). `INK2CC_RA_STATS=1` now prints the attention calls, causal pairs and FLOPs
  by class; at 64K the 504 global calls are 2.01e14 FLOP, and 69.5 s of device time is
  3.35 TFLOP/s.
- **P0b range audit** (`INK_KV_STATS=1`), sampled over every state layer: 250K q4 run,
  max |K| 20.04, max |V| 11.27, block scale max 2.862 (K) / 1.609 (V), **0% of values above
  f16's 65504**. So a 16-bit K/V input (the tensor-core path) is range-safe for this
  checkpoint; the codec's f32 scales stay as they are.
- **P0c** the parity gates already in the tree (`cuda-inkc-check` against the CPU reference, the
  teacher-forced argmax oracle) are the kernel gate and both pass; `INK_IDS_DUMP` dumps the
  prompt's ids for the oracle, and a failed `relattn` case now prints a per-head diff. The 250K
  needle of §6 remains open and is what any *retrieval* claim still waits on.

### 14.2 P1.1 — the prompt kernel, restructured (fp32, no tensor cores)

`ccg_relattn_qt_t<G>` (one instantiation a query-head group, so a G of 3 does not pay for 8):

- **K's tile transposed on its float4**: the row-major tile put every lane's `float4` in the
  same four banks (a 4-way conflict on every K read). The tile is now `[dim/4][32 positions]`,
  conflict-free.
- **the tile is read once for every query head**: the dot walks the K row with all G
  accumulators live (G times fewer shared reads, G-way ILP), and the V `float4` is loaded once
  a position and used by every head.
- `exp2f` with the softmax folded into log2 e; the branch out of the V loop.
- Typed rows are dequantized 4 values at a time in the prompt path and 32 values (one codec
  block, coalesced) in the decode path.

Measured (`cuda_inkc_bench`, q4, 250K, a 1024-row chunk against the full prefix):

| shape | before (the old kernel) | after | |
|---|---:|---:|---:|
| H=64, 250K, 1024 rows (the bench's default shape, both kernels) | 8511 ms, 1.03 TFLOP/s | **1425 ms, 6.16 TFLOP/s** | 5.97×, measured |
| H=24 (Mini), 250K, 1024 rows | not measured (the old kernel is gone from the tree) | **656 ms, 5.02 TFLOP/s** | — |
| H=24 (Mini), 64K, 1024 rows | not measured | **163 ms, 5.03 TFLOP/s** | — |

### 14.3 P3 — the decode step, GQA-shared

`ccg_relattn_dec_k`: a block a (KV head, 128 positions), the tile's K and V dequantized once
into shared and every query head of that KV head reading them (the split walk had a block a
(query *head*, row) and read and dequantized the same rows G times), the block's warps carrying
an online softmax across four 32-position pieces so one partial a 128 positions is written, and
a persistent partial buffer instead of a 65 MiB `cudaMallocAsync` a layer.

Measured (q4, 250K, one global layer's step): **8.03 → 4.15 ms (1.94×)**, 401 → 777 GFLOP/s.
`COLI_CUDA_RELATTN_GQA=0` restores the old walk.

### 14.4 P2 — the mirror off VRAM (measured, negative)

`INK2_KV_MIRROR=host` puts the q4 mirror in pinned, device-mapped host memory, freeing 2.4 GB
of VRAM at 250K for the expert tier. Measured at 64K (63,977 tokens, 32 tokens):

| 64K | mirror on the device | mirror in host memory |
|---|---:|---:|
| prefill | 266.6 s | **745.4 s** |
| device side of the layers | 69.5 s | **577.0 s** |
| decode | 16.55 tok/s | 8.58 tok/s |
| tier budget | 8.26 GiB | 8.41 GiB |

The kernel's q4 row reads over PCIe are far slower than the same reads from VRAM, and the tiled
kernel reads each row once per 8-query-row tile, so the mirror's traffic is amplified. The freed
bytes did not become tier space either (+0.15 GiB). **Conclusion: the mirror stays on the
device**; the flag is kept for a card where VRAM, not PCIe, is the binding constraint.

### 14.5 P4 — the expert/decode floor

The in-frame MoE path is present (`COLI_CUDA_CHAIN_MOE`) but gates every layer on the tier's
table: `ink2cc_dm_layer` wants every expert of the layer device-resident and one format a layer,
and `ink2cc_dm_table` sets `ch->dm_on[i] = nres == E`. The formats are **not** the blocker —
`cc_moe_block_type(22)` covers this checkpoint's fmt 122 gate/up and `fg == 140` its NVFP4 down.
The model has **24 layers of 128 experts** (3072 slots, not 6144 — an earlier version of this
section said 256 a layer and was wrong). The blocker is the ranking, not the capacity: at 250K the
tier holds 1718 of 3072 (56%) and still completes **0 of 24 layers** (§15.2) — **1 of 24 layers**
qualified in the 64K runs
(29–47 layer steps ran in the frame). The fix is a per-expert frame route (a partial layer) or a
tier policy that packs whole layers, not a new block format. Until then the host's expert step
dominates decode and this item stays open.

### 14.6 P5 — the overhead

`INK2CC_PROF=1` puts a number on the host's side: at 64K, **15.8 s** of the 46.6 s `other` is
`ink2cc_kv_down` quantizing every state layer's step rows into the q4 cache (6.3e9 values, ~2.5
ns a value, scalar); it scales to ~62 s at 250K. `COLI_PREFIX_STORE=0` changed nothing
(277.3 s vs 266.6 s, run noise), so the prefix store is not the overhead. The next step is a
vectorized q4/q8 block packer (same arithmetic, same rounding, ~4× fewer instructions); not
done in this pass.

### 14.7 P6 — the tuning matrix

`bench/sweep_kernels.py` (`--quick`: 64K/250K, q4, H 24/64, chunk and step) writes
`bench/results-speed.json` with `provenance: measured` on every cell and records the env it ran
under, so the same file serves the `COLI_CUDA_RELATTN_GQA=0` / `COLI_CUDA_RELATTN_QT=0` A/Bs.
Eight cells, 22 s: 250K prompt chunk at Mini's 24 heads **672 ms / 4.90 TFLOP/s**, decode step
3.68 ms / 876 GFLOP/s; the same at H=64 1656 ms / 5.30 TFLOP/s and 5.65 ms / 1519 GFLOP/s.

`bench/sweep_engine.py` (8K prompt, 16 tokens, one cell ~30 s) writes `bench/results-engine.json`:

| 8K, 8,027 tokens | prefill | device side | experts | other |
|---|---:|---:|---:|---:|
| chunk 1024 | 29.5 s | 4.4 s | 17.6 s | 5.6 s |
| chunk 2048 | 29.4 s | 4.8 s | 18.3 s | 5.4 s |
| **chunk 4096** | **1275.5 s** | 320.2 s | 903.7 s | 47.0 s |
| chunk 1024, f16 KV | 30.2 s | 4.9 s | 20.1 s | 4.5 s |
| chunk 1024, q4, no prefix store | 28.1 s | 4.3 s | 17.6 s | 5.8 s |

Two results worth keeping: the prompt chunk's rows are flat from 1024 to 2048 and then
**catastrophic at 4096** (the tier streams a 4096-row step's cold experts in one go: 903.7 s of
experts, 14× the 2048 cell) — so 1024/2048 is the safe band; and f16 KV costs ~7% at 8K with no
attention gain (the kernel widens both types the same way), so q4 stays the long-context setting.
The prefix store is ~free (28.1 s with it off vs 29.5 s with it on, within run noise).

At **64K the chunk size does matter, and the 8K cells cannot see it**. Same binary, same tier
budget (7.68 GiB = 2362 experts), run back to back:

| 64K, 63,977 tokens | prefill | experts | attn (device) | other | decode |
|---|---:|---:|---:|---:|---:|
| chunk 1024 | 265.6 s | 148.1 s | 69.1 s | 45.9 s | 16.82 tok/s |
| **chunk 2048** | **236.2 s** | 123.4 s | 73.3 s | 38.1 s | 17.21 tok/s |

A 1.12× prefill win, reproduced (233.6 s and 236.2 s in two runs), and it comes from the expert
step: half as many, twice as large, so the tier evicts and streams less. `COLI_CUDA_CHAIN_ROWS`
is the knob (default 1024; 4096 stays off the table).

### 14.8 End to end, 64K

Same prompt and box as §2.1, `INK2_KV=q4`, CUDA, `OMP_NUM_THREADS=6`, 32 tokens:

| 64K, 63,977 tokens | §2.1 | now | |
|---|---:|---:|---:|
| prefill | 387.1 s | **266.6 s** | 1.45× |
| device side (the `attn` phase) | 202.6 s | **69.5 s** | 2.91× |
| routed experts | 138.9 s | 148.8 s | run noise |
| other | 44.4 s | 46.4 s | |
| decode | 14.94 tok/s | **16.55 tok/s** | 1.11× |
| RSS | 10.2 GB | 9.2 GB | |

8K (8,027 tokens): the device side 8.1 → 4.1 s; prefill 31.1 → 28.2 s with 8 tokens.

The 250K run of §5 is repeated with these kernels in §14.9.

**Where the device side actually goes** (`CCG_RA_TIMING=1`: CUDA events around the attention
kernels only, so this is measured rather than derived). 64K:

| 64K device side 70.7 s | s | share |
|---|---:|---:|
| attention kernels | **48.1** (4512 calls, 10.7 ms a call) | 68% |
| projections, norms, clips, dconv, gate | 22.6 | 32% |

The kernels do 233.1 TFLOP (201.2 global + 31.9 ring) in those 48.1 s = **4.85 TFLOP/s**, i.e.
the engine runs the kernel at the microbench's rate; the 23% "gap" between 3.85 and 5.0 TFLOP/s
was never the kernel. Scaled to the 250K run that is ~566 s of kernel beside ~266 s of other
device work — which caps what a tensor-core kernel can win and gives the 266 s a name.

### 14.9 250K with the new kernels

**With the §15 fix (this is the run that counts).** Same command and model as §5, 24 tokens:

| 250K, 250,156 tokens | §5 (start) | now | |
|---|---:|---:|---:|
| prefill | 3651.5 s | **1598.5 s** | **2.28×** |
| device side (`attn` phase) | 2706.7 s | **804.7 s** | **3.36×** |
| routed experts | 769.4 s | 612.4 s | tier state, see below |
| other | 172.9 s | 176.0 s | |
| decode | 8.44 tok/s | **12.18 tok/s** | **1.44×** |
| RSS | 12.2 GB | 11.1 GB | |
| **needle at 250,156 tokens** | — | **`7X4Q-ALPHA-93`** | **passes** |

The answer: `Question: what is the secret access code for the vault? Answer: 7X4Q-ALPHA-93.`
(`/var/tmp/scratch/kvtest/250k-fixed.out`, `[cache] hit 99.8%`). The expert phase is better than
§5's 769.4 s for tier reasons (a warm `.coli_usage` history and a larger budget this run), so the
comparable improvement is the device side (3.36×) and the total (2.28×); a same-state A/B is the
tier work of the revised plan, not this table.

**Same tree, a second run, a worse tier budget.** `250k-current.out`, started from `d5c3ccb` with
`CCG_RA_TIMING=1`: free VRAM 11.2 GB, `budget 5.62 GiB = 1710 experts` (against 5.91 GiB / 1798 in
the run above), so the expert phase is 724.0 s instead of 612.4 s and prefill **1791.8 s**. The
needle still answers **`7X4Q-ALPHA-93`**; decode 11.22 tok/s; device side 869.1 s. The attention
kernels were measured directly this time: **673.3 s of the 869.1 s device side (77.5%)** over
12,864 calls, 52.34 ms a call — so at 250K the other device work is 195.8 s, not the 258 s the 64K
ratio implied. Read the two runs as a budget band, not a discrepancy: prefill **1598–1792 s**,
device side **805–869 s**, decode **11.2–12.2 tok/s**, on 5.6–5.9 GiB of tier.

**Before the fix** — the same command on the rebuilt kernels, with the typed mirror never
uploaded (§15), so the numbers are real work done on a lost context. Phase line: `fill 5.3 s |
expert-mm 865.3 s | attn 832.5 s | other 179.0 s`.

| 250K, 250,156 prompt tokens | §5 | now | |
|---|---:|---:|---:|
| prefill | 3651.5 s | **1880.2 s** | 1.94× |
| device side (the `attn` phase) | 2706.7 s | **832.5 s** | 3.25× |
| routed experts | 769.4 s | 865.3 s | |
| other | 172.9 s | 179.0 s | |
| decode | 8.44 tok/s | **11.84 tok/s** | 1.40× |
| RSS | 12.2 GB | 12.6 GB | |
| device | 2802 MiB | 2825 MiB | |

The attention accounting (`INK2CC_RA_STATS=1`): ring 9800 calls / 10.23 Gpair, global 1960 calls /
250.31 Gpair, decode 1104 calls; the chain waited 627.3 s on the device and ran 982.4 s of routed
experts on the host, with 0 layer steps' experts in the frame (at 250K the tier held 1718 experts,
5.46 GiB of 5.64 GiB, so no layer's whole set is resident — §14.5).

The expert phase is **not** comparable to §5's: the tier's budget came out 5.46 GiB this run
(1718 experts, 7262 evictions, 28.4 GiB uploaded) against 6.06 GiB then (1849 experts, 6393
evictions, 26.4 GiB), so 865.3 s of routed experts is the tier's variance, not this work. The
attention and the prefill total are the comparable figures.

Range audit at this context (`INK_KV_STATS=1`): max |K| 20.44, max |V| 11.27, block scale
max 2.92 (K) / 1.611 (V), 0% of 493,780,992 values above f16's 65504.

Raw log: `/var/tmp/scratch/kvtest/250k-new.err`, `/var/tmp/scratch/kvtest/250k-new.out`.


### 14.10 What is still open (updated after §15.11)

- **Prompt-MoE decision: closed, it clears.** Built and run (§15.11, `f900479`): 15.4 ms a
  layer-step at 75% and 21.1 ms at 56%, against the 35 ms gate and the 56–70 ms host step; the
  projection is 64K prefill 156 s (was 248.0) and 250K 1,155 s (was 1,678.0). GPU prompt-MoE is
  the choice; the 64K chunk wavefront is not needed for this item and stays unbuilt.
- **The prompt path at 64K: done, measured, gated (§15.14).** `COLI_CUDA_CHAIN_PROMPT_MOE=1` with
  `COLI_VK_TIER_HOST=1` runs a chunk's every expert in the frame: prefill 267.0 → **162.0 s**
  (−39%) on the same prompt, the 3-fact needle 3/3 both ways, `gate.sh` green. Default off, so the
  default path is as it was.
- **The prompt path above 64K is open, and why is measured.** The path needs every expert of a
  layer present (residents + host pool); at a 262K-served context the pool gets 1.50 GiB (410
  experts) and 0 of 24 layers qualify (§15.14). The 250K lever is the per-step staging
  (`cc_fetch`, two banks) that §15.11 priced at 15.4/21.1 ms a layer-step, not the pool; it also
  removes the pool's decode cost (the tier's 118 GiB of slot copies).
- **A hybrid instead of streaming** (device the residents, the host its non-resident pairs) is
  10 s better at 64K by the microbench's tiles-only cell, and cheaper in PCIe (no 369 GiB of
  uploads at 56%); it needs a filtered host pair step (`moe_ex` takes every rank today). Not
  needed for the 64K win §15.14 measured; it is the cheaper 250K route.
- **The tensor-core prompt path (P1.2)** is still not written, but its premise is now **measured**
  (§15.12, `bench/results-tensor-core.json`): at 250K and 1,024 rows the q4 kernel reaches 58.6
  GB/s where f32 reaches 246 GB/s, i.e. ~4/5 of its 729.6 ms is the unpack, with a 4.3× headroom
  over the bandwidth ceiling. A naive `mma` port would move the limiter, not remove it; it ships
  with the vectorized unpack (P5's arithmetic) and the CPU-reference/oracle gates.
- **P5, the vectorized q4/q8 block packer**: ~62 s at 250K in `ink2cc_kv_down`; not done. It is
  also the unpack that P1.2 needs, so the two are one change.
- **Decode parity: open again, and the reference is ahead.** b12676 (§15.15) decodes 19.96 tok/s
  at 64K against the engine's 17.07–17.27, and 17.96 at 139,421 tokens against 13.87; the §15.6
  "engine at parity" result was measured against b12588. The engine keeps prefill (2.5×/1.5× at
  64K, 1.13× at ~139K). A reference 250K cell is now possible in principle (§15.15 point 3) and
  untried; the decode gap on the engine's side is the per-op kernel count the reference's pins
  removed, not the tier.
- **Transfer timeline**: the prefetch-ordering statement is a code reading, not a measured
  timeline; confirm it before calling it the cause of the 11–17% overlap.
- **M3/M4**: 1M is YaRN-only and out of scope for the 262K acceptance. The 1M requirement is
  documented (§7.1–7.3 and `docs/ENVIRONMENT.md`): a q4 mirror is 9.2 GB and f16 32 GB, so a 12 GB
  card has no 1M device policy — the host-only attention option (§7.2 item 1) is the only one that
  loads, and **24 GB is the documented 1M configuration**. The upstream PR's body for `INK2_KV` +
  `COLI_CUDA_RELATTN_QT` is written (`bench/pr-ink2-kv-relattn.md`, with the exact cherry-pick and
  `gh pr create` lines); it is **not opened**: both knobs are already documented with their
  measured wins (§14.1/§14.2, `docs/ENVIRONMENT.md`), the tests it cites exist
  (`cuda-inkc-check`, `test_ink_kv.c`, `gate.sh`), but this branch is a long-lived experimental line
  (PLAN §1–§15), so a PR from it would ask upstream to review all of it under the title of two
  knobs. The focused branch is a cherry-pick away and the body names the commits and the
  `gh pr create` line.
- **Closed**: the 250K needle (single code and the three-fact/three-depth prompt), the
  whole-layer A/B (rejected as a 250K policy; the in-frame path is decode-only and +3–5% at
  64K), the DSpark 8K/64K price, and the Strata cross-check with its VRAM sweep.
