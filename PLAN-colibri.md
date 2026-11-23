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

## Appendix — benchmark commands

```bash
# build (CUDA primary; Vulkan fallback is `VK=1`)
make -C c inkling CUDA=1 CUDA_HOME=/opt/cuda

# tests
cc -O2 -march=native -I c c/tests/test_ink_kv.c -lm -o /tmp/test_ink_kv && /tmp/test_ink_kv
make -C c cuda-inkc-check CUDA=1 CUDA_HOME=/opt/cuda

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


### 14.10 What is still open

- **The tensor-core prompt path (P1.2)** was not written. The gate it needed is measured: f16
  is range-safe (§14.1) and the kernel/other split is now measured too (§14.8): at 250K the
  kernel is ~566 s and the other device work ~266 s, so a 2.5× kernel wins ~340 s and no more.
  A naive `mma` port would leave the q4 dequant (≈1280 instructions a position per KV head
  against 768 FMAs) as the limiter; it only pays with a vectorized unpack in the same change.
  It must still pass the CPU-reference and oracle gates.
- **P4** waits on a `cc_moe` IQ2_S path.
- **P5** waits on a vectorized block packer (~62 s at 250K).
- **§6's needle** is still the gate on any retrieval claim at 250K.
