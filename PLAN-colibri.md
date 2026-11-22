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
