# PR body — Inkling-2 long context: the KV type (`INK2_KV`) and the tiled prompt kernel (`COLI_CUDA_RELATTN_QT`)

Status: **prepared, not opened.** `gh` is authenticated (RazanKai) and the branch is pushed to
`fork/RazanKai kv-attention-longctx`, but this body describes a focused change (the two knobs, their
docs and tests) while the branch carries a long-lived experimental line of long-context work
(PLAN-colibri.md §1–§15, 250K/needle/tier/MoE measurements). Opening it as-is would ask upstream to
review the whole branch under the title of two knobs. To open it:

```bash
git checkout -b ink2-kv-relattn-qt origin/dev
git cherry-pick <the commits that add INK2_KV, the tiled kernel and its docs/tests>
gh pr create --repo JustVugg/colibri --base dev \
  --head RazanKai:ink2-kv-relattn-qt --title "inkling2: a typed KV cache (INK2_KV) and the tiled prompt kernel (COLI_CUDA_RELATTN_QT)" \
  --body-file bench/pr-ink2-kv-relattn.md
```

## What it changes

1. **`INK2_KV` — the KV cache's type for Inkling-2's attention.** `f16`, `q8`, `q4` rows through
   one put/get pair (`c/ink_kv.h`), the CUDA chain's mirror in the same type (`cc_relattn`'s
   `k_rowb`/`kv_type`), and the engine's host cache in it too, so a row's bytes and rounding are
   the same on both sides. q4 is what makes 250K fit a 12 GB card (250K q4 is ~1.2 GiB a side
   against 4.6 GiB at f16). The f32 path is untouched (`INK2_KV=f32`).
2. **`COLI_CUDA_RELATTN_QT` — the tiled prompt kernel.** `c/backend_cuda.cu`'s
   `ccg_relattn_qt_t` for prompt chunks (a tile of 8 query rows a block, K/V read once a tile, K's
   tile transposed on its float4, the tile read once for every query head), the per-(head,row)
   scalar kernel otherwise; the threshold is the env var (default 8 rows, `0` = never). Same box,
   q4, 250K, a 1,024-row chunk: 1.03 → 5.02 TFLOP/s.

## Why it is safe to take

- The kernels are checked against the CPU reference row by row: `make -C c cuda-inkc-check`
  (`tests/test_cuda_inkc.cu`), every KV type, decode and prompt, plus the relative-position and
  tau/sink inputs.
- The end-to-end gate is `bench/gate.sh`: the same checks, then a needle at 1K/2K/4K/8K through the
  engine on the CUDA path — the test that caught the typed-mirror bug (PLAN §15), where every
  kernel passed and a long prompt still returned garbage.
- `tests/test_ink_kv.c` pins the codec's rounding and row bytes; `INK2_KV=f32` reproduces the
  pre-change bits.

## Numbers (RTX 5070 12 GB, IQ3_XXS snapshot, one process)

- 64K, 63,977 tokens, q4: prefill 248.0 s (expert-mm 127.0 / attn 71.9 / other 47.5), decode
  17.77–18.09 tok/s.
- 250K, 250,156 tokens, q4: prefill 1,678.0 s (expert 658.1 / attn 836.4 / other 180.1), decode
  12.22 tok/s, the 250K needle found.
- The tiled kernel alone: 5.02 TFLOP/s against 1.03 for the scalar one at 250K, 1,024 rows (rows
  in `bench/results-speed.json`).

## What is deliberately not in it

- The q4/q8 **vectorized unpack** and the tensor-core attention: `cuda_inkc_bench` measures q4 at
  58.6 GB/s against f32's 246 GB/s, so ~4/5 of that kernel's time is the unpack, not the loads
  (PLAN §15.12). Shipping `mma` without the unpack would move the limiter, not remove it.
- The prompt expert path (PLAN §15.11): measured at 15.4 ms a layer-step against the host's 56–70,
  still unwired in the engine.
- 1M: YaRN-only and out of scope for the 262K acceptance; a 24 GB card is the documented 1M
  configuration (PLAN §7.2/§7.3, `docs/ENVIRONMENT.md`).
