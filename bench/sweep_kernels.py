#!/usr/bin/env python3
"""P6 of the 250K speed plan: the kernel tuning matrix.

Runs c/cuda_inkc_bench over the contexts, prompt rows, cache types and head counts the plan
tunes, and writes one JSON row a cell with the measured ms, GFLOP/s and GB/s. Every row is
tagged `measured` (the microbench drives the real kernels through the chain's ops table); no
number here is derived.

    python3 bench/sweep_kernels.py --bin c/cuda_inkc_bench --out bench/results-speed.json

`--quick` keeps it to the cells that change a decision (64K and 250K, q4, Mini's 24 heads, and
the decode step); the full matrix adds the other cache types and a G=8 shape.
Set env `COLI_CUDA_RELATTN_GQA=0` to sweep the old decode walk, or `COLI_CUDA_RELATTN_QT=0` to
sweep the scalar prompt kernel, for an A/B on the same binary.
"""
import argparse
import json
import os
import re
import subprocess
import sys

LINE = re.compile(
    r"(?P<what>prefill chunk|decode step)\s+(?P<ms>[0-9.]+) ms \(mean\s+(?P<mean>[0-9.]+)\)\s+"
    r"(?P<gflops>[0-9.]+) GFLOP/s\s+(?:(?P<gbs>[0-9.]+) GB/s mirror\s+(?P<mib>[0-9.]+) MiB read"
    r"|(?P<walk>[0-9.]+) MiB x (?P<heads>[0-9]+) heads walked)"
)


def run(binary, ctx, rows, reps, kvtype, heads, env):
    cmd = [binary, str(ctx), str(rows), str(reps), str(kvtype), str(heads)]
    e = dict(os.environ)
    e.update(env)
    out = subprocess.run(cmd, capture_output=True, text=True, env=e, timeout=3600)
    if out.returncode != 0:
        return {"error": out.stderr.strip()[-400:] or "exit %d" % out.returncode}
    m = LINE.search(out.stdout)
    if not m:
        return {"error": "unparsed: " + out.stdout.strip()[-300:]}
    row = {
        "what": "prefill" if rows > 0 else "decode",
        "ctx": ctx,
        "rows": rows,
        "kv": {0: "f32", 1: "f16", 2: "q8", 3: "q4"}[kvtype],
        "query_heads": heads,
        "reps": reps,
        "ms": float(m.group("ms")),
        "mean_ms": float(m.group("mean")),
        "gflop_s": float(m.group("gflops")),
        "provenance": "measured",
    }
    if m.group("gbs"):
        row["mirror_gb_s"] = float(m.group("gbs"))
        row["mirror_mib_read"] = float(m.group("mib"))
    else:
        row["walk_mib"] = float(m.group("walk"))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="c/cuda_inkc_bench")
    ap.add_argument("--out", default="bench/results-speed.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE, repeated")
    a = ap.parse_args()
    env = dict(kv.split("=", 1) for kv in a.env)
    if not os.path.exists(a.bin):
        sys.exit("sweep: %s is not built (make -C c cuda-inkc-bench CUDA=1)" % a.bin)
    cells = []
    if a.quick:
        for ctx in (65536, 262144):
            for heads in (24, 64):
                cells.append((ctx, 0, 7, 3, heads))       # the decode step
                cells.append((ctx, 1024, a.reps, 3, heads))
    else:
        for kv in (0, 1, 2, 3):
            for ctx in (8192, 32768, 65536, 262144):
                for heads in (24, 64):
                    cells.append((ctx, 0, 5, kv, heads))
                    cells.append((ctx, 1024, a.reps, kv, heads))
    rows = []
    for ctx, rows_, reps, kv, heads in cells:
        r = run(a.bin, ctx, rows_, reps, kv, heads, env)
        r["env"] = env
        rows.append(r)
        print("%-8s ctx %6d rows %4d %-3s H %2d: %s" % (
            r.get("what", "?"), ctx, rows_, r.get("kv", "?"), heads,
            ("%.2f ms, %.0f GFLOP/s" % (r["ms"], r["gflop_s"])) if "ms" in r else "FAIL " + r.get("error", "")),
            flush=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)
    print("sweep: %d cells -> %s" % (len(rows), a.out))


if __name__ == "__main__":
    main()
