#!/usr/bin/env python3
"""P6 of the 250K speed plan: the engine-level cells the kernel microbench cannot see.

Runs the engine itself (a short 8K prompt, so a cell is ~30 s) over the knobs that only the
engine decides -- the prompt chunk's rows, the KV type, the mirror's home -- and writes the
phase line's numbers a JSON row. `attn` here is the whole device side of a layer (projections,
norms, attention, gate), not the attention kernel: for the kernel, use sweep_kernels.py.

    python3 bench/sweep_engine.py --snap /path/iq3xxs.colibri --out bench/results-engine.json
"""
import argparse
import json
import os
import re
import subprocess

PHASES = re.compile(r"\[phases\] fill (?P<fill>[0-9.]+)s \| expert-mm (?P<expert>[0-9.]+)s \| "
                    r"shared (?P<shared>[0-9.]+)s \| attn (?P<attn>[0-9.]+)s \| other (?P<other>[0-9.]+)s")
PREFILL = re.compile(r"\[prefill (?P<s>[0-9.]+)s \| (?P<gen>[0-9]+) tokens in (?P<ds>[0-9.]+)s = (?P<tps>[0-9.]+) tok/s")
TUNE = re.compile(r"TUNE decode: (?P<n>[0-9]+) tokens in (?P<s>[0-9.]+)s")

CELLS = [
    ("chunk 1024", {"COLI_CUDA_CHAIN_ROWS": ""}),
    ("chunk 2048", {"COLI_CUDA_CHAIN_ROWS": "2048"}),
    ("chunk 4096", {"COLI_CUDA_CHAIN_ROWS": "4096"}),
    ("chunk 1024, f16 kv", {"COLI_CUDA_CHAIN_ROWS": "", "INK2_KV": "f16"}),
    ("chunk 1024, q4, no store", {"COLI_CUDA_CHAIN_ROWS": "", "COLI_PREFIX_STORE": "0"}),
]


def run(binary, snap, prompt, n, env, timeout):
    e = dict(os.environ)
    e.update({"SNAP": snap, "COLI_CUDA": "1", "OMP_NUM_THREADS": e.get("OMP_NUM_THREADS", "6"),
              "INK2_KV": "q4", "CTX_MAX": "10000"})
    e.update({k: v for k, v in env.items() if v != ""})
    for k, v in env.items():
        if v == "":
            e.pop(k, None)
    out = subprocess.run([binary, "-f", prompt, "-n", str(n)], capture_output=True, text=True,
                         env=e, timeout=timeout)
    row = {"env": env, "provenance": "measured"}
    for name, rx in (("phases", PHASES), ("prefill", PREFILL), ("tune", TUNE)):
        for line in out.stdout.splitlines():
            m = rx.search(line)
            if m:
                row[name] = m.groupdict()
                break
    if "prefill" not in row:
        row["error"] = (out.stderr or out.stdout).strip()[-400:]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="c/inkling")
    ap.add_argument("--snap", required=True)
    ap.add_argument("--prompt", default="bench/prompt-8k.txt")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", default="bench/results-engine.json")
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()
    rows = []
    for name, env in CELLS:
        r = run(a.bin, a.snap, a.prompt, a.n, env, a.timeout)
        r["cell"] = name
        rows.append(r)
        p = r.get("phases", {})
        print("%-24s prefill %-8s attn %-7s expert %-7s other %-7s %s" % (
            name, r.get("prefill", {}).get("s", "?"), p.get("attn", "?"), p.get("expert", "?"),
            p.get("other", "?"), r.get("error", "")), flush=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)
    print("sweep_engine: %d cells -> %s" % (len(rows), a.out))


if __name__ == "__main__":
    main()
