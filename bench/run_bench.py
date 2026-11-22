#!/usr/bin/env python3
"""Benchmark Colibri inkling2 on the local box at several contexts."""
import json
import os
import re
import subprocess
import sys
import time

ROOT = "/home/nazar/src/colibri"
BIN = os.path.join(ROOT, "c", "inkling")
SNAP = "/run/media/nazar/Gaaaymes/models/Inkling-2-Mini-GGUF/iq3xxs.colibri"
CASES = [
    ("8k", 10000, os.path.join(ROOT, "bench", "prompt-8k.txt")),
    ("32k", 36000, os.path.join(ROOT, "bench", "prompt-32k.txt")),
    ("64k", 70000, os.path.join(ROOT, "bench", "prompt-64k.txt")),
]

def main():
    results = []
    for name, ctx, path in CASES:
        prompt = open(path, encoding="utf-8").read().strip()
        env = os.environ.copy()
        env.update({
            "SNAP": SNAP,
            "CTX_MAX": str(ctx),
            "COLI_CUDA": "1",
            "OMP_NUM_THREADS": "6",
        })
        t0 = time.time()
        try:
            cmd = [BIN, "-p", prompt, "-n", "32"]
            if len(prompt) > 130000:
                cmd = [BIN, "-f", path, "-n", "32"]
            p = subprocess.run(
                cmd,
                cwd=ROOT, env=env, text=True, capture_output=True,
                timeout=3600,
            )
            wall = time.time() - t0
            out, err, rc = p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:
            wall = time.time() - t0
            out = (e.stdout or "") if isinstance(e.stdout, str) else ""
            err = (e.stderr or "") if isinstance(e.stderr, str) else ""
            rc = "timeout"
        log = os.path.join(ROOT, "bench", f"out-{name}.log")
        with open(log, "w", encoding="utf-8") as f:
            f.write(out)
            f.write("\n--- stderr ---\n")
            f.write(err)
        pre = re.search(
            r"\[prefill ([\d.]+)s \| (\d+) tokens in ([\d.]+)s = ([\d.]+) tok/s \| RSS ([\d.]+) GB\]",
            out)
        tune = re.search(r"TUNE decode: (\d+) tokens in ([\d.]+)s", out)
        prompt_tokens = re.search(r"\[(\d+) prompt tokens", out)
        cfg = re.search(r"cfg: D=(\d+) L=(\d+) V=(\d+)\((\d+)\) heads=(\d+)/(\d+) kv=(\d+)/(\d+) hd=(\d+) win=(\d+) d_rel=(\d+) ext=(\d+) E=(\d+)\+(\d+) topk=(\d+)", out)
        kv = re.search(r"\[kv\] ([\d.]+) MiB", err) or re.search(r"\[kv\] ([\d.]+) MiB", out)
        row = {
            "case": name,
            "ctx_max": ctx,
            "prompt_chars": len(prompt),
            "prompt_tokens": int(prompt_tokens.group(1)) if prompt_tokens else None,
            "prefill_s": float(pre.group(1)) if pre else None,
            "gen_tokens": int(pre.group(2)) if pre else None,
            "decode_s": float(pre.group(3)) if pre else None,
            "decode_tok_s": float(pre.group(4)) if pre else None,
            "rss_gb": float(pre.group(5)) if pre else None,
            "tune_tokens": int(tune.group(1)) if tune else None,
            "tune_s": float(tune.group(2)) if tune else None,
            "kv_mib": float(kv.group(1)) if kv else None,
            "wall_s": round(wall, 1),
            "rc": rc,
            "cfg": cfg.groups() if cfg else None,
            "log": log,
        }
        results.append(row)
        print(json.dumps(row), flush=True)
    with open(os.path.join(ROOT, "bench", "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

if __name__ == "__main__":
    main()
