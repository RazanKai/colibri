#!/usr/bin/env python3
"""DSpark gate (NEXT-SESSION step 3 / PLAN 15.3): price the official `dflash` drafter in
llama.cpp before building anything in this engine. Same prompt, temp 0, cache_prompt=false,
same ncmoe and q4 KV, with and without `-md dspark-Inkling-2-Mini-Q8_0.gguf`.

    python3 bench/dspark_gate.py --sizes 8,64

One server at a time (the caller must not run two GPU processes). The server log's
"draft acceptance = X (A accepted / G generated), mean len = L" line gives accepted tokens per
verify step (L - 1); the /completion timings give prompt/decode tok/s.
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = "/home/nazar/opt/llama.cpp/unsloth-b12588"
BIN = os.path.join(ROOT, "llama-server")
MODEL = "/run/media/nazar/Gaaaymes/models/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-IQ3_XXS.gguf"
DRAFT = "/run/media/nazar/Gaaaymes/models/Inkling-2-Mini-GGUF/dspark-Inkling-2-Mini-Q8_0.gguf"
PROMPTS = {8: "/home/nazar/src/colibri/bench/prompt-8k.txt",
           64: "/home/nazar/src/colibri/bench/prompt-64k.txt",
           128: "/home/nazar/src/colibri/bench/prompt-128k.txt"}
RX_ACC = re.compile(r"draft acceptance = (?P<rate>[0-9.]+) \(\s*(?P<acc>[0-9]+) accepted /\s*(?P<gen>[0-9]+) generated\), "
                    r"mean len =\s*(?P<len>[0-9.]+)")


def one(size, with_md, a, port):
    ctx = size * 1024
    prompt = PROMPTS[size]
    args = [BIN, "-m", MODEL, "-ngl", str(a.ngl), "--n-cpu-moe", str(a.ncmoe),
            "-c", str(ctx), "-ctk", "q4_0", "-ctv", "q4_0", "-fa", "on",
            "-b", str(a.batch), "-ub", str(a.ubatch), "--parallel", "1",
            "--host", "127.0.0.1", "--port", str(port)]
    if with_md:
        args += ["-md", DRAFT, "--spec-type", "draft-dspark", "--spec-draft-n-max", "3",
                 "--n-gpu-layers-draft", "99"]
    log_path = "/var/tmp/scratch/opencode/colibri-runs/dspark-%dK-%s.log" % (size, "md" if with_md else "nomd")
    log = open(log_path, "w", encoding="utf-8")
    log.write(" ".join(args) + "\n")
    log.flush()
    t0 = time.time()
    proc = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=log, text=True)
    base = "http://127.0.0.1:%d" % port
    healthy = False
    while time.time() - t0 < a.timeout:
        if proc.poll() is not None:
            break
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as r:
                if r.status == 200:
                    healthy = True
                    break
        except Exception:
            pass
        time.sleep(2)
    if not healthy:
        proc.terminate()
        log.close()
        return {"size_k": size, "with_md": with_md, "error": "server not healthy",
                "log": log_path, "tail": open(log_path, errors="replace").read()[-800:]}
    load_s = time.time() - t0
    text = open(prompt, encoding="utf-8").read().strip()
    body = json.dumps({"prompt": text, "n_predict": a.n_predict, "temperature": 0.0,
                       "top_k": 0, "top_p": 1.0, "min_p": 0.0,
                       "cache_prompt": False, "stream": False}).encode()
    req = urllib.request.Request(base + "/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    t1 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=a.timeout) as r:
            data = json.loads(r.read())
    except Exception as e:
        proc.terminate()
        log.close()
        return {"size_k": size, "with_md": with_md, "error": "request failed: %s" % e,
                "log": log_path, "tail": open(log_path, errors="replace").read()[-800:]}
    wall = time.time() - t1
    tim = data.get("timings") or {}
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()
    logtext = open(log_path, errors="replace").read()
    acc = RX_ACC.search(logtext)
    row = {"size_k": size, "with_md": with_md, "ctx": ctx, "ncmoe": a.ncmoe,
           "load_s": round(load_s, 1), "wall_s": round(wall, 1),
           "prompt_n": tim.get("prompt_n"), "prompt_per_second": tim.get("prompt_per_second"),
           "predicted_n": tim.get("predicted_n"), "predicted_per_second": tim.get("predicted_per_second"),
           "prompt_ms": tim.get("prompt_ms"), "predicted_ms": tim.get("predicted_ms"),
           "content_head": (data.get("content") or "")[:80], "log": log_path}
    if acc:
        row.update({"draft_rate": float(acc["rate"]), "draft_accepted": int(acc["acc"]),
                    "draft_generated": int(acc["gen"]), "mean_len": float(acc["len"]),
                    "accepted_per_verify": round(float(acc["len"]) - 1.0, 2)})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="8,64")
    ap.add_argument("--ncmoe", type=int, default=12)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--ubatch", type=int, default=128)
    ap.add_argument("--n-predict", type=int, default=32)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--out", default="/home/nazar/src/colibri/bench/results-dspark.json")
    a = ap.parse_args()
    rows = []
    for s in [int(x) for x in a.sizes.split(",") if x]:
        for md in (False, True):
            r = one(s, md, a, a.port + (1 if md else 0))
            rows.append(r)
            print(json.dumps(r), flush=True)
            with open(a.out, "w") as f:
                json.dump(rows, f, indent=1)
    print("dspark_gate: %d cells -> %s" % (len(rows), a.out))


if __name__ == "__main__":
    sys.exit(main())
