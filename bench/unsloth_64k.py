#!/usr/bin/env python3
"""Baseline: Unsloth b12588, IQ3_XXS + DSpark, 64K context, --n-cpu-moe."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = "/home/nazar/opt/llama.cpp/unsloth-b12588"
BIN = os.path.join(ROOT, "llama-server")
MODEL = "/run/media/nazar/Gaaaymes/models/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-IQ3_XXS.gguf"
DRAFT = "/run/media/nazar/Gaaaymes/models/Inkling-2-Mini-GGUF/dspark-Inkling-2-Mini-Q8_0.gguf"
PROMPT = "/home/nazar/src/colibri/bench/prompt-64k.txt"
LOG = "/home/nazar/src/colibri/bench/unsloth-64k.log"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncmoe", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--ubatch", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=1800)
    a = ap.parse_args()

    args = [
        BIN,
        "-m", MODEL,
        "-md", DRAFT,
        "--spec-type", "draft-dspark",
        "--spec-draft-n-max", "3",
        "--n-gpu-layers-draft", "99",
        "-ngl", str(a.ngl),
        "--n-cpu-moe", str(a.ncmoe),
        "-c", str(a.ctx),
        "-ctk", "q4_0", "-ctv", "q4_0", "-fa", "on",
        "-b", str(a.batch), "-ub", str(a.ubatch),
        "--parallel", "1",
        "--host", "127.0.0.1", "--port", str(a.port),
    ]
    log = open(LOG, "w", encoding="utf-8")
    log.write(" ".join(args) + "\n")
    log.flush()
    print("start:", " ".join(args), flush=True)
    t0 = time.time()
    proc = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=log, text=True)
    base = f"http://127.0.0.1:{a.port}"
    healthy = False
    while time.time() - t0 < a.timeout:
        if proc.poll() is not None:
            print("server exited", proc.returncode, flush=True)
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
        print("not healthy; log tail:", flush=True)
        log.flush()
        os.system(f"tail -30 {LOG}")
        proc.terminate()
        return 1
    load_s = time.time() - t0
    print(f"healthy after {load_s:.1f}s", flush=True)

    prompt = open(PROMPT, encoding="utf-8").read().strip()
    body = json.dumps({
        "prompt": prompt,
        "n_predict": 32,
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "min_p": 0.0,
        "cache_prompt": False,
        "stream": False,
    }).encode()
    req = urllib.request.Request(base + "/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    t1 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=a.timeout) as r:
            data = json.loads(r.read())
    except Exception as e:
        print("request failed:", e, flush=True)
        log.flush()
        os.system(f"tail -30 {LOG}")
        proc.terminate()
        return 1
    wall = time.time() - t1
    tim = data.get("timings") or {}
    row = {
        "backend": "unsloth-b12588",
        "ncmoe": a.ncmoe,
        "ctx": a.ctx,
        "load_s": round(load_s, 1),
        "wall_s": round(wall, 1),
        "prompt_n": tim.get("prompt_n"),
        "prompt_per_second": tim.get("prompt_per_second"),
        "predicted_n": tim.get("predicted_n"),
        "predicted_per_second": tim.get("predicted_per_second"),
        "prompt_ms": tim.get("prompt_ms"),
        "predicted_ms": tim.get("predicted_ms"),
        "content_head": (data.get("content") or "")[:120],
    }
    print(json.dumps(row), flush=True)
    with open("/home/nazar/src/colibri/bench/unsloth-64k.json", "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)
        f.write("\n")
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
