#!/usr/bin/env python3
"""Same prompt, same snapshot, two engines: is the bad output ours or the model's?

Runs the PLAN section 6 needle prompt (bench/needle_check.py's filler) through
  (a) Unsloth's llama.cpp server (bench/unsloth_64k.py's build, raw /completion, temperature 0), and
  (b) this fork's engine (raw continuation, the same text),
and prints both answers. It says whether the vault code came back in each, so a failure can be
attributed: llama.cpp coherent + ours not = our prompt/template/sampling path; both incoherent =
the snapshot or the model's instruction-following.

    python3 bench/ab_needle.py [--chars 4000] [--n 32] [--port 8083]
"""
import argparse
import json
import os
import random
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench"))
from needle_check import CODE, prompt as needle_prompt  # noqa: E402

LLAMA_ROOT = "/home/nazar/opt/llama.cpp/unsloth-b12588"
LLAMA_BIN = os.path.join(LLAMA_ROOT, "llama-server")
GGUF = "/run/media/nazar/Gaaaymes/models/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-IQ3_XXS.gguf"
SNAP = "/run/media/nazar/Gaaaymes/models/Inkling-2-Mini-GGUF/iq3xxs.colibri"


def llama_answer(prompt, n, port, timeout=900):
    args = [LLAMA_BIN, "-m", GGUF, "-ngl", "99", "--n-cpu-moe", "8", "-c", "8192",
            "-ctk", "q4_0", "-ctv", "q4_0", "-fa", "on", "-b", "1024", "-ub", "256",
            "--parallel", "1", "--host", "127.0.0.1", "--port", str(port)]
    log = open("/var/tmp/scratch/kvtest/ab-llama.log", "w", encoding="utf-8")
    log.write(" ".join(args) + "\n"); log.flush()
    proc = subprocess.Popen(args, cwd=LLAMA_ROOT, stdout=log, stderr=log, text=True)
    base = "http://127.0.0.1:%d" % port
    t0 = time.time()
    try:
        while time.time() - t0 < timeout:
            if proc.poll() is not None:
                return None, "server exited %d; see /var/tmp/scratch/kvtest/ab-llama.log" % proc.returncode
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            time.sleep(2)
        else:
            return None, "no health after %ds" % timeout
        body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0.0,
                           "top_k": 0, "top_p": 1.0, "min_p": 0.0, "cache_prompt": False,
                           "stream": False}).encode()
        req = urllib.request.Request(base + "/completion", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        return (data.get("content") or ""), None
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()


def ours_answer(prompt_path, n, ctx, timeout=900):
    env = dict(os.environ, SNAP=SNAP, CTX_MAX=str(ctx), INK2_KV="q4", COLI_CUDA="1",
               OMP_NUM_THREADS="6")
    out = subprocess.run([os.path.join(ROOT, "c", "inkling"), "-f", prompt_path, "-n", str(n)],
                         capture_output=True, text=True, env=env, timeout=timeout)
    text = out.stdout
    i = text.rfind("Answer:")
    ans = text[i + len("Answer:"):] if i >= 0 else text
    j = ans.find("\n[prefill")
    if j >= 0:
        ans = ans[:j]
    return ans.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=int, default=4000)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--port", type=int, default=8083)
    a = ap.parse_args()
    rng = random.Random(1234)
    p = needle_prompt(a.chars, rng)
    os.makedirs("/var/tmp/scratch/kvtest", exist_ok=True)
    path = "/var/tmp/scratch/kvtest/ab-needle.txt"
    open(path, "w", encoding="utf-8").write(p)
    print("prompt: %d chars, %d sentences, code %s at 90%%\n" % (len(p), p.count(". "), CODE))
    print("== Unsloth llama.cpp (raw completion, temp 0)")
    t0 = time.time()
    lc, err = llama_answer(p, a.n, a.port)
    if err:
        print("  FAILED:", err)
    else:
        print("  %.1fs, %d chars:" % (time.time() - t0, len(lc)))
        print("  " + lc[:400].replace("\n", " "))
        print("  vault code present:", CODE in lc)
    print("\n== this fork (raw continuation, greedy)")
    t0 = time.time()
    try:
        oo = ours_answer(path, a.n, 8192)
        print("  %.1fs, %d chars:" % (time.time() - t0, len(oo)))
        print("  " + oo[:400].replace("\n", " "))
        print("  vault code present:", CODE in oo)
    except subprocess.TimeoutExpired:
        print("  FAILED: timeout")


if __name__ == "__main__":
    main()
