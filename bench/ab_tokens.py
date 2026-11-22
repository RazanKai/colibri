#!/usr/bin/env python3
"""Tokenize the same text in both engines and diff the ids.

The needle A/B (bench/ab_needle.py) shows llama.cpp reads the code from a 4K-char prompt while
this fork emits unrelated tokens, on the same GGUF. The first thing to rule in or out is the
tokenizer: this starts the Unsloth server, POSTs /tokenize, runs this fork with INK_IDS_DUMP on
the identical bytes, and prints both lengths and the first index where the ids differ.

    python3 bench/ab_tokens.py [--chars 4000] [--port 8083]
"""
import argparse
import array
import json
import os
import random
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench"))
from needle_check import prompt as needle_prompt  # noqa: E402

LLAMA_ROOT = "/home/nazar/opt/llama.cpp/unsloth-b12588"
LLAMA_BIN = os.path.join(LLAMA_ROOT, "llama-server")
GGUF = "/run/media/nazar/Gaaaymes/models/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-IQ3_XXS.gguf"
SNAP = "/run/media/nazar/Gaaaymes/models/Inkling-2-Mini-GGUF/iq3xxs.colibri"
IDS = "/var/tmp/scratch/kvtest/ours.ids"
TXT = "/var/tmp/scratch/kvtest/ab-needle.txt"


def llama_tokens(text, port, timeout=600):
    args = [LLAMA_BIN, "-m", GGUF, "-ngl", "99", "--n-cpu-moe", "8", "-c", "8192",
            "-ctk", "q4_0", "-ctv", "q4_0", "-fa", "on", "--parallel", "1",
            "--host", "127.0.0.1", "--port", str(port)]
    log = open("/var/tmp/scratch/kvtest/ab-tok-llama.log", "w", encoding="utf-8")
    log.write(" ".join(args) + "\n"); log.flush()
    proc = subprocess.Popen(args, cwd=LLAMA_ROOT, stdout=log, stderr=log, text=True)
    base = "http://127.0.0.1:%d" % port
    try:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if proc.poll() is not None:
                sys.exit("llama-server exited %d (see ab-tok-llama.log)" % proc.returncode)
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            time.sleep(2)
        body = json.dumps({"content": text}).encode()
        req = urllib.request.Request(base + "/tokenize", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()).get("tokens") or []
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=int, default=4000)
    ap.add_argument("--port", type=int, default=8083)
    a = ap.parse_args()
    rng = random.Random(1234)
    text = needle_prompt(a.chars, rng)
    open(TXT, "w", encoding="utf-8").write(text)

    theirs = llama_tokens(text, a.port)
    print("llama.cpp: %d tokens" % len(theirs))

    env = dict(os.environ, SNAP=SNAP, CTX_MAX="8192", INK2_KV="q4", COLI_CUDA="1",
               OMP_NUM_THREADS="6", INK_IDS_DUMP=IDS)
    out = subprocess.run([os.path.join(ROOT, "c", "inkling"), "-f", TXT, "-n", "2"],
                         capture_output=True, text=True, env=env, timeout=900)
    ours = list(array.array("i", open(IDS, "rb").read()))
    print("this fork: %d tokens" % len(ours))

    n = min(len(ours), len(theirs))
    first = next((i for i in range(n) if ours[i] != theirs[i]), None)
    if len(ours) == len(theirs) and first is None:
        print("IDS IDENTICAL up to length: the prompt path agrees, the bug is in the forward pass")
        return
    print("first difference at %s of %d compared" % (first, n))
    if first is not None:
        lo = max(0, first - 4)
        print("  theirs[%d:%d] = %s" % (lo, first + 5, theirs[lo:first + 5]))
        print("  ours  [%d:%d] = %s" % (lo, first + 5, ours[lo:first + 5]))
        # the text around the divergence, by token position in llama's stream
        print("  (ids differ: tokenizer merges, or the bytes the engine was given)")


if __name__ == "__main__":
    main()
