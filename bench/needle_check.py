#!/usr/bin/env python3
"""Run the engine on the needle prompts and say whether the vault code came back.

    python3 bench/needle_check.py [--kv q4] [--sizes 1,2,4,8,16] [--exe c/inkling]

Each size is a generated prompt of that many thousands of tokens (bench/make_prompts.py's
sentence filler), the code at 95.6% of it, the question at the end. The answer is everything
the engine printed after the last "Answer:" up to the timing line. This is the harness
PLAN-colibri.md §6 reports on: the code is NOT found on this IQ3_XXS snapshot even at 1K, so
the failure is the model/prompt, not the KV cache or the context length.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

CODE = "7X4Q-ALPHA-93"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP = os.environ.get("NEEDLE_SNAP", "/run/media/nazar/Gaaaymes/models/Inkling-2-Mini-GGUF/iq3xxs.colibri")
SUBJECTS = ["the archive team", "the night shift", "the survey crew", "the review board", "the lab staff", "the records clerk"]
PREDS = ["checked the sensor logs", "filed the weekly report", "walked the perimeter", "tested the backup pump", "sorted the sample trays", "updated the duty roster"]
TAILS = ["before the storm arrived", "while the generator was running", "after the long meeting", "as the sun came up", "without any delay", "for the third time that week"]


def prompt(chars, rng):
    import random
    parts, cur, at = [" Notes from the station. "], 0, int(chars * 0.9)
    cur = len(parts[0])
    def s():
        return "%s %s %s. " % (rng.choice(SUBJECTS), rng.choice(PREDS), rng.choice(TAILS))
    while cur < at:
        x = s(); parts.append(x); cur += len(x)
    n = " The secret access code for the vault is %s and it must be remembered exactly. " % CODE
    parts.append(n); cur += len(n)
    while cur < chars:
        x = s(); parts.append(x); cur += len(x)
    parts.append(" Question: what is the secret access code for the vault? Answer:")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv", default="q4")
    ap.add_argument("--sizes", default="1,2,4,8,16")
    ap.add_argument("--exe", default=os.path.join(ROOT, "c", "inkling"))
    ap.add_argument("--snap", default=os.environ.get("SNAP", SNAP))
    a = ap.parse_args()
    import random
    for k in [int(x) for x in a.sizes.split(",") if x]:
        rng = random.Random(7)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(prompt(k * 4600, rng) + "\n")
            path = f.name
        env = dict(os.environ, SNAP=a.snap, CTX_MAX=str(k * 1400 + 2000), INK2_KV=a.kv,
                   COLI_CUDA="1", OMP_NUM_THREADS="6")
        p = subprocess.run([a.exe, "-f", path, "-n", "24"], cwd=ROOT, env=env,
                           capture_output=True, timeout=7200)
        os.unlink(path)
        out = p.stdout.decode("utf-8", "replace")
        ntok = re.search(r"\[(\d+) prompt tokens\]", out)
        tail = out.rsplit("Answer:", 1)[-1].split("\n[prefill")[0] if "Answer:" in out else ""
        found = CODE in tail or "ALPHA-93" in tail.upper()
        print("%2dK (%s tokens): %s  %r" % (k, ntok.group(1) if ntok else "?",
                                           "FOUND" if found else "missing", tail[:100]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
