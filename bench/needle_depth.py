#!/usr/bin/env python3
"""Depth-and-type needle: several facts, of different kinds, at 10%, 50% and 90% of a long
prompt, asked together at the end. The single-code needle at ~95.6% is one fact at one depth;
a pass there can still be a lucky attention pattern. This puts a vault code (10%), a pump
serial (50%) and a cipher key (90%) in one 250K prompt and checks all three in the generation.

    python3 bench/needle_depth.py --tokens 250000 [--n 64]

The engine echoes the prompt, so the check reads only what follows the final "Answer:" cue
(the generation) up to the timing line.
"""
import argparse
import os
import random
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP = os.environ.get("NEEDLE_SNAP", "/run/media/nazar/Gaaaymes/models/Inkling-2-Mini-GGUF/iq3xxs.colibri")
FACTS = [
    (0.10, "The secret access code for the vault is 7X4Q-ALPHA-93 and it must be remembered exactly. ",
     "7X4Q-ALPHA-93"),
    (0.50, "The spare pump's maintenance serial is PUMP-41-BRAVO-77 and it must be remembered exactly. ",
     "PUMP-41-BRAVO-77"),
    (0.90, "The backup cipher key is K9-ORBIT-27 and it must be remembered exactly. ",
     "K9-ORBIT-27"),
]
SUBJECTS = ["the archive team", "the night shift", "the survey crew", "the review board", "the lab staff", "the records clerk"]
PREDS = ["checked the sensor logs", "filed the weekly report", "walked the perimeter", "tested the backup pump", "sorted the sample trays", "updated the duty roster"]
TAILS = ["before the storm arrived", "while the generator was running", "after the long meeting", "as the sun came up", "without any delay", "for the third time that week"]


def build(chars, rng):
    parts = [" Notes from the station. "]
    cur = len(parts[0])
    at = {round(f * chars): t for f, t, _ in FACTS}
    stops = sorted(at)
    for stop in stops:
        while cur < stop:
            x = "%s %s %s. " % (rng.choice(SUBJECTS), rng.choice(PREDS), rng.choice(TAILS))
            parts.append(x)
            cur += len(x)
        parts.append(at[stop])
        cur += len(at[stop])
    while cur < chars:
        x = "%s %s %s. " % (rng.choice(SUBJECTS), rng.choice(PREDS), rng.choice(TAILS))
        parts.append(x)
        cur += len(x)
    parts.append(" Question: what are the vault access code, the spare pump's maintenance serial, and the "
                 "backup cipher key, in that order? Answer:")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=250000)
    ap.add_argument("--chars-per-token", type=float, default=5.52)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--exe", default=os.path.join(ROOT, "c", "inkling"))
    ap.add_argument("--snap", default=os.environ.get("SNAP", SNAP))
    ap.add_argument("--log", default="/var/tmp/scratch/opencode/colibri-runs/needle-depth-250k.log")
    a = ap.parse_args()
    rng = random.Random(11)
    text = build(int(a.tokens * a.chars_per_token), rng)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(text + "\n")
        path = f.name
    env = dict(os.environ, SNAP=a.snap, CTX_MAX="262144", INK2_KV="q4",
               COLI_CUDA="1", OMP_NUM_THREADS="6")
    with open(a.log, "w") as log:
        log.write("prompt chars %d (%.0f tokens at %.2f chars/token), facts at 10/50/90%%\n"
                  % (len(text), len(text) / a.chars_per_token, a.chars_per_token))
        log.flush()
        p = subprocess.run([a.exe, "-f", path, "-n", str(a.n)], cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, timeout=7200)
    os.unlink(path)
    out = open(a.log, errors="replace").read()
    ntok = re.search(r"\[(\d+) prompt tokens\]", out)
    tail = out.rsplit("Answer:", 1)[-1].split("\n[prefill")[0] if "Answer:" in out else ""
    print("prompt %s tokens | exit %d | generation tail:\n%s" %
          (ntok.group(1) if ntok else "?", p.returncode, tail.strip()[-400:]))
    ok = 0
    for frac, _, code in FACTS:
        found = code in tail
        ok += found
        print("  %2d%%  %-16s %s" % (round(frac * 100), code, "FOUND" if found else "missing"))
    print("needle_depth: %d of %d facts" % (ok, len(FACTS)))
    return 0 if ok == len(FACTS) else 1


if __name__ == "__main__":
    sys.exit(main())
