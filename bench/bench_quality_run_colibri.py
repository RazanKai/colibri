#!/usr/bin/env python3
"""Run one colibri engine (ours or origin/dev) over bench_q/prompts.txt and save its answers.

    python3 run_colibri.py <engine> <label> [env KEY=VAL ...]

One process, one prompt a line (`-f`), greedy, fixed n. The answers come back from the engine's own
`===== prompt k =====` blocks; the timing of each step is the engine's own line.
"""
import json, os, re, subprocess, sys, time

R = "/var/tmp/scratch/bench/"
engine, label = sys.argv[1], sys.argv[2]
env = dict(os.environ)
env.update({"SNAP": "/run/media/nazar/Gaaaymes/models/Inkling-2-Mini-GGUF/iq3xxs.colibri",
            "CTX_MAX": "4096", "COLI_CUDA": "1", "OMP_NUM_THREADS": "6",
            "COLI_VK_TIER_BALANCE": "0"})
for kv in sys.argv[3:]:
    k, v = kv.split("=", 1)
    env[k] = v
out = R + "out-%s.txt" % label
t0 = time.time()
p = subprocess.run([engine, "-f", R + "prompts.txt", "-n", "192"], cwd=os.path.dirname(engine),
                   env=env, capture_output=True, text=True, timeout=7200)
wall = time.time() - t0
open(out, "w", encoding="utf-8").write(p.stdout)
open(R + "err-%s.txt" % label, "w", encoding="utf-8").write(p.stderr)

blocks = re.split(r"\n===== prompt \d+ =====\n", p.stdout)[1:]
items = [json.loads(l) for l in open(R + "items.jsonl", encoding="utf-8")]
rows, tok_s, gen_tok = [], [], 0
for i, it in enumerate(items):
    b = blocks[i] if i < len(blocks) else ""
    j = b.find("prompt tokens] ")
    if j >= 0:
        b = b[j + len("prompt tokens] "):]
    b = re.split(r"\n\[prefill ", b)[0]
    b = re.sub(r"\n\[eos after \d+ tokens\]", "", b)
    m = re.search(r"\| (\d+) tokens in ([\d.]+)s = ([\d.]+) tok/s", blocks[i] if i < len(blocks) else "")
    if m:
        gen_tok += int(m.group(1)); tok_s.append(float(m.group(3)))
    # the prompt echo is the prompt itself: drop its first occurrence
    q = it["prompt"]
    if b.startswith(q):
        b = b[len(q):]
    rows.append({"id": it["id"], "kind": it["kind"], "gold": it["gold"], "out": b.strip()})
json.dump({"engine": engine, "label": label, "wall_s": round(wall, 1), "gen_tokens": gen_tok,
           "decode_tok_s": round(sum(tok_s) / len(tok_s), 2) if tok_s else None, "rows": rows},
          open(R + "preds-%s.json" % label, "w", encoding="utf-8"), indent=1)
print(json.dumps({"label": label, "wall_s": round(wall, 1), "gen_tokens": gen_tok,
                  "mean_decode_tok_s": round(sum(tok_s) / len(tok_s), 2) if tok_s else None,
                  "blocks": len(blocks)}))
