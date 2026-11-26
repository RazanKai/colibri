#!/usr/bin/env python3
"""Run llama.cpp's server over the same prompts and save its answers (raw completion, greedy)."""
import json, os, re, signal, subprocess, sys, time, urllib.request

R = "/var/tmp/scratch/bench/"
BIN = "/home/nazar/opt/llama.cpp/unsloth-b12676/llama-server"
MODEL = "/run/media/nazar/Gaaaymes/models/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-IQ3_XXS.gguf"
PORT = 8093
label = sys.argv[1] if len(sys.argv) > 1 else "llama"
ctx = sys.argv[2] if len(sys.argv) > 2 else "4096"

items = [json.loads(l) for l in open(R + "items.jsonl", encoding="utf-8")]
args = [BIN, "-m", MODEL, "-ngl", "99", "--n-cpu-moe", "12", "-c", ctx, "-ctk", "q4_0", "-ctv", "q4_0",
        "-fa", "on", "-b", "512", "-ub", "128", "--parallel", "1", "--host", "127.0.0.1",
        "--port", str(PORT), "-np", "1"]
log = open(R + "server-%s.log" % label, "w")
log.write(" ".join(args) + "\n"); log.flush()
t0 = time.time()
proc = subprocess.Popen(args, cwd=os.path.dirname(BIN), stdout=log, stderr=log, text=True)
base = "http://127.0.0.1:%d" % PORT
try:
    up = False
    while time.time() - t0 < 300:
        if proc.poll() is not None: break
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as r:
                if r.status == 200: up = True; break
        except Exception:
            time.sleep(1)
    if not up:
        print("server did not come up"); proc.kill(); sys.exit(1)
    rows, gen_tok, tps = [], 0, []
    for it in items:
        body = json.dumps({"prompt": it["prompt"], "n_predict": 192, "temperature": 0.0,
                           "top_k": 1, "top_p": 1.0, "cache_prompt": False,
                           "stop": ["\n=====", "Question:", "\nQuestion"]}).encode()
        req = urllib.request.Request(base + "/completion", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.loads(r.read())
        rows.append({"id": it["id"], "kind": it["kind"], "gold": it["gold"],
                     "out": (d.get("content") or "").strip()})
        t = d.get("timings") or {}
        gen_tok += int(t.get("predicted_n") or 0)
        if t.get("predicted_per_second"): tps.append(float(t["predicted_per_second"]))
    json.dump({"engine": BIN, "label": label, "wall_s": round(time.time() - t0, 1), "gen_tokens": gen_tok,
               "decode_tok_s": round(sum(tps) / len(tps), 2) if tps else None, "rows": rows},
              open(R + "preds-%s.json" % label, "w", encoding="utf-8"), indent=1)
    print(json.dumps({"label": label, "wall_s": round(time.time() - t0, 1), "gen_tokens": gen_tok,
                      "mean_decode_tok_s": round(sum(tps) / len(tps), 2) if tps else None,
                      "blocks": len(rows)}))
finally:
    proc.send_signal(signal.SIGTERM)
    try: proc.wait(timeout=20)
    except Exception: proc.kill()
    log.close()
