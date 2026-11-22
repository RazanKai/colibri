#!/usr/bin/env python3
"""Test the Strata SQ-10 Inkling-2-Mini pack on the running server: the long-prompt
read (three facts of two kinds at 10/50/90%, and a plain 64K filler), with the server's
own timings (prompt/predicted tok/s, DSpark draft_n/draft_n_accepted).

    python3 bench/strata_test.py --url http://127.0.0.1:8080 --tokens 8,56

Rows go to bench/results-strata.json. The same three-fact prompt at 244,180 tokens is the
colibri check in PLAN 15.7; keep this under the served max_model_len (65,536 here).
"""
import argparse
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench"))
from needle_depth import FACTS, build  # noqa: E402

PLAIN = os.path.join(ROOT, "bench", "prompt-64k.txt")


def ask(url, model, prompt, max_tokens, temperature, timeout):
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": temperature}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    wall = time.time() - t0
    content = (d.get("choices") or [{}])[0].get("message", {}).get("content", "")
    return d, content, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="inkling-2-mini-sq-10")
    ap.add_argument("--tokens", default="8,56", help="needle prompt sizes in thousands")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--out", default=os.path.join(ROOT, "bench", "results-strata.json"))
    ap.add_argument("--plain-64k", action="store_true", help="also send bench/prompt-64k.txt")
    a = ap.parse_args()
    rows = []
    for k in [int(x) for x in a.tokens.split(",") if x]:
        import random
        p = build(k * 1000 * 5.52, random.Random(11))
        d, content, wall = ask(a.url, a.model, p, a.max_tokens, a.temperature, a.timeout)
        tim = d.get("timings") or {}
        facts = {code: (code in content) for _, _, code in FACTS}
        row = {"kind": "needle3", "target_k": k, "wall_s": round(wall, 1),
               "prompt_n": tim.get("prompt_n"), "prompt_per_second": tim.get("prompt_per_second"),
               "predicted_n": tim.get("predicted_n"),
               "predicted_per_second": tim.get("predicted_per_second"),
               "draft_n": tim.get("draft_n"), "draft_n_accepted": tim.get("draft_n_accepted"),
               "facts": facts, "found": sum(facts.values()), "content_head": content[:160]}
        rows.append(row)
        print(json.dumps(row), flush=True)
    if a.plain_64k:
        text = open(PLAIN, encoding="utf-8").read().strip()
        d, content, wall = ask(a.url, a.model, text, a.max_tokens, a.temperature, a.timeout)
        tim = d.get("timings") or {}
        row = {"kind": "plain64k", "wall_s": round(wall, 1),
               "prompt_n": tim.get("prompt_n"), "prompt_per_second": tim.get("prompt_per_second"),
               "predicted_n": tim.get("predicted_n"),
               "predicted_per_second": tim.get("predicted_per_second"),
               "draft_n": tim.get("draft_n"), "draft_n_accepted": tim.get("draft_n_accepted"),
               "content_head": content[:160]}
        rows.append(row)
        print(json.dumps(row), flush=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    print("strata_test: %d rows -> %s" % (len(rows), a.out))


if __name__ == "__main__":
    main()
