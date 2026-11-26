#!/usr/bin/env python3
"""Score every engine's answers and compare them item by item (PLAN-style: measured, not claimed)."""
import json, glob, re

R = "/var/tmp/scratch/bench/"

def norm(s):
    s = s.strip().lower().replace(",", "").replace(" ", "")
    s = s.split("<|end_message|>")[0]
    return s.strip("`'\".")

def answer(kind, out):
    """The answer a grader reads: for both kinds the text after #### (the anchor every prompt asks
    for); GSM8K falls back to the last number, the code items to their first non-empty line."""
    m = re.search(r"####\s*(.+?)(?:<\|end_message\|>|\n|$)", out, re.S)
    if m and m.group(1).strip():
        return m.group(1).strip()
    if kind == "gsm8k":
        nums = re.findall(r"-?\d[\d,]*\.?\d*", out)
        return nums[-1] if nums else out[:40]
    for line in out.splitlines():
        if line.strip():
            return line
    return ""

def equal(kind, a, gold):
    a, g = norm(a), norm(gold)
    if a == g: return True
    if kind == "gsm8k":
        def num(s):
            try: return float(s)
            except Exception: return None
        x, y = num(a), num(g)
        return x is not None and y is not None and abs(x - y) < 1e-9
    return False

rows_by = {}
for f in sorted(glob.glob(R + "preds-*.json")):
    d = json.load(open(f, encoding="utf-8"))
    rows_by[d["label"]] = d

labels = list(rows_by)
print("%-12s %6s %6s %7s %9s %8s" % ("engine", "gsm8k", "code", "all", "decode", "answers"))
score = {}
for lab, d in rows_by.items():
    ok = {"gsm8k": 0, "code": 0}; n = {"gsm8k": 0, "code": 0}
    for r in d["rows"]:
        n[r["kind"]] += 1
        if equal(r["kind"], answer(r["kind"], r["out"]), r["gold"]): ok[r["kind"]] += 1
    score[lab] = ok, n
    tot, ntot = ok["gsm8k"] + ok["code"], n["gsm8k"] + n["code"]
    print("%-12s %5.1f%% %5.1f%% %6.1f%% %8s %8d" % (
        lab, 100.0 * ok["gsm8k"] / max(1, n["gsm8k"]), 100.0 * ok["code"] / max(1, n["code"]),
        100.0 * tot / max(1, ntot), d.get("decode_tok_s"), ntot))

base = labels[0]
print("\nagreement with %s (the same answer for the same item):" % base)
for lab in labels[1:]:
    same = sum(1 for a, b in zip(rows_by[base]["rows"], rows_by[lab]["rows"])
               if norm(answer(a["kind"], a["out"])) == norm(answer(b["kind"], b["out"])))
    print("  %-12s %5.1f%% (%d/%d)" % (lab, 100.0 * same / len(rows_by[base]["rows"]), same,
                                        len(rows_by[base]["rows"])))

print("\nfirst disagreements (id, gold, then each engine's extracted answer):")
shown = 0
for i, a in enumerate(rows_by[base]["rows"]):
    vals = [answer(rows_by[l]["rows"][i]["kind"], rows_by[l]["rows"][i]["out"]).replace("\n", " ")[:48]
            for l in labels]
    if len({norm(v) for v in vals}) > 1 and shown < 10:
        print("  %s gold=%r" % (a["id"], a["gold"]))
        for l, v in zip(labels, vals):
            print("     %-12s %r" % (l, v))
        shown += 1
