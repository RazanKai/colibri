#!/usr/bin/env python3
"""Build the engine-comparison benchmark: real GSM8K items + execution-grounded code tracing.

Every gold answer is either the dataset's own (`gsm8k_test.jsonl`, OpenAI's grade-school-math) or the
output of actually running the snippet, so nothing here is graded by opinion. One prompt a line, for
the engines' `-f` (one prompt per line).
"""
import json, os, random, subprocess, sys, re, urllib.request

R = "/var/tmp/scratch/bench/"          # items.jsonl and prompts.txt land here; the runners read them
os.makedirs(R, exist_ok=True)
if not os.path.exists(R + "gsm8k_test.jsonl"):   # OpenAI's grade-school-math test set (1319 items)
    urllib.request.urlretrieve("https://raw.githubusercontent.com/openai/grade-school-math/master/"
                               "grade_school_math/data/test.jsonl", R + "gsm8k_test.jsonl")
N_GSM = int(sys.argv[1]) if len(sys.argv) > 1 else 60
N_CODE = int(sys.argv[2]) if len(sys.argv) > 2 else 60

PREFIX = ("<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
          "<|message_user|><|content_text|>")
SUFFIX = "<|end_message|><|message_model|><|content_text|>"
items = []

# ---- GSM8K: real items, gold = the number after the dataset's "####" ----
qs = [json.loads(l) for l in open(R + "gsm8k_test.jsonl", encoding="utf-8")]
for i, q in enumerate(qs[:N_GSM]):
    gold = q["answer"].split("####")[-1].strip().replace(",", "")
    prompt = ("Question: " + " ".join(q["question"].split()) +
              " Solve it. End your answer with the final number after ####. Answer:")
    items.append({"id": "gsm%03d" % i, "kind": "gsm8k", "prompt": prompt, "gold": gold,
                  "golds": [gold, gold + ".0"]})

# ---- code tracing: gold = what the snippet prints when run ----
TEMPL = [
    "x = {a}; y = {b}; x += y; print(x)",
    "xs = [{a}, {b}, {c}]; print(xs[1])",
    "print(sum([{a}, {b}, {c}]))",
    "def f(x): return x * 2; print(f({a}))",
    "s = 'abcdef'; print(s[{a}:{b}])",
    "print({a} % {b})",
    "print(str({a}) + str({b}))",
    "print(sorted([{c}, {a}, {b}])[0])",
    "x = [{a}, {b}, {c}]; x.reverse(); print(x)",
    "print(len('hello') + {a})",
    "x = 0; for i in range({a}): x += i; print(x)",
    "print(max({a}, {b}) - min({a}, {b}))",
    "print({a} * {b} - {c})",
    "x = {{'p': {a}, 'q': {b}}}; print(x['q'])",
    "print([i * i for i in range({a})])",
    "print('{c}' * {a})",
    "x = {a}; y = {b}; print(x if x > y else y)",
    "print(float({a}) / {b})",
    "t = ({a}, {b}); print(t[1])",
    "print(abs({a} - {b}) + {c})",
]
rng = random.Random(20261126)
seen = set()
i = 0
while len([x for x in items if x["kind"] == "code"]) < N_CODE:
    t = TEMPL[i % len(TEMPL)]
    a, b, c = rng.randint(2, 40), rng.randint(2, 9), rng.randint(2, 30)
    if "{b}" in t and "%" in t and b == 0:
        b = 3
    code = t.format(a=a, b=b, c=c)
    if code in seen:
        i += 1
        continue
    seen.add(code)
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=5)
    except Exception:
        i += 1
        continue
    if out.returncode != 0:
        i += 1
        continue
    gold = out.stdout.rstrip("\n")
    if not gold or len(gold) > 40:
        i += 1
        continue
    prompt = ("What does this Python print? " + code + " End your answer with the printed value after ####.")
    items.append({"id": "code%03d" % i, "kind": "code", "prompt": prompt, "gold": gold,
                  "golds": [gold], "code": code})
    i += 1

for it in items:
    it["raw"] = it["prompt"]
    it["prompt"] = PREFIX + it["prompt"] + SUFFIX     # the engine's own --chat framing, literal
with open(R + "items.jsonl", "w", encoding="utf-8") as f:
    for it in items:
        f.write(json.dumps(it) + "\n")
with open(R + "prompts.txt", "w", encoding="utf-8") as f:
    for it in items:
        assert "\n" not in it["prompt"]
        f.write(it["prompt"] + "\n")

n_gsm = sum(1 for x in items if x["kind"] == "gsm8k")
print("items", len(items), "gsm", n_gsm, "code", len(items) - n_gsm)
for it in items[:2] + items[n_gsm:n_gsm + 2]:
    print("-", it["id"], "|", it["prompt"][:110].replace("\n", " "), "| gold:", it["gold"])
