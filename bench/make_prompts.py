#!/usr/bin/env python3
"""Generate the long-context needle prompts this box was measured on.

    python3 bench/make_prompts.py [outdir]

Writes prompt-8k.txt / prompt-32k.txt / prompt-64k.txt (the timing prompts, if absent) and
prompt-250k.txt (a ~250K-token single-line prompt with the vault code at 95.6% of the text and
a question at the end). The engine's -f reader keeps one line whole (see PLAN-colibri.md).

The text is sentence-style filler on purpose: it is natural enough to keep the model
continuing, and it makes the prompt size predictable in tokens (~5.5 chars a token).
"""
import os
import random
import sys

CODE = "7X4Q-ALPHA-93"
SUBJECTS = ["the research team", "the archival group", "the engineering staff", "the review board",
            "the field crew", "the analysis unit", "the records office", "the system operators"]
PREDS = ["catalogued the readings from the northern site", "compared the routing statistics of the previous quarter",
         "rebuilt the index of temperature samples", "checked the calibration of the long-baseline sensors",
         "summarised the placement experiments in a short memo", "reviewed the retention policy for old logs",
         "traced the failure to a loose connector on the test bench", "measured the throughput of the storage array",
         "audited the sampling schedule for the winter campaign", "documented the recovery steps in the runbook"]
TAILS = ["so the next shift could continue without repeating the work", "and filed the results under the usual reference number",
         "before the maintenance window closed for the night", "while the instruments were still warm",
         "in a form the reviewers could check line by line", "even though the power budget was tight"]


def sentence(rng):
    return "%s %s %s. " % (rng.choice(SUBJECTS), rng.choice(PREDS), rng.choice(TAILS))


def build(chars, rng, code=CODE):
    parts, cur, at = [], 0, int(chars * 0.956)
    while cur < at:
        s = sentence(rng)
        parts.append(s)
        cur += len(s)
    n = " The secret access code for the vault is %s and it must be remembered exactly. " % code
    parts.append(n)
    cur += len(n)
    while cur < chars:
        s = sentence(rng)
        parts.append(s)
        cur += len(s)
    parts.append(" Question: what is the secret access code for the vault? Answer:")
    return "".join(parts)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__))
    os.makedirs(out, exist_ok=True)
    # ~5.52 chars a token on this prompt family: 1,381,000 chars ≈ 250K tokens
    for name, chars in (("8k", 44_300), ("32k", 177_000), ("64k", 353_000), ("250k", 1_381_000)):
        path = os.path.join(out, "prompt-%s.txt" % name)
        if name != "250k" and os.path.exists(path) and os.path.getsize(path) > 0:
            print("%s: kept (exists)" % path)
            continue
        rng = random.Random(42)
        text = build(chars, rng)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print("%s: %d chars, needle at %.1f%%" %
              (path, len(text), 100.0 * text.find(CODE) / len(text)))


if __name__ == "__main__":
    main()
