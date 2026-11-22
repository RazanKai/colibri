#!/usr/bin/env python3
"""Whole-layer residency A/B (PLAN 15.2-15.3): run the engine with COLI_VK_TIER_WHOLE=n and
record, per cell, the numbers the plan asks for -- budget, launch free VRAM, evictions, uploads,
cold experts kept on the CPU, "N of 24 layers in the frame", the phases, prefill, decode tok/s,
RSS, and the tier report's overlap split. One cell at a time; the caller must not run two.

    python3 bench/sweep_whole.py --snap .../iq3xxs.colibri --mode sweep64
    python3 bench/sweep_whole.py --snap .../iq3xxs.colibri --mode k250
    python3 bench/sweep_whole.py --snap ... --cells n12 --prompt bench/prompt-64k.txt

Every row is written as it finishes: a 64K cell is ~5 min, a 250K cell ~30 min.
"""
import argparse
import json
import os
import re
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RX = {
    "free": re.compile(r"device 0 ready, (?P<gb>[0-9.]+) GB free"),
    "budget": re.compile(r"budget (?P<val>[0-9.]+) (?P<unit>GiB|MiB) = (?P<n>[0-9]+) experts"),
    "layers": re.compile(r"the routed experts of (?P<n>[0-9]+) of (?P<l>[0-9]+) layers in the frame"),
    "phases": re.compile(r"\[phases\] fill (?P<fill>[0-9.]+)s \| expert-mm (?P<expert>[0-9.]+)s \| "
                         r"shared (?P<shared>[0-9.]+)s \| attn (?P<attn>[0-9.]+)s \| other (?P<other>[0-9.]+)s"),
    "prefill": re.compile(r"\[prefill (?P<s>[0-9.]+)s \| (?P<gen>[0-9]+) tokens in (?P<ds>[0-9.]+)s = "
                          r"(?P<tps>[0-9.]+) tok/s \| RSS (?P<rss>[0-9.]+) GB\]"),
    "tune": re.compile(r"TUNE decode: (?P<n>[0-9]+) tokens in (?P<s>[0-9.]+)s"),
    # tier inkling run: device 1718 of 1790 routed experts (96.0%; this run ...) | ... uploads N (X GiB, M warm), evictions N, skipped ...
    "tier": re.compile(r"tier inkling run: device (?P<served>[0-9]+) of (?P<routed>[0-9]+) routed experts "
                       r"\((?P<pct>[0-9.]+)%;.*?uploads (?P<uploads>[0-9]+) \((?P<up_gib>[0-9.]+) (?P<up_unit>GiB|MiB), "
                       r"(?P<warm>[0-9]+) warm\), evictions (?P<evictions>[0-9]+), skipped (?P<qfull>[0-9]+) queue \+ "
                       r"(?P<rated>[0-9]+) rate, failed (?P<failed>[0-9]+) \| device (?P<dev_ms>[0-9.]+) ms, "
                       r"CPU share (?P<cpu_ms>[0-9.]+) ms, waited (?P<wait_ms>[0-9.]+) ms "
                       r"\((?P<hidden_pct>[0-9.]+)% of device time hidden\)"),
    "stream": re.compile(r"stream: (?P<steps>[0-9]+) steps, (?P<cold>[0-9]+) cold experts \((?P<bytes>[^,]*), "
                         r"(?P<gbps>[0-9.]+) GB/s\) and (?P<rows>[0-9]+) rows streamed in (?P<subs>[0-9]+) sub-batches, "
                         r"prefetched (?P<pf>[0-9]+) \((?P<pf_used>[0-9]+) used\), (?P<kept>[0-9]+) cold experts "
                         r"\((?P<kept_rows>[0-9]+) rows\) kept on the CPU"),
    "host": re.compile(r"inkling2 host: (?P<kv_s>[0-9.]+) s quantizing.*?, (?P<host_s>[0-9.]+) s of routed experts"),
    "kv": re.compile(r"inkling2 attention kernels: (?P<s>[0-9.]+) s over (?P<calls>[0-9]+) calls"),
}


def parse(path):
    text = open(path, "r", errors="replace").read()
    row = {}
    for name, rx in RX.items():
        m = None
        for m in rx.finditer(text):
            pass
        if m:
            row[name] = m.groupdict()
    ans = ""
    if "[prefill" in text:
        # stdout and stderr share the log, so the last plain line before the timing line is the
        # generation (the engine's own lines are bracketed / prefixed).
        before = text.rsplit("\n[prefill", 1)[0]
        lines = [l.strip() for l in before.splitlines()
                 if l.strip() and not l.lstrip().startswith(("[", "CMD", "ENV", "START", "EXIT"))
                 and "tier inkling stream:" not in l and "inkling2" not in l]
        ans = lines[-1][:200] if lines else ""
    row["answer"] = ans
    row["found"] = ("7X4Q-ALPHA-93" in ans) or ("ALPHA-93" in ans.upper())
    return row


def run_cell(binary, snap, prompt, ctx, n, env, logdir, timeout):
    name = "%s-n%s" % (os.path.basename(prompt).replace(".txt", ""), env.get("COLI_VK_TIER_WHOLE", "0"))
    log = os.path.join(logdir, "%s-%s.log" % (time.strftime("%H%M%S"), name))
    e = dict(os.environ)
    e.update({"SNAP": snap, "COLI_CUDA": "1", "OMP_NUM_THREADS": e.get("OMP_NUM_THREADS", "6"),
              "INK2_KV": "q4", "CTX_MAX": str(ctx)})
    e.update(env)
    free_before = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader"],
                                 capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    with open(log, "w") as f:
        f.write("CMD %s -f %s -n %d\nENV %s\nSTART %s free=%s\n" %
                (binary, prompt, n, json.dumps(env), time.strftime("%Y-%m-%dT%H:%M:%S"), free_before))
        f.flush()
        p = subprocess.run([binary, "-f", prompt, "-n", str(n)], cwd=ROOT, env=e,
                           stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
        f.write("\nEXIT %d END %s free=%s\n" % (p.returncode, time.strftime("%Y-%m-%dT%H:%M:%S"),
                 subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader"],
                                capture_output=True, text=True).stdout.strip()))
    row = parse(log)
    row.update({"cell": env.get("COLI_VK_TIER_WHOLE", "0"), "prompt": os.path.basename(prompt),
                "ctx": ctx, "n": n, "log": log, "wall_s": round(time.time() - t0, 1),
                "free_before": free_before})
    return row


def summarize(r):
    b = r.get("budget", {})
    f = r.get("free", {})
    p = r.get("prefill", {})
    ph = r.get("phases", {})
    t = r.get("tier", {})
    s = r.get("stream", {})
    l = r.get("layers", {})
    tune = r.get("tune", {})
    dec = ""
    if tune:
        try:
            dec = "%.2f" % ((int(tune["n"])) / float(tune["s"]))
        except Exception:
            pass
    return ("n=%-2s budget %s %s = %s exp | free %s GB | frame %s/%s layers | prefill %s s | dec %s tok/s | "
            "RSS %s GB | evict %s up %s %s %s warm | cold %s kept %s rows | hidden %s%% dev / hostov %s | %s" % (
                r["cell"], b.get("val", "?"), b.get("unit", ""), b.get("n", "?"), f.get("gb", "?"),
                l.get("n", "?"), l.get("l", "?"), p.get("s", "?"), dec, p.get("rss", "?"),
                t.get("evictions", "?"), t.get("uploads", "?"), t.get("up_gib", "?"), t.get("up_unit", ""),
                t.get("warm", "?"), s.get("cold", "?"), s.get("kept_rows", "?"),
                t.get("hidden_pct", "?"), r.get("host_ovl", "?"), r["log"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="c/inkling")
    ap.add_argument("--snap", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "bench", "results-whole.json"))
    ap.add_argument("--logdir", default="/var/tmp/scratch/opencode/colibri-runs")
    ap.add_argument("--mode", default="sweep64", choices=["sweep64", "k250", "cells"])
    ap.add_argument("--cells", default="n12")
    ap.add_argument("--timeout", type=int, default=7200)
    a = ap.parse_args()
    os.makedirs(a.logdir, exist_ok=True)

    if a.mode == "sweep64":
        cells = [("0", "bench/prompt-64k.txt", 70000, 16), ("0", "bench/prompt-64k.txt", 70000, 16),
                 ("6", "bench/prompt-64k.txt", 70000, 16), ("12", "bench/prompt-64k.txt", 70000, 16),
                 ("13", "bench/prompt-64k.txt", 70000, 16), ("17", "bench/prompt-64k.txt", 70000, 16)]
    elif a.mode == "k250":
        cells = [("12", "bench/prompt-250k.txt", 262144, 24), ("0", "bench/prompt-250k.txt", 262144, 24)]
    else:
        cells = [(c, "bench/prompt-64k.txt", 70000, 16) for c in a.cells.split(",") if c]

    rows = []
    existing = []
    if os.path.exists(a.out):
        try:
            existing = json.load(open(a.out))
        except Exception:
            existing = []
    for whole, prompt, ctx, n in cells:
        r = run_cell(a.bin, a.snap, prompt, ctx, n, {"COLI_VK_TIER_WHOLE": whole}, a.logdir, a.timeout)
        t = r.get("tier", {})
        try:  # the fraction of the host CPU share that the device time hid
            dev, wait, cpu = float(t["dev_ms"]), float(t["wait_ms"]), float(t["cpu_ms"])
            r["host_ovl"] = round(max(0.0, dev - wait) / cpu, 3) if cpu > 0 else None
        except Exception:
            r["host_ovl"] = None
        rows.append(r)
        existing.append(r)
        print(summarize(r), flush=True)
        with open(a.out, "w") as f:
            json.dump(existing, f, indent=1)
    print("sweep_whole: %d cells -> %s" % (len(rows), a.out))


if __name__ == "__main__":
    main()
