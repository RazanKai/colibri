#!/usr/bin/env python3
"""Fill PLAN-colibri.md section 14.9 from a 250K engine run's stderr.

The run line is the one the plan quotes elsewhere:

  SNAP=.../iq3xxs.colibri CTX_MAX=262144 INK2_KV=q4 COLI_CUDA=1 OMP_NUM_THREADS=6 \
    ./c/inkling -f bench/prompt-250k.txt -n 24 2> 250k-new.err

Nothing here is computed beyond differences against the recorded baseline, and the raw lines are
kept beside the table so every figure can be checked.

    python3 bench/fill_plan_250k.py /var/tmp/scratch/kvtest/250k-new.err
"""
import re
import sys

PLAN = "PLAN-colibri.md"
PLACEHOLDER = "### 14.9 250K with the new kernels\n\n(filled from the run log: `/var/tmp/scratch/kvtest/250k-new.err`.)\n"

PATTERNS = {
    "prefill": re.compile(r"\[prefill (?P<s>[0-9.]+)s \| (?P<gen>[0-9]+) tokens in (?P<ds>[0-9.]+)s = (?P<tps>[0-9.]+) tok/s \| RSS (?P<rss>[0-9.]+) GB\]"),
    "phases": re.compile(r"\[phases\] fill (?P<fill>[0-9.]+)s \| expert-mm (?P<expert>[0-9.]+)s \| shared (?P<shared>[0-9.]+)s \| attn (?P<attn>[0-9.]+)s \| other (?P<other>[0-9.]+)s"),
    "chain": re.compile(r"\[CUDA\] inkling2 chain: (?P<forwards>[0-9]+) forwards, (?P<frames>[0-9]+) frames \([^)]*\), (?P<wait>[0-9.]+) ms waiting for the device, (?P<host>[0-9.]+) ms of routed experts on the host, (?P<dml>[0-9]+) layer steps' experts in the frame, (?P<mib>[0-9.]+) MiB on the device"),
    "attention": re.compile(r"\[CUDA\] inkling2 attention: ring (?P<rcalls>[0-9]+) calls, (?P<rpairs>[0-9.]+) Gpair, (?P<rflop>[0-9.]+) GFLOP \| global (?P<gcalls>[0-9]+) calls, (?P<gpairs>[0-9.]+) Gpair, (?P<gflop>[0-9.]+) GFLOP \| decode (?P<dcalls>[0-9]+) calls"),
    "kvstats": re.compile(r"\[kvstats\] (?P<kv>\w+) cache: max \|K\| (?P<kmax>[0-9.e+-]+), max \|V\| (?P<vmax>[0-9.e+-]+), block scale max K (?P<ks>[0-9.e+-]+) V (?P<vs>[0-9.e+-]+), (?P<pct>[0-9.e+-]+)% of (?P<n>[0-9]+) values"),
}

BASE = {"prefill": 3651.5, "attn": 2706.7, "expert": 769.4, "other": 172.9, "tps": 8.44,
        "rss": 12.2, "host_kv": 2493, "device": 2802}


def main(path, outpath=None):
    if outpath is None:
        outpath = path[:-4] + ".out" if path.endswith(".err") else path + ".out"
    txt = open(path, errors="replace").read()
    try:   # the phase lines go to stdout
        txt += open(outpath, errors="replace").read()
    except OSError:
        pass
    got = {}
    for k, rx in PATTERNS.items():
        m = None
        for m2 in rx.finditer(txt):
            m = m2
        if m:
            got[k] = m.groupdict()
    missing = [k for k in ("prefill", "phases", "chain") if k not in got]
    if missing:
        sys.exit("fill_plan_250k: %s has no %s line yet (still running?)" % (path, ", ".join(missing)))
    p, ph, ch = got["prefill"], got["phases"], got["chain"]
    f = float
    table = [
        "### 14.9 250K with the new kernels",
        "",
        "Same command and model as §5 (`CTX_MAX=262144`, `INK2_KV=q4`, 24 tokens), on the rebuilt",
        "kernels. Phase line: `fill %s s | expert-mm %s s | attn %s s | other %s s`." % (
            ph["fill"], ph["expert"], ph["attn"], ph["other"]),
        "",
        "| 250K, %s prompt tokens | §5 | now | |" % "{:,}".format(250156),
        "|---|---:|---:|---:|",
        "| prefill | %.1f s | **%.1f s** | %.2f× |" % (BASE["prefill"], f(p["s"]), BASE["prefill"] / f(p["s"])),
        "| device side (the `attn` phase) | %.1f s | **%.1f s** | %.2f× |" % (BASE["attn"], f(ph["attn"]), BASE["attn"] / f(ph["attn"])),
        "| routed experts | %.1f s | %.1f s | |" % (BASE["expert"], f(ph["expert"])),
        "| other | %.1f s | %.1f s | |" % (BASE["other"], f(ph["other"])),
        "| decode | %.2f tok/s | **%.2f tok/s** | %.2f× |" % (BASE["tps"], f(p["tps"]), f(p["tps"]) / BASE["tps"]),
        "| RSS | %.1f GB | %.1f GB | |" % (BASE["rss"], f(p["rss"])),
        "| device | %d MiB | %.0f MiB | |" % (BASE["device"], f(ch["mib"])),
        "",
    ]
    if "attention" in got:
        table += ["The attention accounting (`INK2CC_RA_STATS=1`): ring %s calls / %s Gpair, global %s calls /" % (
            got["attention"]["rcalls"], got["attention"]["rpairs"], got["attention"]["gcalls"]),
            "%s Gpair, decode %s calls; the chain waited %.1f ms on the device and ran %.1f s of routed" % (
            got["attention"]["gpairs"], got["attention"]["dcalls"], f(ch["wait"]) / 1e3, f(ch["host"]) / 1e3),
            "experts on the host, with %s layer steps' experts in the frame." % ch["dml"]]
    if "kvstats" in got:
        k = got["kvstats"]
        table += ["", "Range audit at this context (`INK_KV_STATS=1`): max |K| %s, max |V| %s, block scale" % (k["kmax"], k["vmax"]),
                  "max %s (K) / %s (V), %s%% of %s values above f16's 65504." % (k["ks"], k["vs"], k["pct"], "{:,}".format(int(k["n"])))]
    table += ["", "Raw log: `/var/tmp/scratch/kvtest/250k-new.err`, `/var/tmp/scratch/kvtest/250k-new.out`.",
              "", ""]
    new = "\n".join(table)
    s = open(PLAN).read()
    if PLACEHOLDER not in s:
        sys.exit("fill_plan_250k: the §14.9 placeholder is gone; edit by hand")
    open(PLAN, "w").write(s.replace(PLACEHOLDER, new, 1))
    print(new)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/var/tmp/scratch/kvtest/250k-new.err")
