#!/usr/bin/env python3
"""Estimate the whole-layer tier policy offline, from the routing history.

The tier holds N experts (N = its budget in experts) and the chain can only run a layer's experts
in the frame when *all* of that layer's experts are resident (`ch->dm_on[i] = nres == E`). The
current warm start ranks individual experts by heat, which spreads residency across every layer
and leaves almost no layer whole. Packing whole layers instead would trade some hot-expert
coverage for layers the frame can take.

This reads `.coli_usage` (inkling's text form: `layer expert count` a line, two header lines) and
prints, for a set of capacities:

  policy A  the N hottest experts, any layer   -> how many layers end up whole, their share
  policy B  the K hottest *layers* whole, then fill the rest by expert heat -> how many whole,
            their share

"share" is the fraction of the run's routed activations that those whole layers carry, i.e. the
ceiling on how much of the host expert step moves into the device frame. It is an estimate from
aggregated counts, not a replay of the exact per-step routing -- good enough to decide whether the
deep change is worth it.

    python3 bench/tier_policy_estimate.py --usage .../.coli_usage --capacity 1718,1803,2280
"""
import argparse
from collections import defaultdict


def load(path):
    counts = defaultdict(lambda: defaultdict(int))
    L = E = 0
    with open(path, "r", errors="replace") as f:
        for line in f:
            p = line.split()
            if len(p) != 3:
                continue
            a, b, c = int(p[0]), int(p[1]), int(p[2])
            if a < 0:          # -1 layers experts / -2 a total
                if a == -1:
                    L, E = b, c
                continue
            counts[a][b] += c
    if not L:
        L = max(counts) + 1
        E = max(counts[0]) + 1
    return L, E, counts


def report(L, E, counts, capacity):
    tot = sum(sum(counts[l].values()) for l in range(L))
    if not tot:
        print("no counts")
        return
    flat = sorted(((counts[l][e], l, e) for l in range(L) for e in range(E)), reverse=True)
    # policy A: the N hottest experts
    a = set((l, e) for _, l, e in flat[:capacity])
    a_whole = [l for l in range(L) if all((l, e) in a for e in range(E))]
    a_share = sum(sum(counts[l].values()) for l in a_whole) / tot
    # policy B: whole layers first (hottest layers by their total), then fill
    lorder = sorted(range(L), key=lambda l: -sum(counts[l].values()))
    k = capacity // E
    b = set((l, e) for l in lorder[:k] for e in range(E))
    b_whole = lorder[:k]
    for _, l, e in flat:
        if len(b) >= capacity:
            break
        b.add((l, e))
    b_share = sum(sum(counts[l].values()) for l in b_whole) / tot
    print("capacity %4d experts of %d (%d layers of %d) = %.0f%% of the tier"
          % (capacity, L * E, L, E, 100.0 * capacity / (L * E)))
    print("  policy A (hottest experts): %2d whole layers, %.1f%% of activations in-frame, "
          "%.0f%% of expert slots hot" % (len(a_whole), 100 * a_share,
                                          100.0 * sum(counts[l][e] for _, l, e in flat[:capacity]) / tot))
    print("  policy B (whole layers):    %2d whole layers, %.1f%% of activations in-frame" %
          (len(b_whole), 100 * b_share))
    print("  best layer (heat): %d, coldest whole-layer pick: %d" % (lorder[0], lorder[k - 1] if k else -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usage", default="/run/media/nazar/Gaaaymes/models/Inkling-2-Mini-GGUF/iq3xxs.colibri/.coli_usage")
    ap.add_argument("--capacity", default="1718,1803,2280")
    a = ap.parse_args()
    L, E, counts = load(a.usage)
    print("%s: %d layers x %d experts, %d activations recorded"
          % (a.usage, L, E, sum(sum(counts[l].values()) for l in range(L))))
    for c in [int(x) for x in a.capacity.split(",") if x]:
        report(L, E, counts, c)


if __name__ == "__main__":
    main()
