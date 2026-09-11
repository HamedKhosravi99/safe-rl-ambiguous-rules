"""Ingest filled annotation sheets and compute the human non-dominance study.

Usage:
    python -m saorl.benchmark_sg.ingest_annotations sheetA.txt sheetB.txt ...

Each sheet is annotation_kit/sheet.txt filled in place:
    <n>|<thrA>,<durA>|<thrB>,<durB>|<thrC>,<durC>
Lines starting with '#' and unfilled lines are ignored.

Reports per-annotator non-dominance, pairwise Cohen's kappa, Fleiss
kappa, and the majority-vote rate, over rules ALL annotators completed.
Same dominance convention as the model study: reading i dominates j iff
theta_i <= theta_j and duration_i <= duration_j (fires at least as often
pointwise). Writes results/conformal/benchmark_sg/human_annotation.json.
"""
from __future__ import annotations

import itertools
import json
import os
import statistics as st
import sys


def parse_sheet(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) != 4:
            continue
        try:
            n = int(parts[0])
            reads = [tuple(float(v) for v in seg.split(",")) for seg in parts[1:]]
        except ValueError:
            continue  # unfilled template line
        if len(reads) == 3 and all(len(r) == 2 for r in reads):
            out[n] = reads
    return out


def dominates(i, j):
    return i[0] <= j[0] and i[1] <= j[1]


def nondominated(pool):
    return not any(all(dominates(a, b) for b in pool) for a in pool)


def cohen(x, y, keys):
    n = len(keys)
    po = sum(1 for k in keys if x[k] == y[k]) / n
    p1 = sum(x[k] for k in keys) / n
    p2 = sum(y[k] for k in keys) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main() -> None:
    paths = sys.argv[1:]
    if len(paths) < 2:
        print("need at least 2 filled sheets"); sys.exit(1)
    sheets = {os.path.basename(p): parse_sheet(p) for p in paths}
    for name, s in sheets.items():
        print(f"{name}: {len(s)} rules parsed")
    common = sorted(set.intersection(*[set(s) for s in sheets.values()]))
    if len(common) < 10:
        print(f"only {len(common)} rules completed by all annotators -- too few")
        sys.exit(1)
    lab = {name: {n: nondominated(s[n]) for n in common} for name, s in sheets.items()}
    print(f"\nrules completed by all: {len(common)}")
    for name in sheets:
        f = sum(lab[name].values()) / len(common)
        print(f"  {name}: {sum(lab[name].values())}/{len(common)} = {f:.3f}")
    ks = []
    for a, b in itertools.combinations(sheets, 2):
        k = cohen(lab[a], lab[b], common)
        ks.append(k)
        print(f"  kappa({a},{b}) = {k:.3f}")
    N = len(sheets)
    Pi = []
    for r in common:
        s = sum(lab[k][r] for k in sheets)
        Pi.append((s * (s - 1) + (N - s) * (N - s - 1)) / (N * (N - 1)))
    Pbar = sum(Pi) / len(common)
    pj = sum(sum(lab[k][r] for k in sheets) for r in common) / (len(common) * N)
    Pe = pj ** 2 + (1 - pj) ** 2
    fleiss = (Pbar - Pe) / (1 - Pe) if Pe < 1 else float("nan")
    maj = [sum(lab[k][r] for k in sheets) > N / 2 for r in common]
    print(f"\n  Fleiss kappa = {fleiss:.3f}   mean pairwise = {st.mean(ks):.3f}")
    print(f"  majority-vote non-dominated: {sum(maj)}/{len(common)} = {sum(maj)/len(common):.3f}")
    print(f"  model-interpreter study on same rules: 0.552 (kappa 0.349)")
    print(f"  library on same corpus: 0.784")
    out = os.path.join("results/conformal", "benchmark_sg", "human_annotation.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(n_rules=len(common), n_annotators=N,
                   per_annotator={k: sum(lab[k].values()) / len(common) for k in sheets},
                   mean_pairwise_kappa=st.mean(ks), fleiss_kappa=fleiss,
                   majority_frac=sum(maj) / len(common)),
              open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
