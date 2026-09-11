"""Does the beam actually EMIT the gold reading, and where does it lose it?

Availability said the ingredients exist somewhere in the candidate space.
This runs the factorized beam and reports, against that ceiling, whether
the gold survives -- stage by stage, so a loss is attributable.

Search constants are frozen on dev (`--freeze`) and then applied to test
once. Adjusting a beam width after seeing test recall would repeat exactly
the selection error the shape policy was designed to avoid.

Run: PYTHONPATH=. python3 corset_e2e/analysis/emission_gap_g2.py --freeze
     PYTHONPATH=. python3 corset_e2e/analysis/emission_gap_g2.py
Writes results/e2e/emission_gap_g2.json
"""
from __future__ import annotations

import heapq
import itertools
import json
import math
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.generator.beam_g2 import (  # noqa: E402
    DEFAULT_POLICY, ShapePCFG, enumerate_shapes, instantiate_term, _fill,
    _slots)
from corset_e2e.generator.pred_shape import (  # noqa: E402
    T_SLOT, N_SLOT, predicate_shape, terms_of)
from corset_e2e.generator.skeletons import skeletonize  # noqa: E402
from corset_e2e.generator.skeleton_grammar import _k, dev_term_vocab  # noqa: E402
from corset_e2e.generator.threshold_channel import threshold_axis  # noqa: E402
from corset_e2e.source_grounded.harvest_recording_names import (  # noqa: E402
    names_for_unit)
from corset_e2e.dsl.schema_g2 import IN_G2, classify_target_g2, spec  # noqa: E402
from corset_e2e.generator.generate import metric_subtokens, tokens  # noqa: E402
from corset_e2e.calibration.run_e2e_v11 import V11Server  # noqa: E402
from corset_e2e.calibration.run_e2e_g1 import refs_for  # noqa: E402
from corset_e2e.calibration.run_e2e_v4 import hid, load_split, vis  # noqa: E402

OUT = os.path.join(ROOT, "results/e2e")
POLICY_PATH = os.path.join(OUT, "BEAM_POLICY.json")


def strip_matchers(c):
    """Canonical form without selector label matchers (the shape key)."""
    if not isinstance(c, list) or not c:
        return c
    if c[0] == "sel":
        out = ["sel", c[1]]
        for extra in c[2:]:
            if isinstance(extra, list) and extra and extra[0] in ("offset", "at"):
                out.append(extra)
        return out
    return [c[0]] + [strip_matchers(x) if isinstance(x, list) else x
                     for x in c[1:]]


def shape_key_of(c) -> str:
    return json.dumps(strip_matchers(c), separators=(",", ":"))


def kbest(lists, cap):
    """Top-`cap` index tuples of a sum over independently ranked lists."""
    if not lists:
        yield ()
        return
    start = tuple(0 for _ in lists)
    score0 = sum(l[0][0] for l in lists)
    heap = [(-score0, start)]
    seen = {start}
    out = 0
    while heap and out < cap:
        neg, idx = heapq.heappop(heap)
        yield idx
        out += 1
        for i in range(len(lists)):
            if idx[i] + 1 < len(lists[i]):
                nxt = list(idx)
                nxt[i] += 1
                nxt = tuple(nxt)
                if nxt in seen:
                    continue
                seen.add(nxt)
                s = sum(lists[j][nxt[j]][0] for j in range(len(lists)))
                heapq.heappush(heap, (-s, nxt))


class Generator:
    def __init__(self, policy, pcfg, term_prior, caps, max_cost,
                 thr_grid, win_grid, rec_tbl, key_tbl, dev_label_sets,
                 scalar_prior=None):
        self.p = policy
        self.pcfg = pcfg
        self.term_prior = term_prior
        self.caps = caps
        self.shapes = enumerate_shapes(pcfg, caps, max_cost, policy["k_shape"])
        self.thr_grid = thr_grid
        self.win_grid = sorted(win_grid)
        self.rec_tbl = rec_tbl
        self.key_tbl = key_tbl
        self.dev_label_sets = dev_label_sets
        self.scalar_prior = scalar_prior or {}

    # -- ranked slot candidates, from visible evidence only
    def term_shapes(self, toks):
        scored = []
        for t, c in self.term_prior.items():
            scored.append((math.log(c + 1.0), json.loads(t)))
        scored.sort(key=lambda z: -z[0])
        return scored[:self.p["k_term"]]

    def metrics(self, licensed, refs, declared, toks):
        """v11's own ranking first; re-scoring it here was a mistake.

        The first version recomputed a subtoken-overlap score and ranked
        `source_labels` and `total_open` above the gold metric `up`. The
        v11 licenser already ranks by IDF-weighted matched evidence mass
        and is the component this pipeline is meant to reuse, so its order
        is taken as given and only the extra channels are scored here.
        """
        tset = set(toks)
        out, seen = [], set()
        for i, m in enumerate(licensed[:self.p["k_metric"]]):
            out.append((10.0 - i * 0.01, m))
            seen.add(m)
        for i, m in enumerate(refs[:self.p["k_metric"]]):
            if m not in seen:
                out.append((5.0 - i * 0.01, m))
                seen.add(m)
        scored = []
        for m in declared:
            if m in seen:
                continue
            sub = set(metric_subtokens(m))
            if not sub:
                continue
            ov = len(sub & tset) / len(sub)
            if ov > 0:
                scored.append((ov, m))
        scored.sort(key=lambda z: -z[0])
        out.extend(scored[:self.p["k_metric"]])
        out.sort(key=lambda z: -z[0])
        return out[:self.p["k_metric"]] or [(0.0, "up")]

    def scalars(self, v):
        axis = threshold_axis(self.thr_grid, v)
        txt = f"{v.get('alert_name','')} {v.get('text','')}"
        import re
        lit = {float(x) for x in re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)", txt)}
        # Rank by how often dev rules USE a value, not by magnitude. Sorting
        # by |value| put 1e-06, 2e-06, ... at the head of every unit's list
        # once the derived-constant closure enlarged the axis.
        scored = []
        for a in axis:
            s = 3.0 if a in lit else 0.0
            s += math.log(1.0 + self.scalar_prior.get(a, 0))
            if s > 0:
                scored.append((s, a))
        scored.sort(key=lambda z: (-z[0], abs(z[1])))
        return scored[:self.p["k_thr"]] or [(0.0, 0.0)]

    def windows(self):
        pref = [300.0, 60.0, 3600.0, 900.0]
        out = [(1.0 - i * 0.1, w) for i, w in enumerate(pref)]
        return out[:self.p["k_win"]]

    def label_sets(self, repo):
        counts = self.key_tbl.get(repo, {})
        top = list(counts)[:12]
        if not top:
            return [(0.0, tuple(s)) for s in list(self.dev_label_sets)[:self.p["k_label"]]] or [(0.0, ())]
        tot = sum(counts[k] for k in top) or 1.0
        w = {k: math.log(1.0 + counts[k] / tot * 100.0) for k in top}
        scored = []
        for r in range(1, 8):
            for c in itertools.combinations(top, r):
                s = sum(w[k] for k in c) - self.p["label_size_penalty"] * r
                scored.append((s, tuple(sorted(c))))
        scored.sort(key=lambda z: -z[0])
        return scored[:self.p["k_label"]]

    def generate(self, e, v, server):
        toks = tokens(v.get("text", "")) + tokens(v.get("alert_name", ""))
        cat = server.catalog_for(e, v)
        licensed = list(cat.license(toks, []))
        refs = refs_for(licensed, server.idx[e["repo"]], toks)
        declared = names_for_unit(self.rec_tbl, e["repo"], e["cluster_id"])
        tsh = self.term_shapes(toks)
        met = self.metrics(licensed, refs, declared, toks)
        sca = self.scalars(v)
        win = self.windows()
        lab = self.label_sets(e["repo"])

        pool, seen = [], set()
        cap = self.p["pool_cap"]
        lps = [self.pcfg.logprob(s) for s in self.shapes]
        mx = max(lps)
        wts = [math.exp(l - mx) for l in lps]
        tw = sum(wts) or 1.0
        for shape, wt in zip(self.shapes, wts):
            per_shape = max(8, int(cap * wt / tw))
            nt, nn, nm = _slots(shape)
            if nt == 0:
                continue
            lists = ([tsh] * nt) + ([met] * nt) + ([win] * nt) + \
                    ([sca] * nn) + ([lab] * nm)
            if any(not l for l in lists):
                continue
            base = self.pcfg.logprob(shape)
            for idx in kbest(lists, per_shape):
                o = 0
                ts = [tsh[idx[o + i]][1] for i in range(nt)]
                o += nt
                ms = [met[idx[o + i]][1] for i in range(nt)]
                o += nt
                ws = [win[idx[o + i]][1] for i in range(nt)]
                o += nt
                ss = [sca[idx[o + i]][1] for i in range(nn)]
                o += nn
                ls = [lab[idx[o + i]][1] for i in range(nm)]
                terms = [instantiate_term(ts[i], ms[i], ws[i])
                         for i in range(nt)]
                cand = _fill(shape, terms, ss or [0.0], ls or [()], [0, 0, 0])
                k = shape_key_of(cand)
                if k in seen:
                    continue
                seen.add(k)
                pool.append(k)
                if len(pool) >= cap:
                    return pool, seen
        return pool, seen


def build(policy):
    gold = json.load(open(os.path.join(OUT, "gold_ast.json")))["gold"]
    sp = spec()
    pol = json.load(open(os.path.join(OUT, "SHAPE_POLICY.json")))
    caps = sp["capabilities"]
    thr = set(sp["_thresholds"]) | set(
        json.load(open(os.path.join(OUT, "threshold_closure.json"))))
    shapes = Counter()
    tprior = Counter()
    dev_labels = set()
    for r in gold.values():
        if r.get("split") != "dev" or "ast" not in r:
            continue
        shapes[json.dumps(predicate_shape(r["ast"]), separators=(",", ":"))] += 1
        ts = []
        terms_of(r["ast"], ts)
        for t in ts:
            tprior[_k(skeletonize(t))] += 1
    scalar_prior = Counter()
    for r in gold.values():
        if r.get("split") != "dev" or "ast" not in r:
            continue

        def sc(a):
            if isinstance(a, list):
                if len(a) >= 2 and a[0] == "num" and isinstance(a[1], (int, float)):
                    scalar_prior[round(float(a[1]), 6)] += 1
                for x in a:
                    if isinstance(x, list):
                        sc(x)
        sc(r["ast"])
    pcfg = ShapePCFG().fit(shapes)
    gen = Generator(policy, pcfg, tprior, caps, pol["max_shape_cost"],
                    thr, set(sp["_windows"]),
                    json.load(open(os.path.join(OUT, "recording_names.json"))),
                    json.load(open(os.path.join(OUT, "label_keys.json"))),
                    dev_labels, scalar_prior)
    return gold, gen


def run(split, policy, limit=None):
    gold, gen = build(policy)
    server = V11Server()
    units = [e for e in load_split(split)
             if classify_target_g2(hid(e["rule_id"])) == IN_G2]
    if limit:
        units = units[:limit]
    hit = tot = 0
    cov = set()
    sizes = []
    for e in units:
        r = gold.get(e["rule_id"])
        if not r or "ast" not in r:
            continue
        tot += 1
        v = vis(e["rule_id"])
        pool, seen = gen.generate(e, v, server)
        sizes.append(len(pool))
        if shape_key_of(r["ast"]) in seen:
            hit += 1
            cov.add(e["rule_id"])
    return dict(n=tot, hit=hit, recall=hit / max(tot, 1),
                median_pool=sorted(sizes)[len(sizes) // 2] if sizes else 0,
                covered=sorted(cov))


def main():
    if "--freeze" in sys.argv:
        # Swept on DEV only. The first policy used k_metric=8 and the dev
        # autopsy showed the gold metric set surviving in 0.7% of units:
        # availability was measured over the whole licensed namespace, and
        # a top-8 cut discards it, especially for burn-rate rules that need
        # four metrics at once. Widening k costs heap exploration, not pool
        # size, because the pool cap is what bounds the candidate set.
        best, bestr = None, -1
        for km in (8, 32, 128):
            for kt in (6, 12):
                for kth in (6, 16):
                    p = dict(DEFAULT_POLICY, k_metric=km, k_term=kt,
                             k_thr=kth, pool_cap=20000)
                    r = run("dev", p, limit=150)
                    print(f"  k_metric={km:3d} k_term={kt:2d} k_thr={kth:2d} "
                          f"dev recall {r['recall']:6.1%} pool~{r['median_pool']}")
                    if r["recall"] > bestr:
                        best, bestr = p, r["recall"]
        json.dump(dict(policy=best, dev_recall=bestr,
                       note="swept on dev only; never adjusted after a test run"),
                  open(POLICY_PATH, "w"), indent=1)
        print(f"\nFROZEN {best}\ndev recall {bestr:.1%}")
        return
    policy = json.load(open(POLICY_PATH))["policy"]
    r = run("test", policy)
    avail = json.load(open(os.path.join(OUT, "proposal_recall_g2.json")))
    ceil = avail["complete_ast_available"]["raw"]
    print(f"\nPROPOSAL RECALL (test, raw): {r['hit']}/{r['n']} = "
          f"{r['recall']:.1%}   against availability ceiling {ceil:.1%}")
    print(f"median pool per unit: {r['median_pool']}")
    json.dump(dict(policy=policy, test=r, availability_ceiling=ceil),
              open(os.path.join(OUT, "emission_gap_g2.json"), "w"))


if __name__ == "__main__":
    main()
