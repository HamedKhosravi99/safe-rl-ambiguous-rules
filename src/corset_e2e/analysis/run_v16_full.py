"""V16 stages D-E: slot assignment and full AST composition, dev OOF.

Consumes the reranked group pools from run_v16_groups.py and measures what
REGISTRATION_V16 calls the final number: complete faithful AST proposal
recall, with the stage-survival table that says where losses happen.

Two representation details matter more than any ranking choice here, and
both were silent losses in the retired independent-slot beam:

  * a TERM slot can contain several metric occurrences -- the classic
    ratio `sum(rate(A_err[5m])) / sum(rate(A_tot[5m])) > x` is ONE term
    slot with TWO distinct series -- so terms are instantiated from an
    ordered tuple of metrics (the assignment ranges over occurrence
    positions, not term slots);
  * aggregation GROUPING labels (`sum by (cluster, pod)`) are part of the
    canonical key. The old instantiate_term wrote `by ()` always, so every
    grouped gold was unmatchable no matter what the search did. Groupings
    are now a ranked slot from the repository label-key table.

All conditionals (shape PCFG, term-shape | slot role, window | function,
grouping prior, scalar prior, duration-order direction) are fit on
dev-minus-fold; composition constants are frozen a priori in
REGISTRATION_V16 and not tuned.

Run: PYTHONPATH=. python3 corset_e2e/analysis/run_v16_full.py [limit]
Writes results/e2e/v16_full_report.json
"""
from __future__ import annotations

import itertools
import json
import math
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUT = os.path.join(ROOT, "results/e2e")
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.dsl.ast_canon import _num  # noqa: E402
from corset_e2e.generator.beam_g2 import ShapePCFG, enumerate_shapes  # noqa: E402
from corset_e2e.generator.joint_metric_sets import (  # noqa: E402
    FOLDS, ROLE_SETS, dur_of, fold_of, gold_metric_seq, walk)
from corset_e2e.generator.pred_shape import (  # noqa: E402
    CMP, SETOP, N_SLOT, T_SLOT, predicate_shape, terms_of)
from corset_e2e.generator.skeletons import skeletonize  # noqa: E402
from corset_e2e.generator.skeleton_grammar import _k  # noqa: E402
from corset_e2e.generator.threshold_channel import retrieved_thresholds  # noqa: E402
from corset_e2e.analysis.emission_gap_g2 import kbest, shape_key_of  # noqa: E402
from corset_e2e.dsl.schema_g2 import IN_G2, classify_target_g2, spec  # noqa: E402
from corset_e2e.calibration.run_e2e_v4 import hid, load_split, vis  # noqa: E402

# frozen a priori (REGISTRATION_V16)
K_SHAPE = 32
N_GROUPS = 25
K_TS = 4
K_W = 2
K_GRP = 3
K_A = 3
K_THR = 16
K_LABEL = 6
N_COMBO = 80
N_TSCOMBO = 8
N_CONFIG = 400
POOL_CAP = 6000
R_MAX = 5


# ---------------------------------------------------------- gold analysis

def slot_roles(shape):
    """Roles of term slots, left to right, aligned with terms_of order."""
    out = []
    if shape == T_SLOT:
        return ["bare"]
    if not (isinstance(shape, list) and shape and shape[0] == "bin"):
        return out
    op = shape[1]
    if op in SETOP:
        return slot_roles(shape[2]) + slot_roles(shape[3])
    if op in CMP:
        if shape[2] == T_SLOT:
            out.append("cmp_lhs")
        if shape[3] == T_SLOT:
            out.append("cmp_rhs")
        return out
    return out


def term_info(term):
    """(sel names in order, range windows, leading func, agg groupings)."""
    sels, wins, grps = [], [], []
    func = None
    for nd in walk(term):
        if not isinstance(nd, list) or not nd:
            continue
        if nd[0] == "sel" and isinstance(nd[1], str):
            sels.append(nd[1])
        elif nd[0] == "range" and len(nd) >= 3:
            wins.append(float(nd[2]))
        elif nd[0] == "call" and func is None:
            func = nd[1]
        elif nd[0] == "agg" and len(nd) >= 6 and isinstance(nd[5], list):
            grps.append(tuple(nd[5]))
    return sels, wins, func, grps


def holes_of(ts) -> int:
    return _k(ts).count('["sel","?"]')


# ------------------------------------------------------------- fold stats

def fit_fold_stats(units, fold):
    """Conditionals from dev-minus-fold golds."""
    shapes = Counter()
    ts_by_role = defaultdict(Counter)
    ts_global = Counter()
    win_by_func = defaultdict(Counter)
    win_global = Counter()
    grp_prior = Counter()
    scalar_prior = Counter()
    dur_pairs = [0, 0]
    for u in units:
        if u["fold"] == fold:
            continue
        ast = u["ast"]
        sh = predicate_shape(ast)
        shapes[json.dumps(sh, separators=(",", ":"))] += 1
        roles = slot_roles(sh)
        terms = []
        terms_of(ast, terms)
        if len(roles) == len(terms):
            for role, t in zip(roles, terms):
                key = _k(skeletonize(t))
                ts_by_role[role][key] += 1
                ts_global[key] += 1
                _sels, wins, func, grps = term_info(t)
                for w in wins:
                    win_by_func[func][w] += 1
                    win_global[w] += 1
                for g in grps:
                    grp_prior[g] += 1
        for nd in walk(ast):
            if (isinstance(nd, list) and len(nd) >= 2 and nd[0] == "num"
                    and isinstance(nd[1], (int, float))):
                scalar_prior[round(float(nd[1]), 6)] += 1
        seq = u["seq"]
        durs = [dur_of(m) for m in seq]
        for a, b in zip(durs, durs[1:]):
            if a is not None and b is not None and a != b:
                dur_pairs[0] += a >= b
                dur_pairs[1] += 1
    dur_dir = 1.0 if (dur_pairs[1] == 0
                      or dur_pairs[0] * 2 >= dur_pairs[1]) else -1.0
    return dict(shapes=shapes, ts_by_role=ts_by_role, ts_global=ts_global,
                win_by_func=win_by_func, win_global=win_global,
                grp_prior=grp_prior, scalar_prior=scalar_prior,
                dur_dir=dur_dir)


def _top(counter, k, backoff=None):
    tot = sum(counter.values())
    out = [(math.log((c + 0.5) / (tot + 1.0)), key)
           for key, c in counter.most_common(k)]
    if not out and backoff is not None:
        tot = sum(backoff.values()) or 1
        out = [(math.log((c + 0.5) / (tot + 1.0)), key)
               for key, c in backoff.most_common(k)]
    return out


# ------------------------------------------------------------ composition

def instantiate(ts, metrics, window, grouping):
    """Fill a term skeleton with an ordered metric tuple, window, grouping."""
    it = iter(metrics)

    def go(t):
        if not isinstance(t, list) or not t:
            return t
        if t[0] == "sel":
            return ["sel", next(it, metrics[-1]), []]
        if t[0] == "range":
            return ["range", go(t[1]),
                    _num(window) if window is not None else t[2]]
        if t[0] == "num":
            return t
        if t[0] == "call":
            return ["call", t[1]] + [go(x) for x in t[2:]]
        if t[0] == "agg":
            return ["agg", t[1], go(t[2]),
                    None if t[3] is None else go(t[3]),
                    t[4] if t[4] in ("by", "without") else "by",
                    sorted(grouping)]
        if t[0] == "bin":
            return ["bin", t[1], go(t[2]), go(t[3]),
                    None if t[4] is None else t[4], t[5]]
        return t
    return go(ts)


def build_pred(shape, terms, nums, labels, ctr):
    if shape == T_SLOT:
        v = terms[ctr[0]]
        ctr[0] += 1
        return v
    if shape == N_SLOT:
        v = ["num", _num(nums[ctr[1]])]
        ctr[1] += 1
        return v
    if isinstance(shape, list) and shape and shape[0] == "bin":
        m = shape[4]
        mm = None
        if m is not None:
            lab = labels[ctr[2]]
            ctr[2] += 1
            mm = [m[0], sorted(lab), m[2], []]
        return ["bin", shape[1],
                build_pred(shape[2], terms, nums, labels, ctr),
                build_pred(shape[3], terms, nums, labels, ctr), mm, shape[5]]
    return shape


def count_slots(shape):
    t = n = m = 0
    if shape == T_SLOT:
        return 1, 0, 0
    if isinstance(shape, list) and shape and shape[0] == "bin":
        if shape[4] is not None:
            m += 1
        for side in (shape[2], shape[3]):
            st, sn, sm = count_slots(side)
            t += st
            n += sn
            m += sm
        if shape[1] in CMP:
            if shape[2] == N_SLOT:
                n += 1
            if shape[3] == N_SLOT:
                n += 1
    return t, n, m


def assignments_for(group, R, dur_dir, text_pos, cap=400):
    """Ranked surjections of occurrence positions onto the group."""
    g = sorted(group)
    q = len(g)
    if q > R:
        return []
    out = []
    for tup in itertools.islice(itertools.product(g, repeat=R), cap * 4):
        if len(set(tup)) != q:
            continue
        durs = [dur_of(m) for m in tup]
        ok = tot = 0
        for a, b in zip(durs, durs[1:]):
            if a is not None and b is not None and a != b:
                tot += 1
                ok += (a >= b) if dur_dir > 0 else (a <= b)
        s_dur = (ok / tot) if tot else 0.5
        s_txt = 0.0
        pos = [text_pos.get(m) for m in tup]
        pp = [p for p in pos if p is not None]
        if len(pp) >= 2:
            mono = sum(1 for a, b in zip(pp, pp[1:]) if a <= b)
            s_txt = mono / (len(pp) - 1)
        out.append((1.0 * s_dur + 0.5 * s_txt, tup))
        if len(out) >= cap:
            break
    out.sort(key=lambda z: -z[0])
    return out[:K_A]


def label_set_options(key_tbl, repo):
    counts = key_tbl.get(repo, {})
    top = list(counts)[:12]
    if not top:
        return [(0.0, ())]
    tot = sum(counts[k] for k in top) or 1.0
    w = {k: math.log(1.0 + counts[k] / tot * 100.0) for k in top}
    scored = []
    for r in range(1, 8):
        for c in itertools.combinations(top, r):
            scored.append((sum(w[k] for k in c) - 0.35 * r,
                           tuple(sorted(c))))
    scored.sort(key=lambda z: -z[0])
    return scored[:K_LABEL]


def grouping_options(st, key_tbl, repo):
    opts = _top(st["grp_prior"], K_GRP - 1)
    have = {g for _s, g in opts}
    if () not in have:
        opts.append((-3.0, ()))
    counts = key_tbl.get(repo, {})
    for k in list(counts)[:3]:
        t = (k,)
        if t not in have and len(opts) < K_GRP + 2:
            opts.append((-2.5, t))
    return opts[:K_GRP + 1]


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    gold = json.load(open(os.path.join(OUT, "gold_ast.json")))["gold"]
    pools = json.load(open(os.path.join(OUT, "v16_pools.json")))
    key_tbl = json.load(open(os.path.join(OUT, "label_keys.json")))
    sp = spec()
    pol = json.load(open(os.path.join(OUT, "SHAPE_POLICY.json")))
    closure = set(json.load(open(os.path.join(OUT,
                                              "threshold_closure.json"))))
    thr_grid = set(sp["_thresholds"]) | closure
    for_axis = sp["_fors"]
    caps = sp["capabilities"]

    units = []
    for e in load_split("dev"):
        rid = e["rule_id"]
        if rid not in pools or classify_target_g2(hid(rid)) != IN_G2:
            continue
        g = gold.get(rid)
        if not g or "ast" not in g:
            continue
        seq = gold_metric_seq(g["ast"])
        if not seq:
            continue
        v = vis(rid)
        toks_text = f"{v.get('alert_name', '')} {v.get('text', '')}".lower()
        units.append(dict(
            rid=rid, repo=e["repo"], fold=fold_of(e["cluster_id"]),
            textkey=(v.get("alert_name", "").strip(),
                     " ".join((v.get("text") or "").split())),
            ast=g["ast"], for_s=float(g["for_s"]), seq=seq,
            M=frozenset(seq), text=toks_text, vrec=v))
    if limit:
        units = units[:limit]
    print(f"[load] {len(units)} units with pools")

    print("[fit] per-fold conditionals + shapes")
    stats = {f: fit_fold_stats(units, f) for f in range(FOLDS)}
    shapes_by_fold = {}
    for f in range(FOLDS):
        pcfg = ShapePCFG().fit(stats[f]["shapes"])
        shapes_by_fold[f] = [(pcfg.logprob(s), s) for s in enumerate_shapes(
            pcfg, caps, pol["max_shape_cost"], K_SHAPE)]

    flags = defaultdict(set)
    n = 0
    for ui, u in enumerate(units):
        n += 1
        f = u["fold"]
        st = stats[f]
        gold_key = shape_key_of(u["ast"])
        gold_shape_key = json.dumps(predicate_shape(u["ast"]),
                                    separators=(",", ":"))

        # ---- survival rows (diagnostics, all OOF)
        shape_list = shapes_by_fold[f]
        shape_keys = {json.dumps(s, separators=(",", ":"))
                      for _lp, s in shape_list}
        if gold_shape_key in shape_keys:
            flags["shape"].add(u["rid"])
        pool_groups = [frozenset(g) for g, _s in pools[u["rid"]]]
        if u["M"] in pool_groups[:N_GROUPS]:
            flags["group"].add(u["rid"])
        roles = slot_roles(predicate_shape(u["ast"]))
        gterms = []
        terms_of(u["ast"], gterms)
        ts_ok = True
        if len(roles) == len(gterms):
            for role, t in zip(roles, gterms):
                cands = {k for _s, k in _top(st["ts_by_role"][role], K_TS,
                                             st["ts_global"])}
                if _k(skeletonize(t)) not in cands:
                    ts_ok = False
        else:
            ts_ok = False
        if ts_ok:
            flags["term"].add(u["rid"])
        axis = dict()
        for val in retrieved_thresholds(u["vrec"].get("text", "") or "",
                                        u["vrec"].get("alert_name", "") or ""):
            axis[round(val, 6)] = 3.0
        for val, c in st["scalar_prior"].most_common(200):
            if round(val, 6) in thr_grid:
                axis[round(val, 6)] = axis.get(round(val, 6), 0.0) + \
                    math.log1p(c)
        scal_opts = sorted(((s, v) for v, s in axis.items()),
                           key=lambda z: (-z[0], abs(z[1])))[:K_THR]
        scal_set = {v for _s, v in scal_opts}
        gscal = {round(float(nd[1]), 6) for nd in walk(u["ast"])
                 if isinstance(nd, list) and len(nd) >= 2 and nd[0] == "num"
                 and isinstance(nd[1], (int, float))}
        if gscal <= scal_set:
            flags["numeric"].add(u["rid"])
        lab_opts = label_set_options(key_tbl, u["repo"])
        lab_set = {t for _s, t in lab_opts}
        glabs = {tuple(nd[4][1]) for nd in walk(u["ast"])
                 if isinstance(nd, list) and nd and nd[0] == "bin"
                 and len(nd) >= 5 and nd[4] and isinstance(nd[4][1], list)}
        if all(t in lab_set for t in glabs):
            flags["label"].add(u["rid"])

        # ---- composition
        text_pos = {}
        for m in u["M"]:
            i = u["text"].find(m.split("_")[0].split(":")[0])
            if i >= 0:
                text_pos[m] = i
        grp_opts = grouping_options(st, key_tbl, u["repo"])
        win_g = st["win_global"]

        configs = []
        for lp_sh, shape in shape_list:
            nt, nn, nm = count_slots(shape)
            if nt == 0:
                continue
            roles_s = slot_roles(shape)
            for gnames, gscore in pools[u["rid"]][:N_GROUPS]:
                q = len(gnames)
                if q > nt * 2 or q > R_MAX:
                    continue
                configs.append((lp_sh * 0.6 + gscore, shape, roles_s,
                                frozenset(gnames)))
        configs.sort(key=lambda z: -z[0])
        configs = configs[:N_COMBO]

        pool_keys = set()
        emitted = 0
        full = []
        for base_score, shape, roles_s, group in configs:
            # ranked term options per slot: (ts, window, grouping)
            slot_opts = []
            for role in roles_s:
                opts = []
                for lp_ts, tsk in _top(stats[f]["ts_by_role"][role], K_TS,
                                       stats[f]["ts_global"]):
                    ts = json.loads(tsk)
                    h = holes_of(ts)
                    ranged = '"range"' in tsk
                    has_agg = '"agg"' in tsk
                    wins = ([w for _s, w in _top(win_g, K_W)] or [None]) \
                        if ranged else [None]
                    grps = grp_opts[:K_GRP] if has_agg else [(0.0, ())]
                    for w in wins:
                        for lp_g, gt in (grps if has_agg else [(0.0, ())]):
                            opts.append((lp_ts + (lp_g if has_agg else 0.0),
                                         (ts, h, w, gt)))
                opts.sort(key=lambda z: -z[0])
                slot_opts.append(opts[:5])
            if any(not o for o in slot_opts):
                continue
            for idx in kbest(slot_opts, N_TSCOMBO):
                chosen = [slot_opts[i][idx[i]] for i in range(len(idx))]
                R = sum(c[1][1] for c in chosen)
                if R < len(group) or R > R_MAX:
                    continue
                tscore = sum(c[0] for c in chosen)
                for a_s, tup in assignments_for(group, R, st["dur_dir"],
                                                text_pos):
                    full.append((base_score + tscore + 0.3 * a_s,
                                 shape, chosen, tup))
        full.sort(key=lambda z: -z[0])
        full = full[:N_CONFIG]

        nt_cache = {}
        for score, shape, chosen, tup in full:
            if len(pool_keys) >= POOL_CAP:
                break
            # instantiate terms with their occurrence metrics
            terms = []
            off = 0
            for _s, (ts, h, w, gt) in chosen:
                terms.append(instantiate(ts, list(tup[off:off + h]) or
                                         [tup[-1]], w, gt))
                off += h
            key = json.dumps(shape, separators=(",", ":"))
            nt, nn, nm = nt_cache.get(key) or count_slots(shape)
            nt_cache[key] = (nt, nn, nm)
            lists = ([scal_opts] * nn) + ([lab_opts] * nm)
            if lists:
                per = max(4, (POOL_CAP - len(pool_keys)) // max(
                    1, len(full)))
                for idx in kbest(lists, per):
                    nums = [scal_opts[idx[i]][1] for i in range(nn)]
                    labs = [lab_opts[idx[nn + i]][1] for i in range(nm)]
                    cand = build_pred(shape, terms, nums, labs, [0, 0, 0])
                    k = shape_key_of(cand)
                    if k not in pool_keys:
                        pool_keys.add(k)
                        emitted += 1
            else:
                cand = build_pred(shape, terms, [], [], [0, 0, 0])
                k = shape_key_of(cand)
                if k not in pool_keys:
                    pool_keys.add(k)
                    emitted += 1

        if gold_key in pool_keys and u["for_s"] in for_axis:
            flags["complete"].add(u["rid"])
        if (ui + 1) % 150 == 0:
            print(f"    composed {ui + 1}/{len(units)}  "
                  f"(complete so far {len(flags['complete'])})")

    def weights_of(hit):
        raw = len(hit) / n
        seen, dn, dh = set(), 0, 0
        per = defaultdict(lambda: [0, 0])
        for u in units:
            if u["textkey"] not in seen:
                seen.add(u["textkey"])
                dn += 1
                dh += u["rid"] in hit
            c = per[u["repo"]]
            c[0] += 1
            c[1] += u["rid"] in hit
        macro = sum(c[1] / c[0] for c in per.values()) / len(per)
        return dict(raw=raw, distinct=dh / dn, repo_macro=macro)

    rep = dict(n=n, stages={})
    print(f"\n{'stage':<26s} {'raw':>8s} {'distinct':>10s} {'repo-macro':>11s}")
    for k in ("shape", "group", "term", "numeric", "label", "complete"):
        w = weights_of(flags[k])
        rep["stages"][k] = w
        print(f"{k:<26s} {w['raw']:8.1%} {w['distinct']:10.1%} "
              f"{w['repo_macro']:11.1%}")
    json.dump(rep, open(os.path.join(OUT, "v16_full_report.json"), "w"),
              indent=1)
    print("\nwrote results/e2e/v16_full_report.json")


if __name__ == "__main__":
    main()
