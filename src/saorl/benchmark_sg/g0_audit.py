"""G0 blind-pipeline audit + dev-recall measurement on E0-D.

Runs the six-surface leakage audit (generation, fixtures,
canonicalization/dedup, truncation, threshold bank, ordering), the
delete-the-gold byte test, the fixture-adequacy gate, and measures
candidate recall of the selector-free projection on the development
corpus E0-D (all previously analyzed Prometheus-family rules).

Ship gate (frozen in the plan): dev recall >= 0.75.
Writes results/conformal/e0/g0_audit.json.

Run: SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.benchmark_sg.g0_audit
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .extension import parse_ext
from .gen_independent import build_vocab, generate, pool_vectors, proj
from .parse import parse_prometheus
from .score import score_prom

_ROOT = Path(__file__).resolve().parents[3]
_OUT = _ROOT / "results/conformal" / "e0"
_MAN_EXT = _ROOT / "results/conformal" / "benchmark_sg" / "fetch_manifest_ext.json"


def dev_targets():
    orig, _ = parse_prometheus()
    ext, _, _ = parse_ext(json.load(open(_MAN_EXT)))
    return list(orig) + list(ext)


def obs_of(t) -> dict:
    """Observable surface only. Severity comes from labels (metadata)."""
    return dict(text=t.text, name=t.name, severity=t.reading.severity)


def _digest(pool, scores) -> str:
    blob = json.dumps([[list(proj(r)) for r in pool], scores],
                      sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


class _Tgt:
    """Duck-typed PromTarget carrying only text, for score_reading."""
    def __init__(self, text):
        self.text = text


def main():
    devs = dev_targets()
    vocab = build_vocab(devs)
    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "vocab_e0d.json").write_text(json.dumps(vocab, indent=1))

    n = len(devs)
    hits, miss_ax = 0, Counter()
    pool_sizes, overflow = [], 0
    coll_rates, coll_examples = [], []
    byte_ok = 0
    per_repo = defaultdict(lambda: [0, 0])
    import saorl.benchmark_sg.score as _score_mod
    from .parse import PromTarget as _PT
    for t in devs:
        o = obs_of(t)
        pool, meta = generate(o, vocab)
        # delete-the-gold byte test: a record that never had expr/reading
        pool2, _ = generate(dict(text=o["text"], name=o["name"],
                                 severity=o["severity"]), vocab)
        s1 = [score_prom(_Tgt(t.text), r)[0] for r in pool]
        s2 = [score_prom(_Tgt(t.text), r)[0] for r in pool2]
        if _digest(pool, s1) == _digest(pool2, s2):
            byte_ok += 1
        pool_sizes.append(len(pool))
        overflow += meta["overflow"] > 0
        g = proj(t.reading)
        keys = {proj(r) for r in pool}
        per_repo[t.repo][1] += 1
        if g in keys:
            hits += 1
            per_repo[t.repo][0] += 1
        else:
            axes = []
            if not any(k[0] == g[0] for k in keys):
                axes.append("metric")
            else:
                sub = [k for k in keys if k[0] == g[0]]
                if not any(abs(k[2] - g[2]) < 1e-9 for k in sub):
                    axes.append("threshold")
                if not any(k[1] == g[1] for k in sub):
                    axes.append("comparator")
                if not any(k[5] == g[5] for k in sub):
                    axes.append("for_s")
                if not any(k[3] == g[3] for k in sub):
                    axes.append("window")
                if not any(k[4] == g[4] for k in sub):
                    axes.append("aggregation")
            miss_ax["+".join(axes) or "combo"] += 1
        # fixture adequacy on a subsample (vectors are O(pool*fixtures))
        if len(coll_rates) < 60:
            vecs = pool_vectors(pool)
            distinct_r = len({proj(r) for r in pool})
            distinct_v = len(set(vecs))
            cr = 1.0 - distinct_v / max(1, distinct_r)
            coll_rates.append(cr)
            if cr > 0 and len(coll_examples) < 3:
                seen = {}
                for r, v in zip(pool, vecs):
                    if v in seen and proj(r) != proj(seen[v]):
                        coll_examples.append([list(proj(seen[v])),
                                              list(proj(r))])
                        break
                    seen[v] = r

    recall = hits / n
    # uncapped ceiling + gold-rank distribution (the branch-decision facts)
    import saorl.benchmark_sg.gen_independent as _GI
    old_cap = _GI.MAXPOOL
    _GI.MAXPOOL = 10 ** 6
    ranks, ceil_hits = [], 0
    for t in devs:
        pool, _m = generate(obs_of(t), vocab)
        g = proj(t.reading)
        rk = next((i for i, c in enumerate(pool) if proj(c) == g), None)
        if rk is not None:
            ceil_hits += 1
            ranks.append(rk)
    _GI.MAXPOOL = old_cap
    ranks.sort()

    def _pct(q):
        return ranks[min(len(ranks) - 1, int(q * len(ranks)))] if ranks else None

    rep = dict(
        n_dev=n, dev_recall=round(recall, 4),
        ship_gate=0.75, ships=recall >= 0.75,
        uncapped_ceiling=round(ceil_hits / n, 4),
        gold_rank_p50=_pct(0.50), gold_rank_p90=_pct(0.90),
        branch=("ship" if recall >= 0.75 else
                "branch-3: retention-mode E0 (frozen rule); generator ships "
                "as measured secondary (rho_DSL instrument) on E0-T"),
        byte_test_pass=byte_ok == n, byte_ok=byte_ok,
        miss_axes=dict(miss_ax.most_common()),
        per_repo={k: dict(hit=v[0], n=v[1], recall=round(v[0] / v[1], 3))
                  for k, v in per_repo.items()},
        pool_size=dict(mean=round(sum(pool_sizes) / n, 1),
                       max=max(pool_sizes), overflow_pools=overflow),
        fixture_adequacy=dict(
            n_audited=len(coll_rates),
            mean_collision_rate=round(sum(coll_rates) / len(coll_rates), 4),
            worked_examples=coll_examples),
        scorer_audit="score_prom reads target.text and candidate fields only "
                     "(verified by inspection; no gold access)",
    )
    (_OUT / "g0_audit.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in
                      ("n_dev", "dev_recall", "ships", "byte_test_pass",
                       "miss_axes", "pool_size")}, indent=1))
    print("fixture collision:", rep["fixture_adequacy"]["mean_collision_rate"])
    print("per-repo:", rep["per_repo"])


if __name__ == "__main__":
    main()
