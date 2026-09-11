"""Source-grounded benchmark, step 2: parse artifacts into canonical targets.

Prometheus alert rules and Kyverno policies are parsed into *canonical
structured target representations*.  The target of each rule is the executable
structure itself (no person or model decides it -- plan 3.1/3.2 step "define
the source-grounded target as the canonical executable AST").

We deliberately parse only a *structured subset* that fits the frozen candidate
transformation families, and log how many rules are skipped as out-of-DSL
(complex PromQL joins/topk/group_left, or Kyverno rules whose validate block is
not one of the modelled families).  Everything is deterministic.

Data structures
---------------
PromReading / KyvReading : one candidate reading (the parsed rule is the
    identity reading).  Same type is reused by candidates.py for transforms.
PromTarget / KyvTarget   : identity reading + source text + provenance.

Run:  PYTHONPATH=. python3 -m saorl.benchmark_sg.parse
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from .fetch import load_manifest

# --------------------------------------------------------------------------
# Canonical readings (shared by candidates.py)
# --------------------------------------------------------------------------

COMPARATORS = (">=", "<=", ">", "<", "==", "!=")
_DUR_UNITS = {"ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0, "w": 604800.0, "y": 31536000.0}

# PromQL constructs we do NOT model structurally -> out-of-DSL
_PROM_UNMODELLED = [
    r"\bunless\b", r"\bgroup_left\b", r"\bgroup_right\b", r"\btopk\s*\(",
    r"\bbottomk\s*\(", r"\bhistogram_quantile\s*\(", r"\bpredict_linear\s*\(",
    r"\babsent\s*\(", r"\babsent_over_time\s*\(", r"\bcount_values\s*\(",
    r"\blabel_replace\s*\(", r"\blabel_join\s*\(", r"\bon\s*\(",
    r"\bignoring\s*\(", r"\bquantile\s*\(", r"\bclamp", r"\bvector\s*\(", r"@",
]
_AGG_FUNCS = ("sum", "avg", "max", "min", "count")
_RATE_FUNCS = ("rate", "irate", "increase", "delta", "idelta", "deriv")
_OVERTIME_RE = re.compile(r"\b(\w+)_over_time\s*\(")


@dataclass(frozen=True)
class PromReading:
    metric: str
    selectors: Tuple[Tuple[str, str, str], ...]   # (key, op, value) sorted
    comparator: str
    threshold: float
    rate_window_s: Optional[float]
    aggregation: str                              # none|sum|avg|max|min|count
    agg_by: Tuple[str, ...]                        # grouping labels
    for_s: float
    severity: str
    logic: str = "base"                           # base|conj|disj
    axis: str = "identity"
    op_tag: str = "identity"

    def struct_key(self) -> tuple:
        """Structural identity (ignores axis/op_tag provenance tags)."""
        return (self.metric, self.selectors, self.comparator, round(self.threshold, 6),
                self.rate_window_s, self.aggregation, self.agg_by, self.for_s,
                self.severity, self.logic)


@dataclass(frozen=True)
class KyvReading:
    family: str
    match_kinds: Tuple[str, ...]
    match_mode: str                               # any|all
    exclude_ns: Tuple[str, ...]                   # exception set (namespaces)
    required_labels: Tuple[str, ...]
    allowed_registries: Tuple[str, ...]
    allowed_caps: Tuple[str, ...]
    forbidden: Tuple[str, ...]                     # e.g. ('privileged',) ('hostPath',)
    resource_reqs: Tuple[str, ...]                 # required resource fields
    action: str                                   # Enforce|Audit|Warn
    axis: str = "identity"
    op_tag: str = "identity"

    def struct_key(self) -> tuple:
        return (self.family, self.match_kinds, self.match_mode, self.exclude_ns,
                self.required_labels, self.allowed_registries, self.allowed_caps,
                self.forbidden, self.resource_reqs, self.action)


@dataclass
class PromTarget:
    name: str
    group: str
    reading: PromReading
    text: str
    repo: str
    commit: str
    file: str
    sub_source: str            # group name -> source-held-out split
    last_commit_ts: Optional[int]
    raw_expr: str


@dataclass
class KyvTarget:
    policy_name: str
    rule_name: str
    reading: KyvReading
    text: str
    repo: str
    commit: str
    file: str
    sub_source: str            # top-level category dir
    last_commit_ts: Optional[int]


# --------------------------------------------------------------------------
# Small parsing helpers
# --------------------------------------------------------------------------

def dur_to_s(s) -> Optional[float]:
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s == "0":
        return 0.0
    total, matched = 0.0, False
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h|d|w|y)", s):
        total += float(num) * _DUR_UNITS[unit]
        matched = True
    return total if matched else None


def _strip_template(text: str) -> str:
    """Drop Go-template {{...}} and collapse whitespace in annotation text."""
    text = re.sub(r"\{\{.*?\}\}", " ", text, flags=re.DOTALL)
    return re.sub(r"\s+", " ", text).strip()


def _depth0_positions(expr: str, token_re: re.Pattern) -> List[re.Match]:
    """Matches of token_re that occur at parenthesis depth 0."""
    depths = []
    d = 0
    for ch in expr:
        if ch == "(":
            d += 1
        depths.append(d)
        if ch == ")":
            d -= 1
    return [m for m in token_re.finditer(expr) if depths[m.start()] == 0]


_CMP_RE = re.compile(r"(>=|<=|==|!=|>|<)\s*(-?\d+(?:\.\d+)?)")
_METRIC_RE = re.compile(r"([a-zA-Z_:][a-zA-Z0-9_:]*)\s*\{")
_BARE_METRIC_RE = re.compile(r"\b([a-z_][a-z0-9_]*_[a-z0-9_]+)\b")


def _parse_selectors(seg: str) -> Tuple[Tuple[str, str, str], ...]:
    out = []
    for k, op, v in re.findall(r'([a-zA-Z_][\w]*)\s*(=~|!~|!=|=)\s*"([^"]*)"', seg):
        out.append((k, op, v))
    return tuple(sorted(out))


def _first_selector_block(expr: str, metric: str) -> str:
    i = expr.find(metric + "{")
    if i < 0:
        i = expr.find(metric)
        if i < 0 or "{" not in expr[i:]:
            return ""
    j = expr.find("{", i)
    k = expr.find("}", j)
    return expr[j + 1:k] if (j >= 0 and k > j) else ""


def parse_prom_expr(expr: str) -> Optional[dict]:
    """Parse a PromQL alert expr into structured fields, or None if out-of-DSL.

    Accepts a single numeric-threshold comparison (integer 0/1 equality guards
    are tolerated and dropped); rejects joins, topk, group_left, quantiles, etc.
    """
    flat = re.sub(r"\s+", " ", expr.replace("\n", " ")).strip()
    for pat in _PROM_UNMODELLED:
        if re.search(pat, flat):
            return None
    cmps = _depth0_positions(flat, _CMP_RE) or list(_CMP_RE.finditer(flat))
    if not cmps:
        return None
    # classify comparisons: integer 0/1 (in)equality are guards; keep real ones
    real = []
    for m in cmps:
        op, rhs = m.group(1), float(m.group(2))
        is_guard = (op in ("==", "!=", ">=", "<=", ">", "<")) and rhs in (0.0, 1.0) and \
                   op in ("==", "!=", ">=")
        if not is_guard:
            real.append(m)
    primary = real[0] if len(real) == 1 else (cmps[0] if len(cmps) == 1 else None)
    if primary is None:
        return None
    comparator, threshold = primary.group(1), float(primary.group(2))
    lhs = flat[:primary.start()]
    # metric
    mm = _METRIC_RE.search(lhs)
    if mm:
        metric = mm.group(1)
    else:
        cands = [b for b in _BARE_METRIC_RE.findall(lhs) if b not in _AGG_FUNCS + _RATE_FUNCS]
        if not cands:
            return None
        metric = cands[0]
    if metric in _AGG_FUNCS + _RATE_FUNCS:
        return None
    selectors = _parse_selectors(_first_selector_block(lhs, metric))
    # aggregation (outermost agg function at depth reachable in lhs)
    aggregation, agg_by = "none", tuple()
    for af in _AGG_FUNCS:
        am = re.search(rf"\b{af}\s*(by|without)?\s*\(([^)]*)\)?\s*\(", lhs) or \
             re.search(rf"\b{af}\s*\(", lhs)
        if am:
            aggregation = af
            gm = re.search(rf"\b{af}\s*(?:by|without)\s*\(([^)]*)\)", lhs)
            if gm:
                agg_by = tuple(sorted(x.strip() for x in gm.group(1).split(",") if x.strip()))
            break
    # rate/range window
    rate_window_s = None
    wm = re.search(r"\[(\d+(?:\.\d+)?(?:ms|s|m|h|d|w|y))\]", lhs)
    if wm and (any(rf + "(" in lhs.replace(" ", "") for rf in _RATE_FUNCS) or _OVERTIME_RE.search(lhs)):
        rate_window_s = dur_to_s(wm.group(1))
    return dict(metric=metric, selectors=selectors, comparator=comparator,
                threshold=threshold, rate_window_s=rate_window_s,
                aggregation=aggregation, agg_by=agg_by)


# --------------------------------------------------------------------------
# Prometheus parsing
# --------------------------------------------------------------------------

def parse_prometheus() -> Tuple[List[PromTarget], dict]:
    man = load_manifest()
    files = [f for f in man["files"] if f["family"] == "prometheus"]
    commit = man["repos"]["kube-prometheus"]["commit"]
    targets: List[PromTarget] = []
    skipped = defaultdict(int)
    seen_keys = set()
    for fe in files:
        doc = yaml.safe_load(open(fe["abs_path"]))
        for grp in (doc.get("spec", {}) or {}).get("groups", []) or []:
            gname = grp.get("name", "")
            for r in grp.get("rules", []) or []:
                if "alert" not in r or "expr" not in r:
                    skipped["not_alert"] += 1
                    continue
                parsed = parse_prom_expr(str(r["expr"]))
                if parsed is None:
                    skipped["out_of_dsl_expr"] += 1
                    continue
                ann = r.get("annotations", {}) or {}
                text = _strip_template(" ".join(
                    str(ann.get(k, "")) for k in ("summary", "description")))
                if not text:
                    skipped["no_text"] += 1
                    continue
                sev = str((r.get("labels", {}) or {}).get("severity", ""))
                reading = PromReading(
                    metric=parsed["metric"], selectors=parsed["selectors"],
                    comparator=parsed["comparator"], threshold=parsed["threshold"],
                    rate_window_s=parsed["rate_window_s"], aggregation=parsed["aggregation"],
                    agg_by=parsed["agg_by"], for_s=dur_to_s(r.get("for")) or 0.0,
                    severity=sev)
                key = (r["alert"], reading.struct_key())
                if key in seen_keys:
                    skipped["duplicate"] += 1
                    continue
                seen_keys.add(key)
                targets.append(PromTarget(
                    name=r["alert"], group=gname, reading=reading, text=text,
                    repo="kube-prometheus", commit=commit, file=fe["rel_path"],
                    sub_source=gname, last_commit_ts=fe.get("last_commit_ts"),
                    raw_expr=re.sub(r"\s+", " ", str(r["expr"]).strip())))
    return targets, dict(skipped)


# --------------------------------------------------------------------------
# Kyverno parsing
# --------------------------------------------------------------------------

def _kyv_match_kinds(match_block: dict) -> Tuple[List[str], str]:
    """Return (kinds, mode) from a match/exclude block."""
    for mode in ("any", "all"):
        if isinstance(match_block, dict) and mode in match_block:
            kinds = []
            for entry in match_block[mode] or []:
                kinds += (entry.get("resources", {}) or {}).get("kinds", []) or []
            return kinds, mode
    # legacy: match.resources.kinds
    kinds = ((match_block or {}).get("resources", {}) or {}).get("kinds", []) or []
    return kinds, "any"


def _kyv_exclude_ns(exclude_block) -> List[str]:
    ns = []
    if not isinstance(exclude_block, dict):
        return ns
    for mode in ("any", "all"):
        for entry in exclude_block.get(mode, []) or []:
            ns += (entry.get("resources", {}) or {}).get("namespaces", []) or []
    ns += ((exclude_block.get("resources", {}) or {}).get("namespaces", []) or [])
    return ns


def _strip_anchor(key: str) -> str:
    """Remove Kyverno pattern anchors: =(x) (x) X(x) +(x) ^(x) -> x."""
    m = re.match(r"^[=X+^~]?\(([^)]*)\)$", str(key))
    return m.group(1) if m else str(key)


def _field_signature(pattern) -> Tuple[str, ...]:
    """Salient constrained leaf field names in a pattern (anchors stripped).

    A leaf field is a key whose value is a scalar constraint (string glob,
    bool, null) rather than a nested container.  These tokens define the
    policy's constrained fields and feed the entity/operator score components.
    """
    toks: List[str] = []

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                key = _strip_anchor(k)
                if isinstance(v, (dict, list)):
                    walk(v)
                else:
                    toks.append(key)
        elif isinstance(node, list):
            for it in node:
                walk(it)

    walk(pattern)
    # drop structural boilerplate keys
    drop = {"name", "spec", "metadata", "containers", "template", "kind",
            "apiVersion", "image"}
    sig = sorted({t for t in toks if t and t not in drop and not t.startswith("=")})
    return tuple(sig[:4])


def _classify_validate(validate: dict, rule_text: str) -> Optional[dict]:
    """Map a validate block to one of the modelled families + its parameters.

    Rich families (required_labels / registry / capabilities / resources)
    carry an allowlist or threshold set and receive set/threshold candidate
    axes.  All other structured pattern/anyPattern rules become the generic
    ``field_constraint`` family, which still receives the field-agnostic
    candidate axes (match.any/all, exception sub/superset, action, kind scope).
    CEL / deny / foreach validate blocks are left unparsed (out-of-DSL).
    """
    if not isinstance(validate, dict):
        return None
    pattern = validate.get("pattern")
    if pattern is None and "anyPattern" in validate:
        pattern = validate.get("anyPattern")
    if pattern is None:
        return None
    blob = json.dumps(pattern, default=str).lower()
    txt = rule_text.lower()

    # required labels: pattern -> metadata.labels : {k: "?*"}
    def _labels_of(pat):
        if isinstance(pat, dict):
            md = pat.get("metadata")
            if isinstance(md, dict) and isinstance(md.get("labels"), dict):
                return md["labels"]
        if isinstance(pat, list):
            for it in pat:
                r = _labels_of(it)
                if r:
                    return r
        return None
    labels = _labels_of(pattern)
    if labels and all(str(v).strip() in ("?*", "*") or "?" in str(v) for v in labels.values()):
        keys = sorted(str(k) for k in labels.keys())
        return dict(family="required_labels", required_labels=keys)

    # image registry allowlist: image glob with '/' (and usually '|' alternatives)
    imgs = re.findall(r'"([^"]*/\*[^"]*)"', blob)
    if imgs and ("registr" in txt or "image" in txt or "|" in imgs[0]):
        regs = sorted({seg.strip().split("/")[0] for g in imgs for seg in g.split("|")
                       if seg.strip()})
        regs = [r for r in regs if r and r != "*" and "." in r or r.endswith("io")]
        regs = sorted({r for r in regs if r and r != "*"})
        if regs:
            return dict(family="registry", allowed_registries=regs)

    # capability allowlist
    if "capabilities" in blob and re.search(r'"[A-Z_]{3,}"', json.dumps(pattern)):
        caps = sorted(set(re.findall(r'"([A-Z_]{3,})"', json.dumps(pattern))))
        return dict(family="capabilities", allowed_caps=caps)

    # resource requests/limits thresholds
    if "resources" in blob and ("limits" in blob or "requests" in blob):
        reqs = [r for r in ("limits", "requests") if r in blob]
        return dict(family="resources", resource_reqs=sorted(reqs))

    # named boolean pod-security fields (nicer family labels for scoring)
    for kw, fam, fld in (
        ("privileged", "privileged", "privileged"),
        ("hostpath", "host_path", "hostPath"),
        ("hostnetwork", "host_namespaces", "hostNetwork"),
        ("hostpid", "host_namespaces", "hostPID"),
        ("hostport", "host_ports", "hostPort"),
        ("readonlyrootfilesystem", "read_only_fs", "readOnlyRootFilesystem"),
        ("runasnonroot", "run_as_non_root", "runAsNonRoot"),
        ("allowprivilegeescalation", "privilege_escalation", "allowPrivilegeEscalation"),
        ("seccompprofile", "seccomp", "seccompProfile"),
        ("nodeport", "node_port", "type"),
    ):
        if kw in blob:
            return dict(family=fam, forbidden=[fld])

    # generic structured field constraint (scope/exception/action axes still apply)
    sig = _field_signature(pattern)
    if sig:
        return dict(family="field_constraint", forbidden=list(sig))
    return None


def parse_kyverno() -> Tuple[List[KyvTarget], dict]:
    man = load_manifest()
    files = [f for f in man["files"] if f["family"] == "kyverno"]
    commit = man["repos"]["kyverno-policies"]["commit"]
    targets: List[KyvTarget] = []
    skipped = defaultdict(int)
    seen_keys = set()
    for fe in files:
        try:
            docs = list(yaml.safe_load_all(open(fe["abs_path"])))
        except yaml.YAMLError:
            skipped["yaml_error"] += 1
            continue
        for doc in docs:
            if not isinstance(doc, dict) or doc.get("kind") not in ("ClusterPolicy", "Policy"):
                continue
            spec = doc.get("spec", {}) or {}
            action = str(spec.get("validationFailureAction", "Audit"))
            action = {"audit": "Audit", "enforce": "Enforce", "warn": "Warn"}.get(action.lower(), action)
            ann = (doc.get("metadata", {}) or {}).get("annotations", {}) or {}
            title = str(ann.get("policies.kyverno.io/title", doc.get("metadata", {}).get("name", "")))
            desc = str(ann.get("policies.kyverno.io/description", ""))
            category = str(ann.get("policies.kyverno.io/category", ""))
            text = _strip_template(f"{title}. {desc}")
            for rule in spec.get("rules", []) or []:
                validate = rule.get("validate")
                if not isinstance(validate, dict):
                    skipped["no_validate"] += 1
                    continue
                fam = _classify_validate(validate, text + " " + json.dumps(rule.get("validate", {}), default=str))
                if fam is None:
                    skipped["unmodelled_validate"] += 1
                    continue
                kinds, mode = _kyv_match_kinds(rule.get("match", {}) or {})
                if not kinds:
                    skipped["no_kinds"] += 1
                    continue
                excl = _kyv_exclude_ns(rule.get("exclude", {}))
                reading = KyvReading(
                    family=fam["family"],
                    match_kinds=tuple(sorted(set(kinds))),
                    match_mode=mode,
                    exclude_ns=tuple(sorted(set(excl))),
                    required_labels=tuple(fam.get("required_labels", [])),
                    allowed_registries=tuple(fam.get("allowed_registries", [])),
                    allowed_caps=tuple(fam.get("allowed_caps", [])),
                    forbidden=tuple(fam.get("forbidden", [])),
                    resource_reqs=tuple(fam.get("resource_reqs", [])),
                    action=action)
                top_cat = fe["rel_path"].split("/")[0]
                key = (doc["metadata"]["name"], rule.get("name", ""), reading.struct_key())
                if key in seen_keys:
                    skipped["duplicate"] += 1
                    continue
                seen_keys.add(key)
                targets.append(KyvTarget(
                    policy_name=doc["metadata"]["name"], rule_name=str(rule.get("name", "")),
                    reading=reading, text=text, repo="kyverno-policies", commit=commit,
                    file=fe["rel_path"], sub_source=top_cat,
                    last_commit_ts=fe.get("last_commit_ts")))
    raw_n = len(targets)
    kept = _curate_kyverno(targets)
    skipped["raw_parsed"] = raw_n
    skipped["curated_kept"] = len(kept)
    return kept, dict(skipped)


# per-family cap keeps the benchmark set balanced and in the ~40-80 target band
_KYV_CAP_PER_FAMILY = {"field_constraint": 14, "required_labels": 11}
_KYV_CAP_DEFAULT = 8


def _curate_kyverno(targets: List[KyvTarget]) -> List[KyvTarget]:
    """Deterministically trim to a balanced, in-range benchmark set.

    Rich families (registry/capabilities/resources/label sets) are kept in
    full; the generic field_constraint family is capped so it does not swamp
    the set.  Selection is by sorted (policy_name, rule_name) -- fully
    reproducible, no randomness.
    """
    by_fam: Dict[str, List[KyvTarget]] = defaultdict(list)
    for t in sorted(targets, key=lambda x: (x.policy_name, x.rule_name)):
        by_fam[t.reading.family].append(t)
    kept: List[KyvTarget] = []
    for fam, ts in by_fam.items():
        cap = _KYV_CAP_PER_FAMILY.get(fam, _KYV_CAP_DEFAULT)
        kept.extend(ts[:cap])
    return sorted(kept, key=lambda x: (x.sub_source, x.policy_name, x.rule_name))


# --------------------------------------------------------------------------
# Threshold bank (adjacent values present in the repo, per metric family)
# --------------------------------------------------------------------------

def prometheus_oodsl() -> List[dict]:
    """Out-of-DSL Prometheus alerts (expr not structurally parseable) with text,
    for the overall-coverage denominator (their gold is unrepresentable)."""
    man = load_manifest()
    files = [f for f in man["files"] if f["family"] == "prometheus"]
    out = []
    for fe in files:
        doc = yaml.safe_load(open(fe["abs_path"]))
        for grp in (doc.get("spec", {}) or {}).get("groups", []) or []:
            for r in grp.get("rules", []) or []:
                if "alert" not in r or "expr" not in r:
                    continue
                if parse_prom_expr(str(r["expr"])) is not None:
                    continue
                ann = r.get("annotations", {}) or {}
                text = _strip_template(" ".join(str(ann.get(k, "")) for k in ("summary", "description")))
                if text:
                    out.append(dict(name=r["alert"], text=text, sub_source=grp.get("name", ""),
                                    last_commit_ts=fe.get("last_commit_ts")))
    return out


def kyverno_oodsl() -> List[dict]:
    """Out-of-DSL Kyverno rules (CEL / deny / foreach validate) with text."""
    man = load_manifest()
    files = [f for f in man["files"] if f["family"] == "kyverno"]
    out = []
    for fe in files:
        try:
            docs = list(yaml.safe_load_all(open(fe["abs_path"])))
        except yaml.YAMLError:
            continue
        for doc in docs:
            if not isinstance(doc, dict) or doc.get("kind") not in ("ClusterPolicy", "Policy"):
                continue
            ann = (doc.get("metadata", {}) or {}).get("annotations", {}) or {}
            title = str(ann.get("policies.kyverno.io/title", doc.get("metadata", {}).get("name", "")))
            desc = str(ann.get("policies.kyverno.io/description", ""))
            text = _strip_template(f"{title}. {desc}")
            for rule in (doc.get("spec", {}) or {}).get("rules", []) or []:
                v = rule.get("validate")
                if not isinstance(v, dict):
                    continue
                if any(k in v for k in ("cel", "deny", "foreach")) and text:
                    out.append(dict(name=doc["metadata"]["name"], text=text,
                                    sub_source=fe["rel_path"].split("/")[0],
                                    last_commit_ts=fe.get("last_commit_ts")))
                    break
    return out


def prom_threshold_bank(targets: List[PromTarget]) -> Dict[str, List[float]]:
    bank: Dict[str, set] = defaultdict(set)
    for t in targets:
        root = t.reading.metric.split("_")[0]
        bank[root].add(t.reading.threshold)
        bank["__all__"].add(t.reading.threshold)
    return {k: sorted(v) for k, v in bank.items()}


def main() -> None:
    pt, ps = parse_prometheus()
    kt, ks = parse_kyverno()
    print(f"Prometheus: parsed {len(pt)} alerts; skipped {dict(ps)}")
    print(f"  groups: {sorted(set(t.sub_source for t in pt))}")
    from collections import Counter
    print(f"  families(agg): {Counter(t.reading.aggregation for t in pt)}")
    print(f"Kyverno: parsed {len(kt)} rules; skipped {dict(ks)}")
    print(f"  families: {Counter(t.reading.family for t in kt)}")
    print(f"  categories: {Counter(t.sub_source for t in kt).most_common(8)}")


if __name__ == "__main__":
    main()
