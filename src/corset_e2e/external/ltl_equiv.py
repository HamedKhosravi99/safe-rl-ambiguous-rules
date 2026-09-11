"""Exact LTL equivalence in pure Python (no Spot on this machine).

Formulas are parsed into tuples, put in negation normal form, and
translated to a generalized Buchi automaton by the GPVW tableau (Gerth,
Peled, Vardi, Wolper 1995).  phi and psi are equivalent iff both
L(phi & !psi) and L(psi & !phi) are empty; emptiness is an SCC search for
a reachable nontrivial component meeting every acceptance set.

A bounded-lasso fingerprint (deterministic sample of ultimately periodic
words) runs first: a distinguishing word proves inequivalence at once, so
the tableau is only built for pairs the fingerprint cannot separate.

Syntax accepted (Spot / FRET / nuXmv flavours): !, &, &&, |, ||, ->, <->,
xor, ^, X, F, G, U, R, V (release), W (weak until), M (strong release),
bounded F[a,b] / G[a,b] / X[n] (comma or colon), TRUE/FALSE/true/false/1/0,
identifiers with letters, digits, '_', '.', and quoted "names".

The checker is validated in artemis_load.py against the artifact's own
Spot verdicts before any outcome is read.
"""
from __future__ import annotations

import hashlib
import re
import sys
from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

sys.setrecursionlimit(20000)

TRUE = ("true",)
FALSE = ("false",)

# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

_TOK = re.compile(r"""
    (?P<bounded>[FGX]\s*\[\s*\d+\s*(?:[,:]\s*\d+\s*)?\]) |
    (?P<iff><->) | (?P<imp>->) | (?P<and>&&|&) | (?P<or>\|\||\|) |
    (?P<xor>\^|\bxor\b) | (?P<not>!) |
    (?P<lp>\() | (?P<rp>\)) |
    (?P<const>\b(?:TRUE|FALSE|true|false|True|False)\b) |
    (?P<num>\b[01]\b) |
    (?P<id>"[^"]*"|[A-EH-WYZa-z_.][A-Za-z0-9_.]*|[FGX][0-9][A-Za-z0-9_.]*) |
    (?P<opletter>[FGX]) |
    (?P<ws>\s+)
""", re.X)

_UNARY = {"X", "F", "G"}
_BINARY_T = {"U", "R", "V", "W", "M"}
_CONST = {"TRUE": TRUE, "FALSE": FALSE, "true": TRUE, "false": FALSE, "True": TRUE, "False": FALSE, "1": TRUE, "0": FALSE}


class ParseError(ValueError):
    pass


def tokenize(s: str) -> List[Tuple[str, str]]:
    out = []
    pos = 0
    while pos < len(s):
        m = _TOK.match(s, pos)
        if not m:
            raise ParseError(f"bad char at {pos}: {s[pos:pos+20]!r}")
        pos = m.end()
        kind = m.lastgroup
        if kind == "ws":
            continue
        txt = m.group(kind)
        if kind == "opletter":
            kind = "unary"
        elif kind == "id":
            if txt in _BINARY_T:
                kind = "tbin"
            elif txt in _CONST:
                kind = "const"
        out.append((kind, txt))
    return out


class _Parser:
    # precedence (low -> high): iff < imp < xor < or < and < tbin < unary
    def __init__(self, s: str):
        self.toks = tokenize(s)
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else ("eof", "")

    def take(self):
        t = self.peek(); self.i += 1; return t

    def parse(self):
        f = self.p_iff()
        if self.peek()[0] != "eof":
            raise ParseError(f"trailing tokens: {self.toks[self.i:self.i+5]}")
        return f

    def p_iff(self):
        left = self.p_imp()
        while self.peek()[0] == "iff":
            self.take(); right = self.p_imp()
            left = ("iff", left, right)
        return left

    def p_imp(self):
        left = self.p_xor()
        if self.peek()[0] == "imp":
            self.take(); right = self.p_imp()          # right assoc
            return ("imp", left, right)
        return left

    def p_xor(self):
        left = self.p_or()
        while self.peek()[0] == "xor":
            self.take(); right = self.p_or()
            left = ("xor", left, right)
        return left

    def p_or(self):
        left = self.p_and()
        while self.peek()[0] == "or":
            self.take(); right = self.p_and()
            left = ("or", left, right)
        return left

    def p_and(self):
        left = self.p_tbin()
        while self.peek()[0] == "and":
            self.take(); right = self.p_tbin()
            left = ("and", left, right)
        return left

    def p_tbin(self):
        left = self.p_unary()
        if self.peek()[0] == "tbin":
            op = self.take()[1]
            right = self.p_tbin()                        # right assoc
            return (op if op != "V" else "R", left, right)
        return left

    def p_unary(self):
        k, t = self.peek()
        if k == "not":
            self.take(); return ("not", self.p_unary())
        if k == "unary":
            self.take(); return (t, self.p_unary())
        if k == "bounded":
            self.take()
            op = t[0]
            nums = [int(x) for x in re.findall(r"\d+", t)]
            a, b = (nums[0], nums[1]) if len(nums) == 2 else (nums[0], nums[0])
            if op == "X" and len(nums) == 1:
                a = b = nums[0]
            sub = self.p_unary()
            return ("bounded", op, a, b, sub)
        if k == "lp":
            self.take(); f = self.p_iff()
            if self.take()[0] != "rp":
                raise ParseError("expected )")
            return f
        if k == "const":
            self.take(); return _CONST[t]
        if k == "num":
            self.take(); return _CONST[t]
        if k == "id":
            self.take(); return ("ap", t.strip('"'))
        raise ParseError(f"unexpected token {k} {t!r}")


def parse(s: str):
    return _Parser(s).parse()


# ---------------------------------------------------------------------------
# core normal form: {ap,true,false,not(ap only),and,or,X,U,R}
# ---------------------------------------------------------------------------

def _xn(f, n: int):
    for _ in range(n):
        f = ("X", f)
    return f


def _mk_and(a, b):
    if a == FALSE or b == FALSE:
        return FALSE
    if a == TRUE:
        return b
    if b == TRUE:
        return a
    if a == b:
        return a
    return ("and", a, b) if repr(a) <= repr(b) else ("and", b, a)


def _mk_or(a, b):
    if a == TRUE or b == TRUE:
        return TRUE
    if a == FALSE:
        return b
    if b == FALSE:
        return a
    if a == b:
        return a
    return ("or", a, b) if repr(a) <= repr(b) else ("or", b, a)


def nnf(f, neg: bool = False):
    """Negation normal form over the core operators, with constant folding."""
    k = f[0]
    if k == "true":
        return FALSE if neg else TRUE
    if k == "false":
        return TRUE if neg else FALSE
    if k == "ap":
        return ("not", f) if neg else f
    if k == "not":
        return nnf(f[1], not neg)
    if k == "and":
        a, b = nnf(f[1], neg), nnf(f[2], neg)
        return _mk_or(a, b) if neg else _mk_and(a, b)
    if k == "or":
        a, b = nnf(f[1], neg), nnf(f[2], neg)
        return _mk_and(a, b) if neg else _mk_or(a, b)
    if k == "imp":
        return nnf(("or", ("not", f[1]), f[2]), neg)
    if k == "iff":
        return nnf(("and", ("imp", f[1], f[2]), ("imp", f[2], f[1])), neg)
    if k == "xor":
        return nnf(("or", ("and", f[1], ("not", f[2])), ("and", ("not", f[1]), f[2])), neg)
    if k == "X":
        return ("X", nnf(f[1], neg))
    if k == "F":
        return nnf(("U", TRUE, f[1]), neg)
    if k == "G":
        return nnf(("R", FALSE, f[1]), neg)
    if k == "W":                                   # a W b = (a U b) | G a
        return nnf(("or", ("U", f[1], f[2]), ("G", f[1])), neg)
    if k == "M":                                   # a M b = b U (a & b)
        return nnf(("U", f[2], ("and", f[1], f[2])), neg)
    if k == "U":
        a, b = nnf(f[1], neg), nnf(f[2], neg)
        return ("R", a, b) if neg else _mk_U(a, b)
    if k == "R":
        a, b = nnf(f[1], neg), nnf(f[2], neg)
        return _mk_U(a, b) if neg else ("R", a, b)
    if k == "bounded":
        _, op, a, b, sub = f
        if op == "X":
            return nnf(_xn(sub, b), neg)
        terms = [_xn(sub, i) for i in range(a, b + 1)]
        g = terms[0]
        for t in terms[1:]:
            g = ("or", g, t) if op == "F" else ("and", g, t)
        return nnf(g, neg)
    raise ValueError(f"unknown node {k}")


def _mk_U(a, b):
    if b == TRUE:
        return TRUE
    if b == FALSE:
        return FALSE
    return ("U", a, b)


# ---------------------------------------------------------------------------
# lasso-word semantics (fingerprint)
# ---------------------------------------------------------------------------

def atomic_props(f, acc=None) -> FrozenSet[str]:
    if acc is None:
        acc = set()
    if f[0] == "ap":
        acc.add(f[1])
    elif f[0] in ("true", "false"):
        pass
    elif f[0] == "bounded":
        atomic_props(f[4], acc)
    elif f[0] == "andset":
        for p in f[1]:
            atomic_props(p, acc)
    else:
        for sub in f[1:]:
            if isinstance(sub, tuple):
                atomic_props(sub, acc)
    return frozenset(acc)


def eval_lasso(f, word: Sequence[FrozenSet[str]], loop: int) -> bool:
    """Truth of core-NNF formula f at position 0 of the ultimately periodic
    word word[0..n-1] with word[n] = word[loop]."""
    n = len(word)
    memo: Dict[Tuple[int, int], bool] = {}

    def succ(i: int) -> int:
        return i + 1 if i + 1 < n else loop

    def ev(g, i: int) -> bool:
        key = (id(g), i)
        if key in memo:
            return memo[key]
        k = g[0]
        if k == "true":
            r = True
        elif k == "false":
            r = False
        elif k == "ap":
            r = g[1] in word[i]
        elif k == "not":
            r = g[1][1] not in word[i]
        elif k == "and":
            r = ev(g[1], i) and ev(g[2], i)
        elif k == "andset":
            r = all(ev(p, i) for p in g[1])
        elif k == "or":
            r = ev(g[1], i) or ev(g[2], i)
        elif k == "X":
            r = ev(g[1], succ(i))
        elif k == "U":
            r = False
            j = i
            for _ in range(n + 1):
                if ev(g[2], j):
                    r = True
                    break
                if not ev(g[1], j):
                    break
                j = succ(j)
        elif k == "R":                          # a R b = b holds until (and including when) a holds
            r = True
            j = i
            for _ in range(n + 1):
                if not ev(g[2], j):
                    r = False
                    break
                if ev(g[1], j):
                    break
                j = succ(j)
        else:
            raise ValueError(k)
        memo[key] = r
        return r

    return ev(f, 0)


def _sample_words(aps: Sequence[str], n_random: int = 240) -> List[Tuple[List[FrozenSet[str]], int]]:
    aps = sorted(aps)
    words = []
    m = len(aps)
    # all constant words when cheap
    if m <= 5:
        for mask in range(1 << m):
            letter = frozenset(a for j, a in enumerate(aps) if mask >> j & 1)
            words.append(([letter], 0))
    # deterministic pseudo-random lassos
    seed = int(hashlib.sha1(",".join(aps).encode()).hexdigest()[:8], 16)
    state = seed or 1

    def rnd(k: int) -> int:                      # xorshift32
        nonlocal state
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        return state % k

    for _ in range(n_random):
        p = rnd(6)
        c = 1 + rnd(4)
        w = []
        for _t in range(p + c):
            letter = frozenset(a for a in aps if rnd(2))
            w.append(letter)
        words.append((w, p))
    return words


# ---------------------------------------------------------------------------
# GPVW tableau -> generalized Buchi automaton, and emptiness
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ("incoming", "new", "old", "next", "id")

    def __init__(self, incoming, new, old, nxt, nid):
        self.incoming = set(incoming)
        self.new = set(new)
        self.old = set(old)
        self.next = set(nxt)
        self.id = nid


class TableauBudget(Exception):
    pass


def _closure_until(f, acc: set) -> None:
    if f[0] in ("ap", "true", "false", "not"):
        return
    if f[0] == "U":
        acc.add(f)
    if f[0] == "andset":
        for sub in f[1]:
            _closure_until(sub, acc)
        return
    for sub in f[1:]:
        _closure_until(sub, acc)


def _neg_lit(lit):
    return lit[1] if lit[0] == "not" else ("not", lit)


def build_gba(f, max_nodes: int = 60000):
    """Nodes (states), initial states, transitions m->n labelled by n's
    literals, and acceptance sets, for a core-NNF formula f."""
    nodes: Dict[Tuple[FrozenSet, FrozenSet], _Node] = {}
    counter = [0]

    def fresh(incoming, new, old, nxt):
        counter[0] += 1
        if counter[0] > max_nodes:
            raise TableauBudget(counter[0])
        return _Node(incoming, new, old, nxt, counter[0])

    stack = [fresh({"init"}, {f}, set(), set())]
    while stack:
        nd = stack.pop()
        if not nd.new:
            key = (frozenset(nd.old), frozenset(nd.next))
            if key in nodes:
                nodes[key].incoming |= nd.incoming
                continue
            nodes[key] = nd
            stack.append(fresh({nd.id}, set(nd.next), set(), set()))
            continue
        eta = nd.new.pop()
        k = eta[0]
        if k == "false":
            continue
        if k == "true":
            nd.old.add(eta); stack.append(nd); continue
        if k in ("ap", "not"):
            if _neg_lit(eta) in nd.old:
                continue
            nd.old.add(eta); stack.append(nd); continue
        if k == "and" or k == "andset":
            for sub in (eta[1:] if k == "and" else eta[1]):
                if sub not in nd.old:
                    nd.new.add(sub)
            nd.old.add(eta); stack.append(nd); continue
        if k == "X":
            nd.next.add(eta[1]); nd.old.add(eta); stack.append(nd); continue
        if k == "or":
            a, b = eta[1], eta[2]
            n1 = fresh(nd.incoming, nd.new | ({a} - nd.old), nd.old | {eta}, nd.next)
            n2 = fresh(nd.incoming, nd.new | ({b} - nd.old), nd.old | {eta}, nd.next)
            stack.extend([n1, n2]); continue
        if k == "U":
            a, b = eta[1], eta[2]
            n1 = fresh(nd.incoming, nd.new | ({a} - nd.old), nd.old | {eta}, nd.next | {eta})
            n2 = fresh(nd.incoming, nd.new | ({b} - nd.old), nd.old | {eta}, nd.next)
            stack.extend([n1, n2]); continue
        if k == "R":
            a, b = eta[1], eta[2]
            n1 = fresh(nd.incoming, nd.new | ({b} - nd.old), nd.old | {eta}, nd.next | {eta})
            n2 = fresh(nd.incoming, nd.new | ({a, b} - nd.old), nd.old | {eta}, nd.next)
            stack.extend([n1, n2]); continue
        raise ValueError(k)

    # graph over node ids
    by_id = {nd.id: nd for nd in nodes.values()}
    init = [nd.id for nd in nodes.values() if "init" in nd.incoming]
    succs: Dict[int, List[int]] = {nid: [] for nid in by_id}
    for nd in nodes.values():
        for src in nd.incoming:
            if src != "init" and src in by_id:
                succs[src].append(nd.id)
    untils: set = set()
    _closure_until(f, untils)
    acc_sets = []
    for u in untils:
        acc_sets.append({nd.id for nd in nodes.values() if (u not in nd.old) or (u[2] in nd.old)})
    return by_id, init, succs, acc_sets


def gba_nonempty(by_id, init, succs, acc_sets) -> bool:
    """Reachable nontrivial SCC intersecting every acceptance set?"""
    # iterative Tarjan
    index = {}
    low = {}
    onstack = set()
    st: List[int] = []
    idx = [0]
    reachable = set()
    # reachability first
    work = list(init)
    while work:
        v = work.pop()
        if v in reachable:
            continue
        reachable.add(v)
        work.extend(succs[v])
    for root in reachable:
        if root in index:
            continue
        call = [(root, iter(succs[root]))]
        index[root] = low[root] = idx[0]; idx[0] += 1
        st.append(root); onstack.add(root)
        while call:
            v, it = call[-1]
            advanced = False
            for w in it:
                if w not in index:
                    index[w] = low[w] = idx[0]; idx[0] += 1
                    st.append(w); onstack.add(w)
                    call.append((w, iter(succs[w])))
                    advanced = True
                    break
                elif w in onstack:
                    low[v] = min(low[v], index[w])
            if advanced:
                continue
            call.pop()
            if call:
                u = call[-1][0]
                low[u] = min(low[u], low[v])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = st.pop(); onstack.discard(w); comp.append(w)
                    if w == v:
                        break
                cs = set(comp)
                nontrivial = len(comp) > 1 or any(w in cs for w in succs[comp[0]])
                if nontrivial and all(cs & a for a in acc_sets):
                    return True
    return False



# ---------------------------------------------------------------------------
# formula-progression TGBA (Couvreur 1999 style): states are formulas
# ---------------------------------------------------------------------------

def _and_all(parts):
    flat = set()
    for p in parts:
        if p == TRUE:
            continue
        if p == FALSE:
            return FALSE
        if p[0] == "andset":
            flat |= set(p[1])
        else:
            flat.add(p)
    if not flat:
        return TRUE
    if len(flat) == 1:
        return next(iter(flat))
    return ("andset", frozenset(flat))


def _consistent(lits) -> bool:
    for l in lits:
        if l[0] == "not" and l[1] in lits:
            return False
    return True


class _Expander:
    """expand(f) -> list of (lits, next_formula, promises) with f = OR_i (lits_i & X next_i)."""

    def __init__(self, budget: int):
        self.memo = {}
        self.budget = budget
        self.work = 0

    def expand(self, f):
        if f in self.memo:
            return self.memo[f]
        self.work += 1
        if self.work > self.budget:
            raise TableauBudget(self.work)
        k = f[0]
        if k == "true":
            r = [(frozenset(), TRUE, frozenset())]
        elif k == "false":
            r = []
        elif k in ("ap", "not"):
            r = [(frozenset([f]), TRUE, frozenset())]
        elif k == "andset":
            r = [(frozenset(), TRUE, frozenset())]
            for part in sorted(f[1], key=repr):
                ep = self.expand(part)
                nr = []
                for (l1, n1, p1) in r:
                    for (l2, n2, p2) in ep:
                        l = l1 | l2
                        if _consistent(l):
                            nr.append((l, _and_all([n1, n2]), p1 | p2))
                r = _dedupe(nr)
                if len(r) > self.budget:
                    raise TableauBudget(len(r))
        elif k == "and":
            r = self.expand(_and_all([f[1], f[2]]))
        elif k == "or":
            r = _dedupe(self.expand(f[1]) + self.expand(f[2]))
        elif k == "X":
            r = [(frozenset(), f[1], frozenset())]
        elif k == "U":
            g = f
            r = _dedupe(self.expand(f[2]) +
                        [(l, _and_all([n, g]), p | {g}) for (l, n, p) in self.expand(f[1])])
        elif k == "R":
            g = f
            r = _dedupe(self.expand(_and_all([f[1], f[2]])) +
                        [(l, _and_all([n, g]), p) for (l, n, p) in self.expand(f[2])])
        else:
            raise ValueError(k)
        self.memo[f] = r
        return r


def _dedupe(rows):
    seen = set()
    out = []
    for row in rows:
        if row not in seen:
            seen.add(row)
            out.append(row)
    return out


def _untils(f, acc):
    k = f[0]
    if k in ("ap", "true", "false", "not"):
        return
    if k == "U":
        acc.add(f)
    if k == "andset":
        for p in f[1]:
            _untils(p, acc)
        return
    for sub in f[1:]:
        if isinstance(sub, tuple):
            _untils(sub, acc)


def tgba_nonempty(core_f, budget: int = 200000) -> bool:
    """Is L(core_f) nonempty?  States = progression formulas; transition-based
    generalized acceptance: an SCC is accepting iff for every U-subformula u it
    contains an internal edge that does not promise u."""
    core_f = _and_all([core_f])
    if core_f == FALSE:
        return False
    if core_f == TRUE:
        return True
    ex = _Expander(budget)
    untils = set()
    _untils(core_f, untils)
    untils = sorted(untils, key=repr)
    uidx = {u: i for i, u in enumerate(untils)}
    nU = len(untils)
    # explore
    ids = {core_f: 0}
    states = [core_f]
    edges = {}          # sid -> list of (tid, promise_mask)
    stack = [0]
    while stack:
        s = stack.pop()
        f = states[s]
        rows = ex.expand(f)
        out = []
        for (lits, nxt, prom) in rows:
            nxt = _and_all([nxt])
            if nxt == FALSE:
                continue
            if nxt not in ids:
                ids[nxt] = len(states)
                states.append(nxt)
                stack.append(ids[nxt])
                if len(states) > budget:
                    raise TableauBudget(len(states))
            mask = 0
            for u in prom:
                mask |= 1 << uidx[u]
            out.append((ids[nxt], mask))
        edges[s] = out
    # Tarjan with per-SCC promise bookkeeping
    index = {}
    low = {}
    onst = set()
    st = []
    ctr = [0]
    full = (1 << nU) - 1
    for root in range(len(states)):
        if root in index:
            continue
        index[root] = low[root] = ctr[0]; ctr[0] += 1
        st.append(root); onst.add(root)
        call = [(root, iter(edges[root]))]
        while call:
            v, it = call[-1]
            adv = False
            for (w, _m) in it:
                if w not in index:
                    index[w] = low[w] = ctr[0]; ctr[0] += 1
                    st.append(w); onst.add(w)
                    call.append((w, iter(edges[w])))
                    adv = True
                    break
                elif w in onst:
                    low[v] = min(low[v], index[w])
            if adv:
                continue
            call.pop()
            if call:
                u = call[-1][0]
                low[u] = min(low[u], low[v])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = st.pop(); onst.discard(w); comp.append(w)
                    if w == v:
                        break
                cs = set(comp)
                # internal edges: for each U, need an edge NOT promising it
                seen_free = 0
                internal = False
                for a in comp:
                    for (b, m) in edges[a]:
                        if b in cs:
                            internal = True
                            seen_free |= (~m) & full
                if internal and seen_free == full:
                    return True
    return False

# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=200000)
def _core(s: str):
    return nnf(parse(s))


def is_valid_ltl(s: str) -> bool:
    try:
        parse(s)
        return True
    except ParseError:
        return False
    except RecursionError:
        return False


def gpvw_language_empty(core_f, max_nodes: int = 60000) -> bool:
    if core_f == FALSE:
        return True
    if core_f == TRUE:
        return False
    by_id, init, succs, acc = build_gba(core_f, max_nodes=max_nodes)
    return not gba_nonempty(by_id, init, succs, acc)


def language_empty(core_f, max_nodes: int = 200000) -> bool:
    return not tgba_nonempty(core_f, budget=max_nodes)


def fingerprint(core_f, aps: Sequence[str]) -> Tuple[bool, ...]:
    return tuple(eval_lasso(core_f, w, p) for w, p in _sample_words(aps))


_EQ_CACHE: Dict[Tuple[str, str], Optional[bool]] = {}


def equivalent(s1: str, s2: str, max_nodes: int = 200000) -> Optional[bool]:
    """True / False, or None when the tableau budget was exhausted (undecided)."""
    key = (s1, s2) if s1 <= s2 else (s2, s1)
    if key in _EQ_CACHE:
        return _EQ_CACHE[key]
    f, g = _core(s1), _core(s2)
    res: Optional[bool]
    if f == g:
        res = True
    else:
        aps = sorted(atomic_props(f) | atomic_props(g))
        if fingerprint(f, aps) != fingerprint(g, aps):
            res = False
        else:
            try:
                nf, ng = nnf(("not", f)), nnf(("not", g))
                res = language_empty(_mk_and(f, ng), max_nodes) and \
                    language_empty(_mk_and(g, nf), max_nodes)
            except TableauBudget:
                res = None
    _EQ_CACHE[key] = res
    return res


def satisfiable(s: str, max_nodes: int = 200000) -> Optional[bool]:
    try:
        return not language_empty(_core(s), max_nodes)
    except TableauBudget:
        return None


if __name__ == "__main__":       # a few sanity checks
    checks = [
        ("G(a -> F b)", "G(!a | F b)", True),
        ("F G a", "G F a", False),
        ("a U b", "b | (a & X(a U b))", True),
        ("G a", "!F !a", True),
        ("a W b", "(a U b) | G a", True),
        ("F[0,2] p", "p | X p | X X p", True),
        ("F[0,2] p", "F[0,3] p", False),
        ("a R b", "b W (a & b)", True),
        ("a M b", "b U (a & b)", True),
        ("G(a -> X b)", "G(!a | X b)", True),
        ("(a) V (b)", "a R b", True),
        ("TRUE", "1", True),
        ("G F a & G F b", "G F (a & F b)", True),
        ("a xor b", "(a & !b) | (!a & b)", True),
        ("a <-> b", "(a -> b) & (b -> a)", True),
        ("X X a", "X[2] a", True),
    ]
    checks += [
        ("Xa", "X a", True), ("GFa", "G F a", True), ("Fault", "F ault", True),
        ("G((t & s) -> Xq)", "G((t & s) -> X q)", True), ("X1a", "X1a", True),
        ("G(p -> F[0,3] q)", "G(p -> (q | X q | X X q | X X X q))", True),
        ("G(p -> F[0,3] q)", "G(p -> F[0,2] q)", False),
        ("(a) V (b | a)", "b W a", True),
    ]
    bad = 0
    for s1, s2, exp in checks:
        got = equivalent(s1, s2)
        # cross-check the two constructions on the difference formulas
        f, g = _core(s1), _core(s2)
        d1 = _mk_and(f, nnf(("not", g)))
        d2 = _mk_and(g, nnf(("not", f)))
        x1 = gpvw_language_empty(d1) and gpvw_language_empty(d2)
        flag = "ok " if (got == exp and x1 == exp) else "BAD"
        bad += flag == "BAD"
        print(flag, s1, "==", s2, got, "gpvw:", x1)
    print("BAD count", bad)
