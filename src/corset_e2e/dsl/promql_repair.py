"""Recover expressions whose comments lost their terminator when the
corpus was flattened.

The archived `raw_expr` in the hidden store has had its newlines replaced
by spaces. PromQL comments run from `#` to end of line, so in flattened
text a comment has no terminator and swallows the rest of the expression.
This is a property of the frozen corpus, not of the parser, and the
archive is the artifact of record -- so the repair lives here rather than
in the parser or in a re-harvest.

The repair is a bounded deterministic search. For the first `#`, try
ending the comment at each following word boundary, earliest first, and
accept the first cut whose result parses completely. Prose is a poor
expression -- two adjacent words parse as two operands with no operator,
and sentence punctuation does not lex -- so early cuts almost always fail
and the first success is the sentence end. `repair_stats` reports how many
units needed this and how many remained unparseable, so the cost of the
artifact is visible rather than absorbed.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

from corset_e2e.dsl.promql_ast import Node, try_parse, walk

MAX_COMMENTS = 6
MAX_CUTS = 400
# Each comment multiplies the search, so a depth-first search over several
# comments is exponential. A global attempt budget makes the cost linear in
# the budget and the failure explicit: a unit that exhausts it is reported
# unparseable rather than silently taking unbounded time.
MAX_ATTEMPTS = 3000
_WORD_BOUNDARY = re.compile(r"\s+")


def _size(node: Node) -> int:
    return sum(1 for _ in walk(node))


def _search(src: str, depth: int, budget: list):
    """(ast, how, size) for the repair that RETAINS THE MOST CODE.

    Two criteria are wrong here and both were tried. Taking the first cut
    that parses truncates: deleting a comment through to the end of the
    string is always available, and whatever prose word precedes it then
    parses alone as a bare selector -- one real corpus expression was
    reduced from `max_over_time(...) == 0` to the selector `see`. Scoring
    by retained CHARACTERS does not fix it either, because a candidate
    ending in an unterminated comment keeps every one of those characters
    in the string while the parser never sees them. The score has to be
    the size of what was actually parsed, so it is the AST node count.
    """
    if budget[0] <= 0:
        return None, "repair budget exhausted", -1
    budget[0] -= 1
    node, err = try_parse(src)
    if node is not None:
        return node, "direct" if depth == 0 else "repaired", _size(node)
    if depth >= MAX_COMMENTS:
        return None, err or "repair depth exceeded", -1
    h = src.find("#")
    if h < 0:
        return None, err or "parse error", -1
    head, tail = src[:h], src[h:]
    best = (None, err or "parse error", -1)
    for m in list(_WORD_BOUNDARY.finditer(tail))[:MAX_CUTS]:
        got, how, size = _search(head + tail[m.end():], depth + 1, budget)
        if got is not None and size > best[2]:
            best = (got, how, size)
        if budget[0] <= 0:
            break
    return best


def parse_repairing_comments(src: str) -> Tuple[Optional[Node], str]:
    """(ast, how) where how is 'direct', 'repaired', or a failure message."""
    node, how, _size_ = _search(src, 0, [MAX_ATTEMPTS])
    return node, how


def repair_stats(exprs) -> dict:
    n = direct = repaired = 0
    failures = {}
    for x in exprs:
        x = (x or "").strip()
        if not x:
            continue
        n += 1
        node, how = parse_repairing_comments(x)
        if node is None:
            failures[how] = failures.get(how, 0) + 1
        elif how == "direct":
            direct += 1
        else:
            repaired += 1
    return dict(n=n, direct=direct, repaired=repaired,
                parsed=direct + repaired, failures=failures)
