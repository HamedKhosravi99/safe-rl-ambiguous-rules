"""A PromQL tokenizer and precedence-climbing parser producing a typed AST.

Why this exists. Every reach measurement before this module was made with
regular expressions over the raw expression string, and that approach has
now produced a demonstrable error: the G2 conjunction reach was measured
by splitting on `\\band\\b` with `re.split`, which yields
parenthesis-unbalanced fragments, so a fully parenthesised

    ( (A > k) and on(...) (B > k') )

never exposed a depth-0 comparator in either fragment and was scored as
"not a conjunction of comparisons". The published consequence was that the
conjunction axis is worth 1.0% of the corpus. It is not.

Regexes cannot decide questions about nesting. This module parses instead,
and every subsequent grammar-reach question is asked of the AST.

Scope: the PromQL subset that appears in alerting rules -- binary
operators with their full matching modifiers, aggregations with
by/without and a parameter, function calls, instant and range vector
selectors with label matchers, offset and @ modifiers, subqueries,
numeric and string literals, unary sign, and parentheses. Unknown
constructs raise ParseError with a position, so "we cannot parse this" is
distinguishable from "we parsed it and it is out of grammar" -- a
distinction the previous string matching could not make.

This module never opens the hidden store. It is pure syntax.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union


class ParseError(Exception):
    def __init__(self, msg: str, pos: int = -1):
        super().__init__(f"{msg} (at {pos})")
        self.msg = msg
        self.pos = pos


# ---------------------------------------------------------------- tokenizer

AGGREGATORS = frozenset((
    "sum", "min", "max", "avg", "group", "stddev", "stdvar", "count",
    "count_values", "bottomk", "topk", "quantile", "limitk", "limit_ratio"))

# aggregators taking a first scalar/string parameter
PARAM_AGGREGATORS = frozenset(("topk", "bottomk", "quantile", "count_values",
                               "limitk", "limit_ratio"))

KEYWORDS = frozenset((
    "by", "without", "on", "ignoring", "group_left", "group_right", "offset",
    "bool", "and", "or", "unless", "start", "end"))

SET_OPS = frozenset(("and", "or", "unless"))
COMPARISON_OPS = frozenset(("==", "!=", ">", "<", ">=", "<="))
ARITH_OPS = frozenset(("+", "-", "*", "/", "%", "^"))

# lowest binds loosest
_PRECEDENCE = {
    "or": 1,
    "and": 2, "unless": 2,
    "==": 3, "!=": 3, ">": 3, "<": 3, ">=": 3, "<=": 3,
    "+": 4, "-": 4,
    "*": 5, "/": 5, "%": 5,
    "^": 6,
}
_RIGHT_ASSOC = frozenset(("^",))

_DURATION = re.compile(r"(?:\d+(?:\.\d+)?(?:ms|[smhdwy]))+")
_NUMBER = re.compile(
    r"(?:0[xX][0-9a-fA-F]+"
    r"|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?"
    r"|\.\d+(?:[eE][+-]?\d+)?"
    r"|[iI][nN][fF]|[nN][aA][nN])")
# Recording-rule names carry colons (`job:errors:rate5m`), but a colon is
# also the subquery step separator in `[30m:1m]`. Allowing a LEADING colon
# in an identifier makes `:1m` lex as a name and the subquery unparseable,
# so colons are accepted only between name segments and `:` is punctuation
# everywhere else. A metric whose name begins with a colon is legal
# PromQL but does not occur here; it would surface as a parse error rather
# than be mis-parsed silently.
_IDENT = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*(?::[a-zA-Z0-9_]+)*")
# multi-char operators must be tried before their prefixes
_OPERATORS = ("=~", "!~", "==", "!=", ">=", "<=", "=", ">", "<",
              "+", "-", "*", "/", "%", "^")
_PUNCT = ("(", ")", "{", "}", "[", "]", ",", "@", ":")


@dataclass
class Token:
    kind: str          # NUM STR IDENT OP PUNCT DUR
    text: str
    pos: int


def tokenize(src: str) -> List[Token]:
    out: List[Token] = []
    i, n = 0, len(src)
    while i < n:
        ch = src[i]
        if ch in " \t\r\n":
            i += 1
            continue
        if ch == "#":                                   # comment to EOL
            j = src.find("\n", i)
            i = n if j < 0 else j + 1
            continue
        if ch in "\"'`":                                # string literal
            q, j = ch, i + 1
            buf = []
            while j < n:
                if src[j] == "\\" and q != "`" and j + 1 < n:
                    buf.append(src[j:j + 2])
                    j += 2
                    continue
                if src[j] == q:
                    break
                buf.append(src[j])
                j += 1
            if j >= n:
                raise ParseError("unterminated string", i)
            out.append(Token("STR", "".join(buf), i))
            i = j + 1
            continue
        # a duration only where a number could start, and only if the whole
        # run is a duration (5m, 1h30m) -- otherwise it is a number
        if ch.isdigit() or (ch == "." and i + 1 < n and src[i + 1].isdigit()):
            m = _DURATION.match(src, i)
            if m and (m.end() >= n or not src[m.end()].isalnum()):
                out.append(Token("DUR", m.group(0), i))
                i = m.end()
                continue
            m = _NUMBER.match(src, i)
            if not m:
                raise ParseError("bad number", i)
            out.append(Token("NUM", m.group(0), i))
            i = m.end()
            continue
        m = _IDENT.match(src, i)
        if m:
            out.append(Token("IDENT", m.group(0), i))
            i = m.end()
            continue
        for op in _OPERATORS:
            if src.startswith(op, i):
                out.append(Token("OP", op, i))
                i += len(op)
                break
        else:
            if ch in _PUNCT:
                out.append(Token("PUNCT", ch, i))
                i += 1
            else:
                raise ParseError(f"unexpected character {ch!r}", i)
    return out


def duration_seconds(text: str) -> float:
    """`1h30m` -> 5400.0. Years are 365d, as Prometheus defines them."""
    unit = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0,
            "w": 604800.0, "y": 31536000.0}
    total, i = 0.0, 0
    for m in re.finditer(r"(\d+(?:\.\d+)?)(ms|[smhdwy])", text):
        if m.start() != i:
            raise ParseError(f"bad duration {text!r}")
        total += float(m.group(1)) * unit[m.group(2)]
        i = m.end()
    if i != len(text):
        raise ParseError(f"bad duration {text!r}")
    return total


# --------------------------------------------------------------------- AST

@dataclass
class Node:
    def kind(self) -> str:
        return type(self).__name__


@dataclass
class NumberLit(Node):
    value: float


@dataclass
class StringLit(Node):
    value: str


@dataclass
class LabelMatcher:
    name: str
    op: str            # = != =~ !~
    value: str


@dataclass
class VectorSelector(Node):
    metric: Optional[str]
    matchers: List[LabelMatcher] = field(default_factory=list)
    offset_s: Optional[float] = None
    at_: Optional[str] = None


@dataclass
class MatrixSelector(Node):
    vs: VectorSelector
    range_s: float


@dataclass
class Subquery(Node):
    expr: Node
    range_s: float
    step_s: Optional[float] = None
    offset_s: Optional[float] = None


@dataclass
class Call(Node):
    func: str
    args: List[Node]


@dataclass
class Aggregate(Node):
    op: str
    expr: Node
    param: Optional[Node] = None
    grouping: Tuple[str, ...] = ()
    without: bool = False


@dataclass
class Matching:
    on: bool = False               # True => on(...), False => ignoring(...)
    labels: Tuple[str, ...] = ()
    card: Optional[str] = None     # group_left / group_right
    include: Tuple[str, ...] = ()


@dataclass
class Binary(Node):
    op: str
    lhs: Node
    rhs: Node
    matching: Optional[Matching] = None
    bool_: bool = False


@dataclass
class Unary(Node):
    op: str
    expr: Node


@dataclass
class Paren(Node):
    expr: Node


Expr = Union[NumberLit, StringLit, VectorSelector, MatrixSelector, Subquery,
             Call, Aggregate, Binary, Unary, Paren]


# ------------------------------------------------------------------ parser

class Parser:
    def __init__(self, src: str):
        self.src = src
        self.toks = tokenize(src)
        self.i = 0

    # -- token helpers
    def peek(self, k: int = 0) -> Optional[Token]:
        j = self.i + k
        return self.toks[j] if j < len(self.toks) else None

    def at(self, kind: str, text: Optional[str] = None) -> bool:
        t = self.peek()
        if t is None or t.kind != kind:
            return False
        if text is None:
            return True
        cmp_ = t.text.lower() if kind == "IDENT" else t.text
        return cmp_ == text

    def next(self) -> Token:
        t = self.peek()
        if t is None:
            raise ParseError("unexpected end of expression", len(self.src))
        self.i += 1
        return t

    def expect(self, kind: str, text: Optional[str] = None) -> Token:
        if not self.at(kind, text):
            t = self.peek()
            got = "EOF" if t is None else repr(t.text)
            raise ParseError(f"expected {text or kind}, got {got}",
                             len(self.src) if t is None else t.pos)
        return self.next()

    # -- entry point
    def parse(self) -> Node:
        e = self.parse_expr(0)
        if self.peek() is not None:
            raise ParseError(f"trailing input {self.peek().text!r}",
                             self.peek().pos)
        return e

    # -- precedence climbing
    def parse_expr(self, min_prec: int) -> Node:
        lhs = self.parse_unary()
        while True:
            t = self.peek()
            if t is None:
                break
            op = t.text.lower() if t.kind == "IDENT" else t.text
            if t.kind == "IDENT" and op not in SET_OPS:
                break
            if t.kind not in ("OP", "IDENT"):
                break
            prec = _PRECEDENCE.get(op)
            if prec is None or prec < min_prec:
                break
            self.next()
            bool_ = False
            if self.at("IDENT", "bool"):
                self.next()
                bool_ = True
            matching = self.parse_matching()
            nxt = prec if op in _RIGHT_ASSOC else prec + 1
            rhs = self.parse_expr(nxt)
            lhs = Binary(op, lhs, rhs, matching, bool_)
        return lhs

    def parse_matching(self) -> Optional[Matching]:
        m: Optional[Matching] = None
        if self.at("IDENT", "on") or self.at("IDENT", "ignoring"):
            on = self.next().text.lower() == "on"
            m = Matching(on=on, labels=self.parse_label_list())
        if self.at("IDENT", "group_left") or self.at("IDENT", "group_right"):
            card = self.next().text.lower()
            include: Tuple[str, ...] = ()
            if self.at("PUNCT", "("):
                include = self.parse_label_list()
            if m is None:
                m = Matching()
            m = Matching(m.on, m.labels, card, include)
        return m

    def parse_label_list(self) -> Tuple[str, ...]:
        self.expect("PUNCT", "(")
        names: List[str] = []
        while not self.at("PUNCT", ")"):
            names.append(self.expect("IDENT").text)
            if self.at("PUNCT", ","):
                self.next()
        self.expect("PUNCT", ")")
        return tuple(names)

    def parse_unary(self) -> Node:
        t = self.peek()
        if t is not None and t.kind == "OP" and t.text in ("-", "+"):
            self.next()
            return Unary(t.text, self.parse_expr(_PRECEDENCE["*"]))
        return self.parse_postfix()

    # -- primary with postfix [range] offset @ and subquery
    def parse_postfix(self) -> Node:
        e = self.parse_primary()
        while True:
            if self.at("PUNCT", "["):
                self.next()
                d1 = self.expect("DUR").text
                if self.at("PUNCT", ":"):                # subquery
                    self.next()
                    step = None
                    if self.at("DUR"):
                        step = duration_seconds(self.next().text)
                    self.expect("PUNCT", "]")
                    e = Subquery(e, duration_seconds(d1), step)
                else:
                    self.expect("PUNCT", "]")
                    if isinstance(e, VectorSelector):
                        e = MatrixSelector(e, duration_seconds(d1))
                    else:
                        raise ParseError("range on a non-selector")
                continue
            if self.at("IDENT", "offset"):
                self.next()
                sign = 1.0
                if self.at("OP", "-"):
                    self.next()
                    sign = -1.0
                off = sign * duration_seconds(self.expect("DUR").text)
                if isinstance(e, VectorSelector):
                    e.offset_s = off
                elif isinstance(e, MatrixSelector):
                    e.vs.offset_s = off
                elif isinstance(e, Subquery):
                    e.offset_s = off
                else:
                    raise ParseError("offset on a non-selector")
                continue
            if self.at("PUNCT", "@"):
                self.next()
                if self.at("IDENT", "start") or self.at("IDENT", "end"):
                    val = self.next().text.lower()
                    self.expect("PUNCT", "(")
                    self.expect("PUNCT", ")")
                else:
                    neg = ""
                    if self.at("OP", "-"):
                        self.next()
                        neg = "-"
                    val = neg + self.expect("NUM").text
                tgt = e.vs if isinstance(e, MatrixSelector) else e
                if isinstance(tgt, VectorSelector):
                    tgt.at_ = val
                continue
            break
        return e

    def parse_primary(self) -> Node:
        t = self.peek()
        if t is None:
            raise ParseError("unexpected end of expression", len(self.src))
        if t.kind == "NUM":
            self.next()
            txt = t.text.lower()
            if txt == "inf":
                return NumberLit(float("inf"))
            if txt == "nan":
                return NumberLit(float("nan"))
            if txt.startswith("0x"):
                return NumberLit(float(int(txt, 16)))
            return NumberLit(float(txt))
        if t.kind == "STR":
            self.next()
            return StringLit(t.text)
        if t.kind == "DUR":
            # a bare duration in a scalar position (e.g. `... > 5m`) is not
            # valid PromQL; surface it rather than silently coercing
            raise ParseError("bare duration in expression", t.pos)
        if t.kind == "PUNCT" and t.text == "(":
            self.next()
            e = self.parse_expr(0)
            self.expect("PUNCT", ")")
            return Paren(e)
        if t.kind == "PUNCT" and t.text == "{":
            return VectorSelector(None, list(self.parse_matchers()))
        if t.kind == "IDENT":
            name = t.text
            low = name.lower()
            if low in AGGREGATORS and self._is_aggregate_call():
                return self.parse_aggregate()
            nxt = self.peek(1)
            if nxt is not None and nxt.kind == "PUNCT" and nxt.text == "(":
                self.next()
                self.next()
                args: List[Node] = []
                while not self.at("PUNCT", ")"):
                    args.append(self.parse_expr(0))
                    if self.at("PUNCT", ","):
                        self.next()
                self.expect("PUNCT", ")")
                return Call(name, args)
            if low in SET_OPS or low in ("bool", "on", "ignoring", "by",
                                         "without", "group_left",
                                         "group_right", "offset"):
                raise ParseError(f"keyword {name!r} in operand position", t.pos)
            self.next()
            matchers: List[LabelMatcher] = []
            if self.at("PUNCT", "{"):
                matchers = list(self.parse_matchers())
            return VectorSelector(name, matchers)
        raise ParseError(f"unexpected token {t.text!r}", t.pos)

    def _is_aggregate_call(self) -> bool:
        """`sum(` or `sum by (...) (` -- but not a metric literally named sum."""
        nxt = self.peek(1)
        if nxt is None:
            return False
        if nxt.kind == "PUNCT" and nxt.text == "(":
            return True
        return nxt.kind == "IDENT" and nxt.text.lower() in ("by", "without")

    def parse_aggregate(self) -> Node:
        op = self.next().text.lower()
        grouping: Tuple[str, ...] = ()
        without = False
        if self.at("IDENT", "by") or self.at("IDENT", "without"):
            without = self.next().text.lower() == "without"
            grouping = self.parse_label_list()
        self.expect("PUNCT", "(")
        args: List[Node] = []
        while not self.at("PUNCT", ")"):
            args.append(self.parse_expr(0))
            if self.at("PUNCT", ","):
                self.next()
        self.expect("PUNCT", ")")
        if not grouping and (self.at("IDENT", "by") or
                             self.at("IDENT", "without")):
            without = self.next().text.lower() == "without"
            grouping = self.parse_label_list()
        param: Optional[Node] = None
        if op in PARAM_AGGREGATORS:
            if len(args) != 2:
                raise ParseError(f"{op} takes (param, vector)")
            param, expr = args[0], args[1]
        else:
            if len(args) != 1:
                raise ParseError(f"{op} takes one argument")
            expr = args[0]
        return Aggregate(op, expr, param, grouping, without)

    def parse_matchers(self):
        self.expect("PUNCT", "{")
        while not self.at("PUNCT", "}"):
            name = self.expect("IDENT").text
            t = self.peek()
            if t is None or t.kind != "OP" or t.text not in ("=", "!=", "=~",
                                                             "!~"):
                raise ParseError("bad label matcher",
                                 len(self.src) if t is None else t.pos)
            op = self.next().text
            val = self.expect("STR").text
            yield LabelMatcher(name, op, val)
            if self.at("PUNCT", ","):
                self.next()
        self.expect("PUNCT", "}")


def parse(src: str) -> Node:
    return Parser(src).parse()


def try_parse(src: str) -> Tuple[Optional[Node], Optional[str]]:
    try:
        return parse(src), None
    except ParseError as exc:
        return None, exc.msg
    except RecursionError:
        return None, "recursion limit"


# ------------------------------------------------------------- AST helpers

def unparen(e: Node) -> Node:
    while isinstance(e, Paren):
        e = e.expr
    return e


def walk(e: Node):
    yield e
    if isinstance(e, (Paren,)):
        yield from walk(e.expr)
    elif isinstance(e, Unary):
        yield from walk(e.expr)
    elif isinstance(e, Binary):
        yield from walk(e.lhs)
        yield from walk(e.rhs)
    elif isinstance(e, Aggregate):
        if e.param is not None:
            yield from walk(e.param)
        yield from walk(e.expr)
    elif isinstance(e, Call):
        for a in e.args:
            yield from walk(a)
    elif isinstance(e, MatrixSelector):
        yield e.vs
    elif isinstance(e, Subquery):
        yield from walk(e.expr)


def fold_constant(e: Node) -> Optional[float]:
    """Evaluate a purely numeric subtree, or None if it references data.

    `14.4 * 0.001` is a constant expression, not a literal; treating it as
    unparseable is what made multi-window burn-rate rules look like they
    were out of grammar for a reason other than the one that matters.
    """
    e = unparen(e)
    if isinstance(e, NumberLit):
        return e.value
    if isinstance(e, Unary):
        v = fold_constant(e.expr)
        if v is None:
            return None
        return -v if e.op == "-" else v
    if isinstance(e, Binary) and e.op in ARITH_OPS:
        a, b = fold_constant(e.lhs), fold_constant(e.rhs)
        if a is None or b is None:
            return None
        try:
            if e.op == "+":
                return a + b
            if e.op == "-":
                return a - b
            if e.op == "*":
                return a * b
            if e.op == "/":
                return a / b if b else None
            if e.op == "%":
                return a % b if b else None
            if e.op == "^":
                return a ** b
        except (ZeroDivisionError, OverflowError, ValueError):
            return None
    return None


def metric_names(e: Node) -> List[str]:
    out = []
    for nd in walk(e):
        if isinstance(nd, VectorSelector) and nd.metric:
            out.append(nd.metric)
    return out
