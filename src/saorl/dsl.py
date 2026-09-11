"""Finite DSL for SA-ORL semantic candidates (plan v12, sections 2.3-2.4).

A candidate semantics is psi = (g, B): a predicate g over the (augmented) state
and a forbidden-action set B. The induced semantic cost is

    c_psi(s_t, a_t) = 1{ g(s_t) = 1 and a_t in B }.

Predicates are evaluated against (trajectory, t) so that temporal operators
(Persist_m, Within_W, Since, ...) have access to history. The maintenance grammar
(atoms + not/and/or + Persist_m) and the Domain-2 temporal operators (Within_W
past-window and Since(trigger, reset) automaton interval) are all implemented
here; Domain 2 uses Within_W to express "do not advance for the first W steps
after a warning" as a genuinely temporal, window-length ambiguity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, FrozenSet, List, Sequence, Tuple

State = dict  # feature name -> float / int / bool
Trajectory = Sequence[State]


class Predicate:
    """Boolean event predicate g evaluated at index t of a trajectory."""

    def holds(self, traj: Trajectory, t: int) -> bool:  # pragma: no cover
        raise NotImplementedError

    def n_atomic(self) -> int:  # pragma: no cover
        raise NotImplementedError

    def n_temporal(self) -> int:  # pragma: no cover
        raise NotImplementedError


@dataclass(frozen=True)
class Atom(Predicate):
    """Atomic predicate: feature `op` theta, e.g. RUL_hat <= 20 or anom >= 0.8."""
    feature: str
    op: str  # 'le' or 'ge'
    theta: float

    def holds(self, traj, t):
        v = traj[t][self.feature]
        return v <= self.theta if self.op == "le" else v >= self.theta

    def n_atomic(self):
        return 1

    def n_temporal(self):
        return 0

    def __repr__(self):
        sym = "<=" if self.op == "le" else ">="
        return f"({self.feature} {sym} {self.theta})"


@dataclass(frozen=True)
class RegionAtom(Predicate):
    """Generic atom backed by a callable, for s in R or dist(s, H) <= r."""
    name: str
    fn: Callable[[State], bool]

    def holds(self, traj, t):
        return bool(self.fn(traj[t]))

    def n_atomic(self):
        return 1

    def n_temporal(self):
        return 0

    def __repr__(self):
        return f"({self.name})"


@dataclass(frozen=True)
class Not(Predicate):
    g: Predicate

    def holds(self, traj, t):
        return not self.g.holds(traj, t)

    def n_atomic(self):
        return self.g.n_atomic()

    def n_temporal(self):
        return self.g.n_temporal()

    def __repr__(self):
        return f"~{self.g!r}"


@dataclass(frozen=True)
class And(Predicate):
    g1: Predicate
    g2: Predicate

    def holds(self, traj, t):
        return self.g1.holds(traj, t) and self.g2.holds(traj, t)

    def n_atomic(self):
        return self.g1.n_atomic() + self.g2.n_atomic()

    def n_temporal(self):
        return self.g1.n_temporal() + self.g2.n_temporal()

    def __repr__(self):
        return f"({self.g1!r} & {self.g2!r})"


@dataclass(frozen=True)
class Or(Predicate):
    g1: Predicate
    g2: Predicate

    def holds(self, traj, t):
        return self.g1.holds(traj, t) or self.g2.holds(traj, t)

    def n_atomic(self):
        return self.g1.n_atomic() + self.g2.n_atomic()

    def n_temporal(self):
        return self.g1.n_temporal() + self.g2.n_temporal()

    def __repr__(self):
        return f"({self.g1!r} | {self.g2!r})"


@dataclass(frozen=True)
class Persist(Predicate):
    """Persist_m(g): g holds at every index in the window [t-m+1, t]."""
    m: int
    g: Predicate

    def holds(self, traj, t):
        if t - self.m + 1 < 0:
            return False
        return all(self.g.holds(traj, j) for j in range(t - self.m + 1, t + 1))

    def n_atomic(self):
        return self.g.n_atomic()

    def n_temporal(self):
        return self.g.n_temporal() + 1

    def __repr__(self):
        return f"Persist_{self.m}{self.g!r}"


@dataclass(frozen=True)
class Within(Predicate):
    """Within_W(g): g held at some index in the past window [t-W+1, t].

    A past-time "once within W steps" operator. Used for temporal readings such as
    "for W steps after a warning, do not advance"."""
    W: int
    g: Predicate

    def holds(self, traj, t):
        lo = max(0, t - self.W + 1)
        return any(self.g.holds(traj, j) for j in range(lo, t + 1))

    def n_atomic(self):
        return self.g.n_atomic()

    def n_temporal(self):
        return self.g.n_temporal() + 1

    def __repr__(self):
        return f"Within_{self.W}{self.g!r}"


@dataclass(frozen=True)
class Since(Predicate):
    """Since(trigger, reset): a trigger has fired and not yet been reset.

    Fires at t iff the most recent index <= t at which `trigger` holds is at or
    after the most recent index at which `reset` holds (and a trigger occurred at
    all). This is the temporal-automaton "an uncleared alert is active" operator:
    with trigger = warning edge and reset = all-clear edge, it holds exactly on
    the open interval [warning, all-clear). The reset step itself is NOT active."""
    trigger: Predicate
    reset: Predicate

    def holds(self, traj, t):
        last_trig = None
        last_reset = None
        for j in range(t + 1):
            if self.trigger.holds(traj, j):
                last_trig = j
            if self.reset.holds(traj, j):
                last_reset = j
        if last_trig is None:
            return False
        return last_reset is None or last_trig > last_reset

    def n_atomic(self):
        return self.trigger.n_atomic() + self.reset.n_atomic()

    def n_temporal(self):
        return self.trigger.n_temporal() + self.reset.n_temporal() + 1

    def __repr__(self):
        return f"Since({self.trigger!r}<-{self.reset!r})"


@dataclass(frozen=True)
class Candidate:
    """psi = (g, B) plus a language-plausibility ensemble for the rule.

    `plausibility` holds the per-member plausibility scores in [0,1] from a
    frozen ensemble of language judgments (an LLM ensemble run offline and
    cached, or the no-LLM DataDrivenJudge). It replaces the original
    3-annotator audit: plausibility is now a computational, reproducible signal
    measured at ensemble scale, not a survey of recruited humans. Empty means
    "no language signal" -- the construction then treats the plausibility term
    as neutral rather than penalizing the candidate.
    """
    name: str
    predicate: Predicate
    forbidden_actions: FrozenSet[str]
    plausibility: Tuple[float, ...] = field(default=())

    def fires(self, traj: Trajectory, t: int) -> bool:
        return self.predicate.holds(traj, t)

    def cost(self, traj: Trajectory, t: int, action: str) -> int:
        return int(self.predicate.holds(traj, t) and action in self.forbidden_actions)

    def complexity(self) -> int:
        return self.predicate.n_atomic() + self.predicate.n_temporal()

    def plaus_mean(self) -> float:
        """Mean ensemble plausibility; 0.0 if no ensemble is attached."""
        if not self.plausibility:
            return 0.0
        return sum(self.plausibility) / len(self.plausibility)

    def has_plausibility(self) -> bool:
        return len(self.plausibility) > 0
