# REGISTRATION V50 — scope-ambiguous call-limit rule: does ARROW change learned outcomes?

Registered 2026-09-11 before the learned runs (driver: saorl/benchmark_sg/scope_agent.py, scope_agent_learn.py).

## Why this domain
The exact surrogate price is lambda (d - x_C) per competing reading: it is large only when
the readings' firing regions are disjoint, each competing region is rare enough to stay
within budget on its own (so a singleton is certifiable), and value is earned in those
regions.  Strictness ambiguity (thresholds, durations, caps) overlaps or nests and caps the
price at a few percent (monitoring 2-7%; Kyverno firing-set Jaccard 0.75).  Scope ambiguity
("which service is 'the external service'?") gives disjoint regions by construction.

## Environment (fixed before any learned run)
Tool-using agent; sub-tasks resolved by one call each to K services; resolved sub-task = 1,
wasted call = 0, local work = 0.2; gamma = 0.97 (session ends w.p. 0.03 per step).  Demand
per session: primary in {6,8,10}; auxiliaries s_k or s_k+1 with s = light (1,1,1), medium
(2,2,1), heavy (3,3,2).  Reading psi_k charges calls to service k.  K in {2,3,4}, d in
{0.05,0.10,0.15,0.20}.  Exact analysis (scope_agent.json) done before this registration;
certified instances: light d>=0.10, medium d>=0.15, heavy d=0.20, all K (18 instances).

## Arms (same log, seed, budget, learner)
single:<service> for each of the K readings (commit to one interpretation; the certified one
is ARROW's singleton, the others are the practitioner's wrong guesses); surrogate (union /
pointwise-max, the only full-set representation a single-signal learner accepts).

## Learners
The paper's single-signal learners (BCQ-FQI-Lagrangian, CPQ, PID-Lagrangian, neural CQL
at n=20,000), mechanisms unchanged; two adapter generalisations injected without editing
the learner files: the action set is the domain's (K+1 actions), and the per-step cost is
1{call to a charged service} in place of the maintenance-domain 1{fires and a=continue}
(identical semantics there).  Feasibility and return_fn: on-policy empirical-model estimates
from the same log (as V48/V49).  Logs: 0.7 x search-optimal + 0.3 x uniform behaviour, n in
{2,000, 20,000}, 10 seeds.

## Outcomes and analysis (exact, true model)
Return / V_U, cost under every reading, safe = max cost <= d; per instance: unsafe rate of
each wrong single reading, safe rate of surrogate and certified singleton, paired
(singleton - surrogate) return difference, sign test across instances.

## Hypotheses
H1 wrong single readings: unsafe on most runs.  H2 surrogate: safe, return well below V_U.
H3 certified singleton: as safe as the surrogate, return higher by a large fraction of the
exact price (9-32% of V_U across the certified instances).

## Amendment (before the learned runs; the first smoke test showed why)
Two environment changes, both made before any learned result was recorded: (i) primary
sub-tasks differ mildly in importance (with j remaining, the next is worth 0.8 + 0.02 j,
i.e. 0.82-1.0; auxiliaries stay at 1.0), because with identical values the constrained
optimum is a randomised policy that the paper's deterministic greedy learners cannot
express (they returned all-or-nothing policies); (ii) d in {0.05, 0.075, 0.10, 0.15},
because at d = 0.20 the primary limit no longer binds (V_U = V_unc), so no reading could
be unsafe.  Exact surface recomputed (scope_agent.json): certified instances are light
K in {2,3,4} at d in {0.10, 0.15} and medium K in {2,3,4} at d = 0.15 (9 instances, exact
surrogate price 8.7-28.9%); heavy auxiliary demand is never certified at these budgets.
