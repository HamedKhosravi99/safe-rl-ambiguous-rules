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

## Amendment 2 (2026-09-21, after the archived runs; recorded with the earlier archives kept)
Two independent code audits of the archived run found that three learner settings had been
inherited from the maintenance domain (rewards 4-40) without rescaling to this domain (rewards
0.2-1.0, 0/1 costs): the multiplier grid (0, 2, 5, ..., 320), whose smallest non-zero point
already makes every charged call unprofitable, so every Lagrangian learner returned the same
"never call" policy; the BCQ support threshold 0.05, which excluded the zero-cost local action
at rarely visited states; and the CPQ/CAPS cost-limit grids laid out for a 0-100 range. All three
were replaced by rules stated in scope_agent_learn.py next to each value (multiplier step 0.02 =
the reward increment, offset by 0.01 to avoid exact ties, then 2..320; an action admissible iff
logged at the state, exact because transitions are deterministic; limit grids geometric at ratio
1.1 from 1/(1-gamma) to 1, then 0.8, 0.5, 0.3, 0.15, 0.05, 0), shared by every learner and both arms; the offline selection rule is
unchanged. Two learners were added as our own tabular implementations, CAPS (Chemingui et al.,
AAAI 2025; per-state cost-to-go threshold in place of the accumulated-cost budget) and O3SRL
(Chemingui et al., NeurIPS 2025; EXP3 over the grid points up to 5, four rounds per arm, losses scaled to the observed range, iterate returned); PID-Lagrangian
runs with the maintenance gains divided by 100 and 96 dual steps (gain sensitivity x0.5 / x2 in
scope_agent_learn_pid_gains_x*.json). The earlier archives are kept for the sensitivity paragraph
of Appendix D.3: scope_agent_learn_coarsegrid.json (every inherited setting: the maintenance multiplier grid,
tau 0.05, the maintenance limit grids, PID gains 80/60/40 with 24 steps, nine O3SRL arms; six
learners) and scope_agent_learn_tau005.json (rule-defined multiplier grid and PID gains, but tau
0.05 and the maintenance limit grids, so its CPQ and CAPS rows are those of the coarse-grid
archive), with the matching scope_agent_check_*.json. The hypotheses are unchanged; H3's
magnitude moved (median gain 18.7 -> 15.0 points for BCQ-FQI and O3SRL, 18.7 -> 22.0 for PID,
CPQ 20.8 -> 20.7, CQL 36.4 -> 37.3); the sign of every instance's gain is the same in the main,
coarse-grid, tau-0.05 and halved-gain archives, and with doubled PID gains one instance (light
K=2, d=0.10) turns negative by 2.5 points.
