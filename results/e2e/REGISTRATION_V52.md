# REGISTRATION V52 — the checked deployment route on the learned policies of the agent-service experiment

Registered 2026-09-12, before any CHECK outcome on these policies was computed (driver:
saorl/benchmark_sg/scope_agent_check.py; policies: results_e2e/scope_agent_learn.json, V50).

## Question
The learner-aware composition theorem (Proposition S5 (composition with an approximate learner)) certifies a learned policy without
post-training evaluation only when the learner meets its contract; V51 showed that the
tabular single-signal learners do not (calibrated shortfalls 0.06-0.17), while the LP planner
does from n = 20,000.  The paper's general route for arbitrary learners is checked deployment
(Theorem 4.4(a)): the returned policy is evaluated on fresh data against every surviving
reading and deployed only if the bound clears the budget.  V52 applies that CHECK to every
policy of the main agent-service experiment.

## Procedure (the paper's CHECK, unchanged)
CHECK draws n_ev = 20,000 fresh state-action pairs from the returned policy's discounted
occupancy under the true model, reads every reading's cost at those pairs, forms the
empirical-Bernstein upper confidence bound on each reading's cost rate with Bonferroni over
the K readings at delta_ev = 0.05, and deploys iff the largest bound is at most d (the
check_ship routine used by the paper's other checked-route experiments).  Because every
c_k is 0/1 and the K readings fire on disjoint actions, the cost samples are exactly a
multinomial over {reading 1, ..., reading K, none} with the policy's true cost rates,
which the V50 archive records for every policy; CHECK is therefore simulated from those
rates, identical in law to running it on the stored policy.  One CHECK draw per policy,
seeded by (instance, learner, arm, n, seed).

## Policies
All V50 rows: the ARROW-selected sufficient reading, the union surrogate, and every wrong
single reading, for BCQ-FQI-Lagrangian, CPQ, PID-Lagrangian (n in {2,000, 20,000}) and
neural CQL-Lagrangian (n = 20,000), 10 seeds, 9 instances.

## Outcomes (per learner and arm)
Learned return before CHECK (share of V_U); share passing CHECK; unsafe policies among
those passing (the deployment failure event of Theorem 4.4(a)); deployed return conditional
on passing; return that CHECK withholds by rejecting truly safe policies.

## Hypotheses
H1 no unsafe policy is deployed (the Bernstein bound is valid; expected failures at most
delta_ev of the runs).  H2 every wrong-reading policy that is unsafe is rejected.  H3 the
ARROW arm passes on most runs and its deployed return keeps most of its gain over the
surrogate; some truly safe knife-edge policies are rejected (their cost sits at the budget).
