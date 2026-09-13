# REGISTRATION V51 — does the ARROW-certified singleton make a NATIVE multi-constraint learner learn better?

Registered 2026-09-12, before any learned run of this design (driver:
saorl/benchmark_sg/scope_agent_native.py; domain and exact surface unchanged from V50).

## Question
V50 compared the certified singleton with the pointwise-max (union) surrogate, the only
representation a single-signal learner accepts.  A reviewer notes that a learner which
enforces the K constraints separately has, by Theorem 4.1, the same exact optimum under the
full set and under a sufficient singleton, so V50's gain is removal of representation-induced
conservatism.  V51 asks the sharper question: for a learner that CAN enforce all K readings
separately, does replacing the K constraints by the one certified reading make finite-sample
learning easier?  Theorem 4.1 fixes the exact target of both arms to be identical, so any
difference is a learning-efficiency effect, not an objective change.

## Arms (same log, seed, learner code, iteration budget, hyperparameters)
* native: K separate constraints J_{c_k} <= d, one multiplier per reading;
* arrow: the single certified reading, the SAME code with one constraint;
* surrogate (n = 20,000 only, continuity with V50): one multiplier on the pointwise maximum.

## Learners (native multi-constraint versions of the paper's tabular learners)
Both are BCQ-filtered fitted-Q iteration (bcq_tau = 0.05, 80 FQI sweeps, as V50) on the
shaped reward r - sum_k lambda_k c_k with a VECTOR multiplier updated per constraint from the
on-policy P-hat cost estimate of the same log, 24 dual steps:
* vector Lagrangian FQI: integral-only dual ascent, lambda_k <- max(0, lambda_k + 60 e_k);
* vector PID-Lagrangian FQI: PID per constraint with the paper's gains (kp 80, ki 60, kd 40).
Selection rule as in the paper: the offline-feasible iterate (every honoured cost <= d on
P-hat) with the best offline return; if none is feasible, the iterate with the smallest
maximal honoured cost.  The singleton arm runs the identical code with K = 1.

## Learner-aware tolerance (Theorem C.9 / eq. contract)
Calibration seeds 100-104 (disjoint from evaluation seeds 0-9), singleton arm, each learner
L and log size n, on the 9 instances certified at eps = 0.01: eps_r,L(n) = the largest
true-model return shortfall V_psi(d) - J_r(pi_hat) over calibration runs, eta_L(n) = the
largest true-model exceedance (J_{c_psi}(pi_hat) - d)_+; both frozen before evaluation.  ARROW
is re-run exactly at eps'_L(n) = eps_r,L(n) + V_psi(d) - V_psi(d - eta_L(n)) on all 36
instances; evaluation uses only the instances that still certify at eps'_L(n).

## Protocol
Log sizes n in {2,000, 5,000, 20,000}; behaviour 0.7 x psi-optimal + 0.3 x uniform (V50);
evaluation seeds 0-9; every returned policy scored exactly on the true model.

## Outcomes
Exact sanity check: |V_psi(d) - V_U(d)| on every certified instance (must be ~0, Theorem 4.1).
Per run: return / V_U, cost under every reading, safe (all readings <= d), dual step of the
first offline-feasible iterate, number of feasible iterates, final multipliers.  Paired
(arrow - native) return difference per (instance, n, seed); medians by n and by K; sign test
over instances at each n; safety rates of both arms.

## Hypotheses (either direction will be reported)
H1 exact: the two exact optima tie on every certified instance.
H2 finite data: the singleton attains higher learned return than the native full set at small
n and the gap shrinks as n grows.  H3 safety: both arms are safe at comparable rates.
H4 the singleton reaches an offline-feasible iterate in fewer dual steps.
A null or reversed result is reported as such and the main-text claim stays confined to
single-signal learners.

## Amendment (2026-09-12, after the smoke test, before any evaluation run)
The smoke test (one calibration seed, n = 2,000, vector Lagrangian FQI) gave a worst-case
shortfall eps_r = 0.115 (23-31% of V_U on eight of the nine instances) and no instance
certifies at that tolerance.  Recorded before evaluation:
1. Two native multi-constraint learners are added, both closer to optimal than the
   Lagrangian pair: vector CPQ (one tabular cost critic per honoured reading; an in-support
   action is admissible iff every critic's cost-to-go is at most the limit d_lim, the paper's
   loose-to-tight d_lim grid and selection rule; singleton arm = one critic) and the
   occupancy LP on the estimated model P-hat restricted to behaviour-supported pairs (the
   exact planner of the certified-offline route, with K constraints or one).
2. Branch rule: if eps'_L(n) certifies no instance, the learning-efficiency comparison for
   that (L, n) is run on the nine instances certified at eps = 0.01 and reported as such,
   next to the certified count (zero) and the shortfall that caused it; where eps'_L(n)
   certifies a nonempty set, that set is used.
3. Log sizes n in {2,000, 5,000, 20,000, 50,000} (50,000 added to show convergence).

## Exploratory extension (2026-09-12, after the registered runs; labelled exploratory)
Prompted by the LP arms differing at n = 2,000, the same protocol was re-run at n in
{250, 500, 1,000} (results_e2e/scope_agent_native_smalln.json).  Outcome: per-instance median
return gaps are mixed (vector PID -2.3 to -1.8 points, vector CPQ -0.3 to +3.1, LP +2.5 to
+5.3, Lagrangian FQI 0); the singleton arm is LESS safe than the K-constraint arm
for the LP (54-64% vs 77-81% safe runs), PID (87-89% vs 97-100%) and CPQ (90-94% vs 98-100%)
learners, because with an inaccurate estimated model the slack constraints act as a
regularizer.  The "ARROW makes native learning easier" hypothesis is therefore not supported
in any data regime of this domain; the supported statement is the registered one: the
singleton is lossless for native learners once the learner is reasonably accurate, and it
helps the cost-critic mechanism specifically.
