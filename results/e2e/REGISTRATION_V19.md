# REGISTRATION V19 — variance-adaptive deployment gate at larger evaluation budgets

Written 2026-09-02, after the W5 certificate audit (certificate_audit.json)
and before any outcome below exists. The audit established that the
implemented gate `E[W] <= B * CP(k, n)` is range-dominated on C-MAPSS
(B = 53.3): no seed can certify at n = 300 even with zero violations, while
the re-evaluated set-protected seed 0 has true mean W = 0.011 (22% of the
budget). The plan (W5, item 3) allows improving the evaluator only after
the audit; this is that improvement, registered before it is run.

## Objects

For every seed s in 0..49 of the set-protected FQI arm in the two domains
where the CP gate fails (synthetic maintenance, C-MAPSS replay MDP), and
for completeness the single-reading arm of the same seeds:
retrain the archived policy (deterministic; the audit reproduced two seeds
exactly), then roll out n = 20,000 FRESH evaluation episodes with a new
fixed rollout seed (20260903 + s), disjoint from the archived 300-episode
evaluation (seed 321). Per episode record W = max_psi C_psi / Z_psi with the
audit's fixed normalizers and range B.

## Gates evaluated on the same episodes, all at delta_ev = 0.05

1. CP route (the paper's): B * CP_{0.975}(k, n).
2. Union Hoeffding over the w maximal readings: mean W + B sqrt(ln(w/delta)/(2n)).
3. Empirical Bernstein (Maurer & Pontil 2009) at level 1 - delta/w:
   mean + sqrt(2 v ln(2w/delta)/n) + 7 B ln(2w/delta)/(3(n-1)).
A gate certifies when its bound is at most d = 0.05. Also reported: the
seed's true mean W (the 20,000-episode average), the archived 300-episode
verdict, and the smallest n at which each gate would certify on the seed's
own moments.

## Endpoints

- Number of seeds (of 50, per domain and arm) certified by each gate at
  n = 20,000, side by side with the archived n = 300 counts.
- Number of seeds whose true mean W is below d at all (the ceiling any
  valid gate can reach).

## Branch rules (fixed now)

- If Bernstein certifies at n = 20,000 on at least half of the C-MAPSS
  set-protected seeds whose mean W is below d, the paper says a
  variance-adaptive gate makes the deployment certificate operational on
  C-MAPSS at that evaluation budget; otherwise it says the gate remains
  uninformative at 20,000 episodes and reports the n it would need.
- The CP route's count at 20,000 is reported whatever it is; the paper does
  not switch the headline "ships" numbers to the new gate -- both are
  shown, the archived one first.
- No seed, budget, or normalizer is changed after seeing results.

Compute: local or PACE Phoenix (CPU only); results_conformal/lp/evaluator_scale.json.
