# REGISTRATION V53 — decision-specific offline certification on the compiled monitoring rules

Registered 2026-09-12, before the confirmatory run (driver: paper/theory_extension/certificate_v53.py;
results: results/theory_extension/certificate_v53.json; generator: scripts/paper/make_gen_v53.py).

## Provenance
Exploratory runs on 2026-09-12 (session scratchpad: robust_dual_cert.py, occ_cert.py, split_cert.py) compared
several offline certificates on the 28 compiled monitoring rules and motivated this study.  Their numbers are
not reused: the confirmatory run uses fresh seeds, the frozen definitions below, and a fixed reporting plan.

## Question
Table 12 of the paper reports the archived offline certificate (uniform simulation-lemma slack): on the
27 sufficient reading instances at d = 0.05, eps = 0.01, delta = 0.05 it certifies 0/27 at n = 1e7,
22/27 at 3.2e7 and 27/27 only at 1e10 pooled load transitions.  V53 asks (i) how much of that data
requirement is certificate conservatism and how much is intrinsic to the decision, and (ii) what
ARROW can certify from a fixed log when the exact-budget decision is unresolved.

## Setting (as archived, unchanged)
The 28 compiled monitoring rules of results/e2e/control_suite_uncapped.json, gamma = 0.97, two readings
per rule.  The only stochastic component is the load chain M (R = 3 rows, m_z = 3 successors per row);
every monitor counter updates deterministically.  Telemetry of n pooled transitions gives per-row counts
N_z = round(n pi_z), pi the stationary law of M; the empirical chain Mhat is recompiled into every rule.
Truth: the exact verdict of each (rule, reading) at eps = 0.01 from paper/theory_extension/real_rules_exact.json
(d = 0.05: 27 sufficient / 29 insufficient reading instances; d = 0.02 and d = 0.10 as archived).

## Notation for the confidence event
R = 3 uncertain rows; m_z = 3 successors of row z; N_z the count of row z.  The confidence level is
allocated once across rows: alpha_z = sqrt( 2 ( m_z ln 2 + ln(R/delta) ) / N_z ) (Weissman et al. 2003),
and E = { ||M_z - Mhat_z||_1 <= alpha_z for all z } has probability at least 1 - delta.  Every certificate
below is a deterministic function of (Mhat, N, delta); soundness is on E, so no further union bound over
candidates, competitors or witnesses is needed.

## Certificates (frozen)
1. UNIFORM (archived, comparison only): V^- = V^{Mhat}_psi(d - beta_c) - beta_r; empirical face with
   budget d + beta_c and threshold V^- - beta_r - eps; What = max competing cost + beta_c;
   beta = gamma/(2(1-gamma)) * max_z alpha_z * span(f), span_r = 0.7, span_c = 1.  Certify iff What <= d.
2. OCC (ablation): row-specific, occupancy-weighted penalty B_f(x) = gamma span(f)/(2(1-gamma)) sum_{s,a} x(s,a) b(s,a),
   b(s,a) = alpha_z on stochastic rows (load z, action continue), 0 on deterministic rows;
   V_low = max_x r.x - B_r(x) s.t. c_psi.x + B_c(x) <= d;  Ghat = { c_psi.x - B_c(x) <= d, r.x + B_r(x) >= V_low - eps };
   certify iff max_phi max_{x in Ghat} c_phi.x + B_c(x) <= d.
3. DUAL (the certificate under study, "robust-dual ARROW"): V_low as in OCC if that program is feasible, else 0
   (valid since r >= 0); for each competitor phi solve the robust dual
      min (1-gamma) mu0.h + mu d - nu (V_low - eps)   over h, mu >= 0, nu >= 0, t_s >= 0
      s.t. c_phi(s,a) <= mu c_psi(s,a) - nu r(s,a) + h(s) - gamma [ Mhat_z . h_s + (alpha_z/2) t_s ]   (stochastic rows, h_s = h at the successors)
           c_phi(s,a) <= mu c_psi(s,a) - nu r(s,a) + h(s) - gamma h(s')                               (deterministic rows)
           t_s >= h(s') - h(s'') for all successor pairs of s;
   certify iff every competitor's optimum is <= d.
   Tightened variant DUAL(eta): the anchor budget d is replaced by d - eta in V_low, in Ghat's cost constraint and
   in the dual objective; the competitor bound stays d, so it certifies F^eps_psi(d - eta) subset of Pi_C(d).

## Design
- Budgets d in {0.02, 0.05, 0.10}; eps = 0.01; delta = 0.05.
- Grid n in {1e3, 3.16e3, 1e4, 3.16e4, 1e5, 3.16e5, 1e6, 3.16e6, 1e7, 3.16e7, 1e8}.
- Primary draws: 10 independent chain draws per n (multinomial counts per row), identical draws for all
  certificates and readings (draw id = (n, rep)).
- Instance structure (disclosed): at d = 0.05 the 27 sufficient readings form two classes by exact margin
  and by the oracle quantity I*: 22 readings with kappa = 0.0337 and 5 readings with kappa = 0.0028 that share
  one compiled stochastic geometry.  Five readings evaluated on one draw are not five independent draws.
  Therefore 50 additional independent draws are taken for the hard class at n in {1e5, 3.16e5, 1e6} (d = 0.05),
  and certification rates are reported by class.
- Tightening sweep: for every reading not certified by DUAL at (n, d = 0.05), the smallest eta in
  {0.001, 0.002, 0.005, 0.01, 0.02} for which DUAL(eta) certifies, and the exact return price V_psi(d) - V_psi(d - eta)
  computed on the true model.
- Oracle diagnostic (true chain, reported as oracle): for each sufficient reading class the decision-information
  radius I* = inf { sum_z pi_z KL(M_z || Q_z) : Q in closure of B_psi }, B_psi = { Q : Gamma^eps_psi(Q) > 0 } the
  chains under which the reading is insufficient (Gamma = 0 is sufficient; the closure is used for optimization
  and coincides with the infimum by continuity of Gamma in Q under the LP regularity condition), by directional
  bisection and SLSQP refinement; the lower bound n_info = kl(1 - delta, delta) / I* for any procedure that is
  delta-sound on B_psi and certifies at M with probability at least 1 - delta.
- Secondary (robustness of the row-count idealization): counts obtained from one sampled path of the load chain of
  length n started from pi, n in {1e4, 1e5, 1e6, 1e7}, 5 paths per n, d = 0.05, UNIFORM and DUAL only.

## Outcomes
Per (certificate, d, n): sufficient readings certified (by class and total), false certificates among insufficient
readings, mean objective slack; per (reading class, d): n* = smallest grid n with certification in at least 90% of
draws (60 draws for the hard class where available); tightening table; I* and n_info per class; secondary table.

## Hypotheses (reported as outcomes; no post-hoc tuning)
H1 (soundness) no false certificate for any certificate at any n; any false certificate is reported prominently and
   investigated before any result is used.
H2 (conservatism) DUAL certifies the easy class at n <= 1e4 and all 27 readings by n <= 1e6, versus 3.2e7 and 1e10
   for UNIFORM; OCC lies between.
H3 (intrinsic difficulty) the hard class has n_info of order 1e4-1e5 and DUAL certifies it at 3e5-1e6; the easy class
   has n_info of order 1e2-1e3.
H4 (tightening) at n = 1e5 the hard class certifies at eta <= 0.01 with an exact return price below 6%, and at
   n = 3.16e5 at eta <= 0.002 with a price near 1%.
H5 (idealization) the secondary path-sampled counts reproduce the qualitative ordering and scales of the primary run.
Method ordering and monotonicity in n are outcomes, not assertions.  The generator asserts only structural facts:
valid shapes, identical draw ids across certificates, counts within range, and the exact verdicts matching the archive.

- Note (2026-09-18, proof audit). The quantity reported as I* above is the weighted relative entropy of a decision-reversing
  chain found by the search (directional bisection, then SLSQP refinement), and the search does not certify a global
  infimum. That chain lies in B_psi, so the reported value is an upper bound Ibar >= I* and the floor kl(1 - delta, delta) / Ibar
  is at most the bound of Theorem 4.2(b); it remains a valid lower bound on the transitions used by any delta-sound procedure
  that certifies the reading under the true chain with probability at least 1 - delta. The closure-by-continuity sentence
  above is not relied upon. The search now targets a witness margin Gamma >= 1e-6 (it was 1e-7), and verify_witnesses in
  certificate_v53.py re-solves both linear programs at every reported witness with HiGHS dual simplex and interior point at
  feasibility tolerance 1e-10, requiring Gamma >= 1e-6 - 1e-9; the six stored witnesses have recomputed margin 1.000e-6 under
  both, with equality residuals below 3e-17. The refresh (certificate_v53.py --istar-only) changed the six floors by at most
  0.07% (hard class 36,757 -> 36,734); every other archived number is untouched. Paper wording: Table 14, Figure 2 and
  Appendix D now say witness divergence Ibar and witness-based information floor.
