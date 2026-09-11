# Compiler audit: is the syntax-vs-array dominance disagreement an artifact or a bug?

# VERDICT: MIXED — dominated by "RAW CHECK TOO STRONG"

The threshold/duration disagreement that stopped SAFE-KEEP is ENTIRELY an
artifact of checking unreachable Cartesian states (mechanism A). Separately, the
compiler does project two reading fields away (mechanism B), which restricts
scope but does not invalidate any existing compiled experiment.

## A. State-space audit

`compile_instance` enumerates `("safe",)` plus the full CARTESIAN product
`(load, c_1..c_K)`, `c_k in {0..caps[k]}`. Impossible counter combinations are
retained; there is no reachability pruning. But all counters are driven by the
SAME next load level:

    cs2[k] = min(caps[k], cs[k]+1)  if load2 >= lvl_of[k]  else 0

so they are perfectly coupled. The induced invariant is: if
`lvl_of[a] <= lvl_of[b]` and `caps[a] <= caps[b]`, then on every reachable state
`c_a >= c_b`. Reachable fraction of the Cartesian space: MEDIAN 0.200
(min 0.074) -- about 80% of the enumerated states cannot occur.

Which of the three relations implies which:

    phi >=_sem psi   <=>  L_fire(psi) subset L_fire(phi)      (trace semantics)
    phi >=_raw psi   =>   phi >=_reach psi   =>   J-dominance
    phi >=_sem psi   =>   phi >=_reach psi
    phi >=_raw psi   is strictly stronger than needed: unreachable states carry
                     zero occupancy under EVERY policy, so they cannot affect
                     J_{c}(pi) for any pi. Raw dominance is sufficient, never
                     necessary.

`>=_reach` is the correct notion for J-dominance.

## B. The concrete ApcUpsHighTemperature 10-vs-40 case

    thetas=[10,40] fors=[120,120] caps=[2,2] lvl_of=[1,2] L=3  nS=28
    reachable states: 7 / 28
    raw pointwise violations (c_10 < c_40) at action=continue: 6
    of which REACHABLE: 0

The six violating states are (0,0,2), (0,1,2), (1,0,2), (1,1,2), (2,0,2),
(2,1,2) -- every one has `c_10 < c_40`, i.e. the threshold-40 counter ahead of
the threshold-10 counter. That is impossible: any step that increments the
40-counter has `load2 >= lvl_of[40] = 2 >= lvl_of[10] = 1`, so it increments the
10-counter too; and the 10-counter only resets when the 40-counter also resets.
Nine Cartesian states violate the invariant; zero reachable ones do.

## C. Reachable-state dominance checker

Added `reachable_states(m)` (graph reachability over transition SUPPORT under
ANY action, so no policy is privileged) and `reachable_cost_dominates(m,ka,kb)`
to `saorl/benchmark_sg/safe_keep.py`. Population result over 250 subfamilies and
1,743 discarded-candidate certificates:

| syntactic | raw | reachable | count |
|---|---|---|---:|
| True | False | **True** | **1743** |

Every single syntactically-certified pair fails the RAW check and passes the
REACHABLE check. There were no other combinations at all.

## D. Compiler-equivalence tests (exhaustive trace replay)

Reference semantics: `fire_psi(h) = 1` iff the last `cap_psi` observations were
all at level `>= lvl_psi`. Replaying every level history of horizon 7 from every
mu0-support state and comparing against the compiled cost:

| instance | checks | MISMATCHES |
|---|---:|---:|
| thr 10/40, for 120/120 | 61,236 | **0** |
| thr 0/0.05, for 300/0 | 61,236 | **0** |
| thr 1/2, for 60/180 | 61,236 | **0** |

The compiled monitor exactly implements the level-discretised reading semantics.

## E. Field coverage

| field | affects true semantics? | represented by compiler? | safe abstraction? |
|---|---|---|---|
| metric | YES | fixed per instance (subfamily) | yes, by construction |
| selectors | YES | fixed per instance | yes |
| aggregation / agg_by | YES | fixed per instance | yes |
| threshold | YES | YES (ordinal level) | yes |
| for_s | YES | YES (cap = round(for/60), clipped at DUR_CAP) | yes, with a caveat below |
| comparator | YES | **NO** | benign (see F) |
| rate_window_s | YES | **NO** | ABSTRACTION, scope limit (see F) |

CAVEAT on `for_s`: when two distinct `for_s` map to the same cap, the compiler
perturbs one cap by ordinal rank (amendment A3). Caps are therefore not a
monotone function of `for_s` alone, so syntactic duration ordering could in
principle disagree with cap ordering. It did not occur in 1,743 certificates,
but it is the one residual risk in the syntactic test and is cheap to guard by
checking cap ordering directly.

## F. Key collisions

4,766 compiled keys; 561 carry more than one distinct reading. Only two fields
ever differ within a key:

* COMPARATOR `>` vs `>=` -- 553 collisions. BENIGN. The two differ only when the
  metric is exactly at the threshold; the compiled model discretises the metric
  into ordinal levels, in which "exactly at" is not a separate state, and on a
  continuous metric the event has probability zero.
* RATE_WINDOW_S (e.g. 300s vs 60s) -- 109 collisions. INTENTIONAL ABSTRACTION,
  NOT A BUG, but a real SCOPE LIMIT: the compiled model treats the load as an
  exogenous level process, so the smoothing window that produced it is outside
  the model. Two readings differing only in rate window are indistinguishable in
  the decision problem.

No BUG/UNSOUND-COLLAPSE case was found: no collision distinguishes readings on
histories the compiled benchmark claims to represent.

## G. Existing-experiment impact

**COMPILED EXPERIMENTS OK BUT SCOPE MUST BE EXPLICIT.**

`compile_instance` de-duplicates by `(threshold, for_s)` at construction, so
`m["C"].shape[0]` is the number of DISTINCT compiled constraints, never the raw
reading count. No experiment ever treated two rate-window or comparator variants
as separate constraints, so nothing double-counted and no safety claim was
inflated. What must be stated explicitly is that the compiled candidate family
is the (threshold, duration) family, not "all candidate readings".

Specifically NOT affected: the earlier 196-instance rho = 0 result and the
1,743 certificates here, both of which live entirely inside the
(threshold, duration) family.

## H. SAFE-KEEP feasibility, redefined on semantics

Define `phi >=_sem psi` iff `L_fire(psi) subset L_fire(phi)`. For this grammar
this is decidable ANALYTICALLY (section 6's simplest exact method): with levels
and caps,

    lvl_phi <= lvl_psi  AND  cap_phi <= cap_psi   =>   phi >=_sem psi

PROOF. Suppose `fire_psi(h)=1`: the last `cap_psi` observations are all at level
`>= lvl_psi`. Since `cap_phi <= cap_psi`, the last `cap_phi` observations are a
suffix of those, hence all `>= lvl_psi >= lvl_phi`, so `fire_phi(h)=1`. QED

Then `phi >=_sem psi => c_phi(s,a) >= c_psi(s,a)` on every reachable state, hence
`J_{c_phi}(pi) >= J_{c_psi}(pi)` for every policy and environment consistent with
those monitor semantics -- no RL model needed, no full-K compilation needed.

## I. Small-set LP validation (section 12/13)

Avoiding full-K compilation entirely: reduce to the semantic maximal antichain
first (K 7-12 -> 2-3), then compile only `U u {psi}` for each discarded `psi`.

| discarded readings tested | semantic implication certs | LP-CONFIRMED | LP-REFUTED |
|---:|---:|---:|---:|
| 1,743 | 1,743 | **1,743** | **0** |

`W_psi(U,d) = max{J_c_psi(pi) : pi in Pi_U(d)} <= d` held for every discarded
reading at an informative budget (d = 0.5 * M_psi). Zero refutations.

## J. Verdict

MIXED, with the two mechanisms cleanly separated:
* Mechanism A (dominant): the syntax-vs-array disagreement is 100% an
  unreachable-state artifact. Raw pointwise dominance is too strong; reachable
  dominance is the right test and agrees with the syntactic order in 1,743/1,743.
* Mechanism B (secondary): comparator and rate_window_s are projected away.
  Benign and intentional respectively, but the scope must be stated.

The compiler is FAITHFUL to the level-discretised (threshold, duration) semantics
it targets (0 mismatches in ~184k trace checks).

## K. Consequences for the previous SAFE-KEEP report — CORRECTIONS

1. RETRACTED: "the syntactic certificate is unsound (0/4)". The 0/4 was measured
   against RAW dominance, which is the wrong relation. Against reachable
   dominance it is 1,743/1,743.
2. RETRACTED: "the cover invariant is asserted in the wrong order". The
   syntactic/semantic order is correct; the raw array order was the wrong one.
3. STANDS: the 147/150 key-collision observation, but its cause is now known
   (comparator and rate-window projection) and it is benign/scope-limiting, not
   a bug. The end-to-end audit was still weak evidence and should be rerun with
   `psi` genuinely excluded.
4. STANDS UNCHANGED: current score KEEP violates the safety-cover condition on
   98/579 = 16.9% of subfamilies.

## L. Architecture (section 15)

The separation is mathematically coherent because the two operate under
different quantifiers:
* SAFE-KEEP removes `psi` when `L_fire(psi) subset L_fire(phi)` -- true for EVERY
  environment, policy class, reward and budget. Model-free and budget-free.
* ARROW handles readings that remain semantically INCOMPARABLE but may still be
  decision-equivalent FOR THIS policy class, reward, budget and environment.
Nothing ARROW does can be replaced by SAFE-KEEP and vice versa.
