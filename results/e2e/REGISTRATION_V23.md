# REGISTRATION V23 — reward-free decide on the ARTEMIS retained sets

Written 2026-09-03, before any entailment is checked. ARTEMIS resolves a
translation ambiguity by asking a person about a distinguishing behaviour.
Its units carry no reward, cost or budget, so the paper's decide stage
(Theorem 1) cannot be run there; what can be run is its reward-free part.
For every unit and pool (P1 flagship, P2 union), the calibrated retained
set U of the frozen ARROW arm is reconstructed from the archived trials
and the archived per-unit q-hat (its size must equal the archived retained
size, asserted), and the exact LTL checker of V17 decides, over the union
alphabet of the unit's candidates, every ordered pair "phi entails psi"
(emptiness of phi and not psi) with the V17 pair guard, plus the
satisfiability of every retained reading.

Endpoints (fixed now). (i) w = the number of strongest retained readings:
satisfiable readings not strictly entailed by another retained reading
(the entailment antichain; protecting the w strongest protects the whole
set, for every objective). A unit with w = 1 has a reading that stands in
for the set under any reward and budget, so the clarification question is
unnecessary for safety there. (ii) Joint satisfiability of the strongest
readings: whether one controller can satisfy every retained reading at
once; when it cannot, set protection is infeasible and a question is
needed for feasibility, not only for return. (iii) Whether the strongest
reading is expert-plausible. Reported per pool, by group and by
|P_l| bin, with undecided pairs (guard exhaustion) and unsatisfiable
retained readings counted, never silently dropped. Readings that omit an
atom are weaker over the union alphabet; this is stated, not corrected.

Branch rule: reported wherever it lands. If w = 1 on at least a quarter of
the units of the paper's arm (P2), the paper says a deployer can skip the
clarification question for safety on that share of real requirements and
gives the share; if it is rarer, the paper says the question is necessary
on most requirements and that ARROW's reward-free contribution there is w,
the number of readings a set-protecting controller must satisfy at once.
If joint satisfiability fails on a material share, the paper reports that
set protection is infeasible there.

Output: results/e2e/artemis_decide.json; macros via make_gen_artemis.py.

Addendum (2026-09-03, during the run): the single sequential process was
stopped at unit 82 of P2 because a few units spend the full pair guard on
many pairs; P2 (and later P1) are re-run from scratch as five parallel
chunks (units by index modulo five) with identical guards, scores and
thresholds, and merged in registered order. Nothing about the endpoints,
guards or branch rule changed; only the progress lines of the stopped
process had been seen.

Second addendum (2026-09-03): P1 (Keep = the whole pool, up to 50 classes
per unit) was started in five chunks and had not completed when the
revision was committed; units of 40-48 classes spend the full pair guard
on dozens of pairs (one unit took 2,978 s). P1 is therefore not reported.
P2, the paper's arm, is complete. If P1 finishes, its rows are added to the
archive as a follow-up without changing any P2 number.
