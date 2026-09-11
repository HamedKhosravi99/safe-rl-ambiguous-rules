# REGISTRATION V21 — recalibrating Keep on the faithful population

Written 2026-09-02, before the recalibration is run. Revision item W6-3
asks that the faithful-gold restatement not merely filter old calibration
results when the target population changed. The frozen v11 calibration
set q-hat = 0.41 from every in-grammar calibration unit, faithful or not.
This pass recomputes the calibration threshold on the FAITHFUL in-grammar
calibration units only (the audit's strict bucket: no capability beyond
the six-tuple), with the frozen v11 licenser, weights, mode ("complete")
and delta_sem = 0.10 unchanged, and evaluates it on the faithful test
units of the frozen test pass (gold scores already archived).

Endpoints: q-hat_F; the number of faithful in-grammar calibration units;
retention of the faithful test golds under q-hat_F versus under the
frozen q-hat; the faithful end-to-end rate under each; and the recorded
q-hat recomputed from the same calibration rows, which must equal 0.41
(asserted). No test unit is re-licensed; nothing is tuned.

Branch rule: reported wherever it lands; if q-hat_F differs from 0.41 the
paper states both thresholds and the retention under each.

Output: results_e2e/faithful_recalibration.json.
