# Alert-rule interpretation study (annotator instructions)

You are interpreting standard kube-prometheus alert rules from your own
operational experience. Please do not look up any reference
implementation, default values, or this project's paper.

Each entry in sheet.txt gives an alert's name, summary and description,
which is the same information the automated interpreters received. For
each one, write 3 alerting configurations you would consider genuinely
defensible -- each a (threshold, duration_seconds) pair a competent SRE
could argue for in a design review. They should reflect real
disagreement about what the alert means, not small jitter around one
answer. Do not systematically order them (A strictest, C loosest); judge
each on its own merits. If a stricter threshold justifies a longer
confirmation window (or the reverse), let the numbers show it.

Threshold conventions: absolute count for count-like alerts; a ratio
(e.g. 0.9) for quota/capacity alerts; a rate or fraction for error-rate
alerts -- keep each rule internally consistent.

Fill each numbered line in place, leaving the '#' comment lines alone:

  <n>|<thrA>,<durA>|<thrB>,<durB>|<thrC>,<durC>

Example (format only): 7|0.05,300|0.01,900|0.1,120

Expected time: about 45 minutes. Thank you.
