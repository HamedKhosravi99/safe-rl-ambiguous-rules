"""Source-grounded public-artifact benchmark for CORSET (offline safe RL).

Self-contained package implementing the two-family (Prometheus / Kyverno)
source-grounded benchmark of the integrated revision plan, Section 3, and the
singleton-collapse / policy-class non-dominance measurement of Section 11.

Everything is deterministic and executable: no LLM, no human judge, no
fabricated numbers.  Pipeline stages:

    fetch      -> pin & download public policy artifacts (git)
    parse      -> canonical structured targets (Prometheus alerts, Kyverno rules)
    candidates -> frozen deterministic transformation library
    evaluate   -> fire/cost vectors on a deterministic synthetic fixture universe
    score      -> frozen deterministic score s(l, psi)  (plan 3.3)
    dominance  -> policy-class dominance graph, antichain, collapse, nu_Pi (plan 11)
    run_benchmark -> split conformal + report.json + paper/generated/gen_benchmark_sg.tex

Reproduce:  SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.benchmark_sg.run_benchmark
"""
