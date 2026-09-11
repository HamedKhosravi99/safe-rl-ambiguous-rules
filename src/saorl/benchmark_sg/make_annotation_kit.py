"""Regenerate the human annotation kit so it matches the registered protocol.

REGISTRATION_V10 E1 says the annotators see "the same 30 kube-prometheus
rules the LLM interpreters saw, under the same full-documentation
condition". The kit as committed violated both halves: it listed a
DIFFERENT set of 30 alerts, and it showed only the alert NAME while
instructing annotators that the name alone was the point. The LLM
interpreters in saorl/e1b_elicit.py were shown name, summary and
description. A study run from the old kit could not have been compared
to the model rate, which is E1-a's primary endpoint.

This regenerates both files from e1b_elicit.harvest_rules(), the exact
source the interpreters used, so the two arms differ only in who is
answering.

Run:  PYTHONPATH=. python3 -m saorl.benchmark_sg.make_annotation_kit
"""
from __future__ import annotations
from pathlib import Path
from saorl.e1b_elicit import harvest_rules

KIT = Path(__file__).resolve().parent / "annotation_kit"

INSTRUCTIONS = """# Alert-rule interpretation study (annotator instructions)

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
"""


def main() -> None:
    rules = harvest_rules()
    assert len(rules) == 30, len(rules)
    out = ["# Fill the numbered lines. Lines beginning with '#' are ignored.",
           ""]
    for i, r in enumerate(rules, start=1):
        out += [f"# {i}. {r['name']}",
                f"#    summary: {r['summary']}",
                f"#    description: {r['description']}",
                f"{i}| , | , | , ",
                ""]
    (KIT / "sheet.txt").write_text("\n".join(out))
    (KIT / "INSTRUCTIONS.md").write_text(INSTRUCTIONS)
    print(f"regenerated kit for {len(rules)} rules with full documentation")
    print("  ", KIT / "sheet.txt")
    print("  ", KIT / "INSTRUCTIONS.md")


if __name__ == "__main__":
    main()
