"""Emit gen_csuite_bind.tex by joining gen_control_suite.tex (make_corset_tables.py)
with gen_bind.tex (make_corset_tables.py) on the geometry key.  Deterministic, no
hand-transcribed numbers.  Run from the repository root after make_corset_tables.py."""
import os, re
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
GEN = os.path.join(ROOT, "paper", "generated")
PAPER = os.environ.get("ARROW_PAPER_DIR", os.path.join(ROOT, "paper"))
def rows(name):
    out = {}
    for line in open(os.path.join(GEN, name)):
        line = line.strip()
        if not line or line.startswith("%"): continue
        cells = [c.strip() for c in line.rstrip("\\").split("&")]
        key = re.sub(r"\\texttt\{([^}]*)\}", r"\1", cells[0])
        out[key] = cells
    return out
cs, bd = rows("gen_control_suite.tex"), rows("gen_bind.tex")
L = ["% DERIVED by joining gen_control_suite.tex + gen_bind.tex on geometry",
     "% (deterministic; no hand-transcribed numbers)"]
for key, c in cs.items():
    b = bd[key]                       # same geometry order in both fragments
    assert b[1] == c[1], (key, b[1], c[1])
    # the geometry key L<levels>-caps<h1>-<h2> is an internal name; the paper prints the two
    # readings' duration counters h (in 60-s steps), the quantity the model of Appendix B.2 defines
    m = re.fullmatch(r"L(\d+)-caps(\d+)-(\d+)", key)
    assert m, key
    c = [f"$({m.group(2)}, {m.group(3)})$"] + c[1:]
    L.append(" & ".join(c + b[2:]) + " \\\\")
os.makedirs(os.path.join(PAPER, "generated"), exist_ok=True)
out = os.path.join(PAPER, "generated", "gen_csuite_bind.tex")
open(out, "w").write("\n".join(L) + "\n"); print("wrote", out)
