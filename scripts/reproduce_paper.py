#!/usr/bin/env python3
"""Regenerate every table/figure fragment used by the paper and diff against paper/reference/.

Usage (from the repository root):   python3 scripts/reproduce_paper.py [--only NAME ...]

Each generator in scripts/paper/ reads archived experiment outputs under results/ and
writes LaTeX macro/table fragments to paper/generated/ and figures to paper/figure/
(override the destination with ARROW_PAPER_DIR).  Afterwards every file in
paper/reference/ is compared byte-for-byte with the freshly produced copy.  PDF figures
embed a creation timestamp, so they are reported as regenerated rather than compared.
"""
import os, sys, subprocess, hashlib, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.environ.get("ARROW_PAPER_DIR", os.path.join(ROOT, "paper"))
env = dict(os.environ, ARROW_PAPER_DIR=PAPER, MPLBACKEND=os.environ.get("MPLBACKEND", "Agg"),
           OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "1"),
           PYTHONPATH=os.path.join(ROOT, "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
for d in ("generated", "figure"): os.makedirs(os.path.join(PAPER, d), exist_ok=True)

# dependency order: fragments read by a later generator are produced by an earlier one
STEPS = ["make_corset_tables", "make_gen_csuite_bind", "make_gen_revision", "make_gen_artemis", "make_gen_applic",
         "make_gen_finite", "make_gen_extra", "make_gen_v47_51_tables", "make_gen_v47_51", "make_gen_r6",
         "make_gen_safekeep", "make_gen_v13", "make_gen_v14", "make_gen_v43", "make_gen_artemis_methods",
         "make_gen_artemis_decide", "make_gen_sota", "make_gen_live", "make_gen_v50", "make_gen_v51", "make_gen_v52",
         "make_gen_v53", "make_unified_numbers", "make_figs_mpl", "make_figs_appendix", "make_fig_v53"]
only = set(sys.argv[sys.argv.index("--only") + 1:]) if "--only" in sys.argv else set()
failed = []
for name in STEPS:
    if only and name not in only: continue
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "paper", name + ".py")], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    print(f"[{'ok' if r.returncode == 0 else 'FAIL'}] scripts/paper/{name}.py")
    if r.returncode != 0: failed.append(name); print(r.stderr[-1500:])

def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()
same, diff, missing, pdfs = [], [], [], []
for ref in sorted(glob.glob(os.path.join(ROOT, "paper", "reference", "generated", "*.tex")) +
                  glob.glob(os.path.join(ROOT, "paper", "reference", "figure", "*"))):
    sub = "figure" if "/figure/" in ref else "generated"
    got = os.path.join(PAPER, sub, os.path.basename(ref))
    if not os.path.exists(got): missing.append(os.path.basename(ref)); continue
    if ref.endswith(".pdf"): pdfs.append(os.path.basename(ref)); continue
    (same if md5(got) == md5(ref) else diff).append(os.path.basename(ref))
print(f"\nreference fragments identical: {len(same)}   differing: {len(diff)}   not produced: {len(missing)}   pdf figures regenerated: {len(pdfs)}")
if diff: print("DIFFERING:", diff)
if missing: print("NOT PRODUCED:", missing)
if failed: print("FAILED:", failed); sys.exit(1)
