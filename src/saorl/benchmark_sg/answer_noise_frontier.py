"""V49b read-out: per-cell tables under both truth priors, the posterior
assumed-rate sensitivity grid, and the Pareto frontier over (mean answers,
fallback rate, unsafe rate). Run after answer_noise_check.py.
Run: PYTHONPATH=. python3 -m saorl.benchmark_sg.answer_noise_frontier
Writes results/e2e/answer_noise_frontier.json
"""
import json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
R = os.path.join(ROOT, "results/e2e")


def fmt(x, nd=3):
    return "--" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main():
    v = json.load(open(os.path.join(R, "answer_noise_check.json")))
    out = {}
    for name, pop in v["populations"].items():
        out[name] = {}
        print(f"\n==== {name}: {pop['n_pools']} pools, {pop['n_unsafe_at_0']} not safe at q=0 (rates over pools not safe at q=0)")
        for prior in ("gold", "uniform"):
            S = pop["summary"][prior]
            print(f"--- prior = {prior}")
            print(f"{'cell':36s} {'elim':>6s} {'hand':>6s} {'fallb':>6s} {'uns_h':>7s} {'uns_fb':>7s} {'uns_any':>7s} {'ans_mean':>8s} {'ans_med':>7s} {'ret/VU':>7s} {'ret/Vt':>7s}")
            keys = [k for k in S if not k.startswith("flip") or "|bayes|" not in k] + [k for k in S if "|bayes|" in k and ("pa0.1|dp0.05" in k or "pa0.02|dp0.05" in k or "pa0.3|dp0.05" in k or "pa0.1|dp0.2" in k)]
            seen = set()
            for k in keys:
                if k in seen:
                    continue
                seen.add(k); s = S[k]["unsafe_at_0"]
                print(f"{k:36s} {fmt(s['truth_eliminated']):>6s} {fmt(s['handover']):>6s} {fmt(s['fallback']):>6s} {fmt(s['unsafe_handover'],4):>7s} {fmt(s['unsafe_fallback'],4):>7s} {fmt(s['unsafe_any'],4):>7s} {fmt(s['answers_mean'],2):>8s} {fmt(s['answers_median'],1):>7s} {fmt(s['ret_over_VU_any']):>7s} {fmt(s['ret_over_Vtruth_handover']):>7s}")
            # sensitivity grid: assumed rate x true rate (delta_post 0.05 and 0.2)
            grid = {}
            for dp in v["dpost_grid"]:
                print(f"    posterior sensitivity, delta_post={dp}: rows assumed rate, cols true flip rate; cell = unsafe_any / fallback / mean answers")
                hdr = "      pa\\p " + "".join(f"{p:>22}" for p in v["p_grid"]); print(hdr)
                for pa in v["pa_grid"]:
                    line = f"      {pa:<6}"
                    for p in v["p_grid"]:
                        s = S[f"flip|{p}|bayes|pa{pa}|dp{dp}"]["unsafe_at_0"]; grid[f"pa{pa}|dp{dp}|p{p}"] = s
                        line += f"  {fmt(s['unsafe_any'],4)}/{fmt(s['fallback'],2)}/{fmt(s['answers_mean'],1)}"
                    print(line)
                line = "      hard  "
                for p in v["p_grid"]:
                    s = S[f"flip|{p}|hard"]["unsafe_at_0"]; line += f"  {fmt(s['unsafe_any'],4)}/{fmt(s['fallback'],2)}/{fmt(s['answers_mean'],1)}"
                print(line)
            # Pareto frontier per true p over all protocols: minimize (answers, fallback, unsafe_any)
            fr = {}
            for p in v["p_grid"]:
                pts = []
                for k, st in S.items():
                    if not k.startswith(f"flip|{p}|"):
                        continue
                    s = st["unsafe_at_0"]
                    if s["answers_mean"] is None:
                        continue
                    pts.append((k, s["answers_mean"], s["fallback"], s["unsafe_any"]))
                pareto = []
                for a in pts:
                    dominated = any((b[1] <= a[1] and b[2] <= a[2] and b[3] <= a[3]) and (b[1] < a[1] or b[2] < a[2] or b[3] < a[3]) for b in pts)
                    if not dominated:
                        pareto.append(a)
                pareto.sort(key=lambda z: (z[3], z[2], z[1]))
                fr[str(p)] = [dict(cell=k, answers=a, fallback=f, unsafe=u) for k, a, f, u in pareto]
                print(f"    frontier p={p}: " + "; ".join(f"{k} (ans {a:.1f}, fallback {f:.2f}, unsafe {u:.4f})" for k, a, f, u in pareto[:8]))
            out[name][prior] = dict(grid=grid, frontier=fr)
    json.dump(out, open(os.path.join(R, "answer_noise_frontier.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
