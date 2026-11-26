import os, csv, sys
import numpy as np
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("CIARA_RUNS", "runs"), "coupled")
def load(n):
    rs = list(csv.DictReader(open(os.path.join(D, n))))
    for r in rs:
        for k in r:
            try: r[k] = float(r[k])
            except ValueError: pass
    return rs
S, W, Zr = load("summary.csv"), load("width.csv"), load("zeta.csv")
idx = {(r["scen"], int(r["t_mix"]), int(r["N"]), int(r["level"]), r["radius"], r["zmode"], r["design"]): r for r in S}
fm = lambda x: "n/a" if x != x else f"{x:.2f}"
out = []
def P(x=""): out.append(x)
RAD = ("NAIVE", "C1P", "C1R")
nconf = len({(r["scen"], r["t_mix"], r["N"], r["level"]) for r in S})
R = int(S[0]["R"])

P(f"### Validity: {nconf} configurations (4 scenarios, 3 t_mix, N = 5 and 6, 3 coupling levels), R = {R} each, M_c = 24")
P("Cells: false certifications summed over all configurations (of the campaign) / largest 95% upper bound over configurations and designs (campaign, day 4, day 7, day 10) / interval misses at any of the 40 looks summed / largest upper bound.")
P()
P("| radius | zeta mode | false certifications (campaign) | largest upper bound | interval misses (any look) | largest upper bound |")
P("|---|---|---:|---:|---:|---:|")
for m in RAD:
    for zm in ("T", "E", "Z"):
        rs = [r for r in S if r["radius"] == m and r["zmode"] == zm]
        camp = [r for r in rs if r["design"] == "campaign"]
        P(f"| {m} | {zm} | {int(sum(r['fc'] for r in camp))} of {len(camp) * R} | {max(r['fc_ub'] for r in rs):.4f} | {int(sum(r['miss_any'] for r in camp))} | {max(r['miss_any_ub'] for r in rs):.4f} |")
P()
P("### K3X1: the campaign (looks day 4, 7, 10), zeta true (mode T). Cells: fraction of the oracle gain; mean stopping day; switch rate")
P()
P("| t_mix | N | zeta (exceedance) | oracle gain | NAIVE | C1P | C1R |")
P("|---:|---:|---:|---:|---|---|---|")
for t in (3, 5, 8):
    for N in (5, 6):
        for lv, zl in ((0, "0"), (1, "0.01"), (2, "0.03")):
            c = [idx[("K3X1", t, N, lv, m, "T", "campaign")] for m in RAD]
            P(f"| {t} | {N} | {zl} | {c[0]['og']:.3f} | " + " | ".join(f"{fm(x['frac'])}; {x['stop_day']:.1f}; {x['switch']:.2f}" for x in c) + " |")
P()
P("### Other scenarios, N = 5, zeta true: fraction of the oracle gain; mean stopping day; switch rate (null: no gain exists, the switch rate must be 0)")
P()
P("| scenario | t_mix | zeta | NAIVE | C1P | C1R |")
P("|---|---:|---:|---|---|---|")
for s in ("small", "null", "trap"):
    for t in (3, 5, 8):
        for lv, zl in ((0, "0"), (2, "0.03")):
            c = [idx[(s, t, 5, lv, m, "T", "campaign")] for m in RAD]
            P(f"| {s} | {t} | {zl} | " + " | ".join(f"{fm(x['frac'])}; {x['stop_day']:.1f}; {x['switch']:.2f}" for x in c) + " |")
P()
P("### Median radius, K3X1, N = 5, no coupling (arm 0 incumbent, arms 1 and 2 alternatives); exceedance outcome")
P()
P("| t_mix | day | NAIVE arm 0 / 1 / 2 | C1P arm 0 / 1 / 2 | C1R arm 0 / 1 / 2 | C1R / NAIVE |")
P("|---:|---:|---|---|---|---|")
wi = {(r["scen"], int(r["t_mix"]), int(r["N"]), int(r["level"]), r["radius"], int(r["day"]), r["outcome"], int(r["arm"])): r for r in W}
for t in (3, 5, 8):
    for day in (4, 7, 10):
        v = {m: [wi[("K3X1", t, 5, 0, m, day, "exceed", a)]["median_radius"] for a in range(3)] for m in RAD}
        P(f"| {t} | {day} | " + " | ".join(" / ".join(f"{x:.3f}" for x in v[m]) for m in RAD) + " | " + " / ".join(f"{b / a:.2f}" for a, b in zip(v["NAIVE"], v["C1R"])) + " |")
P()
rat = {}
for r in W:
    if r["radius"] != "NAIVE":
        rat.setdefault((r["radius"], int(r["arm"])), []).append(r["ratio_to_naive"])
P("Ratio of the median radius to NAIVE over all scenarios, t_mix, N, coupling levels, days (4, 7, 10) and outcomes: min / median / max")
P()
P("| radius | incumbent (arm 0) | alternative arm 1 | alternative arm 2 |")
P("|---|---|---|---|")
for m in ("C1P", "C1R"):
    P(f"| {m} | " + " | ".join(f"{np.min(rat[(m, a)]):.2f} / {np.median(rat[(m, a)]):.2f} / {np.max(rat[(m, a)]):.2f}" for a in range(3)) + " |")
P()
P("### Effect of the zeta input on the certified gain (K3X1, mean over t_mix and N): fraction of the oracle gain of the campaign, modes T / E / Z")
P()
P("| radius | zeta 0 | zeta 0.01 | zeta 0.03 |")
P("|---|---|---|---|")
for m in RAD:
    cells = []
    for lv in (0, 1, 2):
        cells.append(" / ".join(f"{np.mean([idx[('K3X1', t, N, lv, m, zm, 'campaign')]['frac'] for t in (3, 5, 8) for N in (5, 6)]):.2f}" for zm in ("T", "E", "Z")))
    P(f"| {m} | " + " | ".join(cells) + " |")
P()
P("### Coupling diagnostics (mean over scenarios, t_mix, N)")
P()
P("| level | true zeta, exceedance (pair weighted) | true zeta (round weighted) | Stage 1 estimate, mean | share of replicates with estimate below the truth | within-round correlation of window means, mean (max) |")
P("|---:|---:|---:|---:|---:|---|")
for lv in (0, 1, 2):
    rs = [r for r in Zr if int(r["level"]) == lv]
    P(f"| {lv} | {np.mean([r['zeta_P_exceed'] for r in rs]):.4f} | {np.mean([r['zeta_R_exceed'] for r in rs]):.4f} | {np.mean([r['zhat_exceed_mean'] for r in rs]):.4f} | {np.mean([r['frac_zhat_below_true'] for r in rs]):.2f} | {np.mean([r['rho_c'] for r in rs]):.3f} ({max(r['rho_c'] for r in rs):.3f}) |")
P()
P("### Adoption criteria V and U (selection.csv)")
P()
for r in csv.DictReader(open(os.path.join(D, "selection.csv"))):
    P("- " + ", ".join(f"{k} = {v}" for k, v in r.items()))
open(os.path.join(D, "tables.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
