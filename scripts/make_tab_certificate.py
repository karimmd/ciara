from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=C.DATA)
    ap.add_argument("--out-dir", default=C.TABS)
    a = ap.parse_args()
    D = a.data_dir
    cert = {(r["look_day"], r["band"], r["row"]): r for r in C.read_csv("campaign_certificate_rows.csv", D)}
    looks = {(r["look_day"], r["band"]): r for r in C.read_csv("campaign_looks.csv", D)}
    pol = {}
    for r in C.read_csv("campaign_policies.csv", D):
        pol[(r["look_day"], r["band"], r["arm"])] = C.fnum(r["mass"])
    lk7 = looks[("7", "planned")]
    masses = ", ".join(f"{pol[('7', 'planned', arm)]:.3f} {C.ARM_SHORT[arm]}" for arm in (C.A_INC, C.A_PRIO, C.A_FIFO))
    L = ["\\begin{table*}[!t]", "\\centering",
         "\\caption{Certificate at the final look (day 7, " + lk7["complete_rounds"] + " complete rounds). Upper bounds of the incumbent and of the certified policy (" + masses.replace("%", "\\%") + ") on each budget row, under "
         "the planned carryover band (t\\_bar $=5$), the measured band and t\\_bar $=8$ (infeasible: no certified policy). Quality cost bound " + f"{C.fnum(lk7['quality_cost_bound']):.4f}" + ".}",
         "\\label{tab:certificate}", "\\small", "\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
         "\\textbf{Row} & \\textbf{Budget} & \\textbf{Incumbent upper} & \\textbf{Policy upper} & \\textbf{Policy upper, measured band} & \\textbf{t\\_bar $=8$} \\\\", "\\midrule"]
    for k in C.COST_ROWS:
        r, m = cert[("7", "planned", k)], cert.get(("7", "measured", k))
        t8 = cert.get(("7", "tbar8", k))
        t8s = "infeasible" if looks[("7", "tbar8")]["lp_status"] == "infeasible" else "---"
        L.append(f"{C.tex_escape(C.LABEL[k])} & {C.f3(C.fnum(r['budget']), 2)} & {C.f3(C.fnum(r['incumbent_upper']))} & {C.f3(C.fnum(r['policy_upper']))} & {C.f3(C.fnum(m['policy_upper'])) if m else '---'} & {t8s} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    p1 = C.write_tex(os.path.join(a.out_dir, "tab_certificate.tex"), "\n".join(L), "campaign_certificate_rows.csv, campaign_looks.csv, campaign_policies.csv")
    est = {(r["band"], r["look_day"], r["arm"], r["outcome"]): r for r in C.read_csv("campaign_arm_estimates.csv", D)}
    L = ["\\begin{table}[!t]", "\\centering", "\\caption{Campaign arm estimates at the final look (day 7): round-weighted center with its radius (energy rows in constant wire-overhead units). "
         "Budgets: tail exceedance 0.10 per family, deadline misses 0.10, device energy 0.5 per class, cloud share 0.30.}", "\\label{tab:campaign-arms}", "\\footnotesize", "\\begin{tabular}{@{}lccc@{}}", "\\toprule",
         "\\textbf{Outcome} & \\textbf{I1-E} & \\textbf{c32/priority} & \\textbf{c32/FIFO} \\\\", "\\midrule"]
    for k in C.OUTCOMES:
        cells = []
        for arm in (C.A_INC, C.A_PRIO, C.A_FIFO):
            r = est.get(("planned", "7", arm, k))
            cells.append(f"{C.f3(C.fnum(r['center']))} $\\pm$ {C.f3(C.fnum(r['radius']))}" if r else "---")
        L.append(f"{C.tex_escape(C.LABEL[k])} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    p2 = C.write_tex(os.path.join(a.out_dir, "tab_campaign_arms.tex"), "\n".join(L), "campaign_arm_estimates.csv")
    print("tab_certificate:", p1, p2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
