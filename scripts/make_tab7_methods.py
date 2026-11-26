from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

CTRL = (("CIARA", "CIARA (comparison run)", "certified policy, drawn per node-block, same run as the baselines"), ("CB", "CB", "constrained bandit"), ("I1-E", "I1-E (comparison run)", "profile table, every epoch"), ("I1-S", "I1-S", "profile table, once"), ("I1-oracle", "I1-oracle", "contention-corrected table"),
        ("I2", "I2", "RACE threshold"), ("I3", "I3", "Mix-or-Split routing"), ("RecServe", "RecServe", "edge confidence quantile"), ("RecServe-ft", "RecServe-ft", "edge confidence, fine-tuned"),
        ("I1-E|c32", "I1-E|c32", "I1-E with the 32-token cascade"), ("I1-E|f1prio", "I1-E|f1prio", "I1-E with priority admission"))


def row(name, cfg, lv):
    e = max((lv.get(k) for k in ("Y_E_A", "Y_E_B", "Y_E_C") if lv.get(k) is not None), default=None)
    meets = None
    if all(lv.get(k) is not None for k in C.COST_ROWS):
        meets = all(lv[k] <= C.BUDGETS[k] + 1e-12 for k in C.COST_ROWS)
    return (name, cfg, lv.get("Y_Q"), lv.get("Y_L_f1"), lv.get("Y_L_f2"), lv.get("Y_D"), e, lv.get("Y_C"), meets)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=C.DATA)
    ap.add_argument("--out-dir", default=C.TABS)
    a = ap.parse_args()
    D = a.data_dir
    rows = []
    e4 = C.read_csv("e4_estimates.csv", D)
    pi = {r["arm"]: C.fnum(r["mass"]) for r in C.read_csv("campaign_policies.csv", D) if r["look_day"] == "7" and r["band"] == "planned"}
    cfg = ", ".join(f"{pi[x]:.2f} {C.ARM_SHORT[x]}" for x in (C.A_INC, C.A_PRIO, C.A_FIFO) if pi.get(x))
    for pol, name, c in (("C", "CIARA (certified policy, held-out)", cfg), ("I", "Incumbent I1-E (held-out)", "profile table, every epoch")):
        rows.append(row(name, c, {r["outcome"]: C.fnum(r["mean"]) for r in e4 if r["policy"] == pol and r["unit"] == "constant"}))
    est = {(r["arm"], r["outcome"]): C.fnum(r["center"]) for r in C.read_csv("campaign_arm_estimates.csv", D) if r["band"] == "planned" and r["look_day"] == "7"}
    for arm, nm in ((C.A_PRIO, "CIARA arm c32/priority (campaign)"), (C.A_FIFO, "CIARA arm c32/FIFO (campaign)")):
        rows.append(row(nm, "arm of S, admission order " + ("priority" if arm == C.A_PRIO else "FIFO"), {k: est.get((arm, k)) for k in C.OUTCOMES}))
    if C.have("e1e5_estimates.csv", D):
        e1 = C.read_csv("e1e5_estimates.csv", D)
        for arm, nm, c in CTRL:
            lv = {r["outcome"]: C.fnum(r["mean"]) for r in e1 if r["arm"] == arm}
            if lv:
                rows.append(row(nm, c, lv))
    if C.have("e5off_realized.csv", D):
        e5 = C.read_csv("e5off_realized.csv", D)
        for pol, nm in (("plug_in_lp", "Plug-in LP (offline)"), ("ciara_without_energy", "CIARA without energy row (offline)")):
            lv = {r["outcome"]: C.fnum(r["level"]) for r in e5 if r["policy"] == pol}
            if lv:
                rows.append(row(nm, "offline policy at the reference levels", lv))
    body = ["\\begin{table*}[!t]", "\\centering", "\\caption{" +
            "Chosen configuration and realized constraints per method (means over node-blocks; device energy in constant wire-overhead units, worst class; budgets: tail exceedance 0.10 per family, "
            "deadline misses 0.10, device energy 0.5, cloud share 0.30). Meets: every budget holds at the point estimate. Controllers of the comparison run and offline baselines are listed after the campaign rows.}",
            "\\label{tab:methods}", "\\footnotesize", "\\begin{tabular}{@{}llccccccc@{}}", "\\toprule",
            "\\textbf{Method} & \\textbf{Configuration} & \\textbf{Quality} & \\textbf{L1} & \\textbf{L2} & \\textbf{Deadline} & \\textbf{Energy} & \\textbf{Cloud} & \\textbf{Meets} \\\\", "\\midrule"]
    for n, c, q, l1, l2, d, e, cl, ok in rows:
        body.append(f"{C.tex_escape(n)} & {C.tex_escape(c)} & {C.f3(q)} & {C.f3(l1)} & {C.f3(l2)} & {C.f3(d)} & {C.f3(e)} & {C.f3(cl)} & {'---' if ok is None else ('yes' if ok else 'no')} \\\\")
    body += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    p = C.write_tex(os.path.join(a.out_dir, "tab7_methods.tex"), "\n".join(body), "e4_estimates.csv, campaign_arm_estimates.csv, e1e5_estimates.csv, e5off_realized.csv")
    print("tab7:", p, f"({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
