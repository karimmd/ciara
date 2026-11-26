from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import tabfmt


def widen(path: str) -> None:
    t = open(path, encoding="utf-8").read().replace("\\begin{table}[!t]", "\\begin{table*}[!t]").replace("\\end{table}", "\\end{table*}")
    open(path, "w", encoding="utf-8").write(t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=C.DATA)
    ap.add_argument("--out-dir", default=C.TABS)
    a = ap.parse_args()
    rows = []
    for r in C.read_csv("overhead.csv", a.data_dir):
        rows.append({"id": r["id"], "label": ("  " if r["sub_item"] == "1" else "") + r["label"], "unit": r["unit"], "n": int(float(r["n"] or 0)), "median": C.fnum(r["median"]), "p95": C.fnum(r["p95"]),
                     "p99": C.fnum(r["p99"]), "max": C.fnum(r["max"]), "mean": C.fnum(r["mean"])})
    out = tabfmt.tab_overhead(rows, 404, a.out_dir)
    widen(out["tex"])
    print("tab_overhead:", out["tex"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
