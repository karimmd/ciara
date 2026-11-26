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
    for r in C.read_csv("level_a.csv", a.data_dir):
        d = {k: C.fnum(v) for k, v in r.items() if k not in ("family", "size", "ckpt")}
        d.update({"family": r["family"], "size": r["size"], "ckpt": r["ckpt"], "admissible": r["admissible"]})
        if d.get("kv_tokens") is not None:
            d["kv_tokens"] = int(d["kv_tokens"])
        rows.append(d)
    out = tabfmt.tab_level_a(rows, a.out_dir)
    widen(out["tex"])
    print("tab_level_a:", out["tex"], f"({out['column']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
