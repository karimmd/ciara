from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import pilot_program as GR
from . import screening as S
from .transient import TauTable


def mem_fn_from_table(table: Dict[str, Any]):
    edge = table.get("edge", {})

    def mem(profile: Optional[str], cap: Optional[int]) -> Optional[float]:
        row = edge.get(profile or "")
        if row is None or cap is None:
            return None
        return (row["budget_gib"] - row["kv_gib"]) + cap * (4096 + 1024) * row["kv_gib"] / row["kv_tokens"]
    return mem


def used_profiles(rows: List[Dict[str, Any]], regimes: List[str]) -> Dict[str, List[str]]:
    out: Dict[str, set] = {a: set() for a in regimes}
    for r in rows:
        if r["arm_id"] in out and r.get("profiles_in_window"):
            out[r["arm_id"]].update(str(r["profiles_in_window"]).split("+"))
    return {a: sorted(v) for a, v in out.items() if v}


def analyze(rows_all: List[Dict[str, Any]], arms: List[Dict[str, Any]], tau: TauTable, table: Dict[str, Any],
            requests: Optional[List[Dict[str, Any]]] = None, cloud_share_mode: str = S.CLOUD_PER_CONFIGURATION,
            n_nodes: Optional[int] = None, budgets: Optional[Dict[str, float]] = None, go_rule: bool = False) -> Dict[str, Any]:
    rows = S.usable_rows(rows_all)
    ids = [a["id"] for a in arms]
    est = S.arm_estimates(rows, ids)
    regimes = [a["id"] for a in arms if a["kind"] == "incumbent"]
    kw: Dict[str, Any] = {}
    if n_nodes is not None:
        kw["n_nodes"] = int(n_nodes)
    if budgets is not None:
        kw["budgets"] = dict(budgets)
    cfg = S.ScreenConfig(arms, tau, mem_fn_from_table(table), used_profiles=used_profiles(rows, regimes), cloud_share_mode=cloud_share_mode, **kw)
    inc = S.pick_incumbent(est, cfg)
    res: Dict[str, Any] = {"n_rows_usable": len(rows), "n_rows_total": len(rows_all), "arm_counts": {a: est[a]["n_blocks"] for a in ids},
                           "incumbent": inc, "regime_profiles": {a: cfg.profiles[a] for a in regimes},
                           "estimates": {a: {k: {kk: vv for kk, vv in v.items() if kk != "values"} for k, v in est[a]["outcomes"].items()} | {"n_blocks": est[a]["n_blocks"]}
                                         for a in ids}}
    if inc["a0"] is None:
        res["selection"] = {"status": "no_incumbent", "S": [], "alternatives": [], "history": []}
        res["pass_table"] = {}
        return res
    a0 = inc["a0"]
    sel = S.select(rows, est, cfg, a0)
    res["selection"] = {k: v for k, v in sel.items() if k != "evals"}
    if go_rule:
        res["go_rule"] = GR.evaluate(rows, est, cfg, a0, sel)
    final_members = sel["S"] if sel["status"] == "selected" else [a0]
    zeta = (sel.get("zeta") or {}).get("zeta") or {}
    table_pass: Dict[str, Any] = {}
    for c in cfg.configs:
        members = final_members if c in final_members else [a0, c]
        table_pass[c] = S.eval_member(c, est, members, zeta, cfg, True)
    res["pass_table"] = table_pass
    res["pass_table_zero_zeta"] = {c: S.eval_member(c, est, [a0, c], {}, cfg, True)["pass"] for c in cfg.configs}
    res["node_effects"] = {k: S.node_effects(rows, est, k) for k in ("Y_Q", "Y_L_f1", "Y_L_f2", "Y_D", "Y_C")}
    if requests is not None:
        arm_of = {(r["node"], int(r["block_id"])): r["arm_id"] for r in rows}
        res["carryover"] = S.carryover_by_epoch(requests, arm_of, [c for c in cfg.configs if est[c]["n_blocks"] >= 5])
    res["parameters"] = {"t_bar": cfg.t_bar, "b": cfg.b, "l": cfg.l, "l_dep": cfg.l_dep, "k7": cfg.k7, "n_nodes": cfg.n_nodes, "eps": cfg.eps,
                         "budgets": cfg.budgets, "tie_quality": S.TIE_Q, "min_blocks": S.MIN_BLOCKS, "cloud_share_mode": cloud_share_mode}
    return res
