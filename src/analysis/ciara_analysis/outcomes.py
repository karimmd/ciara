from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .grid import BURN_IN, MEASURE, Grid

FAMILIES = (1, 2)
CLASSES = ("A", "B", "C")


def mean(xs: Iterable[Optional[float]]) -> Optional[float]:
    v = [float(x) for x in xs if x is not None]
    return sum(v) / len(v) if v else None


def quantile(xs: Sequence[float], q: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def cell_outcomes(reqs: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": len(reqs)}
    out["Y_Q"] = mean(r["q"] for r in reqs)
    out["Y_D"] = mean(r["deadline_violation"] for r in reqs)
    out["Y_C"] = mean(1.0 if r["cloud_used"] else 0.0 for r in reqs) if reqs else None
    for f in FAMILIES:
        sub = [r for r in reqs if r["family"] == f]
        out[f"n_f{f}"] = len(sub)
        out[f"Y_L_f{f}"] = mean(r["exceed"] for r in sub)
        out[f"Y_Q_f{f}"] = mean(r["q"] for r in sub)
    for d in CLASSES:
        sub = [r for r in reqs if r["class"] == d]
        out[f"n_{d}"] = len(sub)
        out[f"Y_E_{d}"] = mean(r["e_norm"] for r in sub)
    lat = [r["latency_s"] for r in reqs if r["latency_s"] is not None]
    out["lat_p50_s"] = quantile(lat, 0.5)
    out["lat_p95_s"] = quantile(lat, 0.95)
    out["n_failed"] = sum(1 for r in reqs if not r["ok"])
    out["n_cloud"] = sum(1 for r in reqs if r["cloud_used"])
    out["n_compact_fallback"] = sum(1 for r in reqs if r.get("compact_extracted") is False)
    out["n_truncated"] = sum(1 for r in reqs if r["q_status"] == "truncated")
    out["n_no_gateway_record"] = sum(1 for r in reqs if not r["found_edge"])
    out["n_admitted_earlier_block"] = sum(1 for r in reqs if r["block_id_admit"] is not None and r["block_id_admit"] != r["block_id"])
    out["n_arrived_earlier_block"] = sum(1 for r in reqs if r["block_id_arrival"] is not None and r["block_id_arrival"] != r["block_id"])
    return out


OUTCOME_KEYS = ("Y_Q", "Y_L_f1", "Y_L_f2", "Y_D", "Y_E_A", "Y_E_B", "Y_E_C", "Y_C")


def gpu_window(samples: List[Dict[str, Any]], gpu: int, t0: int, t1: int) -> Dict[str, Any]:
    pts = []
    for s in samples:
        if s.get("event") != "sample" or not (t0 <= s["mono_ns"] < t1):
            continue
        for g in s.get("gpus", []):
            if g.get("gpu") == gpu:
                pts.append((s["mono_ns"], g))
    out: Dict[str, Any] = {"energy_j": None, "power_w_mean": None, "util_mean": None, "throttled_frac": None, "n": len(pts)}
    if len(pts) < 2:
        return out
    e0, e1 = pts[0][1].get("energy_mj"), pts[-1][1].get("energy_mj")
    span = (pts[-1][0] - pts[0][0]) / 1e9
    if e0 is not None and e1 is not None and span > 0:
        out["energy_j"] = (e1 - e0) / 1e3 * ((t1 - t0) / 1e9) / span
    out["power_w_mean"] = mean(g.get("power_w") for _, g in pts)
    out["util_mean"] = mean(g.get("util_gpu") for _, g in pts)
    thr = [g.get("throttled") for _, g in pts if g.get("throttled") is not None]
    out["throttled_frac"] = (sum(1 for x in thr if x) / len(thr)) if thr else None
    return out


def epoch_summary(epochs: List[Dict[str, Any]], lo: int, hi: int) -> Dict[str, Any]:
    sel = [e for e in epochs if lo <= e["epoch"] < hi]
    profiles = sorted({(e.get("config") or {}).get("profile") for e in sel if (e.get("config") or {}).get("profile")})
    return {
        "epochs_logged_window": len(sel),
        "profiles_in_window": "+".join(profiles) if profiles else None,
        "batch_caps_in_window": "+".join(str(c) for c in sorted({(e.get("config") or {}).get("batch_cap") for e in sel
                                                                 if (e.get("config") or {}).get("batch_cap") is not None})) or None,
        "routers_in_window": "+".join(sorted({(e.get("config") or {}).get("router") for e in sel
                                              if (e.get("config") or {}).get("router")})) or None,
        "n_switch_epochs_window": sum(1 for e in sel if e.get("switched")),
        "switch_time_window_s": sum((e.get("switch_time_s") or 0.0) for e in sel),
        "n_config_not_ok_window": sum(1 for e in sel if e.get("config_ok") is False),
        "n_epoch_errors_window": sum(1 for e in sel if e.get("errors")),
    }


def build_tables(requests: List[Dict[str, Any]], blocks: List[Dict[str, Any]], epochs: List[Dict[str, Any]],
                 manifest: Dict[str, Any], gpu_samples: Optional[List[Dict[str, Any]]] = None,
                 node_gpu: Optional[Dict[str, int]] = None, cloud_gpus: Sequence[int] = ()) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grid = Grid.from_manifest(manifest)
    ends = {(r["node"], int(r["block_id"])): r for r in blocks if r.get("rec") == "block_end"}
    starts = [r for r in blocks if r.get("rec") == "block"]
    starts.sort(key=lambda r: (r["node"], int(r["block_id"])))
    by_cell: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = {}
    for r in requests:
        if r["phase"] in (MEASURE, BURN_IN) and r["block_id"] is not None:
            by_cell.setdefault((r["node"], r["block_id"], r["phase"]), []).append(r)
    meas_rows: List[Dict[str, Any]] = []
    burn_rows: List[Dict[str, Any]] = []
    for b in starts:
        node, bid = b["node"], int(b["block_id"])
        end = ends.get((node, bid)) or {}
        cfg = b.get("config") or {}
        for window, rows_out in ((MEASURE, meas_rows), (BURN_IN, burn_rows)):
            lo, hi = grid.measure_range(bid) if window == MEASURE else grid.burn_in_range(bid)
            t0, t1 = grid.epoch_start_ns(lo), grid.epoch_start_ns(hi)
            row: Dict[str, Any] = {
                "run_id": b.get("run_id"), "node": node, "node_index": b.get("node_index"), "block_id": bid,
                "window": window, "epoch_lo": lo, "epoch_hi": hi, "t_lo_mono_ns": t0, "t_hi_mono_ns": t1,
                "arm_id": b.get("arm_id"), "kind": b.get("kind"), "propensity": b.get("propensity"),
                "context_id": b.get("context_id"), "design": b.get("design"), "epsilon": b.get("epsilon"),
                "seed": b.get("seed"), "rng_draw": (b.get("rng") or {}).get("draw"),
                "cfg_profile": cfg.get("profile"), "cfg_batch_cap": cfg.get("batch_cap"),
                "cfg_escalate_family2": cfg.get("escalate_family2"), "cfg_router": cfg.get("router"),
                "profile_switched": b.get("profile_switched"), "switch_time_s": b.get("switch_time_s"),
                "block_status": end.get("status"), "block_switches": end.get("switches"),
                "block_switch_time_total_s": end.get("switch_time_total_s"),
                "block_actuation_failures": end.get("actuation_failures"),
            }
            row.update(epoch_summary([e for e in epochs if e["node"] == node], lo, hi))
            row.update(cell_outcomes(by_cell.get((node, bid, window), [])))
            if gpu_samples is not None and node_gpu and node in node_gpu:
                g = gpu_window(gpu_samples, node_gpu[node], t0, t1)
                row.update({"gpu_energy_j": g["energy_j"], "gpu_power_w_mean": g["power_w_mean"],
                            "gpu_util_mean": g["util_mean"], "gpu_throttled_frac": g["throttled_frac"]})
            if gpu_samples is not None and cloud_gpus:
                es = [gpu_window(gpu_samples, g, t0, t1)["energy_j"] for g in cloud_gpus]
                row["cloud_gpu_energy_j"] = sum(x for x in es if x is not None) if any(x is not None for x in es) else None
            rows_out.append(row)
    return meas_rows, burn_rows
