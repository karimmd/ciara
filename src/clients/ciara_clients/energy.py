from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import tomllib
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

CONFIG_DIR = os.environ.get("CIARA_CONFIG_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "configs")
DEFAULT_ENERGY = os.path.join(CONFIG_DIR, "clients", "energy.toml")
LEVELS = ("low", "nominal", "high")


def load_energy(path: Optional[str] = None) -> Dict[str, Any]:
    with open(path or DEFAULT_ENERGY, "rb") as f:
        return tomllib.load(f)


def coeffs(tbl: Dict[str, Any], radio: str, link: Dict[str, Any], level: str) -> Tuple[float, float, float, float, float]:
    r = tbl["radio"][radio]
    beta = r["beta_mw"][level]
    if r["mode"] == "state":
        a_up = (r["p_tx_mw"][level] - beta) / (link["rate_up_kbps"] / 1000.0)
        a_dn = (r["p_rx_mw"][level] - beta) / (link["rate_down_kbps"] / 1000.0)
    else:
        a_up, a_dn = r["a_up_mw_per_mbps"][level], r["a_dn_mw_per_mbps"][level]
    return a_up, a_dn, beta, r["tail_s"][level], r["p_tail_mw"][level]


def request_energy(tbl: Dict[str, Any], radio: str, cls: str, link: Dict[str, Any], rec: Dict[str, Any], oh_up: float, oh_dn: float,
                   level: str, gap_to_next_s: Optional[float] = None, tail: bool = False) -> Dict[str, float]:
    a_up, a_dn, beta, tail_s, p_tail = coeffs(tbl, radio, link, level)
    dev = tbl["device"][cls]
    p_cpu = dev["p_cpu_mw"][level]
    p_wait = beta + dev["p_dev_idle_mw"][level]
    t_send = rec.get("t_send_mono_ns")
    t_end = rec.get("t_last_byte_mono_ns")
    if t_end is None:
        t_end = rec.get("t_done_mono_ns")
    L = max(0.0, (t_end - t_send) / 1e9) if (t_send is not None and t_end is not None) else 0.0
    b_up = rec.get("bytes_up") or 0
    b_dn = rec.get("bytes_down") or 0
    mbit_up = 8.0 * b_up * oh_up / 1e6
    mbit_dn = 8.0 * b_dn * oh_dn / 1e6
    air = mbit_up / (link["rate_up_kbps"] / 1000.0) + mbit_dn / (link["rate_down_kbps"] / 1000.0)
    t_radio = min(L, air)
    scale = (t_radio / air) if air > 0 else 0.0
    e_radio = (a_up * mbit_up + a_dn * mbit_dn) * scale + beta * t_radio
    e_wait = p_wait * max(0.0, L - t_radio)
    t_cpu = ((rec.get("cpu_prep") or {}).get("dev_ms", 0.0) + (rec.get("cpu_post") or {}).get("dev_ms", 0.0)) / 1000.0
    e_cpu = p_cpu * t_cpu
    e_tail = 0.0
    if tail and tail_s > 0:
        e_tail = p_tail * (min(tail_s, gap_to_next_s) if gap_to_next_s is not None else tail_s)
    return {"radio_mj": e_radio, "wait_mj": e_wait, "cpu_mj": e_cpu, "tail_mj": e_tail,
            "total_mj": e_radio + e_wait + e_cpu + e_tail, "t_radio_s": t_radio, "latency_s": L, "t_cpu_s": t_cpu}


def read_logs(run_dir: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in sorted(glob.glob(os.path.join(run_dir, "clients", "*.jsonl"))):
        base = os.path.basename(p)
        if base.startswith("client-"):
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
    return out


OVERHEAD_RULES = ("constant", "measured")


class MissingTcStats(RuntimeError):
    pass


def wire_overheads(run_dir: str, recs: Sequence[Dict[str, Any]], default: float, rule: str) -> Dict[str, Tuple[float, float]]:
    if rule not in OVERHEAD_RULES:
        raise ValueError(f"overhead rule {rule!r}: expected one of {OVERHEAD_RULES}")
    if rule == "constant":
        return {}
    app_up: Dict[str, int] = {}
    app_dn: Dict[str, int] = {}
    for r in recs:
        app_up[r["device"]] = app_up.get(r["device"], 0) + (r.get("bytes_up") or 0)
        app_dn[r["device"]] = app_dn.get(r["device"], 0) + (r.get("bytes_down") or 0)
    out: Dict[str, Tuple[float, float]] = {}
    files = glob.glob(os.path.join(run_dir, "clients", "*.tc_stats.json"))
    if not files:
        raise MissingTcStats(f"overhead rule 'measured': no {os.path.join(run_dir, 'clients', '*.tc_stats.json')} (the runtime writes it at the end of a clean run)")
    for p in files:
        with open(p) as f:
            doc = json.load(f)
        for dev, st in doc["devices"].items():
            up = st.get("up", {}).get("bytes", 0)
            dn = st.get("down", {}).get("bytes", 0)
            ou = up / app_up[dev] if app_up.get(dev) else default
            od = dn / app_dn[dev] if app_dn.get(dev) else default
            out[dev] = (ou, od)
    missing = sorted(set(app_up) - set(out))
    if missing:
        raise MissingTcStats(f"overhead rule 'measured': {len(missing)} devices have no counters in {os.path.join(run_dir, 'clients')} (first: {missing[:3]})")
    return out


def compute(run_dir: str, sched_dir: str, tail: bool, tbl: Dict[str, Any], sensitivity: bool = False, *, overhead: str) -> Dict[str, Any]:
    with open(os.path.join(sched_dir, "schedule.jsonl"), "r", encoding="utf-8") as f:
        meta = json.loads(f.readline())
    recs = read_logs(run_dir)
    default_oh = tbl["wire"]["overhead_factor"]
    oh = wire_overheads(run_dir, recs, default_oh, overhead)
    by_dev: Dict[str, List[Dict[str, Any]]] = {}
    for r in recs:
        by_dev.setdefault(r["device"], []).append(r)
    rows: List[Dict[str, Any]] = []
    for dev, lst in by_dev.items():
        lst.sort(key=lambda r: r.get("t_send_mono_ns") or r["t_arrival_mono_ns"])
        for i, r in enumerate(lst):
            cls = r["class"]
            radio = meta["classes"][cls]["radio"]
            link = meta["links"][r["link"]]
            ou, od = oh.get(dev, (default_oh, default_oh))
            gap = None
            if i + 1 < len(lst):
                nxt = lst[i + 1].get("t_send_mono_ns") or lst[i + 1]["t_arrival_mono_ns"]
                end = r.get("t_last_byte_mono_ns") or r.get("t_done_mono_ns")
                gap = max(0.0, (nxt - end) / 1e9) if end else None
            e = request_energy(tbl, radio, cls, link, r, ou, od, "nominal", gap, tail)
            row = {"req": r["req"], "device": dev, "class": cls, "node": r["node"], "family": r["family"], "link": r["link"],
                   "error": r.get("error"), "deadline_met": r.get("deadline_met"),
                   "energy_mj": round(e["total_mj"], 4), "radio_mj": round(e["radio_mj"], 4), "wait_mj": round(e["wait_mj"], 4),
                   "cpu_mj": round(e["cpu_mj"], 4), "tail_mj": round(e["tail_mj"], 4), "t_radio_s": round(e["t_radio_s"], 5),
                   "latency_s": round(e["latency_s"], 4), "t_cpu_s": round(e["t_cpu_s"], 5),
                   "e_norm": round(min(1.0, e["total_mj"] / 1000.0 / tbl["budget"][cls]), 6)}
            if sensitivity:
                for lv in ("low", "high"):
                    row[f"energy_mj_{lv}"] = round(request_energy(tbl, radio, cls, link, r, ou, od, lv, gap, tail)["total_mj"], 4)
            rows.append(row)
    return {"rows": rows, "summary": summarize(rows), "overheads": {d: [round(x, 4) for x in v] for d, v in oh.items()},
            "tail": tail, "overhead_rule": overhead, "model": "device energy model, coefficients from configs/clients/energy.toml"}


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    keys = sorted({(r["class"], r["link"]) for r in rows})
    for cls, link in keys + [(c, "*") for c in sorted({r["class"] for r in rows})]:
        sel = [r for r in rows if r["class"] == cls and (link == "*" or r["link"] == link)]
        if not sel:
            continue
        e = sorted(r["energy_mj"] for r in sel)
        s = {"n": len(sel), "mean_mj": round(statistics.fmean(e), 3), "p50_mj": e[len(e) // 2], "p95_mj": e[min(len(e) - 1, int(0.95 * len(e)))],
             "radio_mj": round(statistics.fmean(r["radio_mj"] for r in sel), 3), "wait_mj": round(statistics.fmean(r["wait_mj"] for r in sel), 3),
             "cpu_mj": round(statistics.fmean(r["cpu_mj"] for r in sel), 3), "e_norm_mean": round(statistics.fmean(r["e_norm"] for r in sel), 5)}
        if "energy_mj_low" in sel[0]:
            s["mean_low_mj"] = round(statistics.fmean(r["energy_mj_low"] for r in sel), 3)
            s["mean_high_mj"] = round(statistics.fmean(r["energy_mj_high"] for r in sel), 3)
        out[f"{cls}/{link}"] = s
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Offline device energy from the client logs and the radio model of the supplement, eq. (S4).', formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="run directory that holds clients/")
    ap.add_argument("--schedule", required=True, help="schedule directory (schedule.jsonl)")
    ap.add_argument("--energy-config", default=None)
    ap.add_argument("--overhead", required=True, choices=OVERHEAD_RULES, help="wire overhead rule: the constant of the energy table or the tc counters of the run (fails if they are missing)")
    ap.add_argument("--tail", action="store_true", help="add the RRC tail term (outside eq. 3.4)")
    ap.add_argument("--sensitivity", action="store_true", help="also compute all-low and all-high coefficient totals")
    ap.add_argument("--out", default=None, help="per-request JSONL (default: <run-dir>/energy.jsonl)")
    a = ap.parse_args(argv)
    res = compute(a.run_dir, a.schedule, a.tail, load_energy(a.energy_config), a.sensitivity, overhead=a.overhead)
    out = a.out or os.path.join(a.run_dir, "energy.jsonl")
    with open(out, "w") as f:
        for r in res["rows"]:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    summ = os.path.join(os.path.dirname(out), "energy-summary.json")
    with open(summ, "w") as f:
        json.dump({k: v for k, v in res.items() if k != "rows"}, f, indent=1)
    print(json.dumps(res["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
