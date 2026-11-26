from __future__ import annotations

import glob
import json
import math
import os
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence


def _pct(xs: Sequence[float], q: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def load_schedule_reqs(sched_dir: str, nodes: Optional[Sequence[int]], classes: Optional[Sequence[str]]):
    meta = None
    reqs: Dict[str, Dict[str, Any]] = {}
    with open(os.path.join(sched_dir, "schedule.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["type"] == "meta":
                meta = r
            elif r["type"] == "req" and (nodes is None or r["node"] in nodes) and (classes is None or r["class"] in classes):
                reqs[r["req"]] = r
    return meta, reqs


def load_logs(run_dir: str) -> List[Dict[str, Any]]:
    out = []
    for p in sorted(glob.glob(os.path.join(run_dir, "clients", "n*.jsonl"))):
        with open(p, encoding="utf-8") as f:
            out.extend(json.loads(l) for l in f if l.strip())
    return out


def check_run(run_dir: str, sched_dir: str, mock_log: Optional[str] = None, nodes: Optional[Sequence[int]] = None,
              classes: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    meta, reqs = load_schedule_reqs(sched_dir, nodes, classes)
    logs = load_logs(run_dir)
    res: Dict[str, Any] = {}
    ids = [r["req"] for r in logs]
    seen = set(ids)
    res["completeness"] = {"scheduled": len(reqs), "logged": len(logs), "duplicates": len(ids) - len(seen),
                           "missing": sorted(set(reqs) - seen)[:10], "n_missing": len(set(reqs) - seen),
                           "extra": sorted(seen - set(reqs))[:10]}
    res["completeness"]["ok"] = not (res["completeness"]["n_missing"] or res["completeness"]["duplicates"] or res["completeness"]["extra"])
    st: Dict[str, int] = {}
    errs: Dict[str, int] = {}
    for r in logs:
        st[str(r.get("status"))] = st.get(str(r.get("status")), 0) + 1
        if r.get("error"):
            k = str(r["error"]).split(":")[0]
            errs[k] = errs.get(k, 0) + 1
    res["status"] = {"counts": st, "errors": errs, "ok": st.get("200", 0) == len(logs) and not errs}
    bad = 0
    for r in logs:
        s = reqs.get(r["req"])
        if s and (s["device"], s["class"], s["family"], s["item"], s["deadline_ms"]) != (r["device"], r["class"], r["family"], r["item"], r["deadline_ms"]):
            bad += 1
        h = r["headers"]
        if s and (h["X-Ciara-Req"], h["X-Ciara-Device"], h["X-Ciara-Class"], h["X-Ciara-Family"], h["X-Ciara-Item"], h["X-Ciara-Deadline-Ms"]) != \
                (s["req"], s["device"], s["class"], str(s["family"]), s["item"], str(s["deadline_ms"])):
            bad += 1
    res["schedule_fidelity"] = {"mismatches": bad, "ok": bad == 0}
    prof: Dict[str, Dict[str, Any]] = {}
    for link in sorted({r["link"] for r in logs}):
        sel = [r for r in logs if r["link"] == link and r.get("t_first_byte_mono_ns") and r.get("t_send_mono_ns")]
        if not sel:
            continue
        L = meta["links"][link]
        rtt = L["delay_up_ms"] + L["delay_down_ms"]
        ttfb = [(r["t_first_byte_mono_ns"] - r["t_send_mono_ns"]) / 1e6 for r in sel]
        lat = [r["latency_ms"] for r in sel if r.get("latency_ms")]
        floor = []
        for r in sel:
            ser = r["bytes_up"] * 8.0 / L["rate_up_kbps"]
            floor.append(rtt + ser)
        prof[link] = {"n": len(sel), "ttfb_ms_p50": round(_pct(ttfb, 0.5), 1), "ttfb_ms_p05": round(_pct(ttfb, 0.05), 1),
                      "floor_ms_p50": round(_pct(floor, 0.5), 1), "latency_ms_p50": round(_pct(lat, 0.5), 1),
                      "latency_ms_p95": round(_pct(lat, 0.95), 1), "deadline_met": round(sum(1 for r in sel if r.get("deadline_met")) / len(sel), 3),
                      "ttfb_below_floor_frac": round(sum(1 for t, f in zip(ttfb, floor) if t < 0.8 * f) / len(sel), 4)}
    res["link_consistency"] = {"profiles": prof, "ok": all(p["ttfb_below_floor_frac"] < 0.02 for p in prof.values())}
    order_bad = 0
    for r in logs:
        seq = [r.get(k) for k in ("t_arrival_mono_ns", "t_prep_done_mono_ns", "t_send_mono_ns", "t_first_byte_mono_ns", "t_last_byte_mono_ns", "t_done_mono_ns")]
        seq = [x for x in seq if x]
        if seq != sorted(seq):
            order_bad += 1
    res["timestamps"] = {"out_of_order_records": order_bad, "ok": order_bad == 0,
                         "wall_minus_mono_spread_ms": None}
    off = [(r["t_send_wall_ns"] - r["t_send_mono_ns"]) / 1e6 for r in logs if r.get("t_send_wall_ns")]
    if off:
        res["timestamps"]["wall_minus_mono_spread_ms"] = round(max(off) - min(off), 3)
    lag = [r["arrival_lag_ms"] for r in logs]
    res["arrival_lag_ms"] = {"p50": round(_pct(lag, 0.5), 3), "p99": round(_pct(lag, 0.99), 3), "max": round(max(lag), 3) if lag else None}
    shares: Dict[str, List[float]] = {}
    work_ms: Dict[str, List[float]] = {}
    for r in logs:
        for k in ("cpu_prep", "cpu_post"):
            c = r.get(k)
            if c and c.get("share_eff") is not None and c["dev_ms"] > 5.0:
                shares.setdefault(r["class"], []).append(c["share_eff"])
        if r.get("cpu_prep"):
            work_ms.setdefault(r["class"], []).append(r["cpu_prep"]["dev_ms"])
    res["cpu_share"] = {c: {"configured": meta["classes"][c]["cpu_share"], "achieved_median": round(statistics.median(v), 4),
                            "n": len(v), "prep_dev_ms_p50": round(_pct(work_ms[c], 0.5), 2)} for c, v in shares.items()}
    res["cpu_share_ok"] = all(abs(v["achieved_median"] - v["configured"]) <= 0.35 * v["configured"] for v in res["cpu_share"].values())
    reused = sum(1 for r in logs if r.get("conn_reused"))
    res["connections"] = {"reused_frac": round(reused / max(1, len(logs)), 3), "retries": sum(r.get("retries", 0) for r in logs)}
    lags, thr = [], 0
    for p in glob.glob(os.path.join(run_dir, "clients", "*.monitor.jsonl")):
        with open(p) as f:
            for line in f:
                m = json.loads(line)
                lags.append(m["loop_lag_ms_max"])
                thr = max(thr, m.get("cgroup_cpu", {}).get("nr_throttled", 0))
    res["monitor"] = {"loop_lag_ms_max": round(max(lags), 2) if lags else None, "loop_lag_ms_p50_of_max": round(_pct(lags, 0.5), 2) if lags else None,
                      "cgroup_nr_throttled": thr, "ok": (not lags) or (max(lags) < 250 and thr == 0)}
    tc: Dict[str, Dict[str, float]] = {}
    app_up: Dict[str, int] = {}
    app_dn: Dict[str, int] = {}
    link_of: Dict[str, str] = {}
    for r in logs:
        app_up[r["device"]] = app_up.get(r["device"], 0) + (r.get("bytes_up") or 0)
        app_dn[r["device"]] = app_dn.get(r["device"], 0) + (r.get("bytes_down") or 0)
        link_of[r["device"]] = r["link"]
    for p in glob.glob(os.path.join(run_dir, "clients", "*.tc_stats.json")):
        with open(p) as f:
            doc = json.load(f)
        for dev, s in doc["devices"].items():
            if dev not in link_of:
                continue
            t = tc.setdefault(link_of[dev], {"up_bytes": 0, "dn_bytes": 0, "app_up": 0, "app_dn": 0, "up_drops": 0, "dn_drops": 0, "up_pkts": 0, "dn_pkts": 0})
            t["up_bytes"] += s.get("up", {}).get("bytes", 0)
            t["dn_bytes"] += s.get("down", {}).get("bytes", 0)
            t["up_drops"] += s.get("up", {}).get("drops", 0)
            t["dn_drops"] += s.get("down", {}).get("drops", 0)
            t["up_pkts"] += s.get("up", {}).get("packets", 0)
            t["dn_pkts"] += s.get("down", {}).get("packets", 0)
            t["app_up"] += app_up.get(dev, 0)
            t["app_dn"] += app_dn.get(dev, 0)
    res["tc"] = {k: {"l2_per_app_up": round(v["up_bytes"] / v["app_up"], 3) if v["app_up"] else None,
                     "l2_per_app_down": round(v["dn_bytes"] / v["app_dn"], 3) if v["app_dn"] else None,
                     "drop_frac_up": round(v["up_drops"] / v["up_pkts"], 5) if v["up_pkts"] else None,
                     "drop_frac_down": round(v["dn_drops"] / v["dn_pkts"], 5) if v["dn_pkts"] else None} for k, v in tc.items()}
    if mock_log and os.path.exists(mock_log):
        with open(mock_log) as f:
            ml = [json.loads(l) for l in f if l.strip()]
        by = {m["headers"]["X-Ciara-Req"]: m for m in ml}
        hdr_bad = 0
        body_bad = 0
        for rid, s in reqs.items():
            m = by.get(rid)
            if m is None:
                continue
            h = m["headers"]
            if (h["X-Ciara-Device"], h["X-Ciara-Class"], h["X-Ciara-Family"], h["X-Ciara-Item"], h["X-Ciara-Deadline-Ms"]) != \
                    (s["device"], s["class"], str(s["family"]), s["item"], str(s["deadline_ms"])):
                hdr_bad += 1
            mt = meta["families"][str(s["family"])]["max_tokens"]
            if not (m["stream"] is True and m["temperature"] == 0 and m["seed"] == 0 and m["enable_thinking"] is False and m["max_tokens"] == mt):
                body_bad += 1
        res["server_view"] = {"requests_seen": len(by), "scheduled_unseen": len(set(reqs) - set(by)), "header_mismatches": hdr_bad,
                              "body_param_mismatches": body_bad, "ok": hdr_bad == 0 and body_bad == 0}
    cpath = os.path.join(sched_dir, "context.json")
    if os.path.exists(cpath):
        with open(cpath) as f:
            ctx = json.load(f)
        be = int(ctx.get("block_epochs", 50))
        segs = ctx["segments"]
        contiguous = all(a["end_epoch"] == b["start_epoch"] for a, b in zip(segs, segs[1:])) and (not segs or segs[0]["start_epoch"] == 0)
        aligned = all(x["start_epoch"] % be == 0 and x["end_epoch"] % be == 0 and x["end_epoch"] > x["start_epoch"] for x in segs)
        covers = bool(segs) and segs[-1]["end_epoch"] * float(ctx.get("epoch_s", 30)) >= float(meta["duration_s"])
        res["context"] = {"default_context": ctx["default_context"], "segments": len(segs), "contiguous": contiguous, "aligned": aligned,
                          "covers_horizon": covers, "ok": contiguous and aligned and covers}
    else:
        res["context"] = {"ok": False, "error": "context.json missing"}
    res["ok"] = all(res[k]["ok"] for k in ("completeness", "schedule_fidelity", "link_consistency", "timestamps", "monitor", "context")) \
        and res["cpu_share_ok"] and res.get("server_view", {"ok": True})["ok"]
    return res


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = list(argv if argv is not None else sys.argv[1:])
    if len(a) < 2:
        print('usage: python -m ciara_clients.checks RUN_DIR SCHEDULE_DIR [MOCK_LOG]')
        return 2
    res = check_run(a[0], a[1], a[2] if len(a) > 2 else None)
    print(json.dumps(res, indent=1))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
