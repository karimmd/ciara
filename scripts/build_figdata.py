from __future__ import annotations

import csv
import gzip
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out", "derived")
E1E5_REQ = os.path.join(DATA, "e1e5_requests_reduced.csv.gz")
TARGET = {1: 9.504, 2: 44.331}
A_INC = "I1-E"
POLICY_ROUNDS = {0: "incumbent", 3: "incumbent", 1: "certified", 2: "certified"}
NQ = 201


def e1e5_arm(a):
    return "CIARA" if (a.startswith("q35-") or a.startswith("CIARA")) else a


def rows(name):
    p = name if os.path.isabs(name) else os.path.join(DATA, name)
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", newline="") as fh:
        yield from csv.DictReader(fh)


def f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def write(name, recs, cols=None):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    cols = cols or list(recs[0].keys())
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in recs:
            w.writerow({k: (f"{v:.8g}" if isinstance(v, float) else v) for k, v in r.items()})
    print("wrote", p, len(recs))


PGRID = [i / 100 for i in range(90)] + [0.9 + i / 1000 for i in range(100)] + [0.99 + i / 10000 for i in range(101)]


def quantiles(xs):
    xs = sorted(xs)
    n = len(xs)
    out = []
    for p in PGRID:
        j = min(n - 1, max(0, int(math.ceil(p * n)) - 1))
        out.append((p, xs[j]))
    return out


def e4_timeseries():
    acc = defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0.0, 0])
    for r in rows("e4_requests_reduced.csv.gz"):
        pol = POLICY_ROUNDS[int(r["round"])]
        win = int(r["epoch"]) // 120
        a = acc[(pol, win)]
        a[0] += 1
        a[1] += int(r["cloud_used"])
        if r["family"] == "1":
            a[2] += 1
            a[3] += int(float(r["exceed"] or 0))
        a[4] += int(float(r["deadline_violation"] or 0))
        if r.get("q") not in (None, ""):
            a[6] += float(r["q"])
            a[7] += 1
    out = []
    for (pol, win), a in sorted(acc.items()):
        out.append({"policy": pol, "round": win // 10, "hour": win, "hour_mid": win + 0.5, "requests": a[0], "Y_C": a[1] / a[0], "requests_f1": a[2],
                    "Y_L_f1": a[3] / a[2] if a[2] else "", "Y_D": a[4] / a[0], "requests_q": a[7], "Y_Q": a[6] / a[7] if a[7] else ""})
    write("e4_timeseries.csv", out)


def latency_quantiles():
    out = []
    groups = defaultdict(list)
    exc = defaultdict(lambda: [0, 0])
    for r in rows("e4_requests_reduced.csv.gz"):
        lat = f(r["latency_s"])
        if lat is not None:
            groups[("e4", POLICY_ROUNDS[int(r["round"])], int(r["family"]))].append(lat)
            groups[("e4_class", POLICY_ROUNDS[int(r["round"])] + "|" + r["class"], int(r["family"]))].append(lat)
        e = exc[("e4", POLICY_ROUNDS[int(r["round"])], int(r["family"]))]
        e[0] += 1
        e[1] += int(float(r["exceed"] or 0))
        e = exc[("e4_class", POLICY_ROUNDS[int(r["round"])] + "|" + r["class"], int(r["family"]))]
        e[0] += 1
        e[1] += int(float(r["deadline_violation"] or 0))
    for r in rows("campaign_requests_reduced.csv.gz"):
        lat = f(r["latency_s"])
        if lat is not None:
            groups[("campaign", r["arm"], int(r["family"]))].append(lat)
        e = exc[("campaign", r["arm"], int(r["family"]))]
        e[0] += 1
        e[1] += int(float(r["exceed"] or 0))
    if os.path.exists(E1E5_REQ):
        for r in rows(E1E5_REQ):
            lat = f(r["latency_s"])
            if lat is None and r.get("timeout") == "1":
                lat = float("inf")
            if lat is not None:
                groups[("e1e5", e1e5_arm(r["arm"]), int(r["family"]))].append(lat)
            e = exc[("e1e5", e1e5_arm(r["arm"]), int(r["family"]))]
            e[0] += 1
            e[1] += int(float(r["exceed"] or 0) or r.get("timeout") == "1")
    for (run, unit, fam), xs in sorted(groups.items()):
        above = sum(1 for x in xs if x > TARGET[fam]) / len(xs)
        e = exc[(run, unit, fam)]
        share_exceed = e[1] / e[0] if e[0] else ""
        for p, v in quantiles(xs):
            out.append({"run": run, "unit": unit, "family": fam, "p": p, "latency_s": v if math.isfinite(v) else "inf", "n": len(xs), "share_above_target": above, "share_exceed": share_exceed,
                        "target_s": TARGET[fam]})
    write("latency_quantiles.csv", out)


def cascade():
    acc = defaultdict(lambda: [0, 0, 0])
    for r in rows("campaign_requests_reduced.csv.gz"):
        if r["family"] != "2":
            continue
        for key in ((r["arm"], r["class"]), (r["arm"], "all")):
            a = acc[key]
            a[0] += 1
            a[1] += int(r["served_by"] == "edge")
            a[2] += int(r["escalated"])
    out = [{"arm": arm, "class": c, "requests_f2": a[0], "share_edge": a[1] / a[0], "share_escalated_after_edge": a[2] / a[0]} for (arm, c), a in sorted(acc.items())]
    write("cascade_edge_share.csv", out)


def switching():
    p = os.path.join(DATA, "e1e5_node_blocks.csv")
    if not os.path.exists(p):
        print("skip e1e5_switching: no e1e5_node_blocks.csv")
        return
    to = {r["arm"]: r for r in rows("e1e5_timeouts.csv")} if os.path.exists(os.path.join(DATA, "e1e5_timeouts.csv")) else {}
    acc = defaultdict(lambda: [0, 0.0, 0.0, 0])
    for r in rows(p):
        if r.get("window") != "measure" or r.get("block_status") not in ("complete", "partial"):
            continue
        a = acc[e1e5_arm(r["arm_id"])]
        a[0] += 1
        a[1] += f(r.get("n_switch_epochs_window")) or 0.0
        a[2] += f(r.get("switch_time_window_s")) or 0.0
        a[3] += int(f(r.get("epochs_logged_window")) or 0)
    out = []
    for arm, a in sorted(acc.items()):
        hours = a[3] * 30 / 3600
        t = to.get(arm, {})
        out.append({"arm": arm, "node_blocks": a[0], "measured_hours": hours, "switches_per_hour": a[1] / hours if hours else "", "paused_share": a[2] / (hours * 3600) if hours else "",
                    "timeout_share": f(t.get("share")) if t else ""})
    write("e1e5_switching.csv", out)


def latency_parts():
    if not os.path.exists(E1E5_REQ):
        print("skip e1e5_latency_parts: no reduced request log of the comparison run at", E1E5_REQ)
        return
    parts = ("dev", "up", "edge_wait", "edge_serve", "cloud")
    acc = defaultdict(lambda: {k: [] for k in parts + ("latency_s",)})
    for r in rows(E1E5_REQ):
        if r.get("ok") != "1":
            continue
        g = acc[(e1e5_arm(r["arm"]), int(r["family"]))]
        for k in parts + ("latency_s",):
            v = f(r.get(k))
            g[k].append(v if v is not None else 0.0)
    out = []
    for (arm, fam), g in sorted(acc.items()):
        n = len(g["latency_s"])
        rec = {"arm": arm, "family": fam, "requests_ok": n}
        for k in parts + ("latency_s",):
            xs = sorted(g[k])
            rec[f"mean_{k}"] = sum(xs) / n
            rec[f"median_{k}"] = xs[n // 2]
        out.append(rec)
    write("e1e5_latency_parts.csv", out)


if __name__ == "__main__":
    e4_timeseries()
    latency_quantiles()
    cascade()
    switching()
    latency_parts()
