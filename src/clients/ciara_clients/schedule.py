from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import platform
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import bank as bankmod
from . import config as cfgmod
from .mmpp import MMPPSpec, arrivals_from_path, poisson_arrivals, realized_rates, simulate_path, state_rates, stationary
from .rng import gamma_mean1, pick, pick_u, stream

GEN_VERSION = "ciara-schedule-gen/1"
SCHEMA = "ciara-schedule/1"
PORT_BASE = 10240
PORT_BLOCK = 64


def device_id(node: int, cls: str, idx: int) -> str:
    return f"n{node}{cls}{idx:03d}"


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _public_link(link: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in link.items() if k not in ("src",)}


def context_plan(cfg: Dict[str, Any], horizon_s: float) -> Dict[str, Any]:
    ctx = cfg.get("context") or {"epoch_s": 30, "block_epochs": 50, "mode": "pooled", "pooled_id": "pooled"}
    epoch_s, be = float(ctx["epoch_s"]), int(ctx["block_epochs"])
    n_epochs = int(math.ceil(horizon_s / epoch_s - 1e-9))
    n_blocks = max(1, -(-n_epochs // be))
    if ctx["mode"] == "pooled":
        ids = [ctx["pooled_id"]] * n_blocks
        factors = [1.0] * n_blocks
        default = ctx["pooled_id"]
    else:
        names = sorted(ctx["level_factor"])
        pattern = ctx.get("pattern")
        if pattern:
            ids = [pattern[b % len(pattern)] for b in range(n_blocks)]
        else:
            r = stream(int(cfg["run"]["seed"]), "context")
            probs = [float(ctx["level_prob"].get(n, 0.0)) for n in names]
            ids = [names[pick(r, probs)] for _ in range(n_blocks)]
        factors = [float(ctx["level_factor"][i]) for i in ids]
        default = ctx["default_id"]
    segments: List[Dict[str, Any]] = []
    for b, cid in enumerate(ids):
        if segments and segments[-1]["context_id"] == cid:
            segments[-1]["end_epoch"] = (b + 1) * be
        else:
            segments.append({"start_epoch": b * be, "end_epoch": (b + 1) * be, "context_id": cid})
    return {"epoch_s": epoch_s, "block_epochs": be, "mode": ctx["mode"], "horizon_epochs": n_epochs, "n_blocks": n_blocks,
            "block_ids": ids, "block_factor": factors, "bounds_s": [b * be * epoch_s for b in range(n_blocks + 1)],
            "file": {"default_context": default, "segments": segments, "epoch_s": epoch_s, "block_epochs": be,
                     "mode": ctx["mode"], "horizon_epochs": n_epochs, "covered_epochs": n_blocks * be}}


def build(cfg: Dict[str, Any], items_by_family: Dict[int, List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    run = cfg["run"]
    seed = int(run["seed"])
    horizon = float(run["duration_s"])
    scale = float(run["load_scale"])
    arrival = run["arrival"]
    nodes = int(run["nodes"])
    mm = cfg["mmpp"]
    spec = MMPPSpec(list(mm["states"]), [float(x) for x in mm["rel_rate"]], [float(x) for x in mm["mean_dwell_s"]],
                    mm.get("jump"))
    shared = mm["modulation"] == "shared"
    cv = float(mm["rate_heterogeneity_cv"])
    calibrate = mm.get("calibrate", "realized")
    classes = sorted(cfg["classes"])
    plan = context_plan(cfg, horizon)
    use_scale = any(f != 1.0 for f in plan["block_factor"])
    bounds = plan["bounds_s"] if use_scale else None
    scales = plan["block_factor"] if use_scale else None

    meta = {
        "type": "meta", "schema": SCHEMA, "generator": GEN_VERSION,
        "seed": seed, "duration_s": horizon, "arrival": arrival, "load_scale": scale, "nodes": nodes,
        "preset": run.get("preset"), "model": run["model"], "config_sha256": cfg["_sha256"],
        "request": cfg["request"],
        "families": {k: {"suite": v["suite"], "max_tokens": v["max_tokens"]} for k, v in cfg["families"].items()},
        "mmpp": {"states": spec.states, "rel_rate": spec.rel_rate, "mean_dwell_s": spec.mean_dwell_s,
                 "stationary": stationary(spec), "modulation": mm["modulation"], "rate_heterogeneity_cv": cv,
                 "calibrate": calibrate},
        "classes": {c: {k: v for k, v in cfg["classes"][c].items()} for c in classes},
        "links": {n: _public_link(l) for n, l in cfg["links"].items()},
        "port_base": PORT_BASE, "port_block": PORT_BLOCK,
        "context": {k: plan[k] for k in ("epoch_s", "block_epochs", "mode", "block_ids", "block_factor")},
    }

    groups: List[Dict[str, Any]] = []
    devices: List[Dict[str, Any]] = []
    reqs: List[Dict[str, Any]] = []
    fam_keys = sorted(cfg["families"], key=int)

    for node in range(1, nodes + 1):
        for cls in classes:
            cl = cfg["classes"][cls]
            n = int(cl["devices_per_node"])
            base = float(cl["base_rate_hz"]) * scale
            rates = state_rates(spec, base) if base > 0 else [0.0] * spec.k
            link_names = list(cl["link_mix"])
            link_w = [float(cl["link_mix"][x]) for x in link_names]
            fam_w = [float(cl["family_mix"][f]) for f in fam_keys]
            ids = [device_id(node, cls, i) for i in range(n)]
            mult = [gamma_mean1(stream(seed, "device", d, "rate"), cv) for d in ids]
            mean_m = sum(mult) / n if n else 1.0
            mult = [m / mean_m for m in mult]
            gpath: Optional[List[Tuple[float, int]]] = None
            if arrival == "mmpp" and shared:
                gpath = simulate_path(stream(seed, "mmpp", "chain", node, cls), spec, horizon)
                if calibrate == "realized" and base > 0:
                    rates = realized_rates(spec, base, gpath, horizon)
                groups.append({"type": "group", "node": node, "class": cls, "rates_hz": [round(r, 9) for r in rates],
                               "path": [[round(t, 3), s] for t, s in gpath]})
            for i, did in enumerate(ids):
                link = link_names[pick(stream(seed, "device", did, "link"), link_w)]
                path = gpath
                dev: Dict[str, Any] = {"type": "device", "device": did, "node": node, "class": cls, "idx": i, "link": link,
                                       "rate_mult": round(mult[i], 6)}
                if arrival == "mmpp":
                    if path is None:
                        path = simulate_path(stream(seed, "device", did, "chain"), spec, horizon)
                        dev["path"] = [[round(t, 3), s] for t, s in path]
                    dev_rates = realized_rates(spec, base, path, horizon) if (calibrate == "realized" and base > 0 and gpath is None) else rates
                    if gpath is None:
                        dev["rates_hz"] = [round(r, 9) for r in dev_rates]
                    times = arrivals_from_path(stream(seed, "device", did, "arrivals"), path, dev_rates, horizon, mult[i], bounds, scales)
                    starts = [p[0] for p in path]
                else:
                    times = poisson_arrivals(stream(seed, "device", did, "arrivals-poisson"), base * mult[i], horizon, bounds, scales)
                    starts = []
                items_rng = stream(seed, "device", did, "items")
                dev["n_req"] = len(times)
                devices.append(dev)
                for k, t in enumerate(times):
                    u_f, u_i = items_rng.random(), items_rng.random()
                    fam = int(fam_keys[pick_u(u_f, fam_w)])
                    items = items_by_family[fam]
                    item = items[min(len(items) - 1, int(u_i * len(items)))]
                    dl = cl["deadline_ms"].get(f"family_{fam}", cl["deadline_ms"]["default"])
                    st = path[bisect.bisect_right(starts, t) - 1][1] if starts else -1
                    reqs.append({"type": "req", "t_ms": round(t * 1000.0, 1), "req": f"{did}-{k:05d}", "device": did,
                                 "node": node, "class": cls, "family": fam, "item": item["id"],
                                 "deadline_ms": int(dl), "state": st})
    reqs.sort(key=lambda r: (r["t_ms"], r["device"], r["req"]))
    stats = _stats(cfg, reqs, devices, horizon)
    per_ctx: Dict[str, Dict[str, Any]] = {}
    block_s = plan["bounds_s"][1] - plan["bounds_s"][0]
    for b, cid in enumerate(plan["block_ids"]):
        lo, hi = plan["bounds_s"][b], min(plan["bounds_s"][b + 1], horizon)
        d = per_ctx.setdefault(cid, {"blocks": 0, "seconds": 0.0, "requests": 0})
        d["blocks"] += 1
        d["seconds"] += max(0.0, hi - lo)
    for r in reqs:
        per_ctx[plan["block_ids"][min(plan["n_blocks"] - 1, int(r["t_ms"] / 1000.0 // block_s))]]["requests"] += 1
    for cid, d in per_ctx.items():
        d["rate_per_node_hz"] = round(d["requests"] / d["seconds"] / cfg["run"]["nodes"], 5) if d["seconds"] > 0 else None
    stats["context"] = {"mode": plan["mode"], "per_context": per_ctx}
    stats["context_plan"] = plan["file"]
    return [meta] + groups + devices + reqs, stats


def _stats(cfg: Dict[str, Any], reqs: Sequence[Dict[str, Any]], devices: Sequence[Dict[str, Any]], horizon: float) -> Dict[str, Any]:
    win = float(cfg["run"].get("mmpp_bin_s", 60))
    nb = max(1, int(horizon // win))
    per_group: Dict[str, List[int]] = {}
    fam_by_class: Dict[str, Dict[int, int]] = {}
    link_count: Dict[str, int] = {}
    for d in devices:
        link_count[d["link"]] = link_count.get(d["link"], 0) + 1
    for r in reqs:
        g = f"n{r['node']}{r['class']}"
        per_group.setdefault(g, [0] * nb)
        w = int(r["t_ms"] / 1000.0 // win)
        if w < nb:
            per_group[g][w] += 1
        fam_by_class.setdefault(r["class"], {}).setdefault(r["family"], 0)
        fam_by_class[r["class"]][r["family"]] += 1
    cvs: Dict[str, float] = {}
    for g, cnt in per_group.items():
        m = sum(cnt) / nb
        var = sum((c - m) ** 2 for c in cnt) / nb
        cvs[g] = round(math.sqrt(var) / m, 3) if m > 0 else float("nan")
    by_class: Dict[str, Any] = {}
    for c in sorted(cfg["classes"]):
        n_c = sum(1 for r in reqs if r["class"] == c)
        n_dev = sum(1 for d in devices if d["class"] == c)
        by_class[c] = {"devices": n_dev, "requests": n_c, "mean_rate_hz_per_node": round(n_c / horizon / cfg["run"]["nodes"], 5),
                       "family_counts": fam_by_class.get(c, {})}
    return {"requests": len(reqs), "devices": len(devices), "by_class": by_class, "link_profile_devices": link_count,
            "count_cv_per_group_window": {"window_s": win, "values": cvs,
                                          "mean": round(sum(v for v in cvs.values() if v == v) / max(1, len(cvs)), 3)}}


def write_schedule(path: str, records: Sequence[Dict[str, Any]]) -> str:
    h = hashlib.sha256()
    with open(path, "wb") as f:
        for r in records:
            b = (dumps(r) + "\n").encode("utf-8")
            f.write(b)
            h.update(b)
    return h.hexdigest()


def generate(cfg: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    items = bankmod.load_family_items(cfg)
    records, stats = build(cfg, items)
    sched_path = os.path.join(out_dir, "schedule.jsonl")
    sched_sha = write_schedule(sched_path, records)
    ctx_path = os.path.join(out_dir, "context.json")
    ctx_text = json.dumps(stats["context_plan"], indent=1, sort_keys=True) + "\n"
    with open(ctx_path, "w", encoding="utf-8") as f:
        f.write(ctx_text)
    ctx_sha = hashlib.sha256(ctx_text.encode("utf-8")).hexdigest()
    referenced = {r["item"] for r in records if r["type"] == "req"}
    bank_path = os.path.join(out_dir, "payloads.jsonl")
    bank_sha = bankmod.write_bank(bank_path, items, referenced)
    cfg_public = {k: v for k, v in cfg.items() if not k.startswith("_")}
    manifest = {
        "schema": "ciara-schedule-manifest/1",
        "schedule_id": sched_sha[:12],
        "generator": GEN_VERSION,
        "python": platform.python_version(),
        "files": {
            "schedule.jsonl": {"sha256": sched_sha, "bytes": os.path.getsize(sched_path), "records": len(records)},
            "context.json": {"sha256": ctx_sha, "bytes": len(ctx_text.encode("utf-8")), "segments": len(stats["context_plan"]["segments"])},
            "payloads.jsonl": {"sha256": bank_sha, "bytes": os.path.getsize(bank_path), "items": len(referenced)},
        },
        "config": {"path": cfg["_path"], "file_sha256": cfg["_sha256"], "resolved": cfg_public},
        "frozen_data": {"pilot_data_manifest_sha256": bankmod.data_manifest_sha256(), "split_files_sha256": bankmod.split_sha256()},
        "run": cfg["run"],
        "stats": {k: v for k, v in stats.items() if k != "context_plan"},
        "exogenous": "Arrival times, items, families, deadlines and link profiles are functions of (seed, config) only; "
                     "no serving outcome enters the generator.",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest


def verify(out_dir: str) -> bool:
    with open(os.path.join(out_dir, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    ok = True
    for name, info in man["files"].items():
        h = cfgmod.sha256_file(os.path.join(out_dir, name))
        good = h == info["sha256"]
        ok &= good
        print(f"{name}: {'ok' if good else 'MISMATCH'} {h[:16]}")
    cfg = json.loads(json.dumps(man["config"]["resolved"]))
    cfg["_sha256"] = man["config"]["file_sha256"]
    cfg["_path"] = man["config"]["path"]
    items = bankmod.load_family_items(cfg)
    records, vstats = build(cfg, items)
    h = hashlib.sha256()
    for r in records:
        h.update((dumps(r) + "\n").encode("utf-8"))
    same = h.hexdigest() == man["files"]["schedule.jsonl"]["sha256"]
    ctx_text = json.dumps(vstats["context_plan"], indent=1, sort_keys=True) + "\n"
    ctx_same = hashlib.sha256(ctx_text.encode("utf-8")).hexdigest() == man["files"]["context.json"]["sha256"]
    print(f"regenerated context.json: {'identical' if ctx_same else 'DIFFERENT'}")
    same = same and ctx_same
    print(f"regenerated schedule: {'identical' if same else 'DIFFERENT'} {h.hexdigest()[:16]}")
    data_ok = bankmod.data_manifest_sha256() == man["frozen_data"]["pilot_data_manifest_sha256"]
    print(f"frozen data manifest: {'unchanged' if data_ok else 'CHANGED'}")
    return ok and same and data_ok


def read_schedule(path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"meta": None, "groups": [], "devices": [], "reqs": []}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            t = r["type"]
            if t == "meta":
                out["meta"] = r
            elif t == "group":
                out["groups"].append(r)
            elif t == "device":
                out["devices"].append(r)
            elif t == "req":
                out["reqs"].append(r)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Schedule generator: offline, seeded, deterministic, independent of any serving outcome.', formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--out", required=True)
    g.add_argument("--config", default=None)
    g.add_argument("--preset", default=None)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--duration-s", type=float, default=None)
    g.add_argument("--arrival", choices=["mmpp", "poisson"], default=None)
    g.add_argument("--nodes", type=int, default=None)
    g.add_argument("--load-scale", type=float, default=None)
    g.add_argument("--context-mode", choices=["pooled", "levels"], default=None)
    g.add_argument("--block-epochs", type=int, default=None, help="controller block length in epochs (contexts change on block boundaries)")
    g.add_argument("--context-pattern", default=None, help="comma list of level names, repeated over the blocks (levels mode)")
    v = sub.add_parser("verify")
    v.add_argument("dir")
    a = ap.parse_args(argv)
    if a.cmd == "gen":
        ctx = {"mode": a.context_mode, "block_epochs": a.block_epochs,
               "pattern": a.context_pattern.split(",") if a.context_pattern else None}
        cfg = cfgmod.load(a.config, a.preset, {"seed": a.seed, "duration_s": a.duration_s, "arrival": a.arrival,
                                               "nodes": a.nodes, "load_scale": a.load_scale}, ctx)
        be_s = float(cfg["context"]["epoch_s"]) * int(cfg["context"]["block_epochs"])
        if (float(cfg["run"]["duration_s"]) / be_s) % 1 > 1e-9:
            print(f"warning: duration {cfg['run']['duration_s']} s is not a multiple of the block length {be_s:g} s; "
                  "the last context segment is padded to a full block", file=sys.stderr)
        man = generate(cfg, a.out)
        print(json.dumps({"schedule_id": man["schedule_id"], "files": man["files"], "stats": {
            "requests": man["stats"]["requests"], "devices": man["stats"]["devices"],
            "count_cv_mean": man["stats"]["count_cv_per_group_window"]["mean"]}}, indent=2))
        return 0
    if a.cmd == "verify":
        return 0 if verify(a.dir) else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
