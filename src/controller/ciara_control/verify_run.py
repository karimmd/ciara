from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

from .config import RunConfig, load_config
from .design import make_design_law
from .schedule import BlockGrid, ContextSchedule
from .util import read_jsonl


class Report:
    def __init__(self):
        self.checks: list = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append({"name": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    @property
    def ok(self) -> bool:
        return all(c["ok"] for c in self.checks)

    def render(self) -> str:
        lines = [f"{'PASS' if c['ok'] else 'FAIL'}  {c['name']}" + (f"  ({c['detail']})" if c["detail"] else "")
                 for c in self.checks]
        n_bad = sum(1 for c in self.checks if not c["ok"])
        lines.append(f"{len(self.checks) - n_bad} of {len(self.checks)} checks passed")
        return "\n".join(lines)


def _p(d: str, name: str) -> str:
    return os.path.join(d, name)


def verify(run_dir: str, cfg: RunConfig, resumed: bool = False, expect_saturation: bool = False,
           monitor_interval: float = 0.5) -> Report:
    rep = Report()
    rows = read_jsonl(_p(run_dir, "blocks.jsonl"))
    blocks = [r for r in rows if r["rec"] == "block"]
    ends = [r for r in rows if r["rec"] == "block_end"]
    epochs = read_jsonl(_p(run_dir, "epochs.jsonl"))
    events = read_jsonl(_p(run_dir, "events.jsonl"))
    with open(_p(run_dir, "manifest.json")) as fh:
        manifest = json.load(fh)
    grid = BlockGrid(cfg.epoch_ns, cfg.block_epochs, cfg.burn_in_epochs, cfg.n_blocks)
    law = make_design_law(cfg)
    ctx = ContextSchedule.load(cfg.context.ref, cfg.context.default)
    node_ids = [n.id for n in cfg.nodes]
    t0 = manifest["t0_mono_ns"]

    with open(_p(run_dir, "STATUS")) as fh:
        rep.check("STATUS is DONE", fh.read().strip() == "DONE")
    keys = [(b["node"], b["block_id"]) for b in blocks]
    rep.check("no duplicate block records", len(keys) == len(set(keys)), f"{len(keys)} records")
    if not resumed:
        rep.check("one block record per node and block", set(keys) == {(n, i) for n in node_ids for i in range(cfg.n_blocks)})
    ekeys = [(e["node"], e["epoch"]) for e in epochs]
    rep.check("no duplicate epoch records", len(ekeys) == len(set(ekeys)), f"{len(ekeys)} records")
    if not resumed:
        rep.check("one epoch record per node and epoch",
                  set(ekeys) == {(n, e) for n in node_ids for e in range(grid.total_epochs)})
    else:
        missing = {(n, e) for n in node_ids for e in range(grid.total_epochs)} - set(ekeys)
        explained = set()
        for x in ends:
            for e in x.get("missed_epochs") or []:
                explained.add((x["node"], e))
        for x in ends:
            if x["status"] == "interrupted":
                for e in range(grid.block_start(x["block_id"]), grid.block_end(x["block_id"])):
                    explained.add((x["node"], e))
        started = {(b["node"], b["block_id"]) for b in blocks}
        for n, e in missing:
            if (n, grid.block_of(e)) not in started:
                explained.add((n, e))
        rep.check("every missing epoch is flagged in a block_end or belongs to an unstarted block",
                  missing <= explained, f"{len(missing)} missing, {len(missing - explained)} unexplained")
    rep.check("burn-in flags follow the grid", all(e["burn_in"] == grid.is_burn_in(e["epoch"]) for e in epochs))
    rep.check("manifest records the grid", manifest["epoch_ns"] == cfg.epoch_ns and manifest["total_epochs"] == grid.total_epochs)

    bad = []
    for b in blocks:
        d = law.draw(b["node_index"], b["block_id"], ctx.context_at(grid.block_start(b["block_id"])))
        if (d.arm_id, d.propensity, d.u) != (b["arm_id"], b["propensity"], b["rng"]["draw"]):
            bad.append((b["node"], b["block_id"]))
    rep.check("every draw re-derived from the seed", not bad, f"{len(blocks)} blocks, {len(bad)} differ")
    counts: dict = {}
    for b in blocks:
        counts[b["arm_id"]] = counts.get(b["arm_id"], 0) + 1
    rep.check("every arm of the design drawn at least once", set(counts) == set(law.arm_ids), json.dumps(counts))

    off = [e for e in epochs if e["t_start_mono_ns"] != t0 + e["epoch"] * cfg.epoch_ns]
    rep.check("epoch starts sit on the monotonic grid", not off, f"{len(off)} off-grid")
    dur = [(e["t_end_mono_ns"] - e["t_start_mono_ns"]) / 1e9 for e in epochs]
    rep.check("epoch durations equal the epoch length (poll jitter below 0.3 s)",
              all(abs(x - cfg.epoch_s) < 0.3 for x in dur), f"min {min(dur):.3f} max {max(dur):.3f} s" if dur else "")
    late = sorted(e["late_ms"] for e in epochs)
    rep.check("boundary work starts within 500 ms of the boundary", late[-1] < 500 if late else False,
              f"median {late[len(late) // 2]:.1f} ms, max {late[-1]:.1f} ms" if late else "")
    act = [(e["t_actuated_mono_ns"] - e["t_start_mono_ns"]) / 1e9 for e in epochs if e["t_actuated_mono_ns"]]
    rep.check("actuation finishes inside its epoch", all(x < cfg.epoch_s for x in act),
              f"max {max(act):.2f} s" if act else "")
    rep.check("no epoch recorded actuation errors", not [e for e in epochs if e["errors"]],
              f"{sum(1 for e in epochs if e['errors'])} epochs with errors")

    mgr = read_jsonl(_p(run_dir, "mock-manager.jsonl"))
    gws = read_jsonl(_p(run_dir, "mock-gateway.jsonl"))
    if mgr or gws:
        seq_ok, order_ok, match_ok, switch_ok = True, True, True, True
        n_act_total = n_sw_epochs = 0
        for n in node_ids:
            acts = sorted((m for m in mgr if m["node"] == n and m["ev"] == "activate"), key=lambda m: m["t_mono_ns"])
            cfgs = sorted((g for g in gws if g["node"] == n and g["ev"] == "config"), key=lambda g: g["t_mono_ns"])
            prev_prof, prev_t, ai = None, 0, 0
            for g in cfgs:
                prof = g["config"]["profile"]
                if prof != prev_prof:
                    cand = [a for a in acts if a["profile"] == prof and prev_t < a["t_mono_ns"] <= g["t_mono_ns"]]
                    if not cand:
                        order_ok = False
                prev_prof, prev_t = prof, g["t_mono_ns"]
            prof_changes = sum(1 for i, g in enumerate(cfgs) if i == 0 or g["config"]["profile"] != cfgs[i - 1]["config"]["profile"])
            if not resumed:
                seq_ok &= (len(acts) == prof_changes)
            n_act_total += len(acts)
            for b in (x for x in blocks if x["node"] == n):
                hit = [g for g in cfgs if g["config"]["block_id"] == b["block_id"] and g["config"]["arm_id"] == b["arm_id"]]
                if not hit or not (b["t_mono_ns"] <= hit[0]["t_mono_ns"] < b["t_mono_ns"] + cfg.epoch_ns):
                    match_ok = False
            logged = sorted(e["switch_time_s"] for e in epochs if e["node"] == n and e["switched"])
            n_sw_epochs += len(logged)
            reported = sorted(a["switch_time_s"] for a in acts)
            if not resumed and logged != reported:
                switch_ok = False
        rep.check("activate calls equal profile changes seen by the gateway", seq_ok, f"{n_act_total} activates")
        rep.check("every profile change: activate completed before the gateway config", order_ok)
        rep.check("every block start reached the gateway with its arm and block id, inside its epoch", match_ok)
        rep.check("switch times logged by the controller equal the manager's", switch_ok, f"{n_sw_epochs} switches")
        mid = [e for e in epochs if e["switched"] and e["epoch_in_block"] > 0]
        rep.check("profile switches also happen mid-block (incumbent regime arm)", len(mid) > 0, f"{len(mid)} mid-block switches")

    inc = [e for e in epochs if e["kind"] == "incumbent"]
    rep.check("incumbent epochs carry the observation given to decide()",
              bool(inc) and all(e["obs"] and e["obs"].get("gateway") is not None and "arrival_rate" in e["obs"] for e in inc),
              f"{len(inc)} incumbent epochs")
    rep.check("telemetry present in epoch records",
              all(e["telemetry"]["gateway_ok"] and e["telemetry"]["engine_ok"] for e in epochs))

    cg = read_jsonl(_p(run_dir, "cloud-gw.jsonl"))
    if cg:
        lg = read_jsonl(_p(run_dir, "mock-loadgen.jsonl"))
        ok_client = sum(1 for r in lg if r["status"] == 200 and not r["error"])
        rep.check("cloud-gw logged every request the clients completed", len(cg) >= ok_client and ok_client > 0,
                  f"{len(cg)} logged, {ok_client} completed at the clients, {len(lg)} sent")
        rep.check("all cloud-gw requests succeeded", all(r["http_status"] == 200 and not r["error"] for r in cg))
        rep.check("cloud-gw records name an originating node of the run", {r["node"] for r in cg} <= set(node_ids),
                  ",".join(sorted({r["node"] for r in cg})))
        rep.check("cloud-gw time fields ordered t_arrive <= t_admit <= t_first_token <= t_done",
                  all(r["t_arrive"] <= r["t_admit"] <= r["t_first_token"] <= r["t_done"] for r in cg))
        srt = sorted(cg, key=lambda r: r["t_arrive"])
        adm = [r["t_admit"] for r in srt]
        rep.check("cloud-gw admission is FIFO", adm == sorted(adm))
        pts = sorted([(r["t_admit"], 1) for r in cg] + [(r["t_done"], -1) for r in cg], key=lambda x: (x[0], x[1]))
        cur = peak = 0
        for _, dlt in pts:
            cur += dlt
            peak = max(peak, cur)
        rep.check("cloud-gw has at most 16 in flight (from its own log)", peak <= 16, f"peak {peak}")
        if expect_saturation:
            rep.check("the burst saturated the bound (peak is 16) and queued requests", peak == 16 and max(r["queue_len_at_arrival"] for r in cg) > 0,
                      f"max queue at arrival {max(r['queue_len_at_arrival'] for r in cg)}")
        stats_path = _p(run_dir, "mock-vllm-stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as fh:
                st = json.load(fh)
            rep.check("the engine itself saw at most 16 concurrent requests", st["max_concurrent"] <= 16,
                      f"engine max {st['max_concurrent']}, auth failures {st['auth_failures']}")
            rep.check("every upstream request carried the run token", st["auth_failures"] == 0)
        rep.check("no usage event leaked to clients that did not ask for it", not any(r["has_usage_event"] for r in lg))
        by_node = {n: sorted((b for b in blocks if b["node"] == n), key=lambda b: b["t_mono_ns"]) for n in node_ids}
        checked = wrong = 0
        for r in cg:
            bl = by_node.get(r["node"], [])
            settled = [b for b in bl if b["t_mono_ns"] + 1_000_000_000 <= r["t_arrive"]]
            fresh = [b for b in bl if r["t_arrive"] - 1_000_000_000 < b["t_mono_ns"] <= r["t_arrive"]]
            if not settled or fresh:
                continue
            checked += 1
            wrong += 0 if (r["block_id"], r["arm_id"]) == (settled[-1]["block_id"], settled[-1]["arm_id"]) else 1
        rep.check("cloud-gw records carry the node's arm and block id as pushed by the controller",
                  checked > 0 and wrong <= 0.02 * checked, f"{checked} checked, {wrong} inconsistent")

    gpu = read_jsonl(_p(run_dir, "monitor-gpu.jsonl"))
    cpu = read_jsonl(_p(run_dir, "monitor-cpu.jsonl"))
    vl = read_jsonl(_p(run_dir, "monitor-vllm.jsonl"))
    if gpu:
        samples = [g for g in gpu if g["event"] == "sample"]
        span = (samples[-1]["mono_ns"] - samples[0]["mono_ns"]) / 1e9 if len(samples) > 1 else 0
        gaps = [(b["mono_ns"] - a["mono_ns"]) / 1e9 for a, b in zip(samples, samples[1:])]
        rep.check("monitor-gpu samples at about 2 Hz", span > 0 and len(samples) >= 0.9 * span / monitor_interval,
                  f"{len(samples)} samples over {span:.0f} s, max gap {max(gaps):.2f} s")
        rep.check("monitor-gpu has no gap above 1 s", max(gaps) < 1.0)
        keys_needed = {"power_w", "util_gpu", "mem_used_mib", "sm_clock_mhz", "throttle_bits", "energy_mj"}
        rep.check("monitor-gpu records power, energy, utilization, memory, clocks and throttle bits",
                  keys_needed <= set(samples[0]["gpus"][0]), f"{len(samples[0]['gpus'])} gpus")
    if cpu:
        gaps = [(b["mono_ns"] - a["mono_ns"]) / 1e9 for a, b in zip(cpu, cpu[1:])]
        rep.check("monitor-cpu per-core CPU time sampled at about 2 Hz",
                  all(k in cpu[0] for k in ("cores", "busy_s", "own_s", "external_s")) and max(gaps) < 1.5,
                  f"{len(cpu)} records over {len(cores_of(cpu))} cores, max gap {max(gaps):.2f} s")
    if vl:
        scrapes = [v for v in vl if v["event"] == "vllm"]
        epochs_seen = sorted({v["epoch"] for v in scrapes})
        per_epoch = {}
        for v in scrapes:
            per_epoch.setdefault(v["epoch"], []).append(v)
        full = [k for k, vs in per_epoch.items() if len(vs) == max(len(x) for x in per_epoch.values())]
        rep.check("monitor-vllm scrapes every engine once per epoch", len(epochs_seen) >= 0.8 * grid.total_epochs and
                  len(full) >= 0.8 * len(epochs_seen), f"{len(epochs_seen)} epochs, {len(scrapes)} records")
        act_per_node = {}
        for v in per_epoch[epochs_seen[len(epochs_seen) // 2]]:
            if v["active"]:
                act_per_node[v["node"]] = act_per_node.get(v["node"], 0) + 1
        rep.check("exactly one active engine per node in a scrape", all(c == 1 for c in act_per_node.values()),
                  json.dumps(act_per_node))
        rep.check("monitor-vllm series are vllm: metrics", all(k.startswith("vllm:") for v in scrapes[:20] for k in v["series"]))
        rep.check("monitor-vllm has no scrape errors", not [v for v in vl if v["event"] == "vllm_error"])
    return rep


def cores_of(cpu: list) -> list:
    return cpu[0]["cores"]


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Check the logs of a dry run against the mock endpoints' own logs (end-to-end verification).")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--resumed", action="store_true")
    ap.add_argument("--expect-saturation", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    rep = verify(args.run_dir, cfg, resumed=args.resumed, expect_saturation=args.expect_saturation)
    print(rep.render())
    with open(os.path.join(args.run_dir, "verify.json"), "w") as fh:
        json.dump(rep.checks, fh, indent=1)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
