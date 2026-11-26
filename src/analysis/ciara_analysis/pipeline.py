from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence

from .energy_bridge import device_energy
from .join import build_requests
from .logs import RunLogs
from .outcomes import OUTCOME_KEYS, build_tables
from .quality import default_scorers


def _cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, float):
        return repr(round(v, 9))
    if isinstance(v, (list, tuple)):
        return "+".join(str(x) for x in v)
    if isinstance(v, bool):
        return "true" if v else "false"
    return v


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([_cell(r.get(c)) for c in cols])


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":"), default=str) + "\n")


def parse_targets(items: Sequence[str]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for it in items:
        k, v = it.split("=", 1)
        out[int(k.strip().lower().lstrip("f"))] = float(v)
    return out


def run(run_dir: str, schedule_dir: str, latency_targets_s: Dict[int, float], out_dir: Optional[str] = None,
        scorers: Optional[Dict[int, Callable]] = None, energy: Optional[Dict[str, Dict[str, Any]]] = None,
        energy_config: Optional[str] = None, node_gpu: Optional[Dict[str, int]] = None, cloud_gpus: Sequence[int] = (),
        truncated_is_wrong: bool = True) -> Dict[str, Any]:
    logs = RunLogs(run_dir)
    manifest = logs.manifest()
    blocks = logs.blocks()
    epochs = logs.epochs()
    scorers = scorers if scorers is not None else default_scorers()
    if energy is None:
        energy = device_energy(run_dir, schedule_dir, energy_config, overhead="constant")
    requests, report = build_requests(logs.client_records(), logs.edge_records(), logs.cloud_records(), blocks, manifest,
                                      scorers, energy, latency_targets_s, truncated_is_wrong)
    gpu = logs.monitor_gpu() if (node_gpu or cloud_gpus) else None
    meas, burn = build_tables(requests, blocks, epochs, manifest, gpu, node_gpu, cloud_gpus)
    out_dir = out_dir or os.path.join(run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    write_csv(os.path.join(out_dir, "outcomes.csv"), meas)
    write_jsonl(os.path.join(out_dir, "outcomes.jsonl"), meas)
    write_csv(os.path.join(out_dir, "outcomes_burnin.csv"), burn)
    write_jsonl(os.path.join(out_dir, "outcomes_burnin.jsonl"), burn)
    write_jsonl(os.path.join(out_dir, "requests.jsonl"), requests)
    report["parameters"] = {
        "latency_targets_s": {str(k): v for k, v in latency_targets_s.items()}, "truncated_is_wrong": truncated_is_wrong,
        "node_gpu": node_gpu, "cloud_gpus": list(cloud_gpus), "run_dir": run_dir, "schedule_dir": schedule_dir,
        "grid": {k: manifest[k] for k in ("t0_mono_ns", "epoch_ns", "block_epochs", "burn_in_epochs", "n_blocks")},
    }
    report["n_measure_rows"], report["n_burnin_rows"] = len(meas), len(burn)
    nulls = {k: sum(1 for r in meas if r.get(k) is None) for k in OUTCOME_KEYS}
    report["null_outcomes_measure"] = nulls
    with open(os.path.join(out_dir, "join_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=str)
    return {"requests": requests, "measure": meas, "burn_in": burn, "report": report, "out_dir": out_dir}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Outcome pipeline: joins the logs of a run and writes the per request table and the block outcome tables.', formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--schedule", required=True, help="schedule directory of the run (schedule.jsonl: classes and links)")
    ap.add_argument("--latency-target", action="append", default=[], metavar="FAMILY=SECONDS",
                    help="latency target of a task family, e.g. 1=8 (repeat for each family); set by the Stage 1 measurement")
    ap.add_argument("--energy-config", default=None)
    ap.add_argument("--node-gpu", action="append", default=[], metavar="NODE=GPU", help="e.g. E1=0 (serving energy of the node)")
    ap.add_argument("--cloud-gpu", action="append", type=int, default=[], help="GPU index of the cloud engine")
    ap.add_argument("--keep-truncated", action="store_true", help="score replies cut off at the cap by their content (default: wrong)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    targets = parse_targets(a.latency_target)
    if not targets:
        print("at least one --latency-target FAMILY=SECONDS is needed", file=sys.stderr)
        return 2
    node_gpu = {k: int(v) for k, v in (x.split("=", 1) for x in a.node_gpu)}
    res = run(a.run_dir, a.schedule, targets, a.out, energy_config=a.energy_config, node_gpu=node_gpu or None,
              cloud_gpus=a.cloud_gpu, truncated_is_wrong=not a.keep_truncated)
    rep = res["report"]
    print(f"requests {rep['n_rows']}, edge matches {rep['n_found_edge']}, cloud matches {rep['n_found_cloud']}; "
          f"{rep['n_measure_rows']} measurement rows, {rep['n_burnin_rows']} burn-in rows -> {res['out_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
