from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from . import layout

IMAGE = "ciara-client:run"
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sh(args: Sequence[str], check: bool = True) -> subprocess.CompletedProcess:
    cp = subprocess.run(list(args), capture_output=True, text=True)
    if check and cp.returncode != 0:
        raise RuntimeError(f"{' '.join(args)}: {cp.stderr.strip()[:1500]}")
    return cp


def read_meta(schedule_dir: str) -> Dict[str, Any]:
    with open(os.path.join(schedule_dir, "schedule.jsonl"), "r", encoding="utf-8") as f:
        return json.loads(f.readline())


def start(a: argparse.Namespace) -> Dict[str, Any]:
    compact = os.path.abspath(a.compact_bank) if a.compact_bank else None
    if a.wire == "compact" and not (compact and os.path.isfile(compact)):
        raise SystemExit("--wire compact needs --compact-bank FILE (ciara_edge.compose_build output); use --wire full for the whole prompt")
    meta = read_meta(a.schedule)
    nodes = [int(x) for x in a.nodes.split(",")] if a.nodes else list(range(1, int(meta["nodes"]) + 1))
    classes = a.classes.split(",") if a.classes else sorted(meta["classes"])
    run_dir = os.path.join(a.runs_root, a.run_id)
    out_dir = os.path.join(run_dir, "clients")
    os.makedirs(out_dir, exist_ok=True)
    sched = os.path.abspath(a.schedule)
    nets = sh(["docker", "network", "ls", "--format", "{{.Name}}"]).stdout.split()
    if a.network not in nets:
        raise SystemExit(f"docker network {a.network!r} does not exist")
    start_at = int(a.start_at_mono_ns) if a.start_at_mono_ns else time.monotonic_ns() + int(a.lead_s * 1e9)
    uid, gid = os.getuid(), os.getgid()
    created: List[Dict[str, Any]] = []
    for node in nodes:
        for cls in classes:
            name = layout.container_name(node, cls, a.prefix)
            cpu = layout.client_cpu(node, cls)
            cmd = ["docker", "run", "-d", "--name", name, "--network", a.network, "--cap-add=NET_ADMIN",
                   *(["--cpuset-cpus", str(cpu)] if cpu is not None else []), "--memory", a.memory, "--pids-limit", "512",
                   "--label", f"ciara.run={a.run_id}", "--label", "ciara.component=clients",
                   "-v", f"{CODE_DIR}:/app:ro", "-v", f"{sched}:/sched:ro", "-v", f"{out_dir}:/out",
                   "-e", "PYTHONPATH=/app", "-e", f"CIARA_UID={uid}", "-e", f"CIARA_GID={gid}"]
            if os.environ.get("CIARA_TOKEN"):
                cmd += ["-e", "CIARA_TOKEN"]
            if a.wire == "compact":
                cmd += ["-v", f"{compact}:/compact/compact-bank.jsonl:ro"]
                bank_args = ["--wire", "compact", "--compact-bank", "/compact/compact-bank.jsonl"]
            else:
                bank_args = ["--wire", "full", "--bank", "/sched/payloads.jsonl"]
            cmd += [IMAGE, "python", "-m", "ciara_clients.runtime", "--schedule", "/sched/schedule.jsonl",
                    *bank_args, "--node", str(node), "--class", cls, "--out", "/out",
                    "--gateway", a.gateway_fmt.format(n=node), "--start-at-mono-ns", str(start_at),
                    "--grace-s", str(a.grace_s)]
            sh(["docker", "rm", "-f", name], check=False)
            sh(cmd)
            created.append({"name": name, "node": node, "class": cls, "cpu": cpu})
    rec = {"run_id": a.run_id, "wire": a.wire, "compact_bank": compact, "schedule_dir": sched,
           "schedule_manifest": os.path.join(sched, "manifest.json"),
           "network": a.network, "image": IMAGE, "start_at_mono_ns": start_at,
           "start_unix": time.time() + (start_at - time.monotonic_ns()) / 1e9,
           "duration_s": meta["duration_s"], "grace_s": a.grace_s, "containers": created}
    with open(os.path.join(run_dir, "clients.launch.json"), "w") as f:
        json.dump(rec, f, indent=1)
    return rec


def load_launch(a: argparse.Namespace) -> Dict[str, Any]:
    with open(os.path.join(a.runs_root, a.run_id, "clients.launch.json")) as f:
        return json.load(f)


def status(a: argparse.Namespace) -> List[Dict[str, Any]]:
    rec = load_launch(a)
    rows = []
    for c in rec["containers"]:
        cp = sh(["docker", "inspect", c["name"], "--format", "{{.State.Status}} {{.State.ExitCode}}"], check=False)
        rows.append({"name": c["name"], "state": cp.stdout.strip() or "absent"})
    return rows


def stop(a: argparse.Namespace) -> None:
    rec = load_launch(a)
    for c in rec["containers"]:
        sh(["docker", "rm", "-f", c["name"]], check=False)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Host-side launcher of the client containers (docker CLI through subprocess, stdlib only).', formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--run-id", required=True)
    s.add_argument("--schedule", required=True, help="directory with schedule.jsonl, manifest.json (and payloads.jsonl for --wire full)")
    s.add_argument("--runs-root", required=True)
    s.add_argument("--network", default="ciara-net")
    s.add_argument("--nodes", default=None, help="comma list, default all nodes of the schedule")
    s.add_argument("--classes", default=None, help="comma list, default all classes")
    s.add_argument("--wire", choices=["compact", "full"], default="compact",
                   help="compact: devices upload their own content only; full: the whole prompt in a chat body")
    s.add_argument("--compact-bank", default=None, help="compact bank file (compose_build output), mounted read only into the containers")
    s.add_argument("--gateway-fmt", default="http://edge-gw-{n}:8000")
    s.add_argument("--lead-s", type=float, default=30.0)
    s.add_argument("--start-at-mono-ns", type=int, default=None,
                   help="schedule time 0 on the shared CLOCK_MONOTONIC (default: now plus --lead-s)")
    s.add_argument("--grace-s", type=float, default=300.0)
    s.add_argument("--memory", default="512m")
    s.add_argument("--prefix", default="client")
    for name in ("status", "stop"):
        p = sub.add_parser(name)
        p.add_argument("--run-id", required=True)
        p.add_argument("--runs-root", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "start":
        print(json.dumps(start(a), indent=1))
    elif a.cmd == "status":
        for r in status(a):
            print(r["name"], r["state"])
    elif a.cmd == "stop":
        stop(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
