from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from typing import Optional

import yaml

from .config import load_config
from .verify_run import verify

TOKEN = "dryrun-token"


def free_ports(n: int) -> list:
    socks, ports = [], []
    for _ in range(n):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        socks.append(s)
        ports.append(s.getsockname()[1])
    for s in socks:
        s.close()
    return ports


def wait_http(url: str, timeout: float = 20.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            urllib.request.urlopen(url, timeout=1.0).read()
            return True
        except Exception:
            time.sleep(0.2)
    return False


class Fleet:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.procs: dict = {}
        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

    def spawn(self, name: str, argv: list, env: Optional[dict] = None) -> subprocess.Popen:
        out = open(os.path.join(self.run_dir, "logs", f"{name}.log"), "ab")
        p = subprocess.Popen([sys.executable] + argv, stdout=out, stderr=subprocess.STDOUT,
                             env={**os.environ, **(env or {})})
        self.procs[name] = p
        pids = {k: v.pid for k, v in self.procs.items()}
        with open(os.path.join(self.run_dir, "dryrun-pids.json"), "w") as fh:
            json.dump(pids, fh)
        return p

    def stop(self, name: str, grace: float = 8.0) -> Optional[int]:
        p = self.procs.get(name)
        if p is None or p.poll() is not None:
            return None if p is None else p.returncode
        os.kill(p.pid, signal.SIGTERM)
        try:
            return p.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            os.kill(p.pid, signal.SIGKILL)
            return p.wait(timeout=5)

    def stop_all(self) -> None:
        for name in list(self.procs):
            self.stop(name)


def build_config(args: argparse.Namespace, run_dir: str, ports: dict) -> dict:
    nodes = [f"E{k}" for k in range(1, args.nodes + 1)]
    return {
        "run_id": os.path.basename(run_dir), "run_dir": run_dir, "seed": args.seed, "epoch_s": args.epoch_s,
        "block_epochs": args.block_epochs, "burn_in_epochs": args.burn_in, "n_blocks": args.blocks,
        "design": {"label": "screening"},
        "arms": [
            {"profile": "q35-4b-fp16", "batch_cap": 8, "escalate_family2": False},
            {"profile": "q35-4b-ih-w8a8", "batch_cap": 16, "escalate_family2": True},
            {"profile": "q35-2b-fp16", "batch_cap": 8, "escalate_family2": True},
            {"id": "I1-E", "kind": "incumbent", "router": "kappa", "profiles": ["q35-4b-fp16"],
             "incumbent": {"factory": "stub", "kwargs": {"profile": "q35-4b-fp16"}}},
            {"id": "I2", "kind": "incumbent", "router": "I2", "router_params": {"tau": 0.4},
             "profiles": ["q35-2b-fp16", "q35-4b-ih-w8a8"],
             "incumbent": {"factory": "scripted", "kwargs": {"script": [
                 {"profile": "q35-2b-fp16", "batch_cap": 16}, {"profile": "q35-4b-ih-w8a8", "batch_cap": 8}]}}},
        ],
        "nodes": [{"id": n, "gateway": f"http://127.0.0.1:{ports['gw0'] + k}",
                   "engine_manager": f"http://127.0.0.1:{ports['mgr0'] + k}"} for k, n in enumerate(nodes)],
        "cloud": {"gateway": f"http://127.0.0.1:{ports['cloud_gw']}", "engine": f"http://127.0.0.1:{ports['vllm']}",
                  "max_in_flight": 16},
        "control": {"start_delay_s": args.start_delay, "resume_margin_s": 2, "retries": 1},
    }


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description='End-to-end dry run: mock engine managers and gateways, cloud gateway on a mock engine, monitor, controller.')
    ap.add_argument("--root", required=True, help="runs root; the run directory is <root>/<run-id>")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--nodes", type=int, default=5)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--block-epochs", type=int, default=2)
    ap.add_argument("--burn-in", type=int, default=1)
    ap.add_argument("--epoch-s", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--start-delay", type=float, default=6.0)
    ap.add_argument("--rate", type=float, default=3.0, help="cloud requests per second per node")
    ap.add_argument("--switch-s", type=float, default=0.3)
    ap.add_argument("--nvml", action="store_true", help="read real NVML (read-only telemetry) instead of the mock GPUs")
    ap.add_argument("--cores", default="0-3", help="cores for the monitor's CPU accounting; also the affinity of this run")
    ap.add_argument("--crash-test", action="store_true")
    ap.add_argument("--crash-frac", type=float, default=0.55, help="fraction of the run after which the controller is killed")
    ap.add_argument("--down-s", type=float, default=None, help="downtime before the restart [1.5 epochs]")
    args = ap.parse_args(argv)

    run_id = args.run_id or time.strftime("dryrun-%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.root, run_id)
    os.makedirs(run_dir)
    if hasattr(os, "sched_setaffinity") and args.cores:
        from .util import parse_int_ranges
        try:
            os.sched_setaffinity(0, set(parse_int_ranges(args.cores)))
        except OSError as exc:
            print(f"affinity not set: {exc}")
    k = args.nodes
    mgr0 = contiguous(k)
    ports = {"mgr0": mgr0, "gw0": contiguous(k, avoid=range(mgr0, mgr0 + k))}
    ports["cloud_gw"], ports["vllm"] = free_ports(2)
    cfg_dict = build_config(args, run_dir, ports)
    cfg_path = os.path.join(run_dir, "dryrun-config.yaml")
    with open(cfg_path, "w") as fh:
        yaml.safe_dump(cfg_dict, fh, sort_keys=False)
    cfg = load_config(cfg_path)
    duration = cfg.total_epochs * cfg.epoch_s
    node_names = ",".join(n.id for n in cfg.nodes)
    fl = Fleet(run_dir)
    env = {"CIARA_TOKEN": TOKEN}
    result = 1
    try:
        fl.spawn("mock-stack", ["-m", "ciara_control.mocks.stack", "--nodes", node_names, "--manager-base-port",
                                str(ports["mgr0"]), "--gateway-base-port", str(ports["gw0"]), "--log-dir", run_dir,
                                "--rate", "1.0", "--switch-s", str(args.switch_s)])
        fl.spawn("mock-vllm", ["-m", "ciara_control.mocks.mock_vllm", "--listen", f"127.0.0.1:{ports['vllm']}",
                               "--ttft-s", "0.15", "--tok-s", "0.01", "--tokens", "20", "--token", TOKEN])
        fl.spawn("cloud-gw", ["-m", "ciara_control.cloud_gateway", "--listen", f"127.0.0.1:{ports['cloud_gw']}",
                              "--upstream", f"http://127.0.0.1:{ports['vllm']}", "--log",
                              os.path.join(run_dir, "cloud-gw.jsonl")], env=env)
        ready = (wait_http(f"http://127.0.0.1:{ports['gw0']}/healthz") and
                 wait_http(f"http://127.0.0.1:{ports['vllm']}/health") and
                 wait_http(f"http://127.0.0.1:{ports['cloud_gw']}/healthz"))
        if not ready:
            print("mock endpoints did not come up; see logs/")
            return 2
        mon_args = ["-m", "ciara_control.monitor", "--run-dir", run_dir, "--config", cfg_path,
                    "--gpu-backend", "nvml" if args.nvml else "mock", "--cores", args.cores, "--exit-when-done",
                    "--wait-manifest-s", "30"]
        fl.spawn("monitor", mon_args, env=env)
        fl.spawn("loadgen", ["-m", "ciara_control.mocks.loadgen", "--url", f"http://127.0.0.1:{ports['cloud_gw']}",
                             "--nodes", node_names, "--rate", str(args.rate), "--duration",
                             str(duration + args.start_delay - 2), "--burst-at", str(args.start_delay + 3),
                             "--burst-n", "80", "--out", os.path.join(run_dir, "mock-loadgen.jsonl")])
        ctl_argv = ["-m", "ciara_control.controller", "run", "--config", cfg_path]
        resumed = False
        if args.crash_test:
            plan = os.path.join(run_dir, "planned-draws.json")
            subprocess.run([sys.executable, "-m", "ciara_control.controller", "plan", "--config", cfg_path,
                            "--out", plan], check=True, stdout=subprocess.DEVNULL)
            c = fl.spawn("controller-1", ctl_argv)
            time.sleep(args.start_delay + args.crash_frac * duration)
            os.kill(c.pid, signal.SIGKILL)
            c.wait(timeout=5)
            print(f"controller killed at about {args.crash_frac * duration:.0f} s of {duration:.0f} s")
            time.sleep(args.down_s if args.down_s is not None else args.epoch_s * 1.5)
            c2 = fl.spawn("controller-2", ctl_argv + ["--resume"])
            code = c2.wait(timeout=duration + 120)
            resumed = True
        else:
            c = fl.spawn("controller", ctl_argv)
            code = c.wait(timeout=duration + args.start_delay + 120)
        print(f"controller exit code {code}")
        lg_code = fl.procs["loadgen"].wait(timeout=duration + 120)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{ports['vllm']}/stats", timeout=3) as r:
                with open(os.path.join(run_dir, "mock-vllm-stats.json"), "w") as fh:
                    fh.write(r.read().decode())
        except Exception as exc:
            print(f"engine stats unavailable: {exc}")
        time.sleep(1.5)
        fl.stop("monitor")
        fl.stop("cloud-gw")
        fl.stop("mock-vllm")
        fl.stop("mock-stack")
        time.sleep(0.5)
        rep = verify(run_dir, cfg, resumed=resumed, expect_saturation=True)
        print(rep.render())
        with open(os.path.join(run_dir, "verify.json"), "w") as fh:
            json.dump(rep.checks, fh, indent=1)
        result = 0 if (rep.ok and code == 0 and lg_code == 0) else 1
        if resumed:
            plan_draws = json.load(open(os.path.join(run_dir, "planned-draws.json")))["draws"]
            want = {(d["node"], d["block_id"]): d["arm_id"] for d in plan_draws}
            got = {(b["node"], b["block_id"]): b["arm_id"] for b in
                   (r for r in _read(os.path.join(run_dir, "blocks.jsonl")) if r["rec"] == "block")}
            same = all(want[k] == v for k, v in got.items())
            print(("PASS" if same else "FAIL") + f"  resumed run drew the planned arms ({len(got)} of {len(want)} node-blocks started)")
            result = result if same else 1
    finally:
        fl.stop_all()
    print("DRY RUN " + ("PASSED" if result == 0 else "FAILED") + f"  run dir {run_dir}")
    return result


def contiguous(n: int, avoid=()) -> int:
    import random
    for _ in range(200):
        base = random.randint(20000, 40000)
        if any(base + i in avoid for i in range(n)):
            continue
        socks = []
        try:
            for i in range(n):
                s = socket.socket()
                s.bind(("127.0.0.1", base + i))
                socks.append(s)
            return base
        except OSError:
            continue
        finally:
            for s in socks:
                s.close()
    raise RuntimeError("no free port block")


def _read(path: str) -> list:
    from .util import read_jsonl
    return read_jsonl(path)


if __name__ == "__main__":
    sys.exit(main())
