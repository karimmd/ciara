#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pwd
import signal
import sys
import threading
import time
from typing import Optional

from . import httpclient, promtext
from .nodeio import engine_endpoints
from .util import JsonlLog, parse_int_ranges

REAL_THROTTLE = 0x4 | 0x8 | 0x20 | 0x40 | 0x80
THROTTLE_NAMES = {0x1: "GpuIdle", 0x2: "AppClocks", 0x4: "SwPowerCap", 0x8: "HwSlowdown", 0x10: "SyncBoost",
                  0x20: "SwThermalSlowdown", 0x40: "HwThermalSlowdown", 0x80: "HwPowerBrake", 0x100: "DisplayClock"}


def uid_of(pid: int, root: str = "/proc") -> Optional[int]:
    try:
        return os.stat(f"{root}/{pid}").st_uid
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def uname_of(uid: Optional[int]) -> Optional[str]:
    if uid is None:
        return None
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


class MockGpuBackend:
    def __init__(self, n_gpus: int = 2):
        self.n_gpus = n_gpus
        self._k = 0

    def sample(self) -> list:
        self._k += 1
        out = []
        for i in range(self.n_gpus):
            out.append({"gpu": i, "util_gpu": (self._k * 7 + i) % 100, "util_mem": 10, "mem_used_mib": 1000.0 + i,
                        "mem_total_mib": 11264.0, "power_w": 60.0 + i, "power_limit_w": 250.0,
                        "energy_mj": 1000 * self._k * (i + 1), "temp_c": 40, "sm_clock_mhz": 1500,
                        "mem_clock_mhz": 7000, "throttle_bits": 1, "throttle_ns": {"sw_power_cap": 0, "hw_thermal": 0},
                        "throttled": False, "n_procs": 0,
                        "n_external_procs": 0})
        return out

    def close(self) -> None:
        pass


class NvmlBackend:
    def __init__(self, with_pcie: bool = False, self_uid: Optional[int] = None, parallel: bool = True,
                 bits_every: int = 4, gpus: Optional[list] = None):
        import concurrent.futures
        import pynvml
        self.nv = pynvml
        pynvml.nvmlInit()
        self.gpu_ids = [int(g) for g in gpus] if gpus else list(range(pynvml.nvmlDeviceGetCount()))
        self.n_gpus = len(self.gpu_ids)
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.gpu_ids]
        self.with_pcie = with_pcie
        self.bits_every = bits_every
        self.self_uid = os.getuid() if self_uid is None else self_uid
        self.power_limit = []
        for h in self.handles:
            try:
                self.power_limit.append(round(pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0, 1))
            except pynvml.NVMLError:
                self.power_limit.append(None)
        self._n = 0
        self._prev_real: dict[int, int] = {}
        self._fields = [(name, getattr(pynvml, const, None)) for name, const in (
            ("sw_power_cap", "NVML_FI_DEV_CLOCKS_EVENT_REASON_SW_POWER_CAP"),
            ("hw_thermal", "NVML_FI_DEV_CLOCKS_EVENT_REASON_HW_THERM_SLOWDOWN"),
            ("sw_thermal", "NVML_FI_DEV_CLOCKS_EVENT_REASON_SW_THERM_SLOWDOWN"),
            ("hw_power_brake", "NVML_FI_DEV_CLOCKS_EVENT_REASON_HW_POWER_BRAKE_SLOWDOWN"),
            ("sync_boost", "NVML_FI_DEV_CLOCKS_EVENT_REASON_SYNC_BOOST"))]
        self._energy_id = getattr(pynvml, "NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION", None)
        self._field_ids = [fid for _, fid in self._fields if fid is not None]
        self.pool = (concurrent.futures.ThreadPoolExecutor(max_workers=min(self.n_gpus, 5), thread_name_prefix="nvml")
                     if parallel and self.n_gpus > 1 else None)

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True)
        try:
            self.nv.nvmlShutdown()
        except Exception:
            pass

    def sample(self) -> list:
        with_bits = self.bits_every > 0 and self._n % self.bits_every == 0
        self._n += 1
        if self.pool is not None:
            return list(self.pool.map(lambda i: self._one(self.handles[i], i, with_bits), range(self.n_gpus)))
        return [self._one(h, i, with_bits) for i, h in enumerate(self.handles)]

    @staticmethod
    def _field_value(f):
        if f.nvmlReturn != 0:
            return None
        v = f.value
        return {0: v.dVal, 1: v.uiVal, 2: v.ulVal, 3: v.ullVal, 4: v.sllVal, 5: v.siVal}.get(f.valueType)

    def _one(self, handle, idx: int, with_bits: bool) -> dict:
        nv = self.nv
        rec: dict = {"gpu": self.gpu_ids[idx]}

        def get(key, fn, tf=lambda v: v):
            try:
                rec[key] = tf(fn())
            except nv.NVMLError:
                rec[key] = None

        try:
            u = nv.nvmlDeviceGetUtilizationRates(handle)
            rec["util_gpu"], rec["util_mem"] = u.gpu, u.memory
        except nv.NVMLError:
            rec["util_gpu"] = rec["util_mem"] = None
        try:
            m = nv.nvmlDeviceGetMemoryInfo(handle)
            rec["mem_used_mib"], rec["mem_total_mib"] = round(m.used / 1048576, 1), round(m.total / 1048576, 1)
        except nv.NVMLError:
            rec["mem_used_mib"] = rec["mem_total_mib"] = None
        get("power_w", lambda: nv.nvmlDeviceGetPowerUsage(handle), lambda v: round(v / 1000.0, 2))
        rec["power_limit_w"] = self.power_limit[idx]
        get("temp_c", lambda: nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU))
        get("sm_clock_mhz", lambda: nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_SM))
        get("mem_clock_mhz", lambda: nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_MEM))
        ids = ([self._energy_id] if self._energy_id is not None else []) + self._field_ids
        rec["energy_mj"], rec["throttle_ns"] = None, None
        if ids:
            try:
                vals = [self._field_value(f) for f in nv.nvmlDeviceGetFieldValues(handle, ids)]
                if self._energy_id is not None:
                    rec["energy_mj"], vals = vals[0], vals[1:]
                names = [n for n, fid in self._fields if fid is not None]
                rec["throttle_ns"] = dict(zip(names, vals))
            except nv.NVMLError:
                pass
        real_now = None
        if rec["throttle_ns"]:
            real_now = sum(v for v in rec["throttle_ns"].values() if v)
            real_now -= rec["throttle_ns"].get("sync_boost") or 0
        prev = self._prev_real.get(idx)
        if real_now is not None:
            self._prev_real[idx] = real_now
        bits = None
        if with_bits:
            try:
                bits = nv.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
            except nv.NVMLError:
                bits = None
        rec["throttle_bits"] = bits
        if real_now is not None and prev is not None:
            rec["throttled"] = real_now > prev
        else:
            rec["throttled"] = None if bits is None else bool(bits & REAL_THROTTLE)
        if self.with_pcie:
            get("pcie_tx_kbs", lambda: nv.nvmlDeviceGetPcieThroughput(handle, nv.NVML_PCIE_UTIL_TX_BYTES))
            get("pcie_rx_kbs", lambda: nv.nvmlDeviceGetPcieThroughput(handle, nv.NVML_PCIE_UTIL_RX_BYTES))
        procs = []
        external = 0
        try:
            running = nv.nvmlDeviceGetComputeRunningProcesses(handle)
        except nv.NVMLError:
            running = []
        for p in running:
            uid = uid_of(p.pid)
            is_external = uid is not None and uid != self.self_uid
            external += 1 if is_external else 0
            procs.append({"pid": p.pid, "user": uname_of(uid), "external": is_external,
                          "mem_mib": round(p.usedGpuMemory / 1048576, 1) if p.usedGpuMemory else None})
        rec["n_procs"] = len(procs)
        rec["n_external_procs"] = external
        if procs:
            rec["procs"] = procs
        return rec


class CpuSampler:
    def __init__(self, cores: Optional[list] = None, own_uids: Optional[set] = None, proc_root: str = "/proc"):
        self.root = proc_root
        self.own_uids = set(own_uids) if own_uids else {os.getuid()}
        self.tck = float(os.sysconf("SC_CLK_TCK"))
        self.cores = list(cores) if cores else None
        self._busy_prev: dict[int, int] = {}
        self._pid_prev: dict[int, int] = {}
        self._tid_prev: dict[tuple, int] = {}
        self._pid_info: dict[int, tuple] = {}
        self._t_prev: Optional[float] = None

    def _read_core_busy(self) -> dict:
        out = {}
        with open(f"{self.root}/stat", "r") as fh:
            for line in fh:
                if not line.startswith("cpu") or line.startswith("cpu "):
                    continue
                f = line.split()
                core = int(f[0][3:])
                v = [int(x) for x in f[1:9]]
                out[core] = v[0] + v[1] + v[2] + v[5] + v[6] + v[7]
        return out

    @staticmethod
    def _parse_stat(text: str) -> Optional[tuple]:
        try:
            lp, rp = text.index("("), text.rindex(")")
            rest = text[rp + 1:].split()
            return text[lp + 1:rp], int(rest[11]) + int(rest[12]), int(rest[36])
        except (ValueError, IndexError):
            return None

    def sample(self, now: Optional[float] = None) -> Optional[dict]:
        t_scan0 = time.monotonic()
        now = time.monotonic() if now is None else now
        busy = self._read_core_busy()
        cores = self.cores if self.cores is not None else sorted(busy)
        own = {c: 0 for c in cores}
        external = {c: 0 for c in cores}
        external_by: dict[tuple, int] = {}
        first = self._t_prev is None
        seen_pids = set()
        try:
            names = os.listdir(self.root)
        except OSError:
            names = []
        for name in names:
            if not name.isdigit():
                continue
            pid = int(name)
            try:
                with open(f"{self.root}/{pid}/stat", "r") as fh:
                    ps = self._parse_stat(fh.read())
            except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
                continue
            if ps is None:
                continue
            seen_pids.add(pid)
            comm, ticks, _ = ps
            prev = self._pid_prev.get(pid)
            self._pid_prev[pid] = ticks
            if pid not in self._pid_info:
                self._pid_info[pid] = (uid_of(pid, self.root), comm)
            if first:
                self._seed_tasks(pid)
                continue
            if prev is not None and ticks == prev:
                continue
            uid, comm = self._pid_info[pid]
            is_own = uid in self.own_uids
            for tid in self._tids(pid):
                try:
                    with open(f"{self.root}/{pid}/task/{tid}/stat", "r") as fh:
                        ts = self._parse_stat(fh.read())
                except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
                    continue
                if ts is None:
                    continue
                _, tt, core = ts
                key = (pid, tid)
                d = tt - self._tid_prev.get(key, 0)
                self._tid_prev[key] = tt
                if d <= 0 or core not in own:
                    continue
                if is_own:
                    own[core] += d
                else:
                    external[core] += d
                    k2 = (uid, comm)
                    external_by[k2] = external_by.get(k2, 0) + d
        for pid in [p for p in self._pid_prev if p not in seen_pids]:
            self._pid_prev.pop(pid, None)
            self._pid_info.pop(pid, None)
        for key in [k for k in self._tid_prev if k[0] not in seen_pids]:
            self._tid_prev.pop(key, None)
        if first:
            self._t_prev, self._busy_prev = now, busy
            return None
        dt = now - self._t_prev
        busy_d = {c: busy.get(c, 0) - self._busy_prev.get(c, 0) for c in cores}
        self._t_prev, self._busy_prev = now, busy
        top = sorted(external_by.items(), key=lambda kv: -kv[1])[:5]
        return {
            "dt_s": round(dt, 4), "cores": cores,
            "busy_s": [round(max(0, busy_d[c]) / self.tck, 3) for c in cores],
            "own_s": [round(own[c] / self.tck, 3) for c in cores],
            "external_s": [round(external[c] / self.tck, 3) for c in cores],
            "external_top": [{"uid": u, "user": uname_of(u), "comm": cm, "cpu_s": round(t / self.tck, 3)}
                            for (u, cm), t in top],
            "scan_ms": round((time.monotonic() - t_scan0) * 1000, 1),
        }

    def _tids(self, pid: int) -> list:
        try:
            return [int(x) for x in os.listdir(f"{self.root}/{pid}/task")]
        except OSError:
            return []

    def _seed_tasks(self, pid: int) -> None:
        for tid in self._tids(pid):
            try:
                with open(f"{self.root}/{pid}/task/{tid}/stat", "r") as fh:
                    ts = self._parse_stat(fh.read())
            except OSError:
                continue
            if ts is not None:
                self._tid_prev[(pid, tid)] = ts[1]


class VllmScraper(threading.Thread):
    def __init__(self, managers: dict, cloud_engine: Optional[str], log: JsonlLog, epoch_s: float,
                 t0_mono_ns: Optional[int], offset_s: float, timeout_s: float, stop: threading.Event,
                 include_buckets: bool = True):
        super().__init__(name="vllm-scraper", daemon=True)
        self.managers = managers
        self.cloud = cloud_engine
        self.log = log
        self.epoch_ns = int(epoch_s * 1e9)
        self.t0 = t0_mono_ns if t0_mono_ns is not None else time.monotonic_ns()
        self.grid = "controller" if t0_mono_ns is not None else "free"
        self.offset_ns = int(offset_s * 1e9)
        self.timeout_s = timeout_s
        self.stop = stop
        self.include_buckets = include_buckets

    def scrape_all(self, epoch: int) -> None:
        targets: list = []
        for node, base in self.managers.items():
            try:
                st = httpclient.request_json("GET", base.rstrip("/") + "/status", timeout=self.timeout_s)
            except Exception as exc:
                self._write({"event": "vllm_error", "node": node, "epoch": epoch, "error": f"status: {exc}"})
                continue
            active = st.get("active_profile") or st.get("profile")
            for prof, ep in engine_endpoints(st).items():
                targets.append((node, prof, ep, prof == active))
        if self.cloud:
            targets.append(("cloud", "cloud", self.cloud, True))
        for node, prof, ep, active in targets:
            t = time.monotonic_ns()
            tok = os.environ.get("CIARA_TOKEN")
            try:
                _, text = httpclient.request("GET", ep.rstrip("/") + "/metrics", timeout=self.timeout_s,
                                             headers={"Authorization": f"Bearer {tok}"} if tok else None)
            except Exception as exc:
                self._write({"event": "vllm_error", "node": node, "profile": prof, "epoch": epoch,
                             "error": f"metrics: {exc}"})
                continue
            series = promtext.keep_vllm(promtext.parse(text))
            if not self.include_buckets:
                series = {k: v for k, v in series.items() if "_bucket" not in k}
            self._write({"event": "vllm", "node": node, "profile": prof, "endpoint": ep, "active": active,
                         "epoch": epoch, "grid": self.grid, "scrape_ms": round((time.monotonic_ns() - t) / 1e6, 1),
                         "series": series})

    def _write(self, rec: dict) -> None:
        rec.setdefault("mono_ns", time.monotonic_ns())
        rec.setdefault("wall_ns", time.time_ns())
        self.log.append(rec)

    def run(self) -> None:
        k = max(0, (time.monotonic_ns() - self.t0 - self.offset_ns) // self.epoch_ns + 1)
        while not self.stop.is_set():
            target = self.t0 + k * self.epoch_ns + self.offset_ns
            wait = (target - time.monotonic_ns()) / 1e9
            if wait > 0 and self.stop.wait(wait):
                break
            self.scrape_all(int(k))
            k += 1


def read_grid(run_dir: str, wait_s: float) -> tuple:
    from .runstate import load_manifest
    deadline = time.monotonic() + wait_s
    while True:
        try:
            m = load_manifest(run_dir)
        except (OSError, ValueError):
            m = None
        if m:
            from .clock import read_boot_id
            if m.get("t0_boot_id") == read_boot_id():
                t0 = int(m["t0_mono_ns"])
            else:
                t0 = time.monotonic_ns() - (time.time_ns() - int(m["t0_wall_ns"]))
            return t0, m["epoch_ns"] / 1e9
        if time.monotonic() >= deadline:
            return None, None
        time.sleep(0.5)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="ciara_control.monitor", description='Monitor: NVML for every GPU, per-core CPU time by owner, vLLM /metrics per epoch.')
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", default=None, help="run configuration: engine managers and cloud engine")
    ap.add_argument("--manager", action="append", default=[], metavar="NODE=URL", help="engine manager base URL")
    ap.add_argument("--cloud-engine", default=None, help="cloud vLLM base URL")
    ap.add_argument("--interval", type=float, default=0.5, help="GPU and CPU sample period, seconds")
    ap.add_argument("--epoch-s", type=float, default=30.0)
    ap.add_argument("--gpu-backend", choices=("nvml", "mock", "none"), default="nvml")
    ap.add_argument("--gpus", default=None, help="GPU indices to sample, e.g. 0,1 (default: every GPU)")
    ap.add_argument("--pcie", action="store_true", help="also sample PCIe throughput (slow NVML calls)")
    ap.add_argument("--throttle-bits-every", type=int, default=4,
                    help="read the instantaneous throttle bits every N samples (24 ms per GPU); 0 turns them off")
    ap.add_argument("--cores", default=None, help="cores for CPU accounting, e.g. 0-7 (default: all)")
    ap.add_argument("--own-uids", default=None, help="comma separated uids that count as the experiment")
    ap.add_argument("--no-cpu", action="store_true")
    ap.add_argument("--no-vllm", action="store_true")
    ap.add_argument("--scrape-offset-s", type=float, default=0.5)
    ap.add_argument("--wait-manifest-s", type=float, default=0.0)
    ap.add_argument("--exit-when-done", action="store_true", help="exit when STATUS is DONE, STOPPED or FAILED")
    args = ap.parse_args(argv)

    managers: dict[str, str] = {}
    cloud_engine = args.cloud_engine
    epoch_s = args.epoch_s
    if args.config:
        from .config import load_config
        cfg = load_config(args.config)
        epoch_s = cfg.epoch_s
        for n in cfg.nodes:
            if n.engine_manager:
                managers[n.id] = n.engine_manager
        cloud_engine = cloud_engine or cfg.cloud.engine
    for item in args.manager:
        node, url = item.split("=", 1)
        managers[node] = url

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    os.makedirs(args.run_dir, exist_ok=True)
    gpu_log = JsonlLog(os.path.join(args.run_dir, "monitor-gpu.jsonl"), durable=False)
    cpu_log = JsonlLog(os.path.join(args.run_dir, "monitor-cpu.jsonl"), durable=False)
    vllm_log = JsonlLog(os.path.join(args.run_dir, "monitor-vllm.jsonl"), durable=False)

    own_uids = {int(x) for x in args.own_uids.split(",")} if args.own_uids else None
    backend = None
    if args.gpu_backend == "nvml":
        backend = NvmlBackend(with_pcie=args.pcie, bits_every=args.throttle_bits_every,
                              gpus=[int(x) for x in args.gpus.split(",")] if args.gpus else None)
    elif args.gpu_backend == "mock":
        backend = MockGpuBackend()
    cpu = None if args.no_cpu else CpuSampler(parse_int_ranges(args.cores) if args.cores else None, own_uids)

    gpu_log.append({"event": "monitor_start", "wall_ns": time.time_ns(), "mono_ns": time.monotonic_ns(),
                    "interval": args.interval, "gpu_backend": args.gpu_backend,
                    "n_gpus": backend.n_gpus if backend else 0, "self_uid": os.getuid(), "pid": os.getpid()})

    scraper = None
    if not args.no_vllm and (managers or cloud_engine):
        t0, es = read_grid(args.run_dir, args.wait_manifest_s)
        scraper = VllmScraper(managers, cloud_engine, vllm_log, es or epoch_s, t0, args.scrape_offset_s,
                              timeout_s=5.0, stop=stop)
        scraper.start()

    status_path = os.path.join(args.run_dir, "STATUS")
    next_t = time.monotonic()
    last_status_check = 0.0
    while not stop.is_set():
        t = time.monotonic()
        mono, wall = time.monotonic_ns(), time.time_ns()
        if backend is not None:
            gpus = backend.sample()
            gpu_log.append({"event": "sample", "mono_ns": mono, "wall_ns": wall, "gpus": gpus,
                            "loadavg": os.getloadavg()[0],
                            "sample_ms": round((time.monotonic_ns() - mono) / 1e6, 1)})
        if cpu is not None:
            rec = cpu.sample()
            if rec is not None:
                rec.update({"event": "cpu", "mono_ns": mono, "wall_ns": wall})
                cpu_log.append(rec)
        if args.exit_when_done and t - last_status_check > 2.0:
            last_status_check = t
            try:
                with open(status_path) as fh:
                    if fh.read().strip() in ("DONE", "STOPPED", "FAILED"):
                        break
            except OSError:
                pass
        next_t += args.interval
        delay = next_t - time.monotonic()
        if delay < -args.interval:
            next_t = time.monotonic()
            delay = 0
        if delay > 0:
            stop.wait(delay)
    stop.set()
    gpu_log.append({"event": "monitor_stop", "wall_ns": time.time_ns(), "mono_ns": time.monotonic_ns()})
    if scraper is not None:
        scraper.join(timeout=5)
    if backend is not None:
        backend.close()
    for lg in (gpu_log, cpu_log, vllm_log):
        lg.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
