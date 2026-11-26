from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import signal
import subprocess
import sys
import time
from typing import Dict, Optional

import aiohttp
from aiohttp import web

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ciara_edge.engines import Engine, EngineSpec, gpu_used_mib, gpus_used_mib, parse_log
    from ciara_edge.profiles import profile_info
else:
    from .engines import Engine, EngineSpec, gpu_used_mib, gpus_used_mib, parse_log
    from .profiles import profile_info

BUDGET = 0.95
DEFAULT_RESIDUAL_MIB = 480.0


def mono_ns() -> int:
    return time.monotonic_ns()


class Manager:
    def __init__(self, args):
        self.args = args
        self.node = args.node
        self.gpus = args.gpus
        self.gpu = self.gpus[0]
        self.budget = args.budget
        self.token = os.environ.get("CIARA_TOKEN", "")
        self.mode = args.mode
        self.restart_only = set(x for x in args.restart_only.split(",") if x)
        self.engines: Dict[str, Engine] = {}
        self.ports: Dict[str, int] = {}
        self.active: Optional[str] = None
        self.last_switch: Optional[dict] = None
        self.switching = False
        self.lock = asyncio.Lock()
        self.gateway_url = args.gateway_url or None
        self.session: Optional[aiohttp.ClientSession] = None
        self.residual: Dict[str, float] = {}
        self.calib: dict = {}
        if args.calib_file and os.path.exists(args.calib_file):
            self.calib = json.load(open(args.calib_file))
            self.residual.update(self.calib.get("residual_mib", {}))
        self.restart_only |= set(self.calib.get("restart_only", []))
        self.explicit_restart_only: set = set(self.restart_only)
        self.auto_restart_only: set = set()
        self.group: list = []
        self.total_mib = args.total_mib
        self.base_used_mib = 0
        self.health_state = "unknown"
        self.started = time.time()
        d = os.path.join(args.log_dir, args.run_id)
        os.makedirs(os.path.join(d, "engines"), exist_ok=True)
        self.run_dir = d
        self.evf = open(os.path.join(d, f"engine-mgr-{self.node}.jsonl"), "a", buffering=1)
        self.pidfile = os.path.join(d, f"engine-mgr-{self.node}.pids")
        self.prepared: list = []
        self.stop_event = asyncio.Event()
        self._health_task: Optional[asyncio.Task] = None

    def event(self, kind: str, **kw) -> None:
        rec = {"event": kind, "node": self.node, "t": mono_ns(), "t_wall": time.time_ns(), **kw}
        self.evf.write(json.dumps(rec, separators=(",", ":"), default=str) + "\n")

    def write_pidfile(self) -> None:
        pgids = {p: e.pgid for p, e in self.engines.items() if e.pgid and e.alive()}
        tmp = self.pidfile + ".tmp"
        json.dump({"manager_pid": os.getpid(), "gpu": self.gpu, "engine_pgids": pgids}, open(tmp, "w"))
        os.replace(tmp, self.pidfile)

    def endpoint(self, profile: str) -> str:
        e = self.engines[profile]
        return e.base

    def port_for(self, profile: str) -> int:
        if profile not in self.ports:
            self.ports[profile] = self.args.engine_port_base + len(self.ports)
        return self.ports[profile]

    def spec(self, profile: str, gmu: float, sleep_mode: bool = True) -> EngineSpec:
        return EngineSpec(profile=profile, gpu=self.gpu, gpus=",".join(str(g) for g in self.gpus) if len(self.gpus) > 1 else None, tp=len(self.gpus), port=self.port_for(profile), host=self.args.bind_host, gmu=gmu,
                          token=self.token, cores=self.args.cores or None, max_model_len=self.args.max_model_len,
                          max_num_seqs=self.args.max_num_seqs, sleep_mode=sleep_mode,
                          log_path=os.path.join(self.run_dir, "engines", f"{self.node}-{profile}.log"))

    def predict_kv_tokens(self, profile: str, gmu: float) -> Optional[float]:
        pts = self.calib.get("kv_points", {}).get(profile)
        if not pts or len(pts) < 2:
            return None
        (g1, t1), (g2, t2) = pts[0], pts[-1]
        if g2 == g1:
            return None
        return t1 + (t2 - t1) * (gmu - g1) / (g2 - g1)

    def kv_fraction(self, profile: str, kv_tokens: Optional[float]) -> Optional[float]:
        solo = self.calib.get("solo_kv_tokens", {}).get(profile)
        return (kv_tokens / solo) if (solo and kv_tokens) else None

    def split_group(self, profiles: list, restart_only: set) -> tuple:
        group = [p for p in profiles if p not in restart_only]
        auto: set = set()
        changed = True
        while changed and len(group) > 1:
            changed = False
            for p in sorted(group, key=lambda q: self.kv_fraction(q, self.predict_kv_tokens(q, self.gmu_for(q, group))) or 9.0):
                frac = self.kv_fraction(p, self.predict_kv_tokens(p, self.gmu_for(p, group)))
                if frac is not None and frac < self.args.min_kv_fraction:
                    group.remove(p)
                    auto.add(p)
                    changed = True
                    break
        return group, auto

    def gmu_for(self, profile: str, resident: list) -> float:
        others = sum(self.residual.get(q, DEFAULT_RESIDUAL_MIB) for q in resident if q != profile)
        return math.floor((self.budget - others / self.total_mib) * 10000) / 10000

    async def gateway_call(self, payload: dict, gateway: Optional[str] = None, timeout: float = 90.0) -> Optional[dict]:
        url = gateway or self.gateway_url
        if not url:
            return None
        try:
            async with self.session.post(url.rstrip("/") + "/admission", json=payload,
                                         timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                return await r.json()
        except Exception as e:
            self.event("gateway_error", error=repr(e), payload=payload)
            return {"error": repr(e)}

    async def _build_group(self, target: Optional[str] = None) -> dict:
        out: dict = {}
        group = self.group
        for p in list(self.engines):
            if p not in group:
                await self.engines[p].stop()
                self.engines.pop(p, None)
                if self.active == p:
                    self.active = None
        for p in list(self.engines):
            if not self.engines[p].alive():
                await self.engines[p].stop()
                self.engines.pop(p, None)
                if self.active == p:
                    self.active = None
        order = [p for p in group if p != target] + ([target] if target in group else [])
        for p in order:
            if p in self.engines:
                out[p] = {"state": self.engines[p].state, "reused": True}
                continue
            if not self.base_used_mib:
                self.base_used_mib = gpus_used_mib(self.gpus)
            for q, e in self.engines.items():
                if e.state == "awake":
                    await e.sleep(level=1, mode="wait")
            gmu = self.gmu_for(p, group)
            eng = Engine(self.spec(p, gmu), self.session)
            self.engines[p] = eng
            ent = {"gmu": gmu, "predicted_kv_tokens": self.predict_kv_tokens(p, gmu)}
            try:
                ent["start"] = await eng.start(ready_timeout=self.args.start_timeout)
                ent["kv_tokens"] = ent["start"].get("kv_tokens")
                ent["kv_fraction_of_solo"] = self.kv_fraction(p, ent["kv_tokens"])
                if p != target:
                    await eng.probe()
                    await eng.sleep(level=1, mode="wait")
                    await asyncio.sleep(1.0)
                    used = gpus_used_mib(self.gpus) - self.base_used_mib
                    others = sum(e.residual_mib or 0 for q, e in self.engines.items() if q != p)
                    eng.residual_mib = int(used - others)
                    self.residual[p] = float(eng.residual_mib)
                    ent["residual_mib"] = eng.residual_mib
            except Exception as e:
                ent["error"] = repr(e)
                self.event("prepare_error", profile=p, error=repr(e))
                await eng.stop()
                self.engines.pop(p, None)
            out[p] = ent
        self.write_pidfile()
        return out

    async def prepare(self, profiles: list, restart_only: Optional[list] = None) -> dict:
        t0 = time.monotonic()
        res = {"mode": self.mode}
        async with self.lock:
            for p in profiles:
                profile_info(p)
            explicit = set(self.restart_only) | set(restart_only or [])
            if self.mode == "restart":
                await self._stop_all()
                self.group = []
                self.prepared = list(profiles)
                res.update(note="restart mode: engines are started on activation", profiles={})
                return res
            self.group, self.auto_restart_only = self.split_group(profiles, explicit)
            self.prepared = list(profiles)
            self.explicit_restart_only = explicit
            res["resident_group"] = list(self.group)
            res["restart_only"] = sorted(explicit & set(profiles))
            res["restart_only_auto"] = sorted(self.auto_restart_only)
            res["profiles"] = await self._build_group(None)
        res["seconds"] = time.monotonic() - t0
        res["gpu_used_mib"] = gpus_used_mib(self.gpus)
        self.event("prepare", **res)
        self.health_state = "ok" if all("error" not in v for v in res["profiles"].values()) else "degraded"
        return res

    def is_restart_only(self, profile: str) -> bool:
        return profile in getattr(self, "explicit_restart_only", set()) or profile in self.auto_restart_only \
            or profile not in self.group

    async def activate(self, profile: str, gateway: Optional[str] = None, mode: Optional[str] = None) -> dict:
        profile_info(profile)
        if gateway:
            self.gateway_url = gateway
        t_req = mono_ns()
        async with self.lock:
            t_lock = mono_ns()
            self.switching = True
            comp: Dict[str, float] = {}
            used_mode = None
            try:
                cur = self.active
                if cur == profile and cur in self.engines and self.engines[cur].state == "awake" and await self.engines[cur].health():
                    out = {"profile": profile, "endpoint": self.endpoint(profile), "mode": "noop", "switch_time_s": 0.0,
                           "switch_s": 0.0,
                           "components": {}, "previous": cur}
                    self.event("activate", **out)
                    return out
                t0 = time.monotonic()
                await self.gateway_call({"paused": True, "drain": True, "timeout_s": self.args.drain_timeout})
                comp["drain_s"] = time.monotonic() - t0
                solo = (mode or self.mode) == "restart" or self.is_restart_only(profile)
                t1 = time.monotonic()
                try:
                    if solo:
                        used_mode = "restart"
                        await self._switch_restart(profile, comp)
                    elif profile in self.engines and self.engines[profile].alive():
                        used_mode = "sleep"
                        await self._switch_sleep(cur, profile, comp)
                    else:
                        used_mode = "rebuild-group"
                        await self._switch_rebuild(profile, comp)
                except Exception as e:
                    self.event("switch_error", profile=profile, mode=used_mode, error=repr(e))
                    if used_mode == "sleep":
                        if profile in self.group:
                            used_mode = "rebuild-after-sleep-failure"
                            await self._stop_all()
                            await self._switch_rebuild(profile, comp)
                        else:
                            used_mode = "restart-after-sleep-failure"
                            await self._switch_restart(profile, comp)
                    else:
                        raise
                comp["engine_s"] = time.monotonic() - t1
                self.active = profile
                ep = self.endpoint(profile)
                t2 = time.monotonic()
                await self.gateway_call({"paused": False, "engine_url": ep, "profile": profile})
                comp["resume_s"] = time.monotonic() - t2
                total = time.monotonic() - t0
                out = {"profile": profile, "endpoint": ep, "mode": used_mode, "switch_time_s": total,
                       "switch_s": comp["engine_s"], "paused_s": total, "components": comp, "previous": cur, "t_start": t_lock, "t_end": mono_ns(),
                       "queue_wait_s": (t_lock - t_req) / 1e9, "gpu_used_mib": gpus_used_mib(self.gpus)}
                self.last_switch = out
                self.event("activate", **out)
                self.write_pidfile()
                self.health_state = "ok"
                return out
            except Exception:
                await self.gateway_call({"paused": False})
                self.health_state = "down"
                raise
            finally:
                self.switching = False

    async def _switch_sleep(self, cur: Optional[str], profile: str, comp: dict) -> None:
        eng = self.engines[profile]
        for q, e in self.engines.items():
            if q != profile and e.state == "awake":
                comp["sleep_s"] = comp.get("sleep_s", 0.0) + await e.sleep(level=1, mode="wait")
        if eng.state == "sleeping":
            comp["wake_s"] = await eng.wake()
        comp["probe_s"] = await eng.probe()

    async def _switch_rebuild(self, profile: str, comp: dict) -> None:
        t = time.monotonic()
        await self._build_group(profile)
        comp["rebuild_s"] = time.monotonic() - t
        if profile not in self.engines:
            raise RuntimeError(f"engine {profile} did not start")
        comp["probe_s"] = await self.engines[profile].probe()

    async def _switch_restart(self, profile: str, comp: dict) -> None:
        t = time.monotonic()
        await self._stop_all()
        comp["stop_s"] = time.monotonic() - t
        t = time.monotonic()
        eng = Engine(self.spec(profile, self.budget, sleep_mode=self.args.solo_sleep_mode), self.session)
        self.engines[profile] = eng
        await eng.start(ready_timeout=self.args.start_timeout)
        comp["start_s"] = time.monotonic() - t
        comp["probe_s"] = await eng.probe()
        self.write_pidfile()

    async def _stop_all(self) -> None:
        for p, e in list(self.engines.items()):
            await e.stop()
            self.engines.pop(p, None)
        self.active = None
        t = time.monotonic()
        while time.monotonic() - t < 60 and gpus_used_mib(self.gpus) > (self.base_used_mib or 0) + 60:
            await asyncio.sleep(0.25)

    async def status(self) -> dict:
        eng = {}
        for p, e in self.engines.items():
            eng[p] = {"state": e.state if e.alive() else "dead", "endpoint": e.base, "port": e.spec.port, "gmu": e.spec.gmu,
                      "pid": e.proc.pid if e.proc else None, "residual_mib": e.residual_mib,
                      "kv_tokens": e.start_info.get("kv_tokens"), "start_to_health_s": e.start_info.get("start_to_health_s"),
                      "kv_fraction_of_solo": self.kv_fraction(p, e.start_info.get("kv_tokens")),
                      "host_rss_mib": e.host_rss_mib()}
        act = self.engines.get(self.active) if self.active else None
        health = None
        if act is not None:
            health = "ok" if (act.alive() and await act.health(timeout=2)) else "down"
        return {"node": self.node, "gpu": self.gpu, "mode": self.mode, "active_profile": self.active,
                "endpoint": act.base if act else None, "engine_health": health, "manager_health": self.health_state,
                "switching": self.switching, "last_switch": self.last_switch, "engines": eng,
                "resident_group": list(self.group), "restart_only": sorted(getattr(self, "explicit_restart_only", set())),
                "restart_only_auto": sorted(self.auto_restart_only), "gateway": self.gateway_url,
                "gpu_used_mib": gpus_used_mib(self.gpus), "total_mib": self.total_mib,
                "residual_mib_table": self.residual, "uptime_s": time.time() - self.started}

    async def health_loop(self) -> None:
        while not self.stop_event.is_set():
            await asyncio.sleep(self.args.health_interval)
            if self.switching:
                continue
            for p, e in list(self.engines.items()):
                if e.proc is not None and not e.alive() and e.state != "dead":
                    e.state = "dead"
                    self.event("engine_died", profile=p, returncode=e.proc.returncode)
                    if self.active == p:
                        self.health_state = "down"
            act = self.engines.get(self.active) if self.active else None
            if act is not None and act.alive() and not await act.health(timeout=3):
                self.event("engine_unhealthy", profile=self.active)
                self.health_state = "down"
            elif act is not None and act.alive():
                self.health_state = "ok"

    async def startup(self) -> None:
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0))
        if not self.total_mib:
            code = "import torch;print(torch.cuda.mem_get_info()[1])"
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.gpus[0]))
            out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=180).stdout
            self.total_mib = int(out.split()[-1]) / (1 << 20)
        self.base_used_mib = gpus_used_mib(self.gpus)
        self._health_task = asyncio.create_task(self.health_loop())
        self.event("start", gpu=self.gpu, gpus=self.gpus, total_mib=self.total_mib, base_used_mib=self.base_used_mib, mode=self.mode,
                   restart_only=sorted(self.restart_only))

    async def shutdown(self) -> None:
        self.stop_event.set()
        if self._health_task:
            self._health_task.cancel()
        await self.gateway_call({"paused": False}, timeout=5)
        for p, e in list(self.engines.items()):
            await e.stop()
        self.engines.clear()
        self.event("stop")
        if self.session:
            await self.session.close()
        try:
            os.remove(self.pidfile)
        except OSError:
            pass
        self.evf.close()


def build_app(m: Manager) -> web.Application:
    async def j(request):
        try:
            return await request.json()
        except ValueError:
            raise web.HTTPBadRequest(text='{"error": "JSON body required"}', content_type="application/json")

    async def h_prepare(request):
        js = await j(request)
        if not isinstance(js.get("profiles"), list) or not js["profiles"]:
            return web.json_response({"error": "profiles must be a non-empty list"}, status=400)
        if js.get("gateway"):
            m.gateway_url = js["gateway"]
        try:
            return web.json_response(await m.prepare(js["profiles"], js.get("restart_only")))
        except KeyError as e:
            return web.json_response({"error": str(e)}, status=400)
        except Exception as e:
            return web.json_response({"error": repr(e)}, status=500)

    async def h_activate(request):
        js = await j(request)
        if not isinstance(js.get("profile"), str):
            return web.json_response({"error": "profile (checkpoint key) required"}, status=400)
        try:
            return web.json_response(await m.activate(js["profile"], js.get("gateway"), js.get("mode")))
        except KeyError as e:
            return web.json_response({"error": str(e)}, status=400)
        except Exception as e:
            return web.json_response({"error": repr(e)}, status=500)

    async def h_status(request):
        return web.json_response(await m.status())

    async def h_shutdown(request):
        m.stop_event.set()
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_post("/prepare", h_prepare)
    app.router.add_post("/activate", h_activate)
    app.router.add_get("/status", h_status)
    app.router.add_post("/shutdown", h_shutdown)
    return app


async def amain(args):
    m = Manager(args)
    await m.startup()
    runner = web.AppRunner(build_app(m), access_log=None)
    await runner.setup()
    await web.TCPSite(runner, args.ctl_host, args.ctl_port).start()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, m.stop_event.set)
    print(f"engine-mgr node {args.node} gpu {args.gpu}: control {args.ctl_host}:{args.ctl_port}, mode {args.mode}", flush=True)
    await m.stop_event.wait()
    await runner.cleanup()
    await m.shutdown()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="CIARA edge engine manager")
    ap.add_argument("--node", type=int, required=True)
    ap.add_argument("--gpu", required=True, help="physical GPU index (CUDA_VISIBLE_DEVICES of the engines); a comma pair such as 0,1 runs one tensor parallel engine (cloud replica)")
    ap.add_argument("--budget", type=float, default=BUDGET, help="gpu_memory_utilization of the engine (edge 0.95, cloud 0.90)")
    ap.add_argument("--cores", default="", help="taskset cpu list for the engines, e.g. 0-1,32-33")
    ap.add_argument("--run-id", default=os.environ.get("RUN_ID", "adhoc"))
    ap.add_argument("--log-dir", default=os.environ.get("LOG_DIR") or os.path.join(os.environ.get("CIARA_ROOT", "."), "runs"))
    ap.add_argument("--ctl-host", default="127.0.0.1")
    ap.add_argument("--ctl-port", type=int, default=None, help="default 9200 + 10 * node")
    ap.add_argument("--bind-host", required=True, help="address the engines listen on (the bridge gateway address)")
    ap.add_argument("--engine-port-base", type=int, default=None, help="default ctl port + 1")
    ap.add_argument("--mode", choices=["sleep", "restart"], default="sleep")
    ap.add_argument("--restart-only", default="", help="comma list of profiles served by cold restart")
    ap.add_argument("--min-kv-fraction", type=float, default=0.5,
                    help="a profile whose predicted KV capacity beside the other resident engines is below this fraction of its solo capacity is restart-only")
    ap.add_argument("--gateway-url", default="")
    ap.add_argument("--calib-file", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration.json"))
    ap.add_argument("--total-mib", type=float, default=float(os.environ.get("CIARA_TOTAL_MIB", "0")), help="torch total memory of the card in MiB (measured when 0; the environment variable CIARA_TOTAL_MIB sets the default, for dry runs with fake engines)")
    ap.add_argument("--solo-sleep-mode", action="store_true", help="keep --enable-sleep-mode on restart-only engines")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--start-timeout", type=float, default=600.0)
    ap.add_argument("--drain-timeout", type=float, default=60.0)
    ap.add_argument("--health-interval", type=float, default=5.0)
    a = ap.parse_args(argv)
    a.gpus = [int(x) for x in str(a.gpu).split(",") if x != ""]
    a.gpu = a.gpus[0]
    a.ctl_port = a.ctl_port or 9200 + 10 * a.node
    a.engine_port_base = a.engine_port_base or a.ctl_port + 1
    return a


if __name__ == "__main__":
    asyncio.run(amain(parse_args()))
