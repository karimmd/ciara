from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp

from .profiles import PILOT_ROOT, profile_info

PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mono() -> float:
    return time.monotonic()


@dataclass
class EngineSpec:
    profile: str
    gpu: int
    port: int
    host: str
    gmu: float
    token: str
    log_path: str
    cores: Optional[str] = None
    max_model_len: int = 8192
    max_num_seqs: int = 16
    sleep_mode: bool = True
    prefix_caching: bool = True
    extra_args: list = field(default_factory=list)
    gpus: Optional[str] = None
    tp: int = 1


def vllm_bin() -> str:
    return os.environ.get("VLLM_BIN", os.path.join(os.path.dirname(sys.executable), "vllm"))


def build_command(spec: EngineSpec) -> list:
    p = profile_info(spec.profile)
    cmd = []
    if spec.cores:
        cmd += ["taskset", "-c", spec.cores]
    cmd += [vllm_bin(), "serve", p.model_path, "--served-model-name", spec.profile,
            "--host", spec.host, "--port", str(spec.port), "--dtype", "float16", "--seed", "0",
            "--max-model-len", str(spec.max_model_len), "--max-num-seqs", str(spec.max_num_seqs),
            "--gpu-memory-utilization", f"{spec.gmu:.4f}", "--api-key", spec.token]
    if spec.tp > 1:
        cmd += ["--tensor-parallel-size", str(spec.tp)]
    cmd.append("--enable-prefix-caching" if spec.prefix_caching else "--no-enable-prefix-caching")
    if spec.sleep_mode:
        cmd.append("--enable-sleep-mode")
    if p.language_model_only:
        cmd.append("--language-model-only")
    cmd += list(spec.extra_args)
    return cmd


NO_NVML = os.environ.get("CIARA_NO_NVML") == "1"


def gpu_used_mib(gpu: int) -> int:
    if NO_NVML:
        return 0
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu)],
                         capture_output=True, text=True, timeout=20).stdout.strip()
    return int(out.splitlines()[0])


def gpu_total_mib(gpu: int) -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", str(gpu)],
                         capture_output=True, text=True, timeout=20).stdout.strip()
    return int(out.splitlines()[0])


def gpus_used_mib(gpus) -> int:
    return max(gpu_used_mib(g) for g in gpus)


LOG_PATTERNS = {
    "weights_gib": r"Model loading took ([0-9.]+) GiB",
    "available_kv_gib": r"Available KV cache memory: ([0-9.]+) GiB",
    "kv_tokens": r"GPU KV cache size: ([0-9,]+) tokens",
    "max_concurrency_8192": r"Maximum concurrency for [0-9,]+ tokens per request: ([0-9.]+)x",
    "consumed_gib": r"Actual usage is ([0-9.]+) GiB for consumed memory",
    "peak_activation_gib": r"GiB for consumed memory \(weights \+ non-torch\), ([0-9.]+) GiB for peak activation",
    "cudagraph_gib": r"for peak activation, and ([0-9.]+) GiB for CUDAGraph memory",
    "free_at_startup_gib": r"Free memory on device \(([0-9.]+)/[0-9.]+ GiB\) on startup",
    "total_gib": r"Free memory on device \([0-9.]+/([0-9.]+) GiB\) on startup",
    "sleep_freed_gib": r"Sleep mode freed ([0-9.]+) GiB memory",
    "sleep_remaining_gib": r"Sleep mode freed [0-9.]+ GiB memory, ([0-9.]+) GiB memory is still in use",
}


def parse_log(path: str) -> dict:
    try:
        txt = open(path, errors="replace").read()
    except OSError:
        return {}
    out = {}
    for k, pat in LOG_PATTERNS.items():
        m = re.findall(pat, txt)
        if m:
            v = m[-1].replace(",", "")
            out[k] = int(v) if k == "kv_tokens" else float(v)
    return out


def descendants(pid: int) -> list:
    import psutil
    try:
        return [pid] + [c.pid for c in psutil.Process(pid).children(recursive=True)]
    except psutil.NoSuchProcess:
        return []


def process_gpu_mib(pids, gpu: int) -> int:
    if NO_NVML:
        return 0
    uuid = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", str(gpu)],
                          capture_output=True, text=True, timeout=20).stdout.strip()
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=20).stdout
    tot = 0
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 3 and parts[0] == uuid and int(parts[1]) in set(pids):
            tot += int(parts[2])
    return tot


class Engine:
    def __init__(self, spec: EngineSpec, session: aiohttp.ClientSession):
        self.spec = spec
        self.session = session
        self.proc: Optional[subprocess.Popen] = None
        self.pgid: Optional[int] = None
        self.state = "absent"
        self.start_info: dict = {}
        self.residual_mib: Optional[int] = None
        self.awake_mib: Optional[int] = None

    @property
    def base(self) -> str:
        return f"http://{self.spec.host}:{self.spec.port}"

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.spec.token}"}

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    async def start(self, ready_timeout: float = 600.0) -> dict:
        env = dict(os.environ)
        env.update({"CUDA_VISIBLE_DEVICES": self.spec.gpus or str(self.spec.gpu), "VLLM_SERVER_DEV_MODE": "1",
                    "CIARA_TOKEN": self.spec.token, "PYTHONUNBUFFERED": "1",
                    "PYTHONPATH": PKG_PARENT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")})
        os.makedirs(os.path.dirname(self.spec.log_path), exist_ok=True)
        logf = open(self.spec.log_path, "ab")
        cmd = build_command(self.spec)
        t0 = mono()
        self.state = "starting"
        self.proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     start_new_session=True)
        logf.close()
        self.pgid = self.proc.pid
        deadline = t0 + ready_timeout
        while True:
            if self.proc.poll() is not None:
                self.state = "dead"
                raise RuntimeError(f"engine {self.spec.profile} exited with {self.proc.returncode} during start; "
                                   f"see {self.spec.log_path}")
            if await self.health(timeout=2.0):
                break
            if mono() > deadline:
                await self.stop()
                raise TimeoutError(f"engine {self.spec.profile} not healthy after {ready_timeout:.0f} s")
            await asyncio.sleep(0.5)
        t_health = mono() - t0
        self.state = "awake"
        self.start_info = {"start_to_health_s": t_health, "pid": self.proc.pid, "pgid": self.pgid, **parse_log(self.spec.log_path)}
        return self.start_info

    async def stop(self, grace: float = 30.0) -> float:
        t0 = mono()
        if self.proc is not None and self.pgid is not None:
            try:
                os.killpg(self.pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            while mono() - t0 < grace and self._group_alive():
                await asyncio.sleep(0.2)
            if self._group_alive():
                try:
                    os.killpg(self.pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                while mono() - t0 < grace + 15 and self._group_alive():
                    await asyncio.sleep(0.2)
            try:
                self.proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        self.state = "absent"
        self.residual_mib = None
        return mono() - t0

    def _group_alive(self) -> bool:
        try:
            os.killpg(self.pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        if self.proc is not None:
            self.proc.poll()
        try:
            os.killpg(self.pgid, 0)
            return True
        except ProcessLookupError:
            return False

    async def health(self, timeout: float = 3.0) -> bool:
        try:
            async with self.session.get(self.base + "/health", timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                return r.status == 200
        except Exception:
            return False

    async def is_sleeping(self) -> Optional[bool]:
        try:
            async with self.session.get(self.base + "/is_sleeping", headers=self.headers,
                                        timeout=aiohttp.ClientTimeout(total=5)) as r:
                return (await r.json())["is_sleeping"]
        except Exception:
            return None

    async def sleep(self, level: int = 1, mode: str = "wait", timeout: float = 180.0) -> float:
        t0 = mono()
        async with self.session.post(f"{self.base}/sleep?level={level}&mode={mode}", headers=self.headers,
                                     timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            if r.status != 200:
                raise RuntimeError(f"/sleep returned {r.status}: {(await r.text())[:200]}")
        dt = mono() - t0
        self.state = "sleeping"
        return dt

    async def wake(self, timeout: float = 240.0) -> float:
        t0 = mono()
        async with self.session.post(self.base + "/wake_up", headers=self.headers,
                                     timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            if r.status != 200:
                raise RuntimeError(f"/wake_up returned {r.status}: {(await r.text())[:200]}")
        dt = mono() - t0
        self.state = "awake"
        return dt

    async def reset_prefix_cache(self) -> None:
        async with self.session.post(self.base + "/reset_prefix_cache", headers=self.headers,
                                     timeout=aiohttp.ClientTimeout(total=30)) as r:
            await r.read()

    async def chat(self, messages, max_tokens: int = 1, logprobs: bool = False, timeout: float = 120.0) -> dict:
        return await chat_once(self.session, self.base, self.spec.token, self.spec.profile, messages, max_tokens,
                               logprobs=logprobs, timeout=timeout)

    async def probe(self, timeout: float = 60.0) -> float:
        r = await self.chat([{"role": "user", "content": "ping"}], max_tokens=1, timeout=timeout)
        return r["latency_s"]

    def host_rss_mib(self) -> Optional[int]:
        import psutil
        if self.proc is None:
            return None
        tot = 0
        for pid in descendants(self.proc.pid):
            try:
                tot += psutil.Process(pid).memory_info().rss
            except psutil.NoSuchProcess:
                pass
        return int(tot / (1 << 20))

    def gpu_mib(self) -> int:
        if self.proc is None:
            return 0
        pids = descendants(self.proc.pid)
        gl = [int(g) for g in self.spec.gpus.split(",")] if self.spec.gpus else [self.spec.gpu]
        return sum(process_gpu_mib(pids, g) for g in gl)


async def chat_once(session, base: str, token: str, model: str, messages, max_tokens: int, logprobs: bool = False,
                    timeout: float = 120.0) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0, "seed": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    if logprobs:
        body["logprobs"] = True
    t0 = mono()
    async with session.post(base + "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {token}"},
                            timeout=aiohttp.ClientTimeout(total=timeout)) as r:
        js = await r.json()
        if r.status != 200:
            raise RuntimeError(f"chat returned {r.status}: {str(js)[:200]}")
    dt = mono() - t0
    ch = js["choices"][0]
    out = {"text": ch["message"].get("content"), "finish_reason": ch.get("finish_reason"), "latency_s": dt,
           "usage": js.get("usage", {})}
    if logprobs and ch.get("logprobs"):
        toks = [(c["token"], c["logprob"]) for c in ch["logprobs"]["content"]]
        out["lp_digest"] = hashlib.sha256(json.dumps(toks).encode()).hexdigest()[:16]
        out["n_tokens"] = len(toks)
    return out
