from __future__ import annotations

import argparse
import asyncio
import errno
import json
import os
import signal
import socket
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from . import bank as bankmod
from . import cpuwork
from . import httpclient as hc
from . import shaping

RUNTIME_VERSION = "ciara-clients-runtime/2"
CHAT_PATH = "/v1/chat/completions"
COMPACT_PATH = "/v1/ciara/compose"


def _chown(path: str) -> None:
    try:
        os.chown(path, int(os.environ["CIARA_UID"]), int(os.environ.get("CIARA_GID", os.environ["CIARA_UID"])))
    except (KeyError, OSError):
        pass


def _mk(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        _chown(path)


def load_group(path: str, node: int, cls: str) -> Dict[str, Any]:
    meta: Optional[Dict[str, Any]] = None
    devices: List[Dict[str, Any]] = []
    reqs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            t = r["type"]
            if t == "meta":
                meta = r
            elif t == "device" and r["node"] == node and r["class"] == cls:
                devices.append(r)
            elif t == "req" and r["node"] == node and r["class"] == cls:
                reqs.append(r)
    if meta is None:
        raise SystemExit("schedule has no meta record")
    devices.sort(key=lambda d: d["idx"])
    reqs.sort(key=lambda r: (r["t_ms"], r["req"]))
    return {"meta": meta, "devices": devices, "reqs": reqs}


class Device:
    def __init__(self, rec: Dict[str, Any], link: Dict[str, Any], cls_cfg: Dict[str, Any], out_dir: str,
                 port_base: int, port_block: int) -> None:
        self.rec = rec
        self.id = rec["device"]
        self.idx = rec["idx"]
        self.link = link
        self.share = float(cls_cfg["cpu_share"])
        self.idle: List[hc.Conn] = []
        self.in_use: set = set()
        self.first_port = port_base + port_block * self.idx
        self.block = port_block
        self._rr = 0
        self.path = os.path.join(out_dir, f"{self.id}.jsonl")
        self.fh = open(self.path, "a", buffering=1, encoding="utf-8")
        _chown(self.path)

    def ports(self):
        for k in range(self.block):
            p = self.first_port + (self._rr + k) % self.block
            if p not in self.in_use:
                yield p

    def log(self, rec: Dict[str, Any]) -> None:
        self.fh.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n")

    def close(self) -> None:
        for c in self.idle:
            c.close()
        self.idle.clear()
        self.fh.close()


class Runtime:
    def __init__(self, a: argparse.Namespace) -> None:
        self.a = a
        grp = load_group(a.schedule, a.node, a.cls)
        self.meta = grp["meta"]
        self.reqs = grp["reqs"]
        self.cls_cfg = self.meta["classes"][a.cls]
        self.name = f"client-{a.node}-{a.cls}"
        self.out = a.out
        _mk(self.out)
        self.host, _, port = a.gateway.replace("http://", "").partition(":")
        self.port = int(port or 80)
        self.token = os.environ.get("CIARA_TOKEN", "")
        port_base, port_block = int(self.meta["port_base"]), int(self.meta["port_block"])
        self.devs: Dict[str, Device] = {}
        links: List[Optional[Dict[str, Any]]] = []
        for d in grp["devices"]:
            link = dict(self.meta["links"][d["link"]])
            link["seed_up"] = shaping.stable_seed(self.meta["seed"], d["device"], "up")
            link["seed_down"] = shaping.stable_seed(self.meta["seed"], d["device"], "down")
            links.append(link)
        self.links = links
        for d, link in zip(grp["devices"], links):
            self.devs[d["device"]] = Device(d, link, self.cls_cfg, self.out, port_base, port_block)
        need = {r["item"] for r in self.reqs}
        self.wire = a.wire
        if self.wire == "compact":
            if not a.compact_bank:
                raise SystemExit("--wire compact needs --compact-bank")
            bank = bankmod.read_bank(a.compact_bank)
        else:
            if not a.bank:
                raise SystemExit("--wire full needs --bank")
            bank = bankmod.read_bank(a.bank)
        self.bank = {k: bank[k] for k in need if k in bank}
        missing = need - set(self.bank)
        if missing:
            raise SystemExit(f"{len(missing)} items missing from the payload bank, for example {sorted(missing)[:3]}")
        wrong = [r["req"] for r in self.reqs if int(self.bank[r["item"]].get("family", r["family"])) != int(r["family"])]
        if wrong:
            raise SystemExit(f"{len(wrong)} requests name an item of another family in the bank, for example {wrong[:3]}")
        self.req_common = self.meta["request"]
        self.tasks: set = set()
        self.counters = {"dispatched": 0, "completed": 0, "errors": 0, "retries": 0, "late_dispatch_ms_max": 0.0}
        self.inflight = 0
        self.t0_mono = 0
        self.ev = open(os.path.join(self.out, f"{self.name}.runtime.jsonl"), "a", buffering=1)
        _chown(os.path.join(self.out, f"{self.name}.runtime.jsonl"))
        self.mon_path = os.path.join(self.out, f"{self.name}.monitor.jsonl")
        self.mon = open(self.mon_path, "a", buffering=1)
        _chown(self.mon_path)
        self.lag_samples: List[float] = []


    def event(self, kind: str, **kw: Any) -> None:
        self.ev.write(json.dumps({"kind": kind, "mono_ns": time.monotonic_ns(), "wall_ns": time.time_ns(), **kw}, separators=(",", ":")) + "\n")

    def body_for(self, r: Dict[str, Any]) -> Any:
        item = self.bank[r["item"]]
        if self.wire == "compact":
            payload = item["payload"]

            def compact_fn() -> bytes:
                return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

            if r["family"] == 1:
                return cpuwork.prep_family1_compact(payload["text"], payload["home"], compact_fn)
            return cpuwork.prep_family2_compact(payload["readings"], payload["q"], compact_fn)
        fam_cfg = self.meta["families"][str(r["family"])]

        def body_fn() -> bytes:
            req = {"model": self.meta["model"], "messages": item["messages"], "stream": bool(self.req_common["stream"]),
                   "temperature": self.req_common["temperature"], "seed": self.req_common["seed"],
                   "max_tokens": int(fam_cfg["max_tokens"]),
                   "chat_template_kwargs": {"enable_thinking": bool(self.req_common["enable_thinking"])}}
            if self.req_common.get("include_usage"):
                req["stream_options"] = {"include_usage": True}
            return json.dumps(req, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

        msgs = item["messages"]
        system = msgs[0]["content"] if msgs and msgs[0]["role"] == "system" else ""
        user = msgs[-1]["content"]
        gen = (cpuwork.prep_family1 if r["family"] == 1 else cpuwork.prep_family2)(system, user, body_fn)
        return gen

    async def connect(self, dev: Device) -> hc.Conn:
        last: Optional[BaseException] = None
        for p in list(dev.ports()):
            try:
                conn = await hc.open_conn(self.ip, self.port, p, float(self.cls_cfg["timeout_s"]))
                dev._rr = (p - dev.first_port + 1) % dev.block
                dev.in_use.add(p)
                return conn
            except TimeoutError:
                raise
            except OSError as e:
                if e.errno not in (errno.EADDRINUSE, errno.EADDRNOTAVAIL):
                    raise
                last = e
                continue
        raise ConnectionError(f"no usable source port in the block of {dev.id}: {last}")

    def release(self, dev: Device, conn: hc.Conn, keep: bool) -> None:
        if keep and self.cls_cfg["connection"]["mode"] == "reuse":
            conn.last_used = time.monotonic()
            conn.uses += 1
            dev.idle.append(conn)
        else:
            conn.close()
            dev.in_use.discard(conn.port)


    async def handle(self, r: Dict[str, Any], due_ns: int) -> None:
        dev = self.devs[r["device"]]
        cl = self.cls_cfg
        t_arr = hc.stamp()
        rec: Dict[str, Any] = {
            "req": r["req"], "device": r["device"], "node": r["node"], "class": r["class"], "family": r["family"], "item": r["item"],
            "deadline_ms": r["deadline_ms"], "link": dev.rec["link"], "sched_state": r["state"], "t_sched_ms": r["t_ms"],
            "headers": {"X-Ciara-Req": r["req"], "X-Ciara-Device": r["device"], "X-Ciara-Class": r["class"],
                        "X-Ciara-Family": str(r["family"]), "X-Ciara-Item": r["item"], "X-Ciara-Deadline-Ms": str(r["deadline_ms"])},
            "t_arrival_mono_ns": t_arr[0], "t_arrival_wall_ns": t_arr[1], "arrival_lag_ms": round((t_arr[0] - due_ns) / 1e6, 3),
            "error": None, "retries": 0, "wire": self.wire,
        }
        self.inflight += 1
        conn: Optional[hc.Conn] = None
        keep = False
        try:
            async with asyncio.timeout(float(cl["timeout_s"])):
                res, st = await cpuwork.run_throttled(dev.share, self.body_for(r))
                body: bytes = res["body"]
                rec["cpu_prep"] = st
                rec["prep_work"] = {k: v for k, v in res.items() if k != "body"}
                t = hc.stamp()
                rec["t_prep_done_mono_ns"], rec["t_prep_done_wall_ns"] = t
                headers = dict(rec["headers"])
                if self.token:
                    headers["Authorization"] = f"Bearer {self.token}"
                keep_alive = cl["connection"]["mode"] == "reuse"
                data = hc.build_request(f"{self.host}:{self.port}", COMPACT_PATH if self.wire == "compact" else CHAT_PATH,
                                        headers, body, keep_alive)
                rec["bytes_up"] = len(data)
                rec["bytes_up_body"] = len(body)
                resp: Optional[hc.Response] = None
                for attempt in range(2):
                    reused = False
                    t_c0 = time.monotonic_ns()
                    while dev.idle:
                        cand = dev.idle.pop()
                        if cand.reader.at_eof():
                            cand.close()
                            dev.in_use.discard(cand.port)
                            continue
                        conn, reused = cand, True
                        break
                    if conn is None:
                        conn = await self.connect(dev)
                    rec["conn_reused"] = reused
                    rec["conn_port"] = conn.port
                    rec["connect_ms"] = round((time.monotonic_ns() - t_c0) / 1e6, 3)
                    ts = hc.stamp()
                    rec["t_send_mono_ns"], rec["t_send_wall_ns"] = ts
                    try:
                        conn.writer.write(data)
                        await conn.writer.drain()
                        resp = await hc.read_response(conn)
                        break
                    except (ConnectionError, OSError):
                        conn.close()
                        dev.in_use.discard(conn.port)
                        conn = None
                        if attempt == 0 and reused:
                            rec["retries"] += 1
                            self.counters["retries"] += 1
                            continue
                        raise
                assert resp is not None and conn is not None
                keep = resp.keep_alive and resp.status < 500
                rec["status"] = resp.status
                rec["bytes_down"] = resp.bytes_down
                rec["t_first_byte_mono_ns"], rec["t_first_byte_wall_ns"] = resp.first_byte or (None, None)
                ft = resp.sse.first_token_stamp
                rec["t_first_token_mono_ns"], rec["t_first_token_wall_ns"] = ft or (None, None)
                rec["t_last_byte_mono_ns"], rec["t_last_byte_wall_ns"] = resp.last_byte or (None, None)
                text = "".join(resp.sse.text) if resp.sse.events or resp.sse.done else resp.body_text
                finish, usage = resp.sse.finish, resp.sse.usage
                if resp.status == 200 and not (resp.sse.events or resp.sse.done):
                    pj = hc.parse_json_reply(resp.body_text)
                    if pj is not None:
                        text, finish, usage = pj
                        rec["reply_format"] = "json"
                rec["reply_text"] = text
                if resp.sse.tool:
                    rec["reply_tool_calls"] = resp.sse.tool
                rec["finish_reason"] = finish
                rec["usage"] = usage
                rec["sse_events"] = resp.sse.events
                rec["sse_done"] = resp.sse.done
                if resp.sse.error:
                    rec["error"] = "sse_error: " + resp.sse.error
                self.release(dev, conn, keep)
                conn = None
                _, st2 = await cpuwork.run_throttled(dev.share, cpuwork.post_reply(text))
                rec["cpu_post"] = st2
        except TimeoutError:
            rec["error"] = "timeout"
        except asyncio.CancelledError:
            rec["error"] = "aborted"
            raise
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        finally:
            self.inflight -= 1
            if conn is not None:
                conn.close()
                dev.in_use.discard(conn.port)
            t_done = hc.stamp()
            rec["t_done_mono_ns"], rec["t_done_wall_ns"] = t_done
            end = rec.get("t_last_byte_mono_ns")
            rec["latency_ms"] = round((end - t_arr[0]) / 1e6, 3) if end else None
            rec["deadline_met"] = bool(end and rec.get("status") == 200 and rec["error"] is None and (end - t_arr[0]) / 1e6 <= r["deadline_ms"])
            dev.log(rec)
            self.counters["completed"] += 1
            if rec["error"]:
                self.counters["errors"] += 1


    async def dispatch(self) -> None:
        for r in self.reqs:
            due = self.t0_mono + int(r["t_ms"] * 1e6)
            delay = (due - time.monotonic_ns()) / 1e9
            if delay > 0:
                await asyncio.sleep(delay)
            late = (time.monotonic_ns() - due) / 1e6
            self.counters["late_dispatch_ms_max"] = max(self.counters["late_dispatch_ms_max"], late)
            task = asyncio.create_task(self.handle(r, due))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
            self.counters["dispatched"] += 1

    async def reaper(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            now = time.monotonic()
            idle_close = float(self.cls_cfg["connection"]["idle_close_s"])
            for dev in self.devs.values():
                keep = []
                for c in dev.idle:
                    if now - c.last_used > idle_close or c.reader.at_eof():
                        c.close()
                        dev.in_use.discard(c.port)
                    else:
                        keep.append(c)
                dev.idle = keep

    async def monitor(self) -> None:
        while True:
            lags = []
            t_end = time.monotonic() + 5.0
            while time.monotonic() < t_end:
                t = time.monotonic()
                await asyncio.sleep(0.1)
                lags.append((time.monotonic() - t - 0.1) * 1000.0)
            lags.sort()
            cpu = _cpu_stat()
            rec = {"mono_ns": time.monotonic_ns(), "wall_ns": time.time_ns(), "inflight": self.inflight,
                   "loop_lag_ms_p50": round(lags[len(lags) // 2], 3), "loop_lag_ms_p99": round(lags[int(len(lags) * 0.99) - 1], 3),
                   "loop_lag_ms_max": round(lags[-1], 3), "cgroup_cpu": cpu, **self.counters}
            self.mon.write(json.dumps(rec, separators=(",", ":")) + "\n")

    async def run(self) -> int:
        a = self.a
        self.event("start", runtime=RUNTIME_VERSION, node=a.node, cls=a.cls, devices=len(self.devs), requests=len(self.reqs),
                   gateway=a.gateway, schedule=os.path.abspath(a.schedule))
        deadline = time.monotonic() + 180.0
        while True:
            try:
                self.ip = socket.gethostbyname(self.host)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise SystemExit(f"name resolution failed for {self.host}")
                await asyncio.sleep(1.0)
        if not a.no_shaping:
            t = time.monotonic()
            cmds = shaping.apply(self.links)
            self.event("shaping_applied", commands=len(cmds), seconds=round(time.monotonic() - t, 3))
            with open(os.path.join(self.out, f"{self.name}.shaping.json"), "w") as f:
                json.dump({"devices": [{"device": d.id, "idx": d.idx, "profile": d.rec["link"], "up": shaping._direction(d.link, "up"),
                                        "down": shaping._direction(d.link, "down"), "first_port": d.first_port} for d in self.devs.values()],
                           "tc_commands": cmds}, f, indent=1)
            _chown(os.path.join(self.out, f"{self.name}.shaping.json"))
        self.t0_mono = a.start_at_mono_ns or time.monotonic_ns() + int(5e9)
        wall_at_start = time.time_ns() + (self.t0_mono - time.monotonic_ns())
        self.event("armed", t0_mono_ns=self.t0_mono, t0_wall_ns=wall_at_start)
        wait = (self.t0_mono - time.monotonic_ns()) / 1e9
        if wait > 0:
            await asyncio.sleep(wait)
        aux = [asyncio.create_task(self.reaper()), asyncio.create_task(self.monitor())]
        disp = asyncio.create_task(self.dispatch())
        stop_ev = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop_ev.set)
        stopper = asyncio.create_task(stop_ev.wait())
        horizon = float(self.meta["duration_s"]) + a.grace_s
        done, _ = await asyncio.wait({disp, stopper}, timeout=horizon, return_when=asyncio.FIRST_COMPLETED)
        if disp not in done:
            disp.cancel()
        end_by = self.t0_mono / 1e9 + horizon
        while self.tasks and not stop_ev.is_set() and time.monotonic_ns() / 1e9 < end_by:
            await asyncio.wait(list(self.tasks), timeout=1.0)
        stopper.cancel()
        pending = list(self.tasks)
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for t in aux:
            t.cancel()
        self.event("done", **self.counters, aborted=len(pending))
        if not a.no_shaping:
            try:
                stats = shaping.read_stats(len(self.links))
                with open(os.path.join(self.out, f"{self.name}.tc_stats.json"), "w") as f:
                    json.dump({"devices": {d.id: stats.get(d.idx, {}) for d in self.devs.values()}, "wall_ns": time.time_ns()}, f)
                _chown(os.path.join(self.out, f"{self.name}.tc_stats.json"))
            except Exception as e:
                self.event("tc_stats_failed", error=str(e))
        for d in self.devs.values():
            d.close()
        with open(os.path.join(self.out, f"{self.name}.status.json"), "w") as f:
            json.dump({"state": "done", **self.counters, "aborted": len(pending), "wall_ns": time.time_ns()}, f)
        _chown(os.path.join(self.out, f"{self.name}.status.json"))
        self.ev.close()
        self.mon.close()
        return 0


def _cpu_stat() -> Dict[str, int]:
    out: Dict[str, int] = {}
    try:
        with open("/sys/fs/cgroup/cpu.stat") as f:
            for line in f:
                k, v = line.split()
                if k in ("usage_usec", "user_usec", "system_usec", "nr_throttled", "throttled_usec"):
                    out[k] = int(v)
    except OSError:
        pass
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Client container runtime: replays the requests of one (edge node, device class) group against its edge gateway.', formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--bank", default=None, help="payload bank with the full messages (wire full only)")
    ap.add_argument("--wire", choices=["compact", "full"], default="compact",
                    help="compact: the device uploads its own content only; full: the whole prompt in a chat body")
    ap.add_argument("--compact-bank", default=None, help="compact bank of ciara_edge.compose_build (wire compact)")
    ap.add_argument("--node", type=int, required=True)
    ap.add_argument("--class", dest="cls", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gateway", required=True, help="http://edge-gw-<n>:8000")
    ap.add_argument("--start-at-mono-ns", type=int, default=0)
    ap.add_argument("--grace-s", type=float, default=300.0,
                    help="time after the schedule horizon before open requests are aborted (300 s = the class A timeout, so no request is cut)")
    ap.add_argument("--no-shaping", action="store_true")
    a = ap.parse_args(argv)
    rt = Runtime(a)
    return asyncio.run(rt.run())


if __name__ == "__main__":
    sys.exit(main())
