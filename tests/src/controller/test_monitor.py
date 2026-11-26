import json
import os
import signal
import subprocess
import sys
import time
import unittest

from ciara_control import promtext
from ciara_control.mocks.sim import SimNode
from ciara_control.mocks.stack import make_manager_app
from ciara_control.gwcore import JsonlWriter
from ciara_control.monitor import CpuSampler, MockGpuBackend, VllmScraper, read_grid
from ciara_control.util import JsonlLog, parse_int_ranges, read_jsonl
from helpers import ServerThread, TmpDir

VLLM_TEXT = """# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="Qwen 3.5 4B"} 3.0
vllm:num_requests_waiting{engine="0",model_name="Qwen 3.5 4B"} 2.0
vllm:kv_cache_usage_perc{engine="0"} 0.25
vllm:prompt_tokens_total{engine="0"} 1000.0
vllm:generation_tokens_total{engine="0"} 400.0
vllm:request_success_total{engine="0",finished_reason="stop"} 8.0
vllm:request_success_total{engine="0",finished_reason="length"} 2.0
vllm:e2e_request_latency_seconds_sum{engine="0"} 20.0
vllm:e2e_request_latency_seconds_count{engine="0"} 10.0
vllm:e2e_request_latency_seconds_bucket{engine="0",le="0.5"} 1.0
vllm:e2e_request_latency_seconds_bucket{engine="0",le="+Inf"} 10.0
process_cpu_seconds_total 12.5
python_gc_objects_collected_total{generation="0"} 99
"""


class Promtext(unittest.TestCase):
    def test_parse_and_collapse(self):
        s = promtext.parse(VLLM_TEXT)
        self.assertEqual(s['vllm:num_requests_running{engine="0",model_name="Qwen 3.5 4B"}'], 3.0)
        c = promtext.sum_by_name(s)
        self.assertEqual(c["vllm:request_success_total"], 10.0)
        self.assertNotIn("vllm:e2e_request_latency_seconds_bucket", c)
        self.assertEqual(set(promtext.keep_vllm(s)) & {"process_cpu_seconds_total"}, set())

    def test_inf_buckets_parse(self):
        s = promtext.parse('x_bucket{le="+Inf"} +Inf\ny NaN\n')
        self.assertEqual(s['x_bucket{le="+Inf"}'], float("inf"))

    def test_summarize_deltas_and_means(self):
        prev = promtext.sum_by_name(promtext.parse(VLLM_TEXT))
        cur = dict(prev)
        cur["vllm:prompt_tokens_total"] += 500
        cur["vllm:request_success_total"] += 5
        cur["vllm:e2e_request_latency_seconds_sum"] += 10.0
        cur["vllm:e2e_request_latency_seconds_count"] += 5
        out = promtext.summarize_vllm(cur, prev)
        self.assertEqual(out["running"], 3.0)
        self.assertEqual(out["prompt_tokens_delta"], 500)
        self.assertEqual(out["requests_finished_delta"], 5)
        self.assertAlmostEqual(out["e2e_mean_s"], 2.0)
        self.assertFalse(out["counter_reset"])

    def test_summarize_detects_restart_and_missing_baseline(self):
        prev = promtext.sum_by_name(promtext.parse(VLLM_TEXT))
        cur = dict(prev)
        cur["vllm:generation_tokens_total"] = 10.0
        out = promtext.summarize_vllm(cur, prev)
        self.assertTrue(out["counter_reset"])
        self.assertIsNone(out["generation_tokens_delta"])
        self.assertIsNone(promtext.summarize_vllm(cur, None)["prompt_tokens_delta"])

    def test_old_metric_names_are_accepted(self):
        out = promtext.summarize_vllm({"vllm:gpu_cache_usage_perc": 0.5}, None)
        self.assertEqual(out["kv_cache_usage"], 0.5)


def write_proc(root, cores, pids):
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "stat"), "w") as fh:
        fh.write("cpu  0 0 0 0 0 0 0 0 0 0\n")
        for c, busy in cores.items():
            fh.write(f"cpu{c} {busy} 0 0 1000 0 0 0 0 0 0\n")
    for pid, tasks in pids.items():
        pdir = os.path.join(root, str(pid))
        os.makedirs(os.path.join(pdir, "task"), exist_ok=True)
        total = sum(t for t, _ in tasks.values())
        _stat(os.path.join(pdir, "stat"), pid, "my proc", total, 0)
        for tid, (ticks, core) in tasks.items():
            tdir = os.path.join(pdir, "task", str(tid))
            os.makedirs(tdir, exist_ok=True)
            _stat(os.path.join(tdir, "stat"), tid, "my proc", ticks, core)


def _stat(path, pid, comm, utime, core):
    rest = ["0"] * 40
    rest[0], rest[11], rest[12], rest[36] = "S", str(utime), "0", str(core)
    with open(path, "w") as fh:
        fh.write(f"{pid} ({comm}) " + " ".join(rest) + "\n")


class CpuAccounting(unittest.TestCase):
    def sample_pair(self, own_uids, first, second, cores=(2, 3)):
        with TmpDir() as root:
            s = CpuSampler(cores=list(cores), own_uids=own_uids, proc_root=root)
            write_proc(root, *first)
            self.assertIsNone(s.sample(now=100.0))
            write_proc(root, *second)
            return s.sample(now=100.5)

    FIRST = ({2: 1000, 3: 500}, {100: {100: (0, 2), 101: (0, 3)}})
    SECOND = ({2: 1080, 3: 520}, {100: {100: (50, 2), 101: (20, 3)}, 200: {200: (30, 2)}})

    def test_threads_attributed_to_their_core_as_external(self):
        r = self.sample_pair({os.getuid() + 7}, self.FIRST, self.SECOND)
        self.assertEqual(r["cores"], [2, 3])
        self.assertEqual(r["dt_s"], 0.5)
        self.assertEqual(r["external_s"], [0.8, 0.2])
        self.assertEqual(r["own_s"], [0.0, 0.0])
        self.assertEqual(r["busy_s"], [0.8, 0.2])
        self.assertEqual(r["external_top"][0]["comm"], "my proc")

    def test_same_threads_count_as_own_for_the_experiment_uid(self):
        r = self.sample_pair({os.getuid()}, self.FIRST, self.SECOND)
        self.assertEqual(r["own_s"], [0.8, 0.2])
        self.assertEqual(r["external_s"], [0.0, 0.0])

    def test_idle_process_and_other_cores_are_ignored(self):
        first = ({2: 10, 3: 10, 4: 10}, {100: {100: (5, 4)}})
        second = ({2: 10, 3: 10, 4: 99}, {100: {100: (55, 4)}})
        r = self.sample_pair({os.getuid() + 7}, first, second)
        self.assertEqual(r["external_s"], [0.0, 0.0])

    def test_exited_process_is_dropped_without_error(self):
        with TmpDir() as root:
            s = CpuSampler(cores=[2], own_uids={0}, proc_root=root)
            write_proc(root, {2: 0}, {100: {100: (0, 2)}})
            s.sample(now=1.0)
            import shutil
            shutil.rmtree(os.path.join(root, "100"))
            write_proc(root, {2: 5}, {})
            r = s.sample(now=1.5)
            self.assertEqual(r["external_s"], [0.0])
            self.assertNotIn(100, s._pid_prev)

    @unittest.skipUnless(hasattr(os, "sched_setaffinity"), "needs sched_setaffinity")
    def test_real_busy_process_pinned_to_one_core(self):
        core = sorted(os.sched_getaffinity(0))[-1]
        child = subprocess.Popen([sys.executable, "-c", "while True: pass"])
        try:
            os.sched_setaffinity(child.pid, {core})
            time.sleep(0.2)
            for own, external_expected in (({os.getuid() + 12345}, True), ({os.getuid()}, False)):
                s = CpuSampler(cores=[core], own_uids=own)
                s.sample()
                time.sleep(1.0)
                r = s.sample()
                busy = r["external_s"][0] if external_expected else r["own_s"][0]
                other = r["own_s"][0] if external_expected else r["external_s"][0]
                self.assertGreater(busy, 0.6)
                self.assertLess(busy, 1.3)
                self.assertLess(other, 0.35)
        finally:
            child.kill()
            child.wait()


class FakeNvml:
    class NVMLError(Exception):
        pass

    NVML_TEMPERATURE_GPU, NVML_CLOCK_SM, NVML_CLOCK_MEM = 0, 1, 2
    NVML_FI_DEV_CLOCKS_EVENT_REASON_SW_POWER_CAP = 10
    NVML_FI_DEV_CLOCKS_EVENT_REASON_HW_THERM_SLOWDOWN = 11
    NVML_FI_DEV_CLOCKS_EVENT_REASON_SW_THERM_SLOWDOWN = 12
    NVML_FI_DEV_CLOCKS_EVENT_REASON_HW_POWER_BRAKE_SLOWDOWN = 13
    NVML_FI_DEV_CLOCKS_EVENT_REASON_SYNC_BOOST = 14
    NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION = 83

    def __init__(self):
        self.counters = {0: [0, 0, 0, 0, 5], 1: [0, 0, 0, 0, 5]}
        self.bits_calls = 0

    def nvmlInit(self): pass
    def nvmlShutdown(self): pass
    def nvmlDeviceGetCount(self): return 2
    def nvmlDeviceGetHandleByIndex(self, i): return i
    def nvmlDeviceGetEnforcedPowerLimit(self, h): return 250000

    def nvmlDeviceGetUtilizationRates(self, h):
        return type("U", (), {"gpu": 40 + h, "memory": 7})()

    def nvmlDeviceGetMemoryInfo(self, h):
        return type("M", (), {"used": 1048576 * 100, "total": 1048576 * 11264})()

    def nvmlDeviceGetPowerUsage(self, h): return 61500
    def nvmlDeviceGetTemperature(self, h, k): return 55
    def nvmlDeviceGetClockInfo(self, h, k): return 1500 if k == self.NVML_CLOCK_SM else 7000

    def nvmlDeviceGetCurrentClocksThrottleReasons(self, h):
        self.bits_calls += 1
        return 0x1

    def nvmlDeviceGetComputeRunningProcesses(self, h): return []

    def nvmlDeviceGetFieldValues(self, h, ids):
        out = []
        for fid in ids:
            v = type("V", (), {})()
            if fid == self.NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION:
                val, typ = 123456, 3
            else:
                val, typ = self.counters[h][fid - 10], 3
            f = type("F", (), {})()
            f.nvmlReturn, f.valueType = 0, typ
            f.value = type("Val", (), {"dVal": 0.0, "uiVal": 0, "ulVal": 0, "ullVal": val, "sllVal": val, "siVal": 0})()
            out.append(f)
        return out


class NvmlBackendLogic(unittest.TestCase):
    def make(self, fake, **kw):
        import ciara_control.monitor as mon
        sys.modules["pynvml"] = fake
        try:
            return mon.NvmlBackend(**kw)
        finally:
            del sys.modules["pynvml"]

    def test_record_fields_and_decoding(self):
        fake = FakeNvml()
        b = self.make(fake, parallel=False)
        r = b.sample()
        self.assertEqual(len(r), 2)
        g = r[1]
        self.assertEqual((g["util_gpu"], g["util_mem"], g["mem_used_mib"], g["mem_total_mib"]), (41, 7, 100.0, 11264.0))
        self.assertEqual((g["power_w"], g["power_limit_w"], g["energy_mj"], g["sm_clock_mhz"]), (61.5, 250.0, 123456, 1500))
        self.assertEqual(g["throttle_ns"]["sync_boost"], 5)
        self.assertEqual(g["throttle_bits"], 0x1)
        self.assertFalse(g["throttled"])

    def test_gpus_option_samples_only_the_listed_cards_and_keeps_physical_indices(self):
        fake = FakeNvml()
        b = self.make(fake, parallel=False, gpus=[1])
        r = b.sample()
        self.assertEqual((b.n_gpus, len(r), r[0]["gpu"]), (1, 1, 1))
        self.assertEqual(r[0]["util_gpu"], 41)

    def test_throttled_follows_the_advance_of_real_reasons_and_bits_are_subsampled(self):
        fake = FakeNvml()
        b = self.make(fake, parallel=False, bits_every=4)
        b.sample()
        r = b.sample()
        self.assertIsNone(r[0]["throttle_bits"])
        self.assertFalse(r[0]["throttled"])
        fake.counters[0][1] += 250_000_000
        fake.counters[1][4] += 250_000_000
        r = b.sample()
        self.assertTrue(r[0]["throttled"])
        self.assertFalse(r[1]["throttled"])
        r = b.sample()
        self.assertFalse(r[0]["throttled"])
        b.sample()
        self.assertEqual(fake.bits_calls, 2 * 2)

    def test_parallel_sampling_returns_gpus_in_order(self):
        b = self.make(FakeNvml(), parallel=True)
        self.assertEqual([x["gpu"] for x in b.sample()], [0, 1])
        b.close()


class Streams(unittest.TestCase):
    def test_mock_gpu_backend_record_shape(self):
        b = MockGpuBackend(3)
        recs = b.sample()
        self.assertEqual([r["gpu"] for r in recs], [0, 1, 2])
        for key in ("power_w", "energy_mj", "util_gpu", "mem_used_mib", "sm_clock_mhz", "throttle_bits",
                    "n_external_procs"):
            self.assertIn(key, recs[0])

    def test_parse_int_ranges(self):
        self.assertEqual(parse_int_ranges("2-5,38-41,7"), [2, 3, 4, 5, 7, 38, 39, 40, 41])

    def test_scraper_writes_one_record_per_resident_engine_and_the_cloud_engine(self):
        with TmpDir() as d:
            sim = SimNode("E1", 1, now_s=time.monotonic, seed=1, rate=2.0)
            log = JsonlWriter(os.path.join(d, "mock.jsonl"))
            with ServerThread(lambda: make_manager_app(sim, log)) as srv:
                sim.endpoint_base = srv.url
                sim.prepare(["q35-4b-fp16", "q35-2b-fp16"])
                sim.activate("q35-4b-fp16", switch_s=0.0)
                out = JsonlLog(os.path.join(d, "monitor-vllm.jsonl"), durable=False)
                import threading
                sc = VllmScraper({"E1": srv.url}, None, out, 1.0, None, 0.0, 3.0, threading.Event())
                sc.scrape_all(5)
                out.close()
            log.close()
            recs = read_jsonl(os.path.join(d, "monitor-vllm.jsonl"))
            self.assertEqual(sorted((r["profile"], r["active"]) for r in recs),
                             [("q35-2b-fp16", False), ("q35-4b-fp16", True)])
            r = recs[0]
            self.assertEqual((r["node"], r["epoch"]), ("E1", 5))
            self.assertTrue(all(k.startswith("vllm:") for k in r["series"]))
            self.assertTrue(any("_bucket" in k for k in r["series"]))
            self.assertIn("mono_ns", r)
            self.assertIn("wall_ns", r)

    def test_read_grid_reconstructs_origin_from_wall_time(self):
        with TmpDir() as d:
            self.assertEqual(read_grid(d, 0.0), (None, None))
            wall0 = time.time_ns() - 7_000_000_000
            with open(os.path.join(d, "manifest.json"), "w") as fh:
                json.dump({"t0_wall_ns": wall0, "epoch_ns": 30_000_000_000}, fh)
            t0, es = read_grid(d, 0.0)
            self.assertEqual(es, 30.0)
            self.assertAlmostEqual((time.monotonic_ns() - t0) / 1e9, 7.0, delta=0.5)

    def test_monitor_process_writes_gpu_cpu_and_stops_on_sigterm(self):
        with TmpDir() as d:
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(
                [os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "src", "controller"), env.get("PYTHONPATH", "")])
            p = subprocess.Popen([sys.executable, "-m", "ciara_control.monitor", "--run-dir", d, "--gpu-backend",
                                  "mock", "--no-vllm", "--interval", "0.25", "--cores", "0"], env=env)
            time.sleep(2.3)
            p.send_signal(signal.SIGTERM)
            self.assertEqual(p.wait(timeout=10), 0)
            gpu = read_jsonl(os.path.join(d, "monitor-gpu.jsonl"))
            cpu = read_jsonl(os.path.join(d, "monitor-cpu.jsonl"))
            self.assertEqual(gpu[0]["event"], "monitor_start")
            self.assertEqual(gpu[-1]["event"], "monitor_stop")
            samples = [g for g in gpu if g["event"] == "sample"]
            self.assertGreaterEqual(len(samples), 6)
            gaps = [(b["mono_ns"] - a["mono_ns"]) / 1e9 for a, b in zip(samples, samples[1:])]
            self.assertLess(max(gaps), 0.5)
            self.assertGreaterEqual(len(cpu), 5)
            self.assertEqual(cpu[0]["cores"], [0])
            for key in ("busy_s", "own_s", "external_s", "dt_s", "mono_ns", "wall_ns"):
                self.assertIn(key, cpu[0])


if __name__ == "__main__":
    unittest.main()
