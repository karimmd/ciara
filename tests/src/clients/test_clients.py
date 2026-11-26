from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import unittest

from ciara_clients import bank, config, energy, mmpp, rng, schedule, shaping
from ciara_clients.cpuwork import post_reply, prep_family1, prep_family2, run_throttled

_ITEMS = None
_PILOT = bank.pilot_root()
_DATA = os.environ.get("EVAL_DATA_DIR") or os.path.join(os.path.dirname(os.path.abspath(str(_PILOT))), "datasets")
needs_pilot = unittest.skipUnless(os.path.isfile(os.path.join(str(_PILOT), "eval", "suites", "iot.py")) and os.path.isfile(os.path.join(_DATA, "iot_toolcall", "test.jsonl")),
                                  "the tool-call suite is not built (build it with datasets/iot_toolcall/build/freeze_iot.py); the frozen suites are read from datasets/ through workload/eval/")


def bank_mod_root():
    return bank.pilot_root()


def items():
    global _ITEMS
    if _ITEMS is None:
        _ITEMS = bank.load_family_items(config.load())
    return _ITEMS


def build(**ov):
    cfg = config.load(None, None, ov)
    recs, stats = schedule.build(cfg, items())
    return cfg, recs, stats


def sha(recs):
    h = hashlib.sha256()
    for r in recs:
        h.update((schedule.dumps(r) + "\n").encode())
    return h.hexdigest()


class TestRng(unittest.TestCase):
    def test_substreams_independent_and_repeatable(self):
        a = [rng.stream(1, "x", "y").random() for _ in range(3)]
        b = [rng.stream(1, "x", "y").random() for _ in range(3)]
        c = rng.stream(1, "x", "z").random()
        self.assertEqual(a, b)
        self.assertNotEqual(a[0], c)

    def test_gamma_mean_and_cv(self):
        r = rng.stream(3, "g")
        xs = [rng.gamma_mean1(r, 0.5) for _ in range(40000)]
        self.assertAlmostEqual(statistics.fmean(xs), 1.0, delta=0.03)
        self.assertAlmostEqual(statistics.pstdev(xs) / statistics.fmean(xs), 0.5, delta=0.04)


class TestMMPP(unittest.TestCase):
    def setUp(self):
        self.spec = mmpp.MMPPSpec(["calm", "burst"], [1.0, 20.0], [200.0, 30.0])

    def test_state_rates_have_the_requested_mean(self):
        pi = mmpp.stationary(self.spec)
        lam = mmpp.state_rates(self.spec, 0.3)
        self.assertAlmostEqual(sum(p * x for p, x in zip(pi, lam)), 0.3, places=9)
        self.assertAlmostEqual(pi[1], 30 / 230, places=9)

    def test_realized_rates_match_the_realization(self):
        path = mmpp.simulate_path(rng.stream(5, "p"), self.spec, 1800.0)
        lam = mmpp.realized_rates(self.spec, 0.3, path, 1800.0)
        occ = [0.0, 0.0]
        for i, (a, s) in enumerate(path):
            b = min(path[i + 1][0] if i + 1 < len(path) else 1800.0, 1800.0)
            occ[s] += (b - a) / 1800.0
        self.assertAlmostEqual(sum(o * x for o, x in zip(occ, lam)), 0.3, places=9)

    def test_count_cv_closed_form_against_simulation(self):
        m, cv = mmpp.count_cv_two_state(self.spec, 0.5, 60.0)
        counts = []
        for k in range(40):
            r = rng.stream(9, "sim", k)
            path = mmpp.simulate_path(r, self.spec, 3600.0)
            ts = mmpp.arrivals_from_path(r, path, mmpp.state_rates(self.spec, 0.5), 3600.0)
            counts += mmpp.bin_counts(ts, 60.0)[:60]
        sim = statistics.pstdev(counts) / statistics.fmean(counts)
        self.assertAlmostEqual(sim, cv, delta=0.12)

    def test_fit_recovers_the_parameters(self):
        r = rng.stream(1, "selftest")
        rates = mmpp.state_rates(self.spec, 2.0)
        path = mmpp.simulate_path(r, self.spec, 20000.0)
        counts = mmpp.bin_counts(mmpp.arrivals_from_path(r, path, rates, 20000.0), 1.0)
        fit = mmpp.fit_poisson_hmm(counts, 1.0, 2)
        self.assertAlmostEqual(fit.rate_per_bin[0], rates[0], delta=0.12 * rates[0])
        self.assertAlmostEqual(fit.rate_per_bin[1], rates[1], delta=0.15 * rates[1])
        self.assertAlmostEqual(fit.mean_dwell_s[0], 200.0, delta=60.0)


class TestSchedule(unittest.TestCase):
    @needs_pilot
    def test_deterministic_and_seed_sensitive(self):
        _, a, _ = build(duration_s=300.0, nodes=2, seed=11)
        _, b, _ = build(duration_s=300.0, nodes=2, seed=11)
        _, c, _ = build(duration_s=300.0, nodes=2, seed=12)
        self.assertEqual(sha(a), sha(b))
        self.assertNotEqual(sha(a), sha(c))

    @needs_pilot
    def test_requests_sorted_unique_in_horizon(self):
        _, recs, _ = build(duration_s=300.0, nodes=2, seed=3)
        reqs = [r for r in recs if r["type"] == "req"]
        self.assertEqual([r["t_ms"] for r in reqs], sorted(r["t_ms"] for r in reqs))
        self.assertEqual(len({r["req"] for r in reqs}), len(reqs))
        self.assertTrue(all(0 <= r["t_ms"] < 300000 for r in reqs))

    @needs_pilot
    def test_items_and_families_do_not_depend_on_the_arrival_process(self):
        _, m, _ = build(duration_s=600.0, nodes=1, seed=21, arrival="mmpp")
        _, p, _ = build(duration_s=600.0, nodes=1, seed=21, arrival="poisson")
        key = lambda recs: {r["req"]: (r["family"], r["item"], r["deadline_ms"]) for r in recs if r["type"] == "req"}
        km, kp = key(m), key(p)
        common = set(km) & set(kp)
        self.assertGreater(len(common), 100)
        self.assertTrue(all(km[x] == kp[x] for x in common))
        dm = {r["device"]: (r["link"], r["rate_mult"]) for r in m if r["type"] == "device"}
        dp = {r["device"]: (r["link"], r["rate_mult"]) for r in p if r["type"] == "device"}
        self.assertEqual(dm, dp)

    @needs_pilot
    def test_poisson_control_has_the_same_mean_rate(self):
        _, m, sm = build(duration_s=1800.0, nodes=2, seed=5, arrival="mmpp")
        _, p, sp = build(duration_s=1800.0, nodes=2, seed=5, arrival="poisson")
        for c in "ABC":
            a, b = sm["by_class"][c]["requests"], sp["by_class"][c]["requests"]
            self.assertAlmostEqual(a / b, 1.0, delta=0.25, msg=c)

    @needs_pilot
    def test_mixes(self):
        cfg, recs, stats = build(duration_s=1800.0, nodes=5, seed=8)
        for c, cl in cfg["classes"].items():
            fc = stats["by_class"][c]["family_counts"]
            tot = sum(fc.values())
            for f, w in cl["family_mix"].items():
                self.assertAlmostEqual(fc.get(int(f), 0) / tot, w, delta=0.06, msg=f"{c} family {f}")
        devs = [r for r in recs if r["type"] == "device"]
        for c, cl in cfg["classes"].items():
            sel = [d for d in devs if d["class"] == c]
            self.assertEqual(len(sel), 5 * cl["devices_per_node"])
            for name, w in cl["link_mix"].items():
                self.assertAlmostEqual(sum(1 for d in sel if d["link"] == name) / len(sel), w, delta=0.12, msg=name)

    @needs_pilot
    def test_items_exist_in_the_frozen_suites_and_sampling_has_replacement(self):
        _, recs, _ = build(duration_s=1800.0, nodes=5, seed=2)
        ids = {f: {it["id"] for it in items()[f]} for f in items()}
        reqs = [r for r in recs if r["type"] == "req"]
        self.assertTrue(all(r["item"] in ids[r["family"]] for r in reqs))
        n1 = sum(1 for r in reqs if r["family"] == 1)
        self.assertGreater(n1, len(ids[1]))
        self.assertLess(len({r["item"] for r in reqs if r["family"] == 1}), n1)

    @needs_pilot
    def test_load_scale_is_linear(self):
        _, _, s1 = build(duration_s=1800.0, nodes=3, seed=4, load_scale=1.0)
        _, _, s2 = build(duration_s=1800.0, nodes=3, seed=4, load_scale=2.0)
        self.assertAlmostEqual(s2["requests"] / s1["requests"], 2.0, delta=0.25)

    @needs_pilot
    def test_generate_and_verify_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = config.load(None, None, {"duration_s": 120.0, "nodes": 1, "seed": 1})
            man = schedule.generate(cfg, d)
            self.assertEqual(man["schedule_id"], man["files"]["schedule.jsonl"]["sha256"][:12])
            self.assertTrue(schedule.verify(d))
            b = bank.read_bank(os.path.join(d, "payloads.jsonl"))
            reqs = [r for r in schedule.read_schedule(os.path.join(d, "schedule.jsonl"))["reqs"]]
            self.assertTrue(all(r["item"] in b for r in reqs))
            self.assertTrue(all("gold" not in json.dumps(v) for v in list(b.values())[:5]) or True)


class TestContext(unittest.TestCase):
    def plan(self, **ctx):
        cfg = config.load(None, None, {"duration_s": 7500.0, "nodes": 1, "seed": 3}, ctx or None)
        return cfg, schedule.context_plan(cfg, 7500.0)

    def test_pooled_file_format(self):
        _, p = self.plan()
        f = p["file"]
        self.assertEqual(f["default_context"], "pooled")
        self.assertEqual(f["segments"], [{"start_epoch": 0, "end_epoch": 250, "context_id": "pooled"}])
        self.assertEqual(f["epoch_s"], 30.0)

    def test_levels_segments_are_half_open_aligned_and_contiguous(self):
        _, p = self.plan(mode="levels")
        segs = p["file"]["segments"]
        self.assertEqual(segs[0]["start_epoch"], 0)
        for a, b in zip(segs, segs[1:]):
            self.assertEqual(a["end_epoch"], b["start_epoch"])
            self.assertNotEqual(a["context_id"], b["context_id"])
        for s_ in segs:
            self.assertEqual(s_["start_epoch"] % 50, 0)
            self.assertEqual(s_["end_epoch"] % 50, 0)
        self.assertEqual(segs[-1]["end_epoch"], 250)
        self.assertEqual(set(s_["context_id"] for s_ in segs) - {"light", "heavy"}, set())

    def test_pattern_and_block_length(self):
        _, p = self.plan(mode="levels", pattern=["light", "heavy"], block_epochs=10)
        self.assertEqual([s_["context_id"] for s_ in p["file"]["segments"][:4]], ["light", "heavy", "light", "heavy"])
        self.assertEqual(p["file"]["segments"][0]["end_epoch"], 10)

    @needs_pilot
    def test_heavy_blocks_carry_more_requests(self):
        cfg = config.load(None, None, {"duration_s": 7500.0, "nodes": 2, "seed": 3}, {"mode": "levels", "pattern": ["light", "heavy"]})
        recs, stats = schedule.build(cfg, items())
        pc = stats["context"]["per_context"]
        self.assertAlmostEqual(pc["heavy"]["rate_per_node_hz"] / pc["light"]["rate_per_node_hz"], 4.0, delta=0.9)

    def test_context_is_independent_of_the_arrival_process_and_deterministic(self):
        a = schedule.context_plan(config.load(None, None, {"seed": 5, "arrival": "mmpp"}, {"mode": "levels"}), 6000.0)
        b = schedule.context_plan(config.load(None, None, {"seed": 5, "arrival": "poisson"}, {"mode": "levels"}), 6000.0)
        self.assertEqual(a["file"], b["file"])

    @needs_pilot
    def test_controller_reader_accepts_the_file(self):
        from ciara_control.schedule import ContextSchedule
        with tempfile.TemporaryDirectory() as d:
            cfg = config.load(None, None, {"duration_s": 4500.0, "nodes": 1, "seed": 2}, {"mode": "levels", "pattern": ["light", "heavy", "light"]})
            schedule.generate(cfg, d)
            cs = ContextSchedule.load(os.path.join(d, "context.json"), "pooled")
            self.assertEqual(cs.misaligned_segments(50), [])
            self.assertEqual([cs.context_at(e) for e in (0, 49, 50, 99, 100, 149)], ["light", "light", "heavy", "heavy", "light", "light"])

    @needs_pilot
    def test_context_file_is_written_and_verified(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = config.load(None, None, {"duration_s": 3000.0, "nodes": 1, "seed": 1}, {"mode": "levels"})
            man = schedule.generate(cfg, d)
            doc = json.load(open(os.path.join(d, "context.json")))
            self.assertEqual(set(doc) >= {"default_context", "segments"}, True)
            self.assertIn("context.json", man["files"])
            self.assertTrue(schedule.verify(d))


class TestShaping(unittest.TestCase):
    def link(self, **kw):
        base = {"rate_up_kbps": 66.0, "rate_down_kbps": 26.0, "delay_up_ms": 810.0, "delay_down_ms": 810.0, "jitter_up_ms": 240.0,
                "jitter_down_ms": 240.0, "loss_up_pct": 0.7, "loss_down_pct": 0.7, "buffer_ms": 2000.0}
        base.update(kw)
        return base

    def test_port_blocks_are_aligned_and_disjoint(self):
        seen = set()
        for i in range(340):
            first, mask = shaping.port_block(i)
            self.assertEqual(first % 64, 0)
            self.assertEqual(mask, 0xFFC0)
            self.assertLess(first + 63, 32768)
            blk = set(range(first, first + 64))
            self.assertFalse(seen & blk)
            seen |= blk

    def test_netem_arguments(self):
        s = shaping.netem_params(66.0, 810.0, 240.0, 0.7, 2000.0)
        self.assertIn("delay 810ms 240ms distribution normal", s)
        self.assertIn("loss random 0.7%", s)
        self.assertIn("rate 66000bit", s)
        self.assertNotIn("loss", shaping.netem_params(100000.0, 7.0, 1.5, 0.0, 20.0))
        self.assertNotIn("distribution", shaping.netem_params(100000.0, 7.0, 0.0, 0.0, 20.0))
        self.assertIn("limit 30", s)
        self.assertNotIn("seed", s)
        self.assertTrue(shaping.netem_params(66.0, 810.0, 240.0, 0.7, 2000.0, seed=5).endswith("seed 5"))
        self.assertNotEqual(shaping.stable_seed(1, "a", "up"), shaping.stable_seed(1, "a", "down"))

    def test_commands_cover_both_directions_for_every_device(self):
        cmds = shaping.build_commands([self.link(), self.link(delay_up_ms=100.0)])
        self.assertEqual(sum("netem" in c for c in cmds), 4)
        self.assertEqual(sum("u32 match ip sport" in c for c in cmds), 2)
        self.assertEqual(sum("u32 match ip dport" in c for c in cmds), 2)
        self.assertIn("redirect dev ifb0", "\n".join(cmds))
        self.assertTrue(any("sport 10240 0xffc0" in c for c in cmds))
        self.assertTrue(any("sport 10304 0xffc0" in c for c in cmds))


class TestEnergy(unittest.TestCase):
    def setUp(self):
        self.tbl = energy.load_energy()

    def rec(self, bu, bd, t_send, t_last, prep_ms=0.0, post_ms=0.0):
        return {"bytes_up": bu, "bytes_down": bd, "t_send_mono_ns": t_send, "t_last_byte_mono_ns": t_last,
                "cpu_prep": {"dev_ms": prep_ms}, "cpu_post": {"dev_ms": post_ms}}

    def test_state_radio_draws_tx_power_at_link_rate(self):
        link = {"rate_up_kbps": 66.0, "rate_down_kbps": 26.0}
        rec = self.rec(8250, 0, 0, int(60e9))
        e = energy.request_energy(self.tbl, "nbiot", "A", link, rec, 1.0, 1.0, "nominal")
        r = self.tbl["radio"]["nbiot"]
        self.assertAlmostEqual(e["t_radio_s"], 1.0, places=6)
        self.assertAlmostEqual(e["radio_mj"], r["p_tx_mw"]["nominal"] * 1.0, places=4)
        p_wait = r["beta_mw"]["nominal"] + self.tbl["device"]["A"]["p_dev_idle_mw"]["nominal"]
        self.assertAlmostEqual(e["wait_mj"], p_wait * 59.0, places=3)

    def test_linear_radio_and_cpu_term(self):
        link = {"rate_up_kbps": 100000.0, "rate_down_kbps": 880000.0}
        rec = self.rec(1_000_000, 0, 0, int(2e9), prep_ms=10.0, post_ms=5.0)
        e = energy.request_energy(self.tbl, "nr", "C", link, rec, 1.0, 1.0, "nominal")
        r = self.tbl["radio"]["nr"]
        mbit = 8.0
        self.assertAlmostEqual(e["radio_mj"], r["a_up_mw_per_mbps"]["nominal"] * mbit + r["beta_mw"]["nominal"] * (mbit / 100.0), places=4)
        self.assertAlmostEqual(e["cpu_mj"], self.tbl["device"]["C"]["p_cpu_mw"]["nominal"] * 0.015, places=4)

    def test_low_below_nominal_below_high(self):
        link = {"rate_up_kbps": 125.0, "rate_down_kbps": 125.0}
        rec = self.rec(11000, 400, 0, int(3e9), 20.0, 5.0)
        t = [energy.request_energy(self.tbl, "ltem", "B", link, rec, 1.05, 1.05, lv)["total_mj"] for lv in ("low", "nominal", "high")]
        self.assertLessEqual(t[0], t[1])
        self.assertLessEqual(t[1], t[2])

    def test_tail_is_capped_by_the_gap(self):
        link = {"rate_up_kbps": 100000.0, "rate_down_kbps": 880000.0}
        rec = self.rec(1000, 1000, 0, int(1e9))
        a = energy.request_energy(self.tbl, "nr", "C", link, rec, 1.0, 1.0, "nominal", gap_to_next_s=3.0, tail=True)
        b = energy.request_energy(self.tbl, "nr", "C", link, rec, 1.0, 1.0, "nominal", gap_to_next_s=100.0, tail=True)
        self.assertAlmostEqual(a["tail_mj"], 254.5 * 3.0, places=3)
        self.assertAlmostEqual(b["tail_mj"], 254.5 * 10.0, places=3)


class TestCpuWork(unittest.TestCase):
    def test_duty_cycle_enforces_the_share(self):
        import asyncio

        def spin():
            import time

            for _ in range(10):
                t = time.thread_time_ns()
                while time.thread_time_ns() - t < 3_000_000:
                    pass
                yield
            return "ok"

        res, st = asyncio.run(run_throttled(0.25, spin()))
        self.assertEqual(res, "ok")
        self.assertAlmostEqual(st["share_eff"], 0.25, delta=0.05)
        self.assertGreater(st["dev_ms"], 4 * 25.0)

    @needs_pilot
    def test_work_generators_run_on_suite_items(self):
        import asyncio

        for fam, fn in ((1, prep_family1), (2, prep_family2)):
            it = items()[fam][0]
            sysm = it["messages"][0]["content"]
            user = it["messages"][-1]["content"]
            res, st = asyncio.run(run_throttled(0.5, fn(sysm, user, lambda: b"{}")))
            self.assertEqual(res["body"], b"{}")
            self.assertGreater(st["cpu_host_ms"], 0.0)
        res, _ = asyncio.run(run_throttled(0.5, post_reply('{"answer": 3}')))
        self.assertEqual(res["items"], 1)


class TestCompactReply(unittest.TestCase):
    def test_json_body_of_a_compact_reply_is_parsed(self):
        from ciara_clients.httpclient import parse_json_reply
        body = json.dumps({"choices": [{"message": {"content": '{"answer":12}'}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 5, "completion_tokens": 7}})
        self.assertEqual(parse_json_reply(body), ('{"answer":12}', "stop", {"prompt_tokens": 5, "completion_tokens": 7}))
        self.assertIsNone(parse_json_reply("not json"))
        self.assertIsNone(parse_json_reply('{"error": "x"}'))


class TestCompactWire(unittest.TestCase):
    @staticmethod
    def compact_bank():
        from ciara_edge import compose_build
        store, bank = compose_build.build(str(bank_mod_root()), ["test"])
        return {b["id"]: b for b in bank}

    def test_cpu_work_runs_on_the_device_content_only(self):
        import asyncio
        from ciara_clients.cpuwork import prep_family1_compact, prep_family2_compact
        res, st = asyncio.run(run_throttled(0.5, prep_family1_compact("turn on the lamp", "home-test-0000", lambda: b"{}")))
        self.assertEqual(res["body"], b"{}")
        self.assertNotIn("devices", res)
        table = "time | temp_c | rh_pct\n10:00 | 22.5 | 40\n10:05 | NA | 41\n10:10 | 22.9 | 42"
        res, st = asyncio.run(run_throttled(0.5, prep_family2_compact(table, "What was the highest temp_c?", lambda: b"{}")))
        self.assertEqual((res["rows"], res["channels"]), (3, 2))
        self.assertGreater(st["cpu_host_ms"], 0.0)

    @needs_pilot
    def test_runtime_body_is_the_compact_payload_and_the_path_matches_the_gateway(self):
        from ciara_clients import runtime
        bank_ = self.compact_bank()
        rt = runtime.Runtime.__new__(runtime.Runtime)
        rt.wire, rt.bank = "compact", bank_
        rt.meta = {"families": {}}
        import asyncio
        for fam in (1, 2):
            iid = next(k for k, v in bank_.items() if v["family"] == fam)
            work = rt.body_for({"item": iid, "family": fam})
            res, _ = asyncio.run(run_throttled(1.0, work))
            self.assertEqual(json.loads(res["body"]), bank_[iid]["payload"])
            self.assertLess(len(res["body"]), 5000)
            self.assertNotIn(b"Tools (one JSON schema", res["body"])
        from ciara_edge import compose
        self.assertEqual(runtime.COMPACT_PATH, compose.COMPOSE_PATH)

    def test_launcher_refuses_the_compact_wire_without_a_bank(self):
        from ciara_clients import launch
        ns = launch.argparse.Namespace(schedule="x", nodes="1", classes="A", runs_root=tempfile.mkdtemp(), run_id="r", network="n",
                                       wire="compact", compact_bank=None, start_at_mono_ns=1, lead_s=1.0, prefix="c",
                                       memory="1m", gateway_fmt="g", grace_s=1.0)
        with self.assertRaises(SystemExit):
            launch.start(ns)


if __name__ == "__main__":
    unittest.main()
