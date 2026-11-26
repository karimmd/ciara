import json
import os
import unittest

from ciara_control.clock import FakeClock
from ciara_control.controller import EXIT_OK, EXIT_RESUME, SimulatedCrash
from ciara_control.runstate import load_run_state
from ciara_control.util import read_jsonl
from helpers import Fleet, TmpDir, base_config, make_controller

S = 1_000_000_000
DELTA = 30 * S


def crash_at(epoch):
    def hook(e):
        if e == epoch:
            raise SimulatedCrash()
    return hook


def logs(d):
    rows = read_jsonl(os.path.join(d, "blocks.jsonl"))
    return ([r for r in rows if r["rec"] == "block"], [r for r in rows if r["rec"] == "block_end"],
            read_jsonl(os.path.join(d, "epochs.jsonl")), read_jsonl(os.path.join(d, "events.jsonl")))


def incumbent_only(d, **over):
    cfg = base_config(d, **over)
    cfg["arms"] = [{"id": "INC", "kind": "incumbent", "profiles": ["q35-4b-fp16"],
                    "incumbent": {"factory": "stub", "kwargs": {"profile": "q35-4b-fp16"}}}]
    return cfg


def crash_and_resume(cfg, crash_epoch, downtime_ns, resume_cfg=None, same_fleet=True):
    clock = FakeClock()
    fleet = Fleet(clock)
    ctl = make_controller(cfg, clock, fleet, fault_hook=crash_at(crash_epoch))
    try:
        ctl.run()
        raise AssertionError("crash expected")
    except SimulatedCrash:
        pass
    clock.advance(downtime_ns)
    ctl2 = make_controller(resume_cfg or cfg, clock, fleet)
    code = ctl2.run(resume=True)
    return ctl, ctl2, fleet, code


class ResumeContinueBlock(unittest.TestCase):
    def test_reenters_block_with_same_arm_and_flags_missed_epochs(self):
        with TmpDir() as d:
            cfg = base_config(d)
            ctl, ctl2, fleet, code = crash_and_resume(cfg, crash_epoch=6, downtime_ns=20 * S)
            self.assertEqual(code, EXIT_OK)
            blocks, ends, epochs, events = logs(d)
            node = "E1"
            mine = sorted(e["epoch"] for e in epochs if e["node"] == node and e["block_id"] == 1)
            self.assertEqual(mine, [4, 7])
            end = [x for x in ends if x["node"] == node and x["block_id"] == 1]
            self.assertEqual(len(end), 1)
            self.assertEqual((end[0]["status"], end[0]["resumed"], end[0]["missed_epochs"]), ("partial", True, [5, 6]))
            ev = [x for x in events if x["event"] == "resume"][0]
            self.assertEqual((ev["from_epoch"], ev["last_logged_epoch"], ev["missed_epochs"]), (7, 4, [5, 6]))
            self.assertEqual(sum(1 for b in blocks if b["node"] == node and b["block_id"] == 1), 1)

    def test_no_duplicate_epochs_or_blocks_and_run_completes(self):
        with TmpDir() as d:
            cfg = base_config(d)
            ctl, ctl2, fleet, code = crash_and_resume(cfg, crash_epoch=10, downtime_ns=35 * S)
            blocks, ends, epochs, events = logs(d)
            keys = [(e["node"], e["epoch"]) for e in epochs]
            self.assertEqual(len(keys), len(set(keys)))
            bkeys = [(b["node"], b["block_id"]) for b in blocks]
            self.assertEqual(len(bkeys), len(set(bkeys)))
            self.assertEqual(len(bkeys), 3 * 8)
            self.assertEqual(len({(x["node"], x["block_id"]) for x in ends}), 3 * 8)
            with open(os.path.join(d, "STATUS")) as fh:
                self.assertEqual(fh.read().strip(), "DONE")
            m = load_run_state(d).manifest
            self.assertEqual([s["n"] for s in m["sessions"]], [1, 2])
            self.assertTrue(m["sessions"][1]["resume"])

    def test_draws_identical_to_an_uninterrupted_run(self):
        with TmpDir() as d1, TmpDir() as d2:
            c1 = base_config(d1, run_id="same")
            clock = FakeClock()
            make_controller(c1, clock, Fleet(clock)).run()
            ref = {(b["node"], b["block_id"]): (b["arm_id"], b["propensity"], b["rng"]["draw"]) for b in logs(d1)[0]}
            c2 = base_config(d2, run_id="same")
            crash_and_resume(c2, crash_epoch=13, downtime_ns=50 * S)
            got = {(b["node"], b["block_id"]): (b["arm_id"], b["propensity"], b["rng"]["draw"]) for b in logs(d2)[0]}
            for key, val in got.items():
                self.assertEqual(ref[key], val)
            self.assertGreaterEqual(len(got), 3 * 6)

    def test_incumbent_block_reentered_with_fresh_decisions_and_hook(self):
        with TmpDir() as d:
            cfg = incumbent_only(d)
            ctl, ctl2, fleet, code = crash_and_resume(cfg, crash_epoch=6, downtime_ns=20 * S)
            inst = ctl2.nodes[0].incumbents["INC"]
            resumed = [b for b in inst.blocks if b["resumed"]]
            self.assertEqual([b["epoch"] for b in resumed], [7])
            _, ends, epochs, _ = logs(d)
            e7 = [e for e in epochs if e["node"] == "E1" and e["epoch"] == 7][0]
            self.assertIsNotNone(e7["obs"])
            self.assertTrue(e7["config_ok"])

    def test_resume_repushes_configuration_idempotently(self):
        with TmpDir() as d:
            cfg = base_config(d)
            ctl, ctl2, fleet, code = crash_and_resume(cfg, crash_epoch=6, downtime_ns=20 * S)
            calls = [c for c in fleet.ios["E1"].calls if c[0] == "config" and c[1]["block_id"] == 1]
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][1], calls[1][1])


class ResumeNextBlock(unittest.TestCase):
    def test_dangling_block_closed_interrupted_and_nodes_idle_to_next_boundary(self):
        with TmpDir() as d:
            cfg = base_config(d)
            cfg["control"]["on_resume"] = "next_block"
            ctl, ctl2, fleet, code = crash_and_resume(cfg, crash_epoch=6, downtime_ns=20 * S)
            self.assertEqual(code, EXIT_OK)
            blocks, ends, epochs, events = logs(d)
            end = [x for x in ends if x["node"] == "E2" and x["block_id"] == 1][0]
            self.assertEqual(end["status"], "interrupted")
            self.assertTrue(end["recovered"])
            mine = sorted(e["epoch"] for e in epochs if e["node"] == "E2")
            self.assertEqual(mine, [0, 1, 2, 3, 4] + list(range(8, 32)))
            self.assertFalse(any(ev["event"] == "reenter_block" for ev in events))

    def test_resume_exactly_at_a_block_boundary(self):
        with TmpDir() as d:
            cfg = base_config(d)
            ctl, ctl2, fleet, code = crash_and_resume(cfg, crash_epoch=6, downtime_ns=53 * S)
            blocks, ends, epochs, events = logs(d)
            ev = [x for x in events if x["event"] == "resume"][0]
            self.assertEqual(ev["from_epoch"], 8)
            self.assertEqual({x["status"] for x in ends if x["block_id"] == 1}, {"interrupted"})
            self.assertEqual({e["epoch"] for e in epochs if e["node"] == "E1"} & {5, 6, 7}, set())
            self.assertEqual(sum(1 for b in blocks if b["block_id"] == 2), 3)


class ResumeEdgeCases(unittest.TestCase):
    def test_downtime_beyond_the_schedule_finishes_and_closes_blocks(self):
        with TmpDir() as d:
            cfg = base_config(d)
            ctl, ctl2, fleet, code = crash_and_resume(cfg, crash_epoch=6, downtime_ns=10_000 * S)
            self.assertEqual(code, EXIT_OK)
            _, ends, epochs, events = logs(d)
            self.assertEqual({x["status"] for x in ends if x["block_id"] == 1}, {"interrupted"})
            self.assertEqual(max(e["epoch"] for e in epochs), 4)
            with open(os.path.join(d, "STATUS")) as fh:
                self.assertEqual(fh.read().strip(), "DONE")

    def test_resume_of_a_finished_run_adds_nothing(self):
        with TmpDir() as d:
            cfg = base_config(d, n_blocks=3)
            clock = FakeClock()
            fleet = Fleet(clock)
            make_controller(cfg, clock, fleet).run()
            n_before = (len(read_jsonl(os.path.join(d, "epochs.jsonl"))), len(read_jsonl(os.path.join(d, "blocks.jsonl"))))
            clock.advance(100 * S)
            self.assertEqual(make_controller(cfg, clock, fleet).run(resume=True), EXIT_OK)
            n_after = (len(read_jsonl(os.path.join(d, "epochs.jsonl"))), len(read_jsonl(os.path.join(d, "blocks.jsonl"))))
            self.assertEqual(n_before, n_after)

    def test_design_change_refuses_to_resume(self):
        with TmpDir() as d:
            cfg = base_config(d)
            clock = FakeClock()
            fleet = Fleet(clock)
            ctl = make_controller(cfg, clock, fleet, fault_hook=crash_at(6))
            with self.assertRaises(SimulatedCrash):
                ctl.run()
            other = base_config(d, seed=cfg["seed"] + 1)
            self.assertEqual(make_controller(other, clock, fleet).run(resume=True), EXIT_RESUME)
            self.assertEqual(max(e["epoch"] for e in read_jsonl(os.path.join(d, "epochs.jsonl"))), 4)

    def test_partial_last_line_is_repaired(self):
        with TmpDir() as d:
            cfg = base_config(d)
            clock = FakeClock()
            fleet = Fleet(clock)
            ctl = make_controller(cfg, clock, fleet, fault_hook=crash_at(6))
            with self.assertRaises(SimulatedCrash):
                ctl.run()
            with open(os.path.join(d, "epochs.jsonl"), "ab") as fh:
                fh.write(b'{"rec":"epoch","node":"E1","epoch":5,"blo')
            clock.advance(20 * S)
            self.assertEqual(make_controller(cfg, clock, fleet).run(resume=True), EXIT_OK)
            with open(os.path.join(d, "epochs.jsonl")) as fh:
                for line in fh:
                    json.loads(line)

    def test_two_crashes_in_a_row(self):
        with TmpDir() as d:
            cfg = base_config(d)
            clock = FakeClock()
            fleet = Fleet(clock)
            with self.assertRaises(SimulatedCrash):
                make_controller(cfg, clock, fleet, fault_hook=crash_at(5)).run()
            clock.advance(40 * S)
            with self.assertRaises(SimulatedCrash):
                make_controller(cfg, clock, fleet, fault_hook=crash_at(14)).run(resume=True)
            clock.advance(40 * S)
            self.assertEqual(make_controller(cfg, clock, fleet).run(resume=True), EXIT_OK)
            blocks, ends, epochs, events = logs(d)
            keys = [(e["node"], e["epoch"]) for e in epochs]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertEqual([x["event"] for x in events].count("resume"), 2)
            self.assertEqual([s["n"] for s in load_run_state(d).manifest["sessions"]], [1, 2, 3])

    def test_grid_survives_a_reboot_through_the_wall_clock(self):
        with TmpDir() as d:
            cfg = base_config(d)
            clock1 = FakeClock(mono_start_ns=5_000_000_000_000, wall_origin_ns=1_760_000_000_000_000_000)
            fleet1 = Fleet(clock1)
            ctl = make_controller(cfg, clock1, fleet1, fault_hook=crash_at(6))
            with self.assertRaises(SimulatedCrash):
                ctl.run()
            t_crash_wall = clock1.wall_ns()
            down = 20 * S
            mono2 = 123 * S
            clock2 = FakeClock(mono_start_ns=mono2, wall_origin_ns=t_crash_wall + down - mono2, boot_id="fake-boot-2")
            fleet2 = Fleet(clock2)
            ctl2 = make_controller(cfg, clock2, fleet2)
            self.assertEqual(ctl2.run(resume=True), EXIT_OK)
            ev = [x for x in logs(d)[3] if x["event"] == "resume"][0]
            self.assertEqual(ev["from_epoch"], 7)
            epochs = logs(d)[2]
            self.assertEqual(sorted({e["epoch"] for e in epochs if e["node"] == "E1"}),
                             [0, 1, 2, 3, 4] + list(range(7, 32)))
            m = load_run_state(d).manifest
            self.assertEqual(m["t0_boot_id"], "fake-boot-2")

    def test_same_boot_resume_keeps_the_exact_grid_even_if_the_wall_clock_stepped(self):
        with TmpDir() as d:
            cfg = base_config(d)
            clock = FakeClock()
            fleet = Fleet(clock)
            with self.assertRaises(SimulatedCrash):
                make_controller(cfg, clock, fleet, fault_hook=crash_at(6)).run()
            t0 = load_run_state(d).manifest["t0_mono_ns"]
            clock.advance(20 * S)
            clock.skew_wall(3 * S)
            self.assertEqual(make_controller(cfg, clock, fleet).run(resume=True), EXIT_OK)
            ev = [x for x in logs(d)[3] if x["event"] == "resume"][0]
            self.assertEqual(ev["from_epoch"], 7)
            self.assertEqual(ev["t0_mono_ns"], t0)
            for e in logs(d)[2]:
                self.assertEqual(e["t_start_mono_ns"], t0 + e["epoch"] * DELTA)


if __name__ == "__main__":
    unittest.main()
