from __future__ import annotations

import copy
import json
import os
import re
import sys
import unittest
from pathlib import Path


from ciara_edge import compose, compose_build

_REPO = str(Path(__file__).resolve().parents[3])
PILOT = os.environ.get("CIARA_PILOT_ROOT") or os.path.join(_REPO, "workload")
_DATA = os.environ.get("EVAL_DATA_DIR") or os.path.join(os.path.dirname(os.path.abspath(PILOT)), "datasets")
PILOT_OK = os.path.isfile(os.path.join(PILOT, "eval", "suites", "iot.py")) and os.path.isfile(os.path.join(_DATA, "iot_toolcall", "test.jsonl"))
NEEDS_PILOT = unittest.skipUnless(PILOT_OK, "the tool-call suite is not built (build it with datasets/iot_toolcall/build/freeze_iot.py); the frozen suites are read from datasets/ through workload/eval/")
SPLITS = ["test", "heldout"]
_CACHE = {}


def built():
    if "b" not in _CACHE:
        store, bank = compose_build.build(PILOT, SPLITS)
        _CACHE["b"] = (store, bank, compose.Composer(compose.ContextStore(json.loads(json.dumps(store, ensure_ascii=False)))))
    return _CACHE["b"]


def frozen(split):
    req = built()[0]["request"]
    return compose_build.load_items(PILOT, split, req["max_tokens"])


@NEEDS_PILOT
class RoundTrip(unittest.TestCase):
    def test_every_item_of_every_split_and_family_round_trips(self):
        store, bank, comp = built()
        by_id = {b["id"]: b for b in bank}
        counts = {}
        for split in SPLITS:
            for fam, items in frozen(split).items():
                n = 0
                for it in items:
                    payload = json.loads(compose.device_body(by_id[it["id"]]["payload"]).decode("utf-8"))
                    body, info = comp.expand(payload, it["id"])
                    self.assertEqual(body["messages"], it["messages"], it["id"])
                    self.assertEqual(compose.canonical_messages(body["messages"]), compose.canonical_messages(it["messages"]))
                    self.assertTrue(info["compose_ok"], it["id"])
                    self.assertEqual(info["sha256"], by_id[it["id"]]["messages_sha256"])
                    n += 1
                counts[(split, fam)] = n
        self.assertEqual(counts, {("test", "1"): 410, ("test", "2"): 400, ("heldout", "1"): 309, ("heldout", "2"): 300})

    def test_request_body_equals_the_full_body_of_the_clients(self):
        store, bank, comp = built()
        r = store["request"]
        it = frozen("test")["1"][0]
        body = comp.request_body(1, it["messages"])
        full = {"model": r["model"], "messages": it["messages"], "stream": True, "temperature": 0, "seed": 0, "max_tokens": 512,
                "chat_template_kwargs": {"enable_thinking": False}, "stream_options": {"include_usage": True}}
        self.assertEqual(json.dumps(body, separators=(",", ":")), json.dumps(full, separators=(",", ":")))
        self.assertEqual(comp.request_body(2, it["messages"])["max_tokens"], 1024)

    def test_frozen_suite_files_are_unchanged(self):
        want = {"sensor_interp/test.jsonl": "1b6cc875b73a98ae184a94a105b3fa7718f0237774d66ac69da641d4c03c68ee",
                "sensor_interp/heldout.jsonl": "2801e37a961b6034baa0b6e6612569086000e4d49bf1e28a4f3f56ba2456f293"}
        import hashlib
        for rel, h in want.items():
            with open(os.path.join(PILOT, "data", rel), "rb") as fh:
                self.assertEqual(hashlib.sha256(fh.read()).hexdigest(), h, rel)

    def test_builder_output_is_deterministic(self):
        a = json.dumps(compose_build.build(PILOT, ["test"])[0], sort_keys=True)
        b = json.dumps(compose_build.build(PILOT, ["test"])[0], sort_keys=True)
        self.assertEqual(a, b)


@NEEDS_PILOT
class Content(unittest.TestCase):
    def test_devices_do_not_upload_the_static_context(self):
        store, bank, comp = built()
        p1 = [b["payload"] for b in bank if b["family"] == 1][0]
        wire = compose.device_body(p1).decode()
        self.assertNotIn("Tools (one JSON schema", wire)
        self.assertNotIn("entity_id", wire)
        self.assertEqual(set(p1), {"v", "f", "home", "text"})
        p2 = [b["payload"] for b in bank if b["family"] == 2][0]
        self.assertEqual(set(p2), {"v", "f", "sensor", "fmt", "readings", "q"})
        w2 = compose.device_body(p2).decode()
        for static in ("analysis assistant", "Site:", "Sampling:", "Channels:", "Answer format"):
            self.assertNotIn(static, w2)
        self.assertTrue(p2["readings"].startswith("time | "))

    def test_family1_upload_is_small_and_family2_upload_is_shorter_than_the_prompt(self):
        store, bank, comp = built()
        f1 = [len(compose.device_body(b["payload"])) for b in bank if b["family"] == 1]
        self.assertLess(max(f1), 1000)
        items = {it["id"]: it for it in frozen("test")["2"]}
        for b in bank:
            if b["family"] == 2 and b["id"] in items:
                sysmsg = items[b["id"]]["messages"][0]["content"]
                self.assertLess(len(compose.device_body(b["payload"])), len(sysmsg) + len(items[b["id"]]["messages"][1]["content"]))

    def test_na_rule_follows_the_na_cells(self):
        store, bank, comp = built()
        seen = {True: 0, False: 0}
        for it in frozen("test")["2"]:
            has_cell = any(c == "NA" for r in it["messages"][0]["content"].split("Readings (one row per time):\n")[1].split("\n")[1:]
                           for c in r.split(" | ")[1:])
            self.assertEqual("NA means that no reading" in it["messages"][0]["content"], has_cell)
            seen[has_cell] += 1
        self.assertTrue(seen[True] > 0 and seen[False] > 0)

    def test_store_holds_what_the_decision_names(self):
        store = built()[0]
        f1, f2 = store["families"]["1"], store["families"]["2"]
        self.assertEqual(len(f1["tools"]), 19)
        self.assertIn("smart-home assistant", f1["instruction"])
        self.assertIn("Output format:", f1["output_format"])
        self.assertTrue(all(k in f2 for k in ("instruction", "na_rule", "formats", "channel_desc", "site_line", "sampling_line")))
        self.assertEqual(len(f2["formats"]), 9)
        s = next(iter(store["sensors"].values()))
        self.assertTrue(all(k in s for k in ("site", "unit", "step_min", "channels")))
        h = next(iter(store["homes"].values()))
        self.assertTrue(all(len(d) == 3 for d in h["devices"]))


@NEEDS_PILOT
class Failures(unittest.TestCase):
    def setUp(self):
        store, bank, comp = built()
        self.store_doc = store
        self.bank = {b["id"]: b for b in bank}
        self.item1 = next(b for b in bank if b["family"] == 1)
        self.item2 = next(b for b in bank if b["family"] == 2)

    def comp(self, mutate=None):
        doc = copy.deepcopy(self.store_doc)
        if mutate:
            mutate(doc)
        return compose.Composer(compose.ContextStore(doc))

    def test_a_changed_context_is_a_mismatch_not_a_pass(self):
        def m(d):
            d["families"]["1"]["tools"][3] = d["families"]["1"]["tools"][3].replace("a", "b", 1)
        body, info = self.comp(m).expand(self.item1["payload"], self.item1["id"])
        self.assertIs(info["compose_ok"], False)

        def m2(d):
            h = next(iter(d["homes"].values()))
            h["devices"][0][2] = "tampered"
        c = self.comp(m2)
        hid = next(iter(self.store_doc["homes"]))
        first = next(b for b in self.bank.values() if b["family"] == 1 and b["payload"]["home"] == hid)
        self.assertIs(c.expand(first["payload"], first["id"])[1]["compose_ok"], False)

    def test_payload_of_another_item_is_a_mismatch(self):
        other = [b for b in self.bank.values() if b["family"] == 1][5]
        body, info = self.comp().expand(other["payload"], self.item1["id"])
        self.assertIs(info["compose_ok"], False)

    def test_item_without_expected_hash_is_not_ok(self):
        body, info = self.comp().expand(self.item1["payload"], "no-such-item")
        self.assertIs(info["compose_ok"], False)
        self.assertIsNone(info["expected"])

    def test_unverified_composition_reports_none(self):
        body, info = self.comp().expand(self.item1["payload"], self.item1["id"], verify=False)
        self.assertIsNone(info["compose_ok"])

    def test_unknown_identifiers_and_bad_payloads_raise(self):
        c = self.comp()
        cases = [({"v": 1, "f": 1, "home": "nope", "text": "x"}, "unknown_home"),
                 (dict(self.item2["payload"], sensor="Snope"), "unknown_sensor"),
                 (dict(self.item2["payload"], fmt="F99"), "unknown_format"),
                 (dict(self.item2["payload"], readings="time | warp_core\n10:00 | 1"), "unknown_channel"),
                 (dict(self.item2["payload"], readings="nonsense"), "bad_readings"),
                 (dict(self.item2["payload"], readings="time | temp_c"), "bad_readings"),
                 ({"v": 2, "f": 1}, "bad_version"), ({"v": 1, "f": 3}, "bad_family"), ({"v": 1, "f": 1, "home": 4, "text": "x"}, "bad_payload"),
                 ([1, 2], "bad_payload")]
        for payload, code in cases:
            with self.assertRaises(compose.ComposeError) as cm:
                c.compose(payload)
            self.assertEqual(cm.exception.code, code, payload)

    def test_builder_reports_an_item_that_does_not_decompose(self):
        with self.assertRaises(compose_build.Undecomposable):
            compose_build.parse_family1("no layout here")
        with self.assertRaises(compose_build.Undecomposable):
            compose_build.parse_family2("no layout here")
        it = frozen("test")["2"][0]["messages"][0]["content"]
        with self.assertRaises(compose_build.Undecomposable):
            compose_build.parse_family2(it.replace("12 readings", "13 readings", 1))
        it1 = frozen("test")["1"][0]["messages"][0]["content"]
        with self.assertRaises(compose_build.Undecomposable):
            compose_build.parse_family1(it1.replace("\n\nDevices (", "\n\nDevices[", 1))


class Style(unittest.TestCase):
    BANNED = re.compile(r"\b(may|might|can|could|yet|never|always)\b", re.IGNORECASE)
    DASH = re.compile("[" + chr(0x2013) + chr(0x2014) + "]")

    def test_files(self):
        root = Path(__file__).resolve().parents[3]
        for rel in ("src/edge/ciara_edge/compose.py", "src/edge/ciara_edge/compose_build.py", "tests/src/edge/test_compose.py"):
            with open(os.path.join(root, rel), encoding="utf-8") as fh:
                for i, line in enumerate(fh, 1):
                    if rel.endswith("test_compose.py") and "BANNED =" in line:
                        continue
                    self.assertIsNone(self.BANNED.search(line), f"{rel}:{i}: {line.strip()}")
                    self.assertIsNone(self.DASH.search(line), f"{rel}:{i}")


if __name__ == "__main__":
    unittest.main()
