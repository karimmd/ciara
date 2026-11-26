from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from .compose import (COMPOSE_PATH, SCHEMA, WIRE_VERSION, Composer, ContextStore, canonical_messages, device_body,
                          messages_sha256)
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ciara_edge.compose import (COMPOSE_PATH, SCHEMA, WIRE_VERSION, Composer, ContextStore, canonical_messages,
                                    device_body, messages_sha256)

BUILDER = "ciara-compose-build/1"
SUITES = {"test": {"1": "iot", "2": "sensor"}, "heldout": {"1": "iot_heldout", "2": "sensor_heldout"}}
SPLIT_FILES = {("1", "test"): "iot_toolcall/test.jsonl", ("1", "heldout"): "iot_toolcall/heldout.jsonl",
               ("2", "test"): "sensor_interp/test.jsonl", ("2", "heldout"): "sensor_interp/heldout.jsonl"}


class Undecomposable(Exception):
    pass


_F1_TOOLS_HDR = "Tools (one JSON schema per line):"
_F1_DEV_HDR = "Devices (entity_id 'friendly name' = state):"
_F1_DEV_LINE = re.compile(r"^(\S+) '(.*)' = (.*)$")


def parse_family1(system: str) -> Dict[str, Any]:
    i_tools = system.find("\n\n" + _F1_TOOLS_HDR + "\n")
    i_dev = system.find("\n\n" + _F1_DEV_HDR + "\n")
    i_fmt = system.rfind("\n\nOutput format: ")
    if not (0 < i_tools < i_dev < i_fmt):
        raise Undecomposable("family 1 layout markers not found in the expected order")
    instruction = system[:i_tools]
    tools_text = system[i_tools + 2 + len(_F1_TOOLS_HDR) + 1:i_dev]
    dev_text = system[i_dev + 2 + len(_F1_DEV_HDR) + 1:i_fmt]
    output_format = system[i_fmt + 2:]
    devices = []
    for ln in (dev_text.split("\n") if dev_text else []):
        m = _F1_DEV_LINE.match(ln)
        if not m:
            raise Undecomposable(f"device line does not parse: {ln[:80]!r}")
        devices.append([m.group(1), m.group(2), m.group(3)])
    return {"instruction": instruction, "tools": tools_text.split("\n"), "devices": devices, "output_format": output_format}


_F2_SITE = re.compile(r"^Site: (.*)\. Unit: (.*)\.$")
_F2_SAMPLING = re.compile(r"^Sampling: one reading every (\d+) minutes, (\d+) readings from (\d\d:\d\d) to (\d\d:\d\d)\.$")
_F2_CHAN = re.compile(r"^- ([a-z0-9_]+): (.*)$")
_F2_NA = ("NA means that no reading was received. Ignore NA readings in every calculation: an NA reading is not above any limit, "
          "and it ends a run of consecutive readings.")
_F2_READ_HDR = "Readings (one row per time):"


def parse_family2(system: str) -> Dict[str, Any]:
    i_fmt = system.rfind("\n\nAnswer format: ")
    i_read = system.find("\n\n" + _F2_READ_HDR + "\n")
    if not (0 < i_read < i_fmt):
        raise Undecomposable("family 2 layout markers not found in the expected order")
    table = system[i_read + 2 + len(_F2_READ_HDR) + 1:i_fmt]
    fmt = system[i_fmt + 2:]
    head = system[:i_read].split("\n")
    if len(head) < 6 or head[1] != "" or head[4] != "Channels:":
        raise Undecomposable("family 2 header block has an unexpected shape")
    m = _F2_SITE.match(head[2])
    ms = _F2_SAMPLING.match(head[3])
    if not m or not ms:
        raise Undecomposable("site or sampling line does not parse")
    body = head[5:]
    has_na = False
    if body and body[-1] == _F2_NA:
        has_na = True
        body = body[:-1]
    chans = []
    for ln in body:
        mc = _F2_CHAN.match(ln)
        if not mc:
            raise Undecomposable(f"channel line does not parse: {ln[:60]!r}")
        chans.append((mc.group(1), mc.group(2)))
    rows = table.split("\n")
    cols = rows[0].split(" | ")
    if cols[0] != "time" or [c for c, _ in chans] != cols[1:]:
        raise Undecomposable("channel list differs from the table header")
    n, t0, t1 = int(ms.group(2)), ms.group(3), ms.group(4)
    if len(rows) - 1 != n or rows[1].split(" | ", 1)[0] != t0 or rows[-1].split(" | ", 1)[0] != t1:
        raise Undecomposable("sampling line differs from the table (reading count or time span)")
    cell_na = any(c == "NA" for r in rows[1:] for c in r.split(" | ")[1:])
    if cell_na != has_na:
        raise Undecomposable("NA rule presence differs from the NA cells of the table")
    return {"instruction": head[0], "site": m.group(1), "unit": m.group(2), "step_min": int(ms.group(1)), "channels": chans,
            "table": table, "format": fmt}


def _pilot(root: str):
    if root not in sys.path:
        sys.path.insert(0, root)
    from eval.suites.registry import get_suite
    return get_suite


def sensor_id(site: str, unit: str) -> str:
    return "S" + hashlib.sha256(f"{site}|{unit}".encode("utf-8")).hexdigest()[:8]


def load_items(pilot_root: str, split: str, max_tokens: Dict[str, int]) -> Dict[str, List[Dict[str, Any]]]:
    get_suite = _pilot(pilot_root)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for fam, suite in SUITES[split].items():
        items = get_suite(suite, {"max_tokens": max_tokens[fam]}).items()
        out[fam] = [{"id": it["id"], "messages": it["messages"]} for it in items]
    return out


def request_defaults(client_config: Optional[str]) -> Dict[str, Any]:
    clients = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "clients")
    if clients not in sys.path:
        sys.path.insert(0, clients)
    from ciara_clients import config as cfgmod
    cfg = cfgmod.load(client_config)
    r = cfg["request"]
    out = {"model": cfg["run"]["model"], "stream": bool(r["stream"]), "temperature": r["temperature"], "seed": r["seed"],
           "chat_template_kwargs": {"enable_thinking": bool(r["enable_thinking"])},
           "max_tokens": {k: int(v["max_tokens"]) for k, v in sorted(cfg["families"].items())}}
    if r.get("include_usage"):
        out["stream_options"] = {"include_usage": True}
    return out


def build(pilot_root: str, splits: Sequence[str], client_config: Optional[str] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    req = request_defaults(client_config)
    fam1: Optional[Dict[str, Any]] = None
    fam2: Optional[Dict[str, Any]] = None
    homes: Dict[str, Any] = {}
    sensors: Dict[str, Any] = {}
    chan_desc: Dict[str, str] = {}
    formats: List[str] = []
    verify: Dict[str, str] = {}
    bank: List[Dict[str, Any]] = []
    parsed2: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    for split in splits:
        items = load_items(pilot_root, split, req["max_tokens"])
        for k, it in enumerate(items["1"]):
            sysmsg, usermsg = it["messages"][0], it["messages"][1]
            if [m["role"] for m in it["messages"]] != ["system", "user"]:
                raise Undecomposable(f"{it['id']}: messages are not [system, user]")
            p = parse_family1(sysmsg["content"])
            shared = {"instruction": p["instruction"], "tools": p["tools"], "output_format": p["output_format"]}
            if fam1 is None:
                fam1 = shared
            elif fam1 != shared:
                raise Undecomposable(f"{it['id']}: shared family 1 parts differ from the first item")
            hid = f"home-{split}-{k:04d}"
            homes[hid] = {"devices": p["devices"]}
            payload = {"v": WIRE_VERSION, "f": 1, "home": hid, "text": usermsg["content"]}
            bank.append({"id": it["id"], "family": 1, "payload": payload, "messages_sha256": messages_sha256(it["messages"])})
            verify[it["id"]] = messages_sha256(it["messages"])
        for it in items["2"]:
            sysmsg, usermsg = it["messages"][0], it["messages"][1]
            if [m["role"] for m in it["messages"]] != ["system", "user"]:
                raise Undecomposable(f"{it['id']}: messages are not [system, user]")
            p = parse_family2(sysmsg["content"])
            if fam2 is None:
                fam2 = {"instruction": p["instruction"]}
            elif fam2["instruction"] != p["instruction"]:
                raise Undecomposable(f"{it['id']}: family 2 instruction differs from the first item")
            for c, d in p["channels"]:
                if chan_desc.setdefault(c, d) != d:
                    raise Undecomposable(f"{it['id']}: channel {c} has two descriptions")
            if p["format"] not in formats:
                formats.append(p["format"])
            sid = sensor_id(p["site"], p["unit"])
            rec = sensors.setdefault(sid, {"site": p["site"], "unit": p["unit"], "step_min": p["step_min"], "channels": []})
            if (rec["site"], rec["unit"], rec["step_min"]) != (p["site"], p["unit"], p["step_min"]):
                raise Undecomposable(f"{it['id']}: sensor id collision for {sid}")
            for c, _ in p["channels"]:
                if c not in rec["channels"]:
                    rec["channels"].append(c)
            parsed2.append((it["id"], {"sid": sid, "format": p["format"], "table": p["table"], "q": usermsg["content"]}, it))
    if fam1 is None or fam2 is None:
        raise Undecomposable("no items")
    formats.sort()
    fmt_ids = {text: f"F{k}" for k, text in enumerate(formats)}
    for rec in sensors.values():
        rec["channels"].sort()
    for iid, p, it in parsed2:
        payload = {"v": WIRE_VERSION, "f": 2, "sensor": p["sid"], "fmt": fmt_ids[p["format"]], "readings": p["table"], "q": p["q"]}
        h = messages_sha256(it["messages"])
        bank.append({"id": iid, "family": 2, "payload": payload, "messages_sha256": h})
        verify[iid] = h
    store = {
        "schema": SCHEMA, "builder": BUILDER, "splits": list(splits), "wire_version": WIRE_VERSION, "compose_path": COMPOSE_PATH,
        "request": req,
        "families": {
            "1": {"instruction": fam1["instruction"], "tools_header": _F1_TOOLS_HDR, "tools": fam1["tools"],
                  "devices_header": _F1_DEV_HDR, "output_format": fam1["output_format"], "sep": "\n\n",
                  "device_line": "{0} '{1}' = {2}"},
            "2": {"instruction": fam2["instruction"], "site_line": "Site: {site}. Unit: {unit}.",
                  "sampling_line": "Sampling: one reading every {step} minutes, {n} readings from {t0} to {t1}.",
                  "channels_header": "Channels:", "channel_line": "- {0}: {1}", "channel_desc": dict(sorted(chan_desc.items())),
                  "na_rule": _F2_NA, "readings_header": _F2_READ_HDR,
                  "formats": {fid: text for text, fid in sorted(fmt_ids.items(), key=lambda kv: kv[1])}},
        },
        "homes": homes, "sensors": dict(sorted(sensors.items())), "verify": {"items": verify},
    }
    return store, bank


def prove(store_doc: Dict[str, Any], bank: List[Dict[str, Any]], pilot_root: str, splits: Sequence[str]) -> Dict[str, Any]:
    store = ContextStore(json.loads(json.dumps(store_doc, ensure_ascii=False)))
    comp = Composer(store)
    req = store_doc["request"]
    by_id = {b["id"]: b for b in bank}
    res: Dict[str, Any] = {}
    failures: List[str] = []
    for split in splits:
        items = load_items(pilot_root, split, req["max_tokens"])
        for fam, lst in items.items():
            n_ok = n_hash = n_body = 0
            for it in lst:
                b = by_id[it["id"]]
                payload = json.loads(device_body(b["payload"]).decode("utf-8"))
                try:
                    body, info = comp.expand(payload, it["id"])
                except Exception as e:
                    failures.append(f"{it['id']}: {type(e).__name__}: {e}")
                    continue
                same = canonical_messages(body["messages"]) == canonical_messages(it["messages"]) and body["messages"] == it["messages"]
                n_ok += 1 if same else 0
                n_hash += 1 if info["compose_ok"] else 0
                full = {"model": req["model"], "messages": it["messages"], "stream": req["stream"], "temperature": req["temperature"],
                        "seed": req["seed"], "max_tokens": int(req["max_tokens"][fam]),
                        "chat_template_kwargs": dict(req["chat_template_kwargs"])}
                if req.get("stream_options"):
                    full["stream_options"] = dict(req["stream_options"])
                n_body += 1 if json.dumps(full, separators=(",", ":"), ensure_ascii=False) == json.dumps(body, separators=(",", ":"), ensure_ascii=False) else 0
                if not (same and info["compose_ok"]):
                    failures.append(f"{it['id']}: composed messages differ from the frozen messages")
            res[f"{split}/family{fam}"] = {"items": len(lst), "messages_identical": n_ok, "hash_ok": n_hash, "request_body_identical": n_body}
    return {"splits": res, "failures": failures[:20], "n_failures": len(failures)}


def sizes(store_doc: Dict[str, Any], bank: List[Dict[str, Any]], pilot_root: str, splits: Sequence[str], text: str) -> Dict[str, Any]:
    import statistics
    req = store_doc["request"]
    out: Dict[str, Any] = {"store_bytes": len(text.encode("utf-8"))}
    by_id = {b["id"]: b for b in bank}
    for split in splits:
        items = load_items(pilot_root, split, req["max_tokens"])
        for fam, lst in items.items():
            full = [len(json.dumps({"model": req["model"], "messages": it["messages"]}, separators=(",", ":"), ensure_ascii=False).encode("utf-8")) for it in lst]
            comp = [len(device_body(by_id[it["id"]]["payload"])) for it in lst]
            out[f"{split}/family{fam}"] = {"n": len(lst), "full_median": statistics.median(full), "full_mean": round(statistics.fmean(full)),
                                           "full_max": max(full), "compact_median": statistics.median(comp),
                                           "compact_mean": round(statistics.fmean(comp)), "compact_max": max(comp)}
    return out


def write_all(out_dir: str, tag: str, store: Dict[str, Any], bank: List[Dict[str, Any]], proof: Dict[str, Any], sz: Dict[str, Any],
              pilot_root: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = {"store": os.path.join(out_dir, f"context-store-{tag}.json"), "bank": os.path.join(out_dir, f"compact-bank-{tag}.jsonl"),
             "proof": os.path.join(out_dir, f"compose-proof-{tag}.json")}
    store_text = json.dumps(store, ensure_ascii=False, separators=(",", ":")) + "\n"
    with open(paths["store"], "w", encoding="utf-8") as f:
        f.write(store_text)
    bank_text = "".join(json.dumps(b, ensure_ascii=False, separators=(",", ":")) + "\n" for b in bank)
    with open(paths["bank"], "w", encoding="utf-8") as f:
        f.write(bank_text)
    data = os.environ.get("EVAL_DATA_DIR") or os.path.join(os.path.dirname(os.path.abspath(pilot_root)), "datasets")
    inputs = {}
    for (fam, split), rel in SPLIT_FILES.items():
        if split in store["splits"]:
            with open(os.path.join(data, rel), "rb") as fh:
                inputs[rel] = hashlib.sha256(fh.read()).hexdigest()
    with open(os.path.join(data, "MANIFEST.json"), "rb") as fh:
        inputs["MANIFEST.json"] = hashlib.sha256(fh.read()).hexdigest()
    doc = {"builder": BUILDER, "splits": store["splits"], "inputs_sha256": inputs,
           "outputs": {"context_store": {"file": os.path.basename(paths["store"]), "bytes": len(store_text.encode("utf-8")),
                                         "sha256": hashlib.sha256(store_text.encode("utf-8")).hexdigest(),
                                         "homes": len(store["homes"]), "sensors": len(store["sensors"]),
                                         "formats": len(store["families"]["2"]["formats"]), "tools": len(store["families"]["1"]["tools"])},
                       "compact_bank": {"file": os.path.basename(paths["bank"]), "bytes": len(bank_text.encode("utf-8")),
                                        "sha256": hashlib.sha256(bank_text.encode("utf-8")).hexdigest(), "items": len(bank)}},
           "proof": proof, "sizes": sz}
    with open(paths["proof"], "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Offline builder of the gateway context store and the device payloads .', formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="test")
    ap.add_argument("--tag", default=None, help="file name tag (default: the splits joined by +)")
    ap.add_argument("--pilot-root", default=os.environ.get("CIARA_PILOT_ROOT") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "workload"))
    ap.add_argument("--client-config", default=None)
    a = ap.parse_args(argv)
    splits = [s for s in a.splits.split(",") if s]
    tag = a.tag or "+".join(splits)
    try:
        store, bank = build(a.pilot_root, splits, a.client_config)
    except Undecomposable as e:
        print(f"build failed: {e}", file=sys.stderr)
        return 1
    proof = prove(store, bank, a.pilot_root, splits)
    text = json.dumps(store, ensure_ascii=False, separators=(",", ":")) + "\n"
    paths = write_all(a.out, tag, store, bank, proof, sizes(store, bank, a.pilot_root, splits, text), a.pilot_root)
    for k, v in proof["splits"].items():
        print(f"{k}: {v['items']} items, messages identical {v['messages_identical']}, hash ok {v['hash_ok']}, "
              f"request body identical {v['request_body_identical']}")
    print(f"store {paths['store']} ({os.path.getsize(paths['store'])} bytes), bank {paths['bank']}, proof {paths['proof']}")
    if proof["n_failures"]:
        print(f"{proof['n_failures']} failures, first: {proof['failures'][:3]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
