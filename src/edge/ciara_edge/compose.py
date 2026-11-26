from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple

SCHEMA = "ciara-context-store/1"
WIRE_VERSION = 1
COMPOSE_PATH = "/v1/ciara/compose"


class ComposeError(Exception):
    def __init__(self, code: str, msg: str):
        super().__init__(f"{code}: {msg}")
        self.code = code


def canonical_messages(messages: List[Dict[str, Any]]) -> bytes:
    return json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def messages_sha256(messages: List[Dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_messages(messages)).hexdigest()


def device_body(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class ContextStore:
    def __init__(self, doc: Dict[str, Any]):
        if doc.get("schema") != SCHEMA:
            raise ValueError(f"not a context store: schema {doc.get('schema')!r}")
        self.doc = doc
        self.request: Dict[str, Any] = doc["request"]
        self.fam: Dict[str, Dict[str, Any]] = doc["families"]
        self.homes: Dict[str, Dict[str, Any]] = doc.get("homes", {})
        self.sensors: Dict[str, Dict[str, Any]] = doc.get("sensors", {})
        self.expected: Dict[str, str] = (doc.get("verify") or {}).get("items", {})

    @classmethod
    def load(cls, path: str) -> "ContextStore":
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def summary(self) -> Dict[str, Any]:
        return {"homes": len(self.homes), "sensors": len(self.sensors), "verify_items": len(self.expected),
                "formats": len(self.fam["2"]["formats"]), "tools": len(self.fam["1"]["tools"])}


class Composer:
    def __init__(self, store: ContextStore):
        self.store = store
        self.t1 = store.fam["1"]
        self.t2 = store.fam["2"]

    def compose(self, payload: Dict[str, Any]) -> List[Dict[str, str]]:
        if not isinstance(payload, dict):
            raise ComposeError("bad_payload", "the compact body must be a JSON object")
        if payload.get("v") != WIRE_VERSION:
            raise ComposeError("bad_version", f"v {payload.get('v')!r}")
        fam = payload.get("f")
        if fam == 1:
            return self._family1(payload)
        if fam == 2:
            return self._family2(payload)
        raise ComposeError("bad_family", f"f {fam!r}")

    def _family1(self, p: Dict[str, Any]) -> List[Dict[str, str]]:
        home, text = p.get("home"), p.get("text")
        if not isinstance(text, str) or not isinstance(home, str):
            raise ComposeError("bad_payload", "family 1 needs string fields home and text")
        rec = self.store.homes.get(home)
        if rec is None:
            raise ComposeError("unknown_home", home)
        t = self.t1
        line = t["device_line"]
        devices = "\n".join(line.format(*d) for d in rec["devices"])
        system = t["sep"].join([
            t["instruction"],
            t["tools_header"] + "\n" + "\n".join(t["tools"]),
            t["devices_header"] + "\n" + devices,
            t["output_format"],
        ])
        return [{"role": "system", "content": system}, {"role": "user", "content": text}]

    def _family2(self, p: Dict[str, Any]) -> List[Dict[str, str]]:
        sid, fmt, table, question = p.get("sensor"), p.get("fmt"), p.get("readings"), p.get("q")
        if not all(isinstance(x, str) for x in (sid, fmt, table, question)):
            raise ComposeError("bad_payload", "family 2 needs string fields sensor, fmt, readings and q")
        s = self.store.sensors.get(sid)
        if s is None:
            raise ComposeError("unknown_sensor", sid)
        t = self.t2
        text = t["formats"].get(fmt)
        if text is None:
            raise ComposeError("unknown_format", fmt)
        lines = table.split("\n")
        head = lines[0].split(" | ")
        rows = lines[1:]
        if head[0] != "time" or len(head) < 2 or not rows:
            raise ComposeError("bad_readings", "the table needs a header row 'time | channel ...' and at least one reading")
        cols = head[1:]
        desc = t["channel_desc"]
        chan_lines = []
        for c in cols:
            if c not in s["channels"] or c not in desc:
                raise ComposeError("unknown_channel", f"{c} for sensor {sid}")
            chan_lines.append(t["channel_line"].format(c, desc[c]))
        t_first = rows[0].split(" | ", 1)[0]
        t_last = rows[-1].split(" | ", 1)[0]
        has_na = False
        for r in rows:
            if "NA" in r and "NA" in r.split(" | ")[1:]:
                has_na = True
                break
        parts = [t["instruction"], "", t["site_line"].format(site=s["site"], unit=s["unit"]),
                 t["sampling_line"].format(step=s["step_min"], n=len(rows), t0=t_first, t1=t_last), t["channels_header"]]
        parts += chan_lines
        if has_na:
            parts.append(t["na_rule"])
        parts += ["", t["readings_header"], table, "", text]
        return [{"role": "system", "content": "\n".join(parts)}, {"role": "user", "content": question}]

    def request_body(self, family: int, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        r = self.store.request
        body: Dict[str, Any] = {"model": r["model"], "messages": messages, "stream": r["stream"], "temperature": r["temperature"],
                                "seed": r["seed"], "max_tokens": int(r["max_tokens"][str(family)]),
                                "chat_template_kwargs": dict(r["chat_template_kwargs"])}
        if r.get("stream_options"):
            body["stream_options"] = dict(r["stream_options"])
        return body

    def expand(self, payload: Dict[str, Any], item: Optional[str], verify: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        t0 = time.perf_counter_ns()
        messages = self.compose(payload)
        t1 = time.perf_counter_ns()
        info: Dict[str, Any] = {"compose_ok": None, "t_compose_ms": (t1 - t0) / 1e6, "t_verify_ms": None}
        if verify:
            got = messages_sha256(messages)
            want = self.store.expected.get(item or "")
            info.update(sha256=got, expected=want, compose_ok=(want is not None and got == want),
                        t_verify_ms=(time.perf_counter_ns() - t1) / 1e6)
        return self.request_body(int(payload["f"]), messages), info
