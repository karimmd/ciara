from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .. import jsonschema_lite
from ..util import data_path, norm_text, read_jsonl, strip_think
from .base import Suite

_TOOLS_CACHE: Optional[List[Dict[str, Any]]] = None


def load_tools() -> List[Dict[str, Any]]:
    global _TOOLS_CACHE
    if _TOOLS_CACHE is None:
        with open(data_path("iot_toolcall", "tools.json"), "r", encoding="utf-8") as f:
            _TOOLS_CACHE = json.load(f)
    return _TOOLS_CACHE


PROMPT_INTRO = (
    "You are a smart-home assistant that controls the devices in a house by calling tools. "
    "Complete the user's request with the tools below and the information provided only."
)

PROMPT_FORMAT = (
    "Output format: reply with one JSON object and nothing else, in the form\n"
    '{"calls": [{"name": "<tool name>", "arguments": {"<argument name>": <value>}}]}\n'
    'Put several objects in "calls" when the request needs several tool calls. '
    'Use the device friendly name as the value of the "name" argument. '
    'If no tool applies (the user asks about the state of a device, or the request is not about '
    'controlling these devices), reply {"calls": []}.'
)

NATIVE_FORMAT = (
    "Use the provided tools to complete the request. If no tool applies (the user asks about the state of a "
    "device, or the request is not about controlling these devices), answer briefly in plain text without "
    "calling a tool."
)


def render_devices(devices: List[Dict[str, str]]) -> str:
    return "\n".join(f"{d['entity_id']} '{d['name']}' = {d['state']}" for d in devices)


def render_system(item: Dict[str, Any], mode: str = "prompt") -> str:
    tools = load_tools()
    dev = "Devices (entity_id 'friendly name' = state):\n" + render_devices(item["devices"])
    if mode == "native":
        return f"{PROMPT_INTRO}\n\n{dev}\n\n{NATIVE_FORMAT}"
    schema = "\n".join(json.dumps(t["function"], separators=(",", ":"), ensure_ascii=False) for t in tools)
    return f"{PROMPT_INTRO}\n\nTools (one JSON schema per line):\n{schema}\n\n{dev}\n\n{PROMPT_FORMAT}"


def output_json_schema() -> Dict[str, Any]:
    variants = []
    for t in load_tools():
        f = t["function"]
        params = json.loads(json.dumps(f["parameters"]))
        params["additionalProperties"] = False
        variants.append({
            "type": "object",
            "properties": {"name": {"type": "string", "enum": [f["name"]]}, "arguments": params},
            "required": ["name", "arguments"],
            "additionalProperties": False,
        })
    return {
        "type": "object",
        "properties": {"calls": {"type": "array", "items": {"anyOf": variants}}},
        "required": ["calls"],
        "additionalProperties": False,
    }


_FENCE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?```$", re.S)
_TOOL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def _first_json(text: str) -> Tuple[Any, bool]:
    dec = json.JSONDecoder()
    for m in re.finditer(r"[\{\[]", text):
        try:
            v, _ = dec.raw_decode(text[m.start():])
            return v, True
        except ValueError:
            continue
    return None, False


def _calls_from_container(obj: Any) -> Tuple[Optional[List[Any]], bool]:
    if isinstance(obj, list):
        return obj, True
    if isinstance(obj, dict):
        for k in ("calls", "tool_calls", "function_calls", "tools"):
            if k in obj and isinstance(obj[k], list):
                return obj[k], True
        if "name" in obj or "function" in obj:
            return [obj], True
    return None, False


def _normalize_call(raw: Any) -> Tuple[Optional[Tuple[str, Dict[str, Any]]], bool]:
    if not isinstance(raw, dict):
        return None, False
    if "function" in raw and isinstance(raw["function"], dict):
        raw = raw["function"]
    name = raw.get("name")
    if not isinstance(name, str):
        return None, False
    args = raw.get("arguments", raw.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return (name, {}), False
    if not isinstance(args, dict):
        return (name, {}), False
    return (name, args), True


def extract_calls(content: str, message: Optional[Dict[str, Any]] = None, native_mode: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {"valid_json": False, "container_ok": False, "calls": [], "well_formed": True, "native": False}
    native = (message or {}).get("tool_calls") or []
    if native:
        out["native"] = True
        out["valid_json"] = True
        out["container_ok"] = True
        for tc in native:
            c, ok = _normalize_call(tc)
            if not ok:
                out["valid_json"] = False
                out["well_formed"] = False
            if c is not None:
                out["calls"].append(c)
        return out

    text = strip_think(content or "")
    if native_mode:
        if not re.search(r"[\{\[]|<tool_call>", text):
            out.update(valid_json=True, container_ok=True)
            return out
    elif not text:
        return out
    tags = _TOOL_TAG.findall(text)
    if tags:
        objs, ok_all = [], True
        for t in tags:
            try:
                objs.append(json.loads(t))
            except ValueError:
                ok_all = False
        out["valid_json"] = ok_all and not native_mode
        out["container_ok"] = True
        for o in objs:
            c, ok = _normalize_call(o)
            out["well_formed"] &= ok
            if c is not None:
                out["calls"].append(c)
        return out

    body = text
    m = _FENCE.match(text)
    if m:
        body = m.group(1).strip()
    obj, pure = None, False
    try:
        obj = json.loads(body)
        pure = True
    except ValueError:
        obj, found = _first_json(text)
        if not found:
            return out
    out["valid_json"] = pure and not native_mode
    raw_calls, ok = _calls_from_container(obj)
    out["container_ok"] = ok
    if raw_calls is not None:
        for rc in raw_calls:
            c, wf = _normalize_call(rc)
            out["well_formed"] &= wf
            if c is not None:
                out["calls"].append(c)
    return out


def _norm_value(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 6)
    if isinstance(v, str):
        return norm_text(v)
    if isinstance(v, list):
        return sorted((_norm_value(x) for x in v), key=lambda x: json.dumps(x, sort_keys=True))
    if isinstance(v, dict):
        return {k: _norm_value(x) for k, x in sorted(v.items())}
    return v


def _resolver(devices: List[Dict[str, str]]):
    by_name = {norm_text(d["name"]): d["entity_id"] for d in devices}
    by_id = {norm_text(d["entity_id"]): d["entity_id"] for d in devices}

    def resolve(s: Any) -> Any:
        if not isinstance(s, str):
            return _norm_value(s)
        k = norm_text(s)
        return by_name.get(k) or by_id.get(k) or k

    return resolve


def canon_call(name: str, args: Dict[str, Any], resolve) -> str:
    a = {}
    for k, v in args.items():
        a[k] = resolve(v) if k == "name" else _norm_value(v)
    return json.dumps([name, a], sort_keys=True, ensure_ascii=False)


def schema_errors(name: str, args: Dict[str, Any]) -> List[str]:
    for t in load_tools():
        if t["function"]["name"] == name:
            return jsonschema_lite.validate(args, t["function"]["parameters"], strict_additional=True)
    return [f"unknown tool {name}"]


def score_output(item: Dict[str, Any], content: str, message: Optional[Dict[str, Any]] = None,
                 native_mode: bool = False) -> Dict[str, Any]:
    p = extract_calls(content, message, native_mode)
    gold = [(g["name"], g["arguments"]) for g in item["gold_calls"]]
    expect_none = not gold
    resolve = _resolver(item["devices"])
    calls = p["calls"]

    errs: List[str] = []
    for n, a in calls:
        errs.extend(schema_errors(n, a))
    structure_ok = p["valid_json"] and p["container_ok"] and p["well_formed"]
    schema_valid = bool(structure_ok and not errs)

    scores: Dict[str, Optional[float]] = {
        "valid_json": 1.0 if p["valid_json"] else 0.0,
        "schema_valid": 1.0 if schema_valid else 0.0,
        "name_match": None,
        "args_exact": None,
        "irrelevance_ok": None,
        "task_acc": None,
    }
    if expect_none:
        ok = len(calls) == 0
        scores["irrelevance_ok"] = 1.0 if ok else 0.0
        scores["task_acc"] = scores["irrelevance_ok"]
    else:
        nm = sorted(n for n, _ in calls) == sorted(n for n, _ in gold)
        ae = sorted(canon_call(n, a, resolve) for n, a in calls) == sorted(canon_call(n, a, resolve) for n, a in gold)
        scores["name_match"] = 1.0 if nm else 0.0
        scores["args_exact"] = 1.0 if ae else 0.0
        scores["task_acc"] = scores["args_exact"]
    return {
        "scores": scores,
        "parsed": {"calls": [{"name": n, "arguments": a} for n, a in calls], "schema_errors": errs[:5],
                   "valid_json": p["valid_json"], "native": p["native"]},
    }


class IotSuite(Suite):
    name = "iot"
    headline = "schema_valid"
    metrics = ["valid_json", "schema_valid", "name_match", "args_exact", "irrelevance_ok", "task_acc"]
    group_key = "category"
    split = "test"

    def items(self) -> List[Dict[str, Any]]:
        mode = self.options.get("tool_mode", "prompt")
        rows = read_jsonl(data_path("iot_toolcall", f"{self.split}.jsonl"))
        limit = self.options.get("limit")
        if limit:
            rows = rows[: int(limit)]
        out = []
        for r in rows:
            item = {
                "id": r["id"],
                "messages": [
                    {"role": "system", "content": render_system(r, mode)},
                    {"role": "user", "content": r["request"]},
                ],
                "params": {"max_tokens": int(self.options.get("max_tokens") or 512)},
                "meta": {"category": r["category"], "source": r["source"], "n_gold": len(r["gold_calls"])},
                "_row": r,
            }
            if mode == "native":
                item["tools"] = load_tools()
            elif self.options.get("constrained"):
                item["params"]["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": "calls", "schema": output_json_schema(), "strict": True}}
            out.append(item)
        return out

    def score(self, item: Dict[str, Any], resp: Dict[str, Any]) -> Dict[str, Any]:
        return score_output(item["_row"], resp.get("content") or "", resp.get("message"),
                            native_mode=self.options.get("tool_mode") == "native")


class IotHeldoutSuite(IotSuite):
    name = "iot_heldout"
    split = "heldout"
