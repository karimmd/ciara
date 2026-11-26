from __future__ import annotations

import json
import re
from decimal import Decimal
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

from ..util import data_path, read_jsonl, strip_think
from .base import Suite

MAX_TOKENS = 1024
LEVELS = [1, 2, 3, 4]
TEMPLATE_IDS = ["T%02d" % i for i in range(1, 18)]

_FENCE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?```$", re.S)
_NUM_HEAD = re.compile(r"^\s*([+-]?)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?")
_UNIT_TAIL = re.compile(r"^[A-Za-z%°/_ ]*$")
_TIME = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*$")
_LETTER = re.compile(r"^\(?([a-e])(?:[).:]|\s|$)", re.I)


def parse_reply(content: str) -> Tuple[bool, Any, bool]:
    text = strip_think(content or "")
    if "<think>" in text:
        return False, None, False
    body = text
    m = _FENCE.match(text)
    if m:
        body = m.group(1).strip()
    try:
        obj = json.loads(body, parse_float=Decimal)
        if isinstance(obj, dict) and "answer" in obj:
            return True, obj["answer"], True
    except ValueError:
        pass
    dec = json.JSONDecoder(parse_float=Decimal)
    for mm in re.finditer(r"\{", text):
        try:
            v, _ = dec.raw_decode(text[mm.start():])
        except ValueError:
            continue
        if isinstance(v, dict) and "answer" in v:
            return False, v["answer"], True
    return False, None, False


def to_fraction(v: Any) -> Optional[Fraction]:
    if isinstance(v, bool):
        return None
    try:
        if isinstance(v, int):
            return Fraction(v)
        if isinstance(v, Decimal):
            return Fraction(v) if v.is_finite() else None
        if isinstance(v, float):
            return Fraction(str(v)) if v == v and v not in (float("inf"), float("-inf")) else None
        if isinstance(v, str):
            m = _NUM_HEAD.match(v)
            if not m or not _UNIT_TAIL.match(v[m.end():]):
                return None
            return Fraction(m.group(1) + m.group(2).replace(",", "") + (m.group(3) or ""))
    except (ValueError, ArithmeticError):
        return None
    return None


def norm_time(v: Any) -> Optional[str]:
    if not isinstance(v, str):
        return None
    m = _TIME.match(v)
    if not m:
        return None
    h, mi, sec = int(m.group(1)), int(m.group(2)), m.group(3)
    if h > 23 or mi > 59 or (sec is not None and int(sec) != 0):
        return None
    return "%02d:%02d" % (h, mi)


def _clean_str(v: Any) -> Optional[str]:
    if not isinstance(v, str):
        return None
    return v.strip().strip("\"'").strip().rstrip(".").strip().casefold()


def is_none(v: Any) -> bool:
    return v is None or _clean_str(v) in ("none", "null")


def norm_choice(v: Any, choices: List[str]) -> Optional[str]:
    s = _clean_str(v)
    if s is None:
        return None
    s = s.replace(" ", "_").replace("-", "_")
    low = [c.casefold() for c in choices]
    if s in low:
        return choices[low.index(s)]
    stems = {}
    for c in low:
        if c != "none":
            stems.setdefault(c.split("_")[0], []).append(c)
    if s in stems and len(stems[s]) == 1:
        return choices[low.index(stems[s][0])]
    return None


def norm_answer(row: Dict[str, Any], v: Any) -> Any:
    t = row["answer_type"]
    if t == "number":
        return to_fraction(v)
    if t in ("time", "time_or_none"):
        if t == "time_or_none" and is_none(v):
            return "none"
        return norm_time(v)
    if t == "yesno":
        if isinstance(v, bool):
            return "yes" if v else "no"
        s = _clean_str(v)
        return {"yes": "yes", "true": "yes", "no": "no", "false": "no"}.get(s) if s else None
    if t in ("label", "channel", "channel_or_none"):
        if t == "channel_or_none" and is_none(v):
            return "none"
        return norm_choice(v, row["choices"])
    if t == "letter":
        if not isinstance(v, str):
            return None
        m = _LETTER.match(v.strip().strip("\"'").strip())
        return m.group(1).upper() if m else None
    raise ValueError(t)


def answers_match(row: Dict[str, Any], value: Any) -> Tuple[bool, Any]:
    pred = norm_answer(row, value)
    if pred is None:
        return False, None
    if row["answer_type"] == "number":
        gold = Fraction(str(row["answer"]))
        tol = Fraction(str(row["tolerance"]))
        return abs(pred - gold) <= tol, pred
    return pred == row["answer"], pred


def score_output(row: Dict[str, Any], content: str) -> Dict[str, Any]:
    valid, value, found = parse_reply(content)
    ok, pred = answers_match(row, value) if found else (False, None)
    acc = 1.0 if ok else 0.0
    lv, tp = "acc_L%d" % row["level"], "acc_" + row["template"]
    scores: Dict[str, Optional[float]] = {"valid_json": 1.0 if valid else 0.0, "answer_acc": acc}
    for k in ["acc_L%d" % x for x in LEVELS] + ["acc_" + t for t in TEMPLATE_IDS]:
        scores[k] = acc if k in (lv, tp) else None
    shown = float(pred) if isinstance(pred, Fraction) else pred
    raw = str(value) if isinstance(value, Decimal) else value
    return {"scores": scores, "parsed": {"answer": raw, "normalized": shown, "gold": row["answer"], "found": found,
                                         "valid_json": valid}}


class SensorSuite(Suite):
    name = "sensor"
    headline = "answer_acc"
    metrics = ["valid_json", "answer_acc"] + ["acc_L%d" % x for x in LEVELS] + ["acc_" + t for t in TEMPLATE_IDS]
    group_key = "level"
    split = "test"

    def items(self) -> List[Dict[str, Any]]:
        rows = read_jsonl(data_path("sensor_interp", f"{self.split}.jsonl"))
        limit = self.options.get("limit")
        if limit:
            rows = rows[: int(limit)]
        mt = int(self.options.get("max_tokens") or MAX_TOKENS)
        out = []
        for r in rows:
            out.append({
                "id": r["id"],
                "messages": [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["question"]}],
                "params": {"max_tokens": mt},
                "meta": {"level": "L%d" % r["level"], "template": r["template"], "answer_type": r["answer_type"],
                         "setting": r["meta"]["knobs"]["setting"]},
                "_row": r,
            })
        return out

    def score(self, item: Dict[str, Any], resp: Dict[str, Any]) -> Dict[str, Any]:
        return score_output(item["_row"], resp.get("content") or "")


class SensorHeldoutSuite(SensorSuite):
    name = "sensor_heldout"
    split = "heldout"


class SensorCalibSuite(SensorSuite):
    name = "sensor_calib"
    split = "calib"
