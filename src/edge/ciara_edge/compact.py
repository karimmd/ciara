from __future__ import annotations

import json
import os
import sys
import types
from decimal import Decimal
from typing import Any, Optional, Tuple

FALLBACK_BYTES = 512
DEFAULT_PILOT = "workload"


def _load(root: str):
    for name, sub in (("eval", ""), ("eval.suites", "suites")):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [os.path.join(root, "eval", sub) if sub else os.path.join(root, "eval")]
            sys.modules[name] = pkg
    if root not in sys.path:
        sys.path.insert(0, root)
    from eval.suites import iot, sensor
    return iot, sensor


class Compactor:
    def __init__(self, pilot_root: Optional[str] = None):
        root = pilot_root or os.environ.get("CIARA_PILOT_ROOT") or DEFAULT_PILOT
        self.iot, self.sensor = _load(root)

    @staticmethod
    def _dumps(obj: Any) -> str:
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False,
                          default=lambda d: float(d) if isinstance(d, Decimal) else str(d))

    def extract(self, family: Optional[int], text: str) -> Optional[str]:
        if family == 1:
            p = self.iot.extract_calls(text or "", None, False)
            if not p["container_ok"]:
                return None
            return self._dumps({"calls": [{"name": n, "arguments": a} for n, a in p["calls"]]})
        if family == 2:
            _, value, found = self.sensor.parse_reply(text or "")
            return self._dumps({"answer": value}) if found else None
        return None

    def compact(self, family: Optional[int], text: str) -> Tuple[str, bool]:
        out = self.extract(family, text)
        if out is not None:
            return out, True
        tail = (text or "").encode("utf-8")[-FALLBACK_BYTES:]
        return tail.decode("utf-8", "ignore"), False
