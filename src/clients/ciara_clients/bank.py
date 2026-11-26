from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

def pilot_root() -> Path:
    env = os.environ.get("CIARA_PILOT_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "workload"


def _suites():
    root = str(pilot_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from eval.suites.registry import get_suite

    return get_suite


def load_family_items(cfg: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    get_suite = _suites()
    out: Dict[int, List[Dict[str, Any]]] = {}
    for key, fam in sorted(cfg["families"].items(), key=lambda kv: int(kv[0])):
        suite = get_suite(fam["suite"], {"max_tokens": fam["max_tokens"]})
        items = []
        for it in suite.items():
            items.append({"id": it["id"], "messages": it["messages"], "max_tokens": int(it["params"]["max_tokens"]),
                          "meta": it.get("meta", {})})
        out[int(key)] = items
    return out


def data_manifest_sha256() -> str:
    path = pilot_root() / "data" / "MANIFEST.json"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_sha256() -> Dict[str, str]:
    base = pilot_root() / "data"
    files = {"iot_test": base / "iot_toolcall" / "test.jsonl", "sensor_test": base / "sensor_interp" / "test.jsonl"}
    return {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in files.items()}


def write_bank(path: str, items_by_family: Dict[int, List[Dict[str, Any]]], referenced: Optional[set] = None) -> str:
    h = hashlib.sha256()
    with open(path, "wb") as f:
        for fam in sorted(items_by_family):
            for it in items_by_family[fam]:
                if referenced is not None and it["id"] not in referenced:
                    continue
                rec = {"id": it["id"], "family": fam, "max_tokens": it["max_tokens"], "messages": it["messages"]}
                line = (json.dumps(rec, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                f.write(line)
                h.update(line)
    return h.hexdigest()


def read_bank(path: str) -> Dict[str, Dict[str, Any]]:
    bank: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                bank[rec["id"]] = rec
    return bank
