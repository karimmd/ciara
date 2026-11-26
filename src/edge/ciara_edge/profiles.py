from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass

PILOT_ROOT = os.environ.get("CIARA_PILOT_ROOT") or os.environ.get("PILOT_ROOT") or "workload"

_common = None


def _load_common():
    global _common
    if _common is None:
        path = os.path.join(PILOT_ROOT, "models", "model_table.py")
        spec = importlib.util.spec_from_file_location("ciara_pilot_common", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["ciara_pilot_common"] = mod
        spec.loader.exec_module(mod)
        _common = mod
    return _common


@dataclass(frozen=True)
class Profile:
    key: str
    model_dir: str
    model_path: str
    base: str
    precision: str
    family: str
    language_model_only: bool


def profile_info(key: str) -> Profile:
    c = _load_common()
    if key not in c.CKPTS:
        raise KeyError(f"unknown profile {key!r}")
    dirname, base, prec = c.CKPTS[key]
    fam = c.family_of(base)
    return Profile(key, dirname, c.model_path(dirname), base, prec, fam, fam in c.MULTIMODAL)
