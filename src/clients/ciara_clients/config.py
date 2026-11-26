from __future__ import annotations

import copy
import hashlib
import os
import tomllib
from typing import Any, Dict, Optional

SCHEMA = "ciara-clients-config/1"
CONFIG_DIR = os.environ.get("CIARA_CONFIG_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "configs")
DEFAULT_PATH = os.path.join(CONFIG_DIR, "clients", "default.toml")

LINK_FIELDS = ["rate_up_kbps", "rate_down_kbps", "delay_up_ms", "delay_down_ms", "jitter_up_ms", "jitter_down_ms",
               "loss_up_pct", "loss_down_pct", "buffer_ms"]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Optional[str] = None, preset: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None,
         context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    path = path or DEFAULT_PATH
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    cfg["_path"] = os.path.abspath(path)
    cfg["_sha256"] = sha256_file(path)
    run = cfg["run"]
    if preset:
        if preset not in cfg.get("presets", {}):
            raise ValueError(f"unknown preset {preset!r}; known: {sorted(cfg.get('presets', {}))}")
        run.update(cfg["presets"][preset])
        run["preset"] = preset
    for k, v in (overrides or {}).items():
        if v is not None:
            run[k] = v
    for k, v in (context or {}).items():
        if v is not None:
            cfg.setdefault("context", {"epoch_s": 30, "block_epochs": 50, "mode": "pooled", "pooled_id": "pooled"})[k] = v
    validate(cfg)
    return cfg


def validate(cfg: Dict[str, Any]) -> None:
    run = cfg["run"]
    if run["arrival"] not in ("mmpp", "poisson"):
        raise ValueError("run.arrival must be 'mmpp' or 'poisson'")
    if run["nodes"] < 1 or run["duration_s"] <= 0 or run["load_scale"] < 0:
        raise ValueError("run.nodes, run.duration_s and run.load_scale out of range")
    fams = cfg["families"]
    for k, fam in fams.items():
        if fam["suite"] not in ("iot", "sensor"):
            raise ValueError(f"families.{k}.suite must be 'iot' or 'sensor'")
    links = cfg["links"]
    for name, link in links.items():
        for fld in LINK_FIELDS:
            if fld not in link:
                raise ValueError(f"links.{name} lacks {fld}")
        if link["class"] not in cfg["classes"]:
            raise ValueError(f"links.{name}.class {link['class']!r} is not a class")
        if "src" not in link or any(f not in link["src"] for f in LINK_FIELDS):
            raise ValueError(f"links.{name} needs a src entry for every parameter")
    for c, cl in cfg["classes"].items():
        mix = cl["family_mix"]
        if any(f not in fams for f in mix) or abs(sum(mix.values()) - 1.0) > 1e-6:
            raise ValueError(f"classes.{c}.family_mix must name known families and sum to 1")
        lm = cl["link_mix"]
        if abs(sum(lm.values()) - 1.0) > 1e-6:
            raise ValueError(f"classes.{c}.link_mix must sum to 1")
        for name in lm:
            if name not in links or links[name]["class"] != c:
                raise ValueError(f"classes.{c}.link_mix names {name!r}, which is not a link of this class")
        if not 0 < cl["cpu_share"] <= 1.0:
            raise ValueError(f"classes.{c}.cpu_share must be in (0, 1]")
        if cl["connection"]["mode"] not in ("reuse", "new"):
            raise ValueError(f"classes.{c}.connection.mode must be 'reuse' or 'new'")
        if cl["devices_per_node"] > 340:
            raise ValueError("at most 340 devices per (node, class) group: 64 source ports per device below 32768")
    ctx = cfg.get("context", {})
    if ctx:
        if ctx["mode"] not in ("pooled", "levels"):
            raise ValueError("context.mode must be 'pooled' or 'levels'")
        if ctx["epoch_s"] <= 0 or ctx["block_epochs"] < 1:
            raise ValueError("context.epoch_s and context.block_epochs must be positive")
        if ctx["mode"] == "levels":
            lv = ctx["level_factor"]
            if ctx["default_id"] not in lv or set(ctx["level_prob"]) - set(lv) or abs(sum(ctx["level_prob"].values()) - 1.0) > 1e-6:
                raise ValueError("context levels: default_id, level_prob and level_factor must agree and level_prob sums to 1")
            if set(ctx.get("pattern", [])) - set(lv):
                raise ValueError("context.pattern names an unknown level")
    m = cfg["mmpp"]
    if not (len(m["states"]) == len(m["rel_rate"]) == len(m["mean_dwell_s"])):
        raise ValueError("mmpp arrays must have equal length")
    if m["modulation"] not in ("shared", "independent"):
        raise ValueError("mmpp.modulation must be 'shared' or 'independent'")


def deep(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(cfg)
