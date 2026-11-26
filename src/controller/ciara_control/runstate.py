from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .util import read_jsonl

MANIFEST = "manifest.json"
BLOCKS = "blocks.jsonl"
EPOCHS = "epochs.jsonl"
EVENTS = "events.jsonl"
STATUS = "STATUS"
PROGRESS = "PROGRESS"


@dataclass
class RunState:
    manifest: dict
    blocks: dict = field(default_factory=dict)
    ended: dict = field(default_factory=dict)
    epochs: dict = field(default_factory=dict)
    last_epoch: int = -1

    def epochs_of(self, node: str) -> set:
        return self.epochs.get(node, set())


def load_manifest(run_dir: str) -> dict | None:
    path = os.path.join(run_dir, MANIFEST)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_run_state(run_dir: str) -> RunState | None:
    manifest = load_manifest(run_dir)
    if manifest is None:
        return None
    st = RunState(manifest=manifest)
    for rec in read_jsonl(os.path.join(run_dir, BLOCKS)):
        key = (rec.get("node"), rec.get("block_id"))
        if rec.get("rec") == "block":
            st.blocks[key] = rec
        elif rec.get("rec") == "block_end":
            st.ended[key] = rec
    for rec in read_jsonl(os.path.join(run_dir, EPOCHS)):
        if rec.get("rec") != "epoch":
            continue
        st.epochs.setdefault(rec["node"], set()).add(rec["epoch"])
        st.last_epoch = max(st.last_epoch, rec["epoch"])
    return st
