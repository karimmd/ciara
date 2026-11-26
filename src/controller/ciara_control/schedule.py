from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BlockGrid:
    epoch_ns: int
    block_epochs: int
    burn_in: int
    n_blocks: int

    @property
    def total_epochs(self) -> int:
        return self.n_blocks * self.block_epochs

    def block_of(self, epoch: int) -> int:
        return epoch // self.block_epochs

    def epoch_in_block(self, epoch: int) -> int:
        return epoch % self.block_epochs

    def block_start(self, block_id: int) -> int:
        return block_id * self.block_epochs

    def block_end(self, block_id: int) -> int:
        return (block_id + 1) * self.block_epochs

    def is_block_start(self, epoch: int) -> bool:
        return epoch % self.block_epochs == 0

    def is_burn_in(self, epoch: int) -> bool:
        return self.epoch_in_block(epoch) < self.burn_in

    def measurement_range(self, block_id: int) -> tuple:
        s = self.block_start(block_id)
        return (s + self.burn_in, s + self.block_epochs)

    def boundary_ns(self, t0_mono_ns: int, epoch: int) -> int:
        return t0_mono_ns + epoch * self.epoch_ns

    def epoch_at(self, t0_mono_ns: int, now_mono_ns: int) -> int:
        return (now_mono_ns - t0_mono_ns) // self.epoch_ns

    def next_boundary_epoch(self, t0_mono_ns: int, now_mono_ns: int, margin_ns: int = 0) -> int:
        d = now_mono_ns + margin_ns - t0_mono_ns
        return max(0, -(-d // self.epoch_ns))


class ContextSchedule:
    def __init__(self, default: str = "pooled", segments: Optional[list] = None, source: Optional[str] = None):
        self.default = default
        self.segments = sorted(
            [(int(s["start_epoch"]), int(s["end_epoch"]), str(s["context_id"])) for s in (segments or [])]
        )
        self.source = source

    def context_at(self, epoch: int) -> str:
        for s, e, cid in self.segments:
            if s <= epoch < e:
                return cid
        return self.default

    def misaligned_segments(self, block_epochs: int) -> list:
        return [(s, e, c) for s, e, c in self.segments if s % block_epochs or e % block_epochs]

    @classmethod
    def load(cls, ref: Optional[str], default: str, base_dir: Optional[str] = None) -> "ContextSchedule":
        if not ref:
            return cls(default)
        path = ref if os.path.isabs(ref) or not base_dir else os.path.join(base_dir, ref)
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        if path.endswith((".yaml", ".yml")):
            import yaml
            raw = yaml.safe_load(text)
        else:
            raw = json.loads(text)
        segs = raw.get("segments") or raw.get("contexts") or []
        return cls(raw.get("default_context", default), segs, source=path)
