from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

PRE, BURN_IN, MEASURE, POST = "pre", "burn_in", "measure", "post"


@dataclass(frozen=True)
class Grid:
    t0_mono_ns: int
    epoch_ns: int
    block_epochs: int
    burn_in: int
    n_blocks: int

    @classmethod
    def from_manifest(cls, m: Dict[str, Any]) -> "Grid":
        return cls(int(m["t0_mono_ns"]), int(m["epoch_ns"]), int(m["block_epochs"]), int(m["burn_in_epochs"]),
                   int(m["n_blocks"]))

    @property
    def total_epochs(self) -> int:
        return self.n_blocks * self.block_epochs

    def epoch_of(self, t_mono_ns: int) -> int:
        return (int(t_mono_ns) - self.t0_mono_ns) // self.epoch_ns

    def block_of(self, epoch: int) -> int:
        return epoch // self.block_epochs

    def epoch_start_ns(self, epoch: int) -> int:
        return self.t0_mono_ns + epoch * self.epoch_ns

    def block_range(self, block_id: int) -> Tuple[int, int]:
        return block_id * self.block_epochs, (block_id + 1) * self.block_epochs

    def burn_in_range(self, block_id: int) -> Tuple[int, int]:
        s = block_id * self.block_epochs
        return s, s + self.burn_in

    def measure_range(self, block_id: int) -> Tuple[int, int]:
        s = block_id * self.block_epochs
        return s + self.burn_in, (block_id + 1) * self.block_epochs

    def locate(self, t_mono_ns: Optional[int]) -> Dict[str, Any]:
        if t_mono_ns is None:
            return {"epoch": None, "block_id": None, "epoch_in_block": None, "phase": None}
        e = self.epoch_of(t_mono_ns)
        if e < 0:
            return {"epoch": e, "block_id": None, "epoch_in_block": None, "phase": PRE}
        if e >= self.total_epochs:
            return {"epoch": e, "block_id": None, "epoch_in_block": None, "phase": POST}
        k = e % self.block_epochs
        return {"epoch": e, "block_id": self.block_of(e), "epoch_in_block": k,
                "phase": BURN_IN if k < self.burn_in else MEASURE}
