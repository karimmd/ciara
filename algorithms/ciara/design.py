from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Draw:
    arm_id: str
    arm_index: int
    propensity: float
    u: float
    entropy: tuple
    label: str
    context_id: str

    def to_dict(self) -> dict:
        return {"seed": self.entropy[0], "node_index": self.entropy[1], "block_id": self.entropy[2],
                "entropy": list(self.entropy), "draw": self.u, "arm_index": self.arm_index}


def certification_propensities(n: int, a0_index: int, eps: float) -> list[float]:
    p = [eps / n] * n
    p[a0_index] += 1.0 - eps
    return p


class DesignLaw:
    def __init__(self, label: str, arm_ids: tuple, seed: int, vectors: dict, default_vector: list,
                 epsilon: Optional[float] = None, incumbent: Optional[str] = None):
        self.label = label
        self.arm_ids = tuple(arm_ids)
        self.seed = int(seed)
        self._vectors = vectors
        self._default = default_vector
        self.epsilon = epsilon
        self.incumbent = incumbent

    def propensities(self, context_id: str) -> list[float]:
        return list(self._vectors.get(context_id, self._default))

    def propensity_of(self, arm_id: str, context_id: str) -> float:
        return self.propensities(context_id)[self.arm_ids.index(arm_id)]

    @staticmethod
    def uniform_variate(seed: int, node_index: int, block_id: int) -> float:
        ss = np.random.SeedSequence([int(seed), int(node_index), int(block_id)])
        return float(np.random.Generator(np.random.PCG64(ss)).random())

    def draw(self, node_index: int, block_id: int, context_id: str) -> Draw:
        p = self.propensities(context_id)
        u = self.uniform_variate(self.seed, node_index, block_id)
        cum: list[float] = []
        s = 0.0
        for x in p:
            s += x
            cum.append(s)
        idx = bisect.bisect_right(cum, u)
        if idx >= len(p):
            idx = max(i for i, x in enumerate(p) if x > 0)
        return Draw(arm_id=self.arm_ids[idx], arm_index=idx, propensity=p[idx], u=u,
                    entropy=(self.seed, int(node_index), int(block_id)), label=self.label,
                    context_id=context_id)

    def describe(self, context_id: str) -> dict:
        return {"label": self.label, "arms": list(self.arm_ids), "propensities": self.propensities(context_id),
                "epsilon": self.epsilon, "incumbent": self.incumbent, "context_id": context_id}


class BalancedLaw(DesignLaw):
    def __init__(self, seed: int, composition: tuple, node_indexes: tuple):
        ids = tuple(dict.fromkeys(composition))
        n = len(composition)
        vec = [composition.count(a) / n for a in ids]
        super().__init__("balanced", ids, seed, {}, vec)
        self.composition = tuple(composition)
        self.node_indexes = tuple(sorted(int(i) for i in node_indexes))
        if len(self.node_indexes) != n:
            raise ValueError(f"composition of {n} arms for {len(self.node_indexes)} nodes")

    @staticmethod
    def permutation(seed: int, block_id: int, n: int) -> list:
        ss = np.random.SeedSequence([int(seed), int(block_id), 1017])
        return [int(x) for x in np.random.Generator(np.random.PCG64(ss)).permutation(n)]

    def draw(self, node_index: int, block_id: int, context_id: str) -> Draw:
        pos = self.node_indexes.index(int(node_index))
        perm = self.permutation(self.seed, block_id, len(self.composition))
        arm = self.composition[perm[pos]]
        idx = self.arm_ids.index(arm)
        u = (perm[pos] + 0.5) / len(self.composition)
        return Draw(arm_id=arm, arm_index=idx, propensity=self._default[idx], u=u, entropy=(self.seed, int(node_index), int(block_id)),
                    label=self.label, context_id=context_id)

    def describe(self, context_id: str) -> dict:
        d = super().describe(context_id)
        d["composition"] = list(self.composition)
        return d


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
