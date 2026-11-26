from __future__ import annotations

import collections
from typing import Any, Deque, Optional

FIFO, F1PRIO = "fifo", "f1prio"
MODES = (FIFO, F1PRIO)


class Waiter:
    __slots__ = ("fut", "family", "arrive_seq", "admit_seq", "overtaken", "queue_len_admit", "mode")

    def __init__(self, fut: Any, family: Optional[int], arrive_seq: int):
        self.fut, self.family, self.arrive_seq = fut, family, arrive_seq
        self.admit_seq = self.overtaken = self.queue_len_admit = self.mode = None


class AdmissionQueue:
    def __init__(self) -> None:
        self.items: Deque[Waiter] = collections.deque()
        self.arrive_seq = 0
        self.admit_seq = 0

    def __len__(self) -> int:
        return len(self.items)

    def push(self, fut: Any, family: Optional[int]) -> Waiter:
        self.arrive_seq += 1
        w = Waiter(fut, family, self.arrive_seq)
        self.items.append(w)
        return w

    def remove(self, w: Waiter) -> bool:
        try:
            self.items.remove(w)
            return True
        except ValueError:
            return False

    def pop(self, mode: str = FIFO) -> Optional[Waiter]:
        if not self.items:
            return None
        if mode not in MODES:
            raise ValueError(f"unknown admission mode {mode!r}; expected one of {MODES}")
        k = 0
        if mode == F1PRIO:
            for i, w in enumerate(self.items):
                if w.family == 1:
                    k = i
                    break
        n = len(self.items)
        self.items.rotate(-k)
        w = self.items.popleft()
        self.items.rotate(k)
        self.admit_seq += 1
        w.admit_seq, w.overtaken, w.queue_len_admit, w.mode = self.admit_seq, k, n, mode
        return w
