from __future__ import annotations

import threading
import time


class Clock:
    def mono_ns(self) -> int:
        raise NotImplementedError

    def wall_ns(self) -> int:
        raise NotImplementedError

    def boot_id(self) -> str:
        raise NotImplementedError

    def wait_until(self, t_mono_ns: int, stop: threading.Event) -> bool:
        raise NotImplementedError


def read_boot_id() -> str:
    try:
        with open("/proc/sys/kernel/random/boot_id") as fh:
            return fh.read().strip()
    except OSError:
        return "unknown"


class RealClock(Clock):
    def boot_id(self) -> str:
        return read_boot_id()

    def mono_ns(self) -> int:
        return time.monotonic_ns()

    def wall_ns(self) -> int:
        return time.time_ns()

    def wait_until(self, t_mono_ns: int, stop: threading.Event) -> bool:
        while True:
            if stop.is_set():
                return False
            remaining = (t_mono_ns - time.monotonic_ns()) / 1e9
            if remaining <= 0:
                return True
            stop.wait(min(remaining, 1.0))


class FakeClock(Clock):
    def __init__(self, mono_start_ns: int = 5_000_000_000_000, wall_origin_ns: int = 1_760_000_000_000_000_000,
                 boot_id: str = "fake-boot-1"):
        self._mono = mono_start_ns
        self._origin = wall_origin_ns
        self._boot = boot_id
        self._lock = threading.Lock()

    def boot_id(self) -> str:
        return self._boot

    def skew_wall(self, ns: int) -> None:
        with self._lock:
            self._origin += int(ns)

    def mono_ns(self) -> int:
        with self._lock:
            return self._mono

    def wall_ns(self) -> int:
        with self._lock:
            return self._origin + self._mono

    def advance(self, ns: int) -> None:
        with self._lock:
            self._mono += int(ns)

    def wait_until(self, t_mono_ns: int, stop: threading.Event) -> bool:
        if stop.is_set():
            return False
        with self._lock:
            if t_mono_ns > self._mono:
                self._mono = t_mono_ns
        return True
