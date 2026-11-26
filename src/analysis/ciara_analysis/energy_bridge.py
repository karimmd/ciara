from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _clients_dir() -> str:
    env = os.environ.get("CIARA_CLIENTS_DIR")
    return env if env else str(Path(__file__).resolve().parents[2] / "clients")


def device_energy(run_dir: str, schedule_dir: str, energy_config: Optional[str] = None,
                  tail: bool = False, *, overhead: str) -> Dict[str, Dict[str, Any]]:
    d = _clients_dir()
    if d not in sys.path:
        sys.path.insert(0, d)
    from ciara_clients import energy as en
    res = en.compute(run_dir, schedule_dir, tail, en.load_energy(energy_config), overhead=overhead)
    return {r["req"]: r for r in res["rows"]}
