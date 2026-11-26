from __future__ import annotations

from typing import Any, Dict, Optional, Type

from .base import Suite
from .iot import IotHeldoutSuite, IotSuite
from .sensor import SensorCalibSuite, SensorHeldoutSuite, SensorSuite

REGISTRY: Dict[str, Type[Suite]] = {
    "iot": IotSuite,
    "iot_heldout": IotHeldoutSuite,
    "sensor": SensorSuite,
    "sensor_heldout": SensorHeldoutSuite,
    "sensor_calib": SensorCalibSuite,
}


def get_suite(name: str, options: Optional[Dict[str, Any]] = None) -> Suite:
    return REGISTRY[name](options)
