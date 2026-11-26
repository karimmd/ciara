from __future__ import annotations

from typing import Any, Dict, List, Optional


class Suite:
    name: str = ""
    headline: str = ""
    metrics: List[str] = []
    ratio_metrics: List[Any] = []
    group_key: Optional[str] = None

    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        self.options = options or {}

    def items(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def score(self, item: Dict[str, Any], resp: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
