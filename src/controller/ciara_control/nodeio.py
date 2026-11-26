from __future__ import annotations

import os
from typing import Optional

from . import httpclient
from .config import ControlSettings, NodeSpec


class NodeIO:
    def prepare(self, profiles: list) -> dict:
        raise NotImplementedError

    def engine_status(self) -> dict:
        raise NotImplementedError

    def activate(self, profile: str) -> dict:
        raise NotImplementedError

    def gateway_state(self) -> dict:
        raise NotImplementedError

    def gateway_config(self, cfg: dict) -> dict:
        raise NotImplementedError

    def engine_metrics(self, endpoint: str) -> str:
        raise NotImplementedError


def _auth_headers() -> dict:
    tok = os.environ.get("CIARA_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


class HttpNodeIO(NodeIO):
    def __init__(self, node: NodeSpec, ctl: ControlSettings):
        self.node = node
        self.ctl = ctl
        self.gw = (node.gateway or "").rstrip("/")
        self.mgr = (node.engine_manager or "").rstrip("/")

    def prepare(self, profiles: list) -> dict:
        return httpclient.request_json("POST", f"{self.mgr}/prepare", {"profiles": list(profiles)},
                                       timeout=self.ctl.prepare_s)

    def engine_status(self) -> dict:
        return httpclient.request_json("GET", f"{self.mgr}/status", timeout=self.ctl.state_s)

    def activate(self, profile: str) -> dict:
        return httpclient.request_json("POST", f"{self.mgr}/activate", {"profile": profile},
                                       timeout=self.ctl.activate_s, retries=self.ctl.retries)

    def gateway_state(self) -> dict:
        return httpclient.request_json("GET", f"{self.gw}/state", timeout=self.ctl.state_s)

    def gateway_config(self, cfg: dict) -> dict:
        return httpclient.request_json("POST", f"{self.gw}/config", cfg, timeout=self.ctl.config_s,
                                       retries=self.ctl.retries)

    def engine_metrics(self, endpoint: str) -> str:
        _, body = httpclient.request("GET", endpoint.rstrip("/") + "/metrics", timeout=self.ctl.metrics_s,
                                     headers=_auth_headers())
        return body


class CloudIO:
    def __init__(self, base_url: Optional[str], ctl: ControlSettings):
        self.base = (base_url or "").rstrip("/")
        self.ctl = ctl

    @property
    def enabled(self) -> bool:
        return bool(self.base)

    def notify(self, nodes: dict) -> dict:
        return httpclient.request_json("POST", f"{self.base}/config", {"nodes": nodes},
                                       timeout=self.ctl.config_s)


def switch_time_s(resp: Optional[dict]) -> Optional[float]:
    if not isinstance(resp, dict):
        return None
    for key, scale in (("switch_time_s", 1.0), ("switch_s", 1.0), ("switch_time_ms", 1e-3),
                       ("switch_ms", 1e-3), ("switch_time", 1.0)):
        v = resp.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v) * scale
    return None


def engine_endpoints(status: Optional[dict]) -> dict:
    out: dict[str, str] = {}
    if not isinstance(status, dict):
        return out
    eng = status.get("engines")
    if isinstance(eng, dict):
        for prof, v in eng.items():
            ep = v.get("endpoint") if isinstance(v, dict) else v
            if isinstance(ep, str):
                out[prof] = ep
    elif isinstance(eng, list):
        for v in eng:
            if isinstance(v, dict) and isinstance(v.get("endpoint"), str):
                out[str(v.get("profile"))] = v["endpoint"]
    prof = status.get("active_profile") or status.get("profile")
    ep = status.get("endpoint")
    if isinstance(ep, str) and prof and prof not in out:
        out[str(prof)] = ep
    return out


def active_profile_of(status: Optional[dict]) -> Optional[str]:
    if not isinstance(status, dict):
        return None
    p = status.get("active_profile") or status.get("profile")
    return str(p) if p else None
