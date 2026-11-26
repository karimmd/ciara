from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Optional

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class HttpError(Exception):
    def __init__(self, message: str, status: Optional[int] = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


def request(method: str, url: str, payload: Any = None, timeout: float = 5.0,
            headers: Optional[dict] = None, retries: int = 0, backoff: float = 0.5) -> tuple:
    data = None
    hdrs = {"Accept": "application/json"}
    if headers:
        hdrs.update(headers)
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
        try:
            with _OPENER.open(req, timeout=timeout) as resp:
                return resp.status, resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last = HttpError(f"{method} {url} -> HTTP {exc.code}", exc.code, body)
            if exc.code < 500:
                raise last
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last = HttpError(f"{method} {url} failed: {exc}")
        if attempt < retries:
            time.sleep(backoff * (2 ** attempt))
    assert last is not None
    raise last


def request_json(method: str, url: str, payload: Any = None, timeout: float = 5.0,
                 headers: Optional[dict] = None, retries: int = 0) -> Any:
    status, body = request(method, url, payload, timeout, headers, retries)
    if not body.strip():
        return {}
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise HttpError(f"{method} {url}: response is not JSON ({exc})", status, body[:200])
