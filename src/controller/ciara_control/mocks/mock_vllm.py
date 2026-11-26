from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import Optional

from aiohttp import web


class MockVllm:
    def __init__(self, ttft_s: float = 0.05, tok_s: float = 0.01, tokens: int = 20, token: Optional[str] = None):
        self.ttft_s, self.tok_s, self.tokens, self.token = ttft_s, tok_s, tokens, token
        self.in_flight = 0
        self.max_concurrent = 0
        self.total = 0
        self.auth_failures = 0
        self.arrival_order: list = []
        self.prompt_tokens_total = 0
        self.generation_tokens_total = 0
        self.last_body: Optional[dict] = None

    def make_app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self.chat)
        app.router.add_get("/stats", self.stats)
        app.router.add_get("/metrics", self.metrics)
        app.router.add_get("/health", self.health)
        return app

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def stats(self, request: web.Request) -> web.Response:
        return web.json_response({"in_flight": self.in_flight, "max_concurrent": self.max_concurrent,
                                  "total": self.total, "auth_failures": self.auth_failures,
                                  "arrival_order": self.arrival_order[-1000:]})

    async def metrics(self, request: web.Request) -> web.Response:
        text = (f"vllm:num_requests_running {self.in_flight}\nvllm:num_requests_waiting 0\n"
                f"vllm:kv_cache_usage_perc 0.1\nvllm:prompt_tokens_total {self.prompt_tokens_total}\n"
                f"vllm:generation_tokens_total {self.generation_tokens_total}\n"
                f"vllm:request_success_total{{finished_reason=\"stop\"}} {self.total}\n")
        return web.Response(text=text, content_type="text/plain")

    async def chat(self, request: web.Request) -> web.StreamResponse:
        if self.token and request.headers.get("Authorization") != f"Bearer {self.token}":
            self.auth_failures += 1
            return web.json_response({"error": "unauthorized"}, status=401)
        body = await request.json()
        self.last_body = body
        self.arrival_order.append(body.get("mock_id"))
        fail = body.get("mock_fail")
        if fail == "http500":
            return web.json_response({"error": "mock failure"}, status=500)
        n = min(int(body.get("max_tokens") or self.tokens), int(body.get("mock_tokens") or self.tokens))
        ttft = float(body.get("mock_ttft_s", self.ttft_s))
        n_prompt = sum(len(str(m.get("content", "")).split()) for m in body.get("messages", [])) + 4
        self.in_flight += 1
        self.max_concurrent = max(self.max_concurrent, self.in_flight)
        self.total += 1
        try:
            await asyncio.sleep(ttft)
            if not body.get("stream"):
                text = " ".join(f"tok{i}" for i in range(n))
                self.prompt_tokens_total += n_prompt
                self.generation_tokens_total += n
                return web.json_response({
                    "id": "mock", "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": n_prompt, "completion_tokens": n, "total_tokens": n_prompt + n}})
            resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            role = {"id": "mock", "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}
            await resp.write(b"data: " + json.dumps(role).encode() + b"\n\n")
            for i in range(n):
                if fail == "drop" and i == 2:
                    request.transport.close()
                    return resp
                ch = {"id": "mock", "object": "chat.completion.chunk",
                      "choices": [{"index": 0, "delta": {"content": f"tok{i} "}, "finish_reason": None}]}
                await resp.write(b"data: " + json.dumps(ch).encode() + b"\n\n")
                await asyncio.sleep(self.tok_s)
            fin = {"id": "mock", "object": "chat.completion.chunk",
                   "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            await resp.write(b"data: " + json.dumps(fin).encode() + b"\n\n")
            if (body.get("stream_options") or {}).get("include_usage"):
                usage = {"id": "mock", "object": "chat.completion.chunk", "choices": [],
                         "usage": {"prompt_tokens": n_prompt, "completion_tokens": n, "total_tokens": n_prompt + n}}
                await resp.write(b"data: " + json.dumps(usage).encode() + b"\n\n")
            await resp.write(b"data: [DONE]\n\n")
            self.prompt_tokens_total += n_prompt
            self.generation_tokens_total += n
            await resp.write_eof()
            return resp
        finally:
            self.in_flight -= 1


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="mock vLLM engine")
    ap.add_argument("--listen", default="127.0.0.1:0")
    ap.add_argument("--ttft-s", type=float, default=0.05)
    ap.add_argument("--tok-s", type=float, default=0.01)
    ap.add_argument("--tokens", type=int, default=20)
    ap.add_argument("--token", default=None)
    args = ap.parse_args(argv)
    host, port = args.listen.rsplit(":", 1)
    eng = MockVllm(args.ttft_s, args.tok_s, args.tokens, args.token)
    web.run_app(eng.make_app(), host=host, port=int(port), access_log=None, print=lambda *_: None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
