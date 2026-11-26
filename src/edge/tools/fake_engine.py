from __future__ import annotations

import argparse
import asyncio
import json
import time

from aiohttp import web


def make_app(args):
    st = {"running": 0, "max_running": 0, "total": 0, "sleeping": False}

    async def chat(request: web.Request):
        body = await request.json()
        st["last_headers"] = {k: v for k, v in request.headers.items() if k.lower().startswith("x-ciara")}
        st["last_body_keys"] = sorted(body)
        st.setdefault("max_tokens_seen", []).append(body.get("max_tokens"))
        high = "UNCERTAIN" in json.dumps(body.get("messages"))
        if st["sleeping"]:
            return web.json_response({"error": "engine is asleep"}, status=503)
        st["running"] += 1
        st["total"] += 1
        st["max_running"] = max(st["max_running"], st["running"])
        try:
            model = body.get("model", "fake")
            n = int(body.get("max_tokens") or args.tokens)
            n = min(n, args.tokens)
            fin = "stop"
            nat_short = getattr(args, "natural_short", None)
            if nat_short:
                nat = getattr(args, "natural_long", 200) if "LONGREPLY" in json.dumps(body.get("messages")) else nat_short
                lim = body.get("max_tokens")
                n = min(int(lim), nat) if lim else nat
                fin = "length" if (lim and int(lim) < nat) else "stop"
            await asyncio.sleep(args.ttft_ms / 1000)
            fam = request.headers.get("X-Ciara-Family", "?")
            toks = [f"{model}:{fam}:{i} " for i in range(n)]
            def lp_entry(tok):
                k = int(body.get("top_logprobs") or 1)
                import math
                if high:
                    tops = [{"token": f"t{j}", "logprob": math.log(1.0 / (k + 1))} for j in range(k)]
                else:
                    tops = [{"token": tok, "logprob": -0.0005}] + [{"token": f"t{j}", "logprob": -9.0 - j} for j in range(k - 1)]
                return {"token": tok, "logprob": tops[0]["logprob"], "top_logprobs": tops}
            if not body.get("stream"):
                await asyncio.sleep(args.tok_ms * n / 1000)
                nn = int(body.get("n") or 1)
                choices = []
                for i in range(nn):
                    text = "".join(toks) + (f" variant{i}" if (high and nn > 1) else "")
                    ch = {"index": i, "message": {"role": "assistant", "content": text}, "finish_reason": fin}
                    if body.get("logprobs"):
                        ch["logprobs"] = {"content": [lp_entry(t) for t in toks]}
                    choices.append(ch)
                return web.json_response({"id": "x", "object": "chat.completion", "model": model, "choices": choices,
                                          "usage": {"prompt_tokens": 10, "completion_tokens": n * nn, "total_tokens": 10 + n * nn}})
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)

            async def ev(d):
                await resp.write(b"data: " + json.dumps(d).encode() + b"\n\n")
            base = {"id": "x", "object": "chat.completion.chunk", "model": model}
            await ev({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
            for t in toks:
                await asyncio.sleep(args.tok_ms / 1000)
                ch = {"index": 0, "delta": {"content": t}, "finish_reason": None}
                if body.get("logprobs"):
                    ch["logprobs"] = {"content": [lp_entry(t)]}
                await ev({**base, "choices": [ch]})
            await ev({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": fin}]})
            if (body.get("stream_options") or {}).get("include_usage"):
                await ev({**base, "choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": n, "total_tokens": 10 + n}})
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp
        finally:
            st["running"] -= 1

    async def stats(request):
        return web.json_response(st)

    async def sleep(request):
        await asyncio.sleep(args.sleep_s)
        st["sleeping"] = True
        return web.Response(status=200)

    async def wake(request):
        await asyncio.sleep(args.wake_s)
        st["sleeping"] = False
        return web.Response(status=200)

    async def is_sleeping(request):
        return web.json_response({"is_sleeping": st["sleeping"]})

    async def reset_prefix(request):
        return web.Response(status=200)

    async def health(request):
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_get("/stats", stats)
    app.router.add_get("/health", health)
    app.router.add_post("/sleep", sleep)
    app.router.add_post("/wake_up", wake)
    app.router.add_get("/is_sleeping", is_sleeping)
    app.router.add_post("/reset_prefix_cache", reset_prefix)
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9990)
    ap.add_argument("--tokens", type=int, default=20)
    ap.add_argument("--tok-ms", type=float, default=20.0)
    ap.add_argument("--ttft-ms", type=float, default=50.0)
    ap.add_argument("--sleep-s", type=float, default=0.3)
    ap.add_argument("--wake-s", type=float, default=0.5)
    a = ap.parse_args()
    web.run_app(make_app(a), host=a.host, port=a.port, print=lambda *x: print("fake engine on", a.host, a.port, flush=True))
