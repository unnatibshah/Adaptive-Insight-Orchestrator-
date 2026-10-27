"""HTTP layer: serves the UI, streams graph runs as server-sent events, publishes the A2A agent cards."""
from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .graph import AGENTS, graph

RUN_TIMEOUT = 45.0
PER_MINUTE, PER_DAY = 6, 60
MAX_CONCURRENT_RUNS = 3

app = FastAPI(title="Adaptive Insight Orchestrator", docs_url="/api/docs", redoc_url=None)
_running = asyncio.Semaphore(MAX_CONCURRENT_RUNS)
# ponytail: per-process rate limit; one instance is enough for a demo. Use Redis if it ever scales out.
_hits: dict[str, deque] = defaultdict(deque)


def client_ip(req: Request) -> str:
    # The hosting proxy appends the real client IP last; earlier entries are client-supplied and spoofable.
    fwd = req.headers.get("x-forwarded-for")
    return fwd.split(",")[-1].strip() if fwd else (req.client.host if req.client else "unknown")


def check_rate(ip: str) -> None:
    now, q = time.time(), _hits[ip]
    while q and now - q[0] > 86400:
        q.popleft()
    if len(q) >= PER_DAY or sum(1 for t in q if now - t < 60) >= PER_MINUTE:
        raise HTTPException(429, "Too many runs. Give it a minute and try again.")
    q.append(now)


class RunRequest(BaseModel):
    query: str = Field(min_length=2, max_length=300)


@app.post("/api/run")
async def run(body: RunRequest, req: Request):
    query = " ".join(body.query.split())
    if len(query) < 2:
        raise HTTPException(422, "Query is empty.")
    check_rate(client_ip(req))
    if _running.locked():
        raise HTTPException(503, "The orchestrator is busy. Try again in a few seconds.")

    async def events():
        async with _running:
            t0 = time.perf_counter()
            try:
                async with asyncio.timeout(RUN_TIMEOUT):
                    async for ev in graph.astream({"query": query}, stream_mode="custom"):
                        yield f"data: {json.dumps(ev)}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'ms': round((time.perf_counter() - t0) * 1000)})}\n\n"
            except TimeoutError:
                yield f"data: {json.dumps({'type': 'error', 'message': 'The run took too long and was stopped.'})}\n\n"
            except Exception:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Something went wrong. Please try again.'})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/agents")
def agents():
    return [a["card"] for a in AGENTS.values()]


@app.get("/agents/{agent_id}/.well-known/agent-card.json")
def agent_card(agent_id: str):
    if agent_id not in AGENTS:
        raise HTTPException(404, "Unknown agent.")
    return AGENTS[agent_id]["card"]


@app.get("/health")
def health():
    return {"status": "ok"}


WEB = Path(__file__).resolve().parent.parent / "web"


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")
