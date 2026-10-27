"""The orchestration graph: plan -> agents (in parallel) -> synthesize, built with LangGraph.

Each agent publishes an A2A-style agent card and owns one MCP server. The planner reads the cards
and the MCP tool schemas to decide which agents to call; the answer is grounded only in tool output.
Every node emits trace events through LangGraph's stream writer so the UI can show the run live.
"""
from __future__ import annotations

import asyncio
import json
import operator
import os
import re
import time
from typing import Annotated, Any, TypedDict

import httpx
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from mcp import Client

from .tools import knowledge_mcp, news_mcp, weather_mcp

PLANNER_MODEL = os.getenv("PLANNER_MODEL", "openai/gpt-oss-20b")
WRITER_MODEL = os.getenv("WRITER_MODEL", "openai/gpt-oss-120b")
TOOL_TIMEOUT = 10.0

# A2A agent cards (https://a2a-protocol.org): served at /agents/<id>/.well-known/agent-card.json
AGENTS: dict[str, dict[str, Any]] = {
    "weather": {"card": {
        "name": "weather-agent", "version": "2.0.0",
        "description": "Current conditions and short-range forecasts for any city.",
        "skills": [{"id": "get_weather", "name": "Weather lookup", "tags": ["weather", "forecast"],
                    "description": "Current temperature, humidity, wind and a 3-day forecast for a city.",
                    "examples": ["weather in Tokyo", "will it rain in London this week?"]}]},
        "mcp": weather_mcp, "tool": "get_weather"},
    "news": {"card": {
        "name": "news-agent", "version": "2.0.0",
        "description": "Recent news headlines on any topic, from the last 3 days.",
        "skills": [{"id": "get_news", "name": "News search", "tags": ["news", "headlines"],
                    "description": "Up to 6 recent English-language articles about a topic.",
                    "examples": ["latest news on renewable energy", "what's happening in Japan"]}]},
        "mcp": news_mcp, "tool": "get_news"},
    "knowledge": {"card": {
        "name": "knowledge-agent", "version": "2.0.0",
        "description": "Encyclopedia background on people, places, organizations and concepts.",
        "skills": [{"id": "get_background", "name": "Background lookup", "tags": ["facts", "background"],
                    "description": "A short encyclopedia summary of a subject.",
                    "examples": ["tell me about Mumbai", "who is Ada Lovelace"]}]},
        "mcp": knowledge_mcp, "tool": "get_background"},
}
for _id, _a in AGENTS.items():
    _a["card"].update({"url": f"/agents/{_id}", "protocolVersion": "0.3.0",
                       "capabilities": {"streaming": False}, "defaultInputModes": ["text/plain"],
                       "defaultOutputModes": ["application/json"]})


class State(TypedDict, total=False):
    query: str
    tasks: list[dict]
    results: Annotated[list[dict], operator.add]
    answer: str


# ---------- LLM (Groq, OpenAI-compatible) ----------

async def chat(model: str, messages: list[dict], *, json_mode: bool = False, max_tokens: int = 500) -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("no GROQ_API_KEY")
    body: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens,
                            "temperature": 0.2, "reasoning_effort": "low"}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
        r = await client.post("https://api.groq.com/openai/v1/chat/completions", json=body,
                              headers={"Authorization": f"Bearer {key}"})
    if r.status_code == 429:
        raise RuntimeError("model rate limit reached")
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


# ---------- Planning ----------

async def tool_schemas() -> dict[str, dict]:
    """Discover each agent's tool input schema from its MCP server (cached after first call)."""
    if not hasattr(tool_schemas, "cache"):
        out = {}
        for aid, a in AGENTS.items():
            async with Client(a["mcp"]) as c:
                t = next(t for t in (await c.list_tools()).tools if t.name == a["tool"])
                out[aid] = {k: v.get("type", "string") for k, v in t.input_schema["properties"].items()}
        tool_schemas.cache = out
    return tool_schemas.cache


def validate_plan(raw: Any, schemas: dict[str, dict]) -> list[dict]:
    """Keep only tasks for known agents whose args match the tool schema. Max 4 tasks."""
    tasks = []
    for t in (raw.get("tasks") if isinstance(raw, dict) else None) or []:
        if not isinstance(t, dict) or t.get("agent") not in schemas or not isinstance(t.get("args"), dict):
            continue
        args = {k: str(v).strip()[:80] for k, v in t["args"].items() if k in schemas[t["agent"]] and str(v).strip()}
        if set(args) == set(schemas[t["agent"]]) and {"agent": t["agent"], "args": args} not in tasks:
            tasks.append({"agent": t["agent"], "args": args})
    return tasks[:4]


_WEATHER = re.compile(r"\b(weather|temperature|forecast|rain|snow|sunny|umbrella|humid|wind|cold|hot)\b", re.I)
_NEWS = re.compile(r"\b(news|headlines?|latest|happening|updates?|stories)\b", re.I)
_PLACE = re.compile(r"\b(?:in|for|at|about|on)\s+([A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*)*)")


def fallback_plan(query: str) -> list[dict]:
    """Keyword routing, used when the planner model is unavailable or returns an invalid plan."""
    m = _PLACE.search(query)
    subject = m.group(1) if m else query.strip(" ?.!")[:80]
    tasks = []
    if _WEATHER.search(query) and m:
        tasks.append({"agent": "weather", "args": {"city": subject}})
    if _NEWS.search(query):
        tasks.append({"agent": "news", "args": {"topic": subject}})
    if not tasks:
        tasks.append({"agent": "knowledge", "args": {"subject": subject}})
    return tasks


PLAN_PROMPT = """You are the planner of a multi-agent system. Pick which agents to call for the user's request.
Agents (from their A2A cards) and their tool arguments:
{agents}
Rules: call only agents that help answer; call the same agent several times for several places or topics;
use [] when no agent is relevant (greetings, unrelated requests). Convert regions to a specific city for weather.
Reply with JSON only: {{"tasks": [{{"agent": "<id>", "args": {{...}}}}], "reason": "<one short sentence>"}}"""


async def plan(state: State) -> dict:
    emit = get_stream_writer()
    emit({"type": "node", "node": "plan", "status": "start"})
    t0 = time.perf_counter()
    schemas = await tool_schemas()
    agents = "\n".join(f'- {aid}: {a["card"]["description"]} args: {json.dumps(schemas[aid])}' for aid, a in AGENTS.items())
    source, reason = "llm", ""
    try:
        raw = json.loads(await chat(PLANNER_MODEL, [{"role": "system", "content": PLAN_PROMPT.format(agents=agents)},
                                                    {"role": "user", "content": state["query"]}], json_mode=True, max_tokens=400))
        tasks, reason = validate_plan(raw, schemas), str(raw.get("reason", ""))[:200]
        if not tasks and raw.get("tasks"):  # model asked for something, but nothing valid survived
            raise ValueError("invalid plan")
    except Exception as e:
        tasks, source, reason = fallback_plan(state["query"]), "fallback", f"keyword routing ({e})"
    emit({"type": "plan", "source": source, "model": PLANNER_MODEL if source == "llm" else None,
          "reason": reason, "tasks": tasks, "ms": round((time.perf_counter() - t0) * 1000)})
    return {"tasks": tasks}


def fan_out(state: State):
    return [Send("agent", {"task": t, "i": i}) for i, t in enumerate(state["tasks"])] or "synthesize"


# ---------- Agents ----------

async def run_agent(payload: dict) -> dict:
    """One agent call: its MCP server is reached through an in-process MCP client."""
    emit = get_stream_writer()
    task, i = payload["task"], payload["i"]
    a = AGENTS[task["agent"]]
    ev = {"type": "tool", "i": i, "agent": a["card"]["name"], "server": a["mcp"].name, "tool": a["tool"], "args": task["args"]}
    emit({**ev, "status": "start"})
    t0 = time.perf_counter()
    try:
        async with Client(a["mcp"]) as c:
            r = await asyncio.wait_for(c.call_tool(a["tool"], task["args"]), TOOL_TIMEOUT)
        data = json.loads(r.content[0].text) if r.content else {}
        if r.is_error or "error" in data:
            raise RuntimeError(data.get("error") if isinstance(data, dict) and "error" in data else "tool failed")
        out = {"ok": True, "data": data}
    except asyncio.TimeoutError:
        out = {"ok": False, "error": f"timed out after {TOOL_TIMEOUT:.0f}s"}
    except Exception as e:
        out = {"ok": False, "error": str(e)[:200] or "tool failed"}
    out["ms"] = round((time.perf_counter() - t0) * 1000)
    emit({**ev, "status": "done", **out})
    return {"results": [{**task, **out}]}


# ---------- Synthesis ----------

WRITE_PROMPT = """Answer the user's request using ONLY the tool results below. Be concise and friendly: 2 to 5 sentences,
plain text, no markdown. Use exact numbers from the data. If a tool failed or returned nothing, say so briefly.
If no tools were called, explain that you can help with weather, news and background on people, places and topics."""


def template_answer(state: State) -> str:
    """Deterministic answer from tool output, used when the writer model is unavailable."""
    lines = []
    for r in state.get("results", []):
        d = r.get("data", {})
        if not r["ok"]:
            lines.append(f"The {r['agent']} agent couldn't help: {r['error']}")
        elif r["agent"] == "weather":
            n = d["now"]
            lines.append(f"{d['place']}: {n['temp_c']}°C and {n['conditions']}, humidity {n['humidity_pct']}%.")
        elif r["agent"] == "news":
            lines.append(f"Recent headlines on {d['topic']}: " + "; ".join(a["title"] for a in d["articles"][:3]) + ".")
        else:
            lines.append(d["summary"].split(". ")[0].rstrip(".") + ".")
    return " ".join(lines) or "I can help with weather, recent news, and background on people, places and topics."


async def synthesize(state: State) -> dict:
    emit = get_stream_writer()
    emit({"type": "node", "node": "synthesize", "status": "start"})
    t0 = time.perf_counter()
    results = sorted(state.get("results", []), key=lambda r: (r["agent"], json.dumps(r["args"])))
    source = "llm"
    try:
        answer = (await chat(WRITER_MODEL, [{"role": "system", "content": WRITE_PROMPT},
                                            {"role": "user", "content": f"Request: {state['query']}\n\nTool results:\n{json.dumps(results)[:9000]}"}],
                             max_tokens=600)).strip()
        if not answer:
            raise ValueError("empty answer")
    except Exception:
        answer, source = template_answer(state), "template"
    emit({"type": "answer", "text": answer, "source": source, "model": WRITER_MODEL if source == "llm" else None,
          "ms": round((time.perf_counter() - t0) * 1000)})
    return {"answer": answer}


def build_graph():
    g = StateGraph(State)
    g.add_node("plan", plan)
    g.add_node("agent", run_agent)
    g.add_node("synthesize", synthesize)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", fan_out, ["agent", "synthesize"])
    g.add_edge("agent", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


graph = build_graph()
