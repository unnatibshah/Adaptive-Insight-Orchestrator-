"""Offline tests: fake MCP servers stand in for the real APIs, and the model is stubbed or absent."""
import json

import pytest
from fastapi.testclient import TestClient
from mcp.server.mcpserver import MCPServer

from app import graph as g
from app import main

SCHEMAS = {"weather": {"city": "string"}, "news": {"topic": "string"}, "knowledge": {"subject": "string"}}


def test_validate_plan_keeps_only_well_formed_tasks():
    raw = {"tasks": [
        {"agent": "weather", "args": {"city": "Paris"}},
        {"agent": "weather", "args": {"city": "Paris"}},           # duplicate
        {"agent": "stocks", "args": {"ticker": "AAPL"}},           # unknown agent
        {"agent": "news", "args": {"query": "AI"}},                # wrong arg name
        {"agent": "knowledge", "args": {"subject": "  "}},         # empty arg
        {"agent": "news", "args": {"topic": "AI", "extra": "x"}},  # extra arg dropped
    ]}
    assert g.validate_plan(raw, SCHEMAS) == [{"agent": "weather", "args": {"city": "Paris"}},
                                             {"agent": "news", "args": {"topic": "AI"}}]
    assert g.validate_plan("not json", SCHEMAS) == []


def test_fallback_plan_routes_by_keywords():
    assert g.fallback_plan("weather in Tokyo") == [{"agent": "weather", "args": {"city": "Tokyo"}}]
    assert g.fallback_plan("latest news about Japan") == [{"agent": "news", "args": {"topic": "Japan"}}]
    assert g.fallback_plan("Who was Ada Lovelace?")[0]["agent"] == "knowledge"


@pytest.fixture
def fake_agents(monkeypatch):
    weather, news, knowledge = MCPServer("weather-mcp"), MCPServer("news-mcp"), MCPServer("knowledge-mcp")

    @weather.tool()
    async def get_weather(city: str) -> dict:
        return {"place": city, "now": {"temp_c": 20, "conditions": "clear sky", "humidity_pct": 50, "wind_kmh": 5}, "forecast": []}

    @news.tool()
    async def get_news(topic: str) -> dict:
        raise RuntimeError("upstream down")

    @knowledge.tool()
    async def get_background(subject: str) -> dict:
        return {"title": subject, "summary": f"{subject} is a test subject. More text.", "url": "https://example.org"}

    for aid, srv in (("weather", weather), ("news", news), ("knowledge", knowledge)):
        monkeypatch.setitem(g.AGENTS[aid], "mcp", srv)
    monkeypatch.setattr(g.tool_schemas, "cache", SCHEMAS, raising=False)


async def run(query):
    return [ev async for ev in g.graph.astream({"query": query}, stream_mode="custom")]


async def test_fans_out_in_parallel_and_survives_a_failed_agent(fake_agents, monkeypatch):
    async def fake_chat(model, messages, **kw):
        if model == g.PLANNER_MODEL:
            return json.dumps({"tasks": [{"agent": "weather", "args": {"city": "Oslo"}},
                                         {"agent": "news", "args": {"topic": "Oslo"}}], "reason": "both"})
        raise RuntimeError("writer down")  # forces the template answer
    monkeypatch.setattr(g, "chat", fake_chat)

    evs = await run("weather and news in Oslo")
    plan = next(e for e in evs if e["type"] == "plan")
    assert plan["source"] == "llm" and len(plan["tasks"]) == 2
    starts = [i for i, e in enumerate(evs) if e["type"] == "tool" and e["status"] == "start"]
    dones = [i for i, e in enumerate(evs) if e["type"] == "tool" and e["status"] == "done"]
    assert len(starts) == 2 and max(starts) < min(dones)  # both calls started before either finished
    answer = next(e for e in evs if e["type"] == "answer")
    assert answer["source"] == "template"
    assert "Oslo: 20°C and clear sky" in answer["text"] and "news agent couldn't help" in answer["text"]


async def test_bad_planner_output_falls_back_to_keywords(fake_agents, monkeypatch):
    async def fake_chat(model, messages, **kw):
        if model == g.PLANNER_MODEL:
            return '{"tasks": [{"agent": "rm -rf", "args": {}}]}'
        return "Ada was a mathematician."
    monkeypatch.setattr(g, "chat", fake_chat)

    evs = await run("Who was Ada Lovelace?")
    plan = next(e for e in evs if e["type"] == "plan")
    assert plan["source"] == "fallback" and plan["tasks"][0]["agent"] == "knowledge"
    assert next(e for e in evs if e["type"] == "answer")["text"] == "Ada was a mathematician."


def test_api_validates_input_and_rate_limits(fake_agents, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    main._hits.clear()
    c = TestClient(main.app)
    assert c.post("/api/run", json={"query": "x"}).status_code == 422
    assert c.post("/api/run", json={"query": "y" * 301}).status_code == 422
    r = c.post("/api/run", json={"query": "Who was Ada Lovelace?"}, headers={"x-forwarded-for": "1.2.3.4"})
    assert r.status_code == 200 and '"type": "done"' in r.text
    for _ in range(main.PER_MINUTE - 1):
        c.post("/api/run", json={"query": "hi there"}, headers={"x-forwarded-for": "1.2.3.4"})
    assert c.post("/api/run", json={"query": "hi there"}, headers={"x-forwarded-for": "1.2.3.4"}).status_code == 429
    # a spoofed leading entry doesn't reset the limit: the proxy-appended last IP is what counts
    assert c.post("/api/run", json={"query": "hi there"}, headers={"x-forwarded-for": "9.9.9.9, 1.2.3.4"}).status_code == 429
    assert c.get("/agents/weather/.well-known/agent-card.json").json()["name"] == "weather-agent"
