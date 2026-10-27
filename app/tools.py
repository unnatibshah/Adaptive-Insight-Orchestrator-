"""MCP servers, one per agent, backed by keyless public APIs.

Each server is a real MCP server; the agents reach it through an in-process MCP client
(mcp.Client(server)), so the protocol is the same as over HTTP, minus the network hop.
"""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from typing import Any

import httpx
from mcp.server.mcpserver import MCPServer

UA = {"User-Agent": "adaptive-insight-orchestrator/2.0 (https://github.com/unnatibshah/Adaptive-Insight-Orchestrator-)"}
TIMEOUT = httpx.Timeout(8.0)

# ponytail: in-process TTL cache, per instance; move to Redis if this ever runs as several replicas.
_cache: dict[tuple, tuple[float, Any]] = {}


async def _get(url: str, params: dict, ttl: float, as_json: bool = True) -> Any:
    key = (url, tuple(sorted(params.items())))
    hit = _cache.get(key)
    if hit and hit[0] > time.monotonic():
        return hit[1]
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=UA) as client:
        r = await client.get(url, params=params)
    host = httpx.URL(url).host
    if r.status_code == 429:
        raise RuntimeError(f"{host} is rate limiting requests right now")
    r.raise_for_status()
    try:
        data = r.json() if as_json else r.text
    except ValueError:
        raise RuntimeError(f"{host} returned an unexpected response ({r.headers.get('content-type', 'unknown type')})") from None
    _cache[key] = (time.monotonic() + ttl, data)
    return data


WMO = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
       51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
       66: "freezing rain", 67: "heavy freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
       80: "light showers", 81: "showers", 82: "violent showers", 85: "snow showers", 86: "heavy snow showers",
       95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm with hail"}

weather_mcp = MCPServer("weather-mcp")


@weather_mcp.tool(description="Current conditions and a 3-day forecast for a city (Open-Meteo).")
async def get_weather(city: str) -> dict:
    geo = await _get("https://geocoding-api.open-meteo.com/v1/search", {"name": city, "count": 1}, ttl=86400)
    if not geo.get("results"):
        return {"error": f"Couldn't find a place called {city!r}."}
    place = geo["results"][0]
    wx = await _get("https://api.open-meteo.com/v1/forecast", {
        "latitude": place["latitude"], "longitude": place["longitude"], "timezone": "auto", "forecast_days": 3,
        "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,weather_code,precipitation_probability_max",
    }, ttl=600)
    cur, daily = wx["current"], wx["daily"]
    return {
        "place": ", ".join(p for p in (place["name"], place.get("admin1"), place.get("country")) if p),
        "now": {"temp_c": cur["temperature_2m"], "humidity_pct": cur["relative_humidity_2m"],
                "wind_kmh": cur["wind_speed_10m"], "conditions": WMO.get(cur["weather_code"], "unknown")},
        "forecast": [{"date": d, "min_c": lo, "max_c": hi, "rain_chance_pct": p, "conditions": WMO.get(c, "unknown")}
                     for d, lo, hi, p, c in zip(daily["time"], daily["temperature_2m_min"], daily["temperature_2m_max"],
                                                daily["precipitation_probability_max"], daily["weather_code"])],
    }


news_mcp = MCPServer("news-mcp")


@news_mcp.tool(description="Recent news headlines about a topic from the last 3 days (Google News).")
async def get_news(topic: str) -> dict:
    xml = await _get("https://news.google.com/rss/search",
                     {"q": f"{topic} when:3d", "hl": "en-US", "gl": "US", "ceid": "US:en"}, ttl=900, as_json=False)
    arts = []
    for item in ET.fromstring(xml).iter("item"):
        src = item.findtext("source") or ""
        title = item.findtext("title") or ""
        arts.append({"title": title.removesuffix(f" - {src}"), "source": src,
                     "url": item.findtext("link"), "published": (item.findtext("pubDate") or "")[:16]})
        if len(arts) == 6:
            break
    return {"topic": topic, "articles": arts} if arts else {"error": f"No recent news found for {topic!r}."}


knowledge_mcp = MCPServer("knowledge-mcp")


@knowledge_mcp.tool(description="Short encyclopedia background on a person, place, organization or concept (Wikipedia).")
async def get_background(subject: str) -> dict:
    hits = await _get("https://en.wikipedia.org/w/api.php", {
        "action": "opensearch", "search": subject, "limit": 1, "namespace": 0, "format": "json"}, ttl=86400)
    if not hits[1]:
        return {"error": f"No encyclopedia entry found for {subject!r}."}
    title = hits[1][0]
    s = await _get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}", {}, ttl=86400)
    return {"title": s["title"], "summary": s.get("extract", ""), "url": s["content_urls"]["desktop"]["page"]}
