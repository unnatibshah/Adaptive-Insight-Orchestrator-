import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from strands.tools.mcp import MCPClient
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.stdio import stdio_client, StdioServerParameters

from agents.shared.config import get_settings
from agents.shared.llm import extract_weather_params
from agents.shared.mcp import MCPToolInvoker

logger = logging.getLogger("weather-agent")
settings = get_settings()


class WeatherQuery(BaseModel):
    city: str
    country_code: str

class WeatherResponse(BaseModel):
    city: str
    temperature: float
    description: str
    humidity: int


class ExtractRequest(BaseModel):
    query: str


def _create_mcp_client() -> MCPClient:
    config = settings.weather
    if config.mcp_url:
        return MCPClient(lambda: streamablehttp_client(config.mcp_url))

    env_vars = {}
    if config.openweather_api_key:
        env_vars["OPENWEATHER_API_KEY"] = config.openweather_api_key

    return MCPClient(
        lambda: stdio_client(
            StdioServerParameters(
                command="python",
                args=["/app/mcp-servers/weather/server.py"],
                env=env_vars or None,
            )
        )
    )


def _get_invoker() -> MCPToolInvoker:
    invoker = getattr(app.state, "weather_tool_invoker", None)
    if invoker is None:
        raise RuntimeError("Weather MCP invoker not initialised")
    return invoker


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = settings.weather
    app.state.weather_config = config
    invoker = MCPToolInvoker(_create_mcp_client)
    app.state.weather_tool_invoker = invoker
    if not config.mcp_url and not config.openweather_api_key:
        logger.warning("OPENWEATHER_API_KEY not configured; local MCP fallback will fail.")
    try:
        yield
    finally:
        await invoker.stop()


app = FastAPI(title="Weather Agent", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/weather", response_model=WeatherResponse)
async def get_weather(query: WeatherQuery):
    invoker = _get_invoker()
    try:
        payload = await invoker.call_tool("get_weather", arguments=query.model_dump())
    except Exception as exc:
        logger.exception("Weather MCP tool invocation failed")
        raise HTTPException(status_code=502, detail=f"Weather provider unavailable: {exc}") from exc

    if "error" in payload:
        logger.warning("Weather MCP error for %s %s: %s", query.city, query.country_code, payload["error"])
        raise HTTPException(status_code=502, detail=payload["error"])

    try:
        return WeatherResponse(**payload)
    except ValidationError as exc:
        logger.exception("Invalid payload returned by weather MCP tool: %s", payload)
        raise HTTPException(status_code=502, detail="Malformed weather response") from exc

@app.post("/extract")
async def extract_params(request: ExtractRequest):
    try:
        return await extract_weather_params(request.query)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.get("/.well-known/agent.json")
async def get_agent_card():
    with Path("/app/agents/weather/agent.json").open() as f:
        return json.load(f)

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "weather-agent"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
