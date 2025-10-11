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
from agents.shared.llm import extract_news_topic
from agents.shared.mcp import MCPToolInvoker

logger = logging.getLogger("news-agent")
settings = get_settings()


class NewsQuery(BaseModel):
    query: str

class NewsArticle(BaseModel):
    title: str
    description: str | None = None
    url: str | None = None


class NewsResponse(BaseModel):
    articles: list[NewsArticle]


class ExtractRequest(BaseModel):
    query: str


def _create_mcp_client() -> MCPClient:
    config = settings.news
    if config.mcp_url:
        return MCPClient(lambda: streamablehttp_client(config.mcp_url))

    env_vars = {}
    if config.news_api_key:
        env_vars["NEWS_API_KEY"] = config.news_api_key

    return MCPClient(
        lambda: stdio_client(
            StdioServerParameters(
                command="python",
                args=["/app/mcp-servers/news/server.py"],
                env=env_vars or None,
            )
        )
    )


def _get_invoker() -> MCPToolInvoker:
    invoker = getattr(app.state, "news_tool_invoker", None)
    if invoker is None:
        raise RuntimeError("News MCP invoker not initialised")
    return invoker


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = settings.news
    app.state.news_config = config
    invoker = MCPToolInvoker(_create_mcp_client)
    app.state.news_tool_invoker = invoker
    if not config.mcp_url and not config.news_api_key:
        logger.warning("NEWS_API_KEY not configured; local MCP fallback will fail.")
    try:
        yield
    finally:
        await invoker.stop()


app = FastAPI(title="News Agent", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/news", response_model=NewsResponse)
async def get_news(query: NewsQuery):
    invoker = _get_invoker()
    try:
        payload = await invoker.call_tool("get_news", arguments=query.model_dump())
    except Exception as exc:
        logger.exception("News MCP tool invocation failed")
        raise HTTPException(status_code=502, detail=f"News provider unavailable: {exc}") from exc

    if "error" in payload:
        logger.warning("News MCP error for query %s: %s", query.query, payload["error"])
        raise HTTPException(status_code=502, detail=payload["error"])

    try:
        return NewsResponse(**payload)
    except ValidationError as exc:
        logger.exception("Invalid payload returned by news MCP tool: %s", payload)
        raise HTTPException(status_code=502, detail="Malformed news response") from exc

@app.post("/extract")
async def extract_params(request: ExtractRequest):
    try:
        return await extract_news_topic(request.query)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.get("/.well-known/agent.json")
async def get_agent_card():
    with Path("/app/agents/news/agent.json").open() as f:
        return json.load(f)

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "news-agent"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
