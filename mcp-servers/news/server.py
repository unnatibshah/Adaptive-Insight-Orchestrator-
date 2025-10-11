#!/usr/bin/env python3
import asyncio
import json

import httpx
import uvicorn
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.fastmcp.server import StreamableHTTPASGIApp
from mcp.types import Tool, TextContent
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from agents.shared.config import get_settings

settings = get_settings()
config = settings.news

server = Server("news-mcp")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_news",
            description="Get latest news articles",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "country": {"type": "string", "description": "Country code", "default": "us"}
                },
                "required": ["query"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "get_news":
        raise ValueError(f"Unknown tool: {name}")
    
    query = arguments.get("query")
    api_key = config.news_api_key
    
    if not api_key:
        return [TextContent(type="text", text=json.dumps({"error": "NewsAPI key not configured"}))]
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://newsapi.org/v2/everything",
                params={"q": query, "apiKey": api_key, "pageSize": 3, "sortBy": "publishedAt"}
            )
            
            if response.status_code == 200:
                data = response.json()
                payload = {
                    "articles": [
                        {
                            "title": article.get("title") or "Untitled article",
                            "description": article.get("description"),
                            "url": article.get("url"),
                        }
                        for article in data.get("articles", [])[:3]
                    ]
                }
                if not payload["articles"]:
                    payload["error"] = "No news found"
                return [TextContent(type="text", text=json.dumps(payload))]
            return [
                TextContent(
                    type="text",
                    text=json.dumps({"error": f"News API returned {response.status_code}"}),
                )
            ]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

async def run_stdio() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

def run_http() -> None:
    session_manager = StreamableHTTPSessionManager(server)
    streamable_app = StreamableHTTPASGIApp(session_manager)

    async def health(_request):
        return JSONResponse({"status": "healthy", "service": config.mcp_service_name})

    app = Starlette(
        routes=[
            Route("/mcp", streamable_app, methods=["GET", "POST", "DELETE", "OPTIONS"]),
            Route("/health", health, methods=["GET"]),
        ],
        lifespan=lambda _app: session_manager.run(),
    )

    host = config.mcp_host
    port = config.mcp_port
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    transport = config.mcp_transport.lower()
    if transport == "http":
        run_http()
    else:
        asyncio.run(run_stdio())
