from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional

from strands.tools.mcp import MCPClient

logger = logging.getLogger(__name__)


class MCPToolInvoker:
    """Manages a single MCP client instance that can be shared safely across requests."""

    def __init__(self, client_factory: Callable[[], MCPClient]) -> None:
        self._client_factory = client_factory
        self._client: Optional[MCPClient] = None
        self._client_lock = asyncio.Lock()
        self._call_lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._client_lock:
            if self._client is None:
                client = self._client_factory()
                client.start()
                self._client = client
                logger.info("Started MCP client %s", getattr(client, "name", "unknown"))

    async def stop(self) -> None:
        async with self._client_lock:
            if self._client is not None:
                try:
                    self._client.stop(None, None, None)
                    logger.info("Stopped MCP client %s", getattr(self._client, "name", "unknown"))
                finally:
                    self._client = None

    async def call_tool(self, name: str, *, arguments: Dict[str, Any], tool_use_id: Optional[str] = None) -> Dict[str, Any]:
        await self.start()
        async with self._call_lock:
            if self._client is None:  # Defensive: start() could fail
                raise RuntimeError("MCP client is not available")
            response = await self._client.call_tool_async(
                tool_use_id=tool_use_id or str(uuid.uuid4()),
                name=name,
                arguments=arguments,
            )
        return _tool_response_to_dict(response)


def _tool_response_to_dict(response: Any) -> Dict[str, Any]:
    """
    Convert various response shapes from the MCP client into a structured dictionary.
    Expected format: {"content": [{"text": "..."}]}
    """
    content_items: List[Any]
    if hasattr(response, "content"):
        content_items = list(getattr(response, "content") or [])
    elif isinstance(response, dict):
        content_items = list(response.get("content") or [])
    else:
        raise ValueError(f"Unexpected MCP response type: {type(response)!r}")

    if not content_items:
        raise ValueError("MCP response did not include any content")

    first = content_items[0]
    text: Optional[str] = None
    if hasattr(first, "text"):
        text = first.text  # type: ignore[attr-defined]
    elif isinstance(first, dict):
        text = first.get("text")

    if text is None:
        raise ValueError("MCP content did not provide text field")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("MCP tool returned non-JSON payload, wrapping in {'raw_text': ...}")
        return {"raw_text": text}
