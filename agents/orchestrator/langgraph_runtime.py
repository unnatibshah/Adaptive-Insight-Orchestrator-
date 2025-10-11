from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urljoin

import httpx
from langgraph.graph import StateGraph, START, END


@dataclass
class GraphState:
    query: str
    history: List[Dict[str, str]]
    needs_weather: bool = False
    needs_news: bool = False
    invoked_agents: List[str] = field(default_factory=list)
    tool_outputs: List[Dict[str, Any]] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    summary_parts: List[str] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    final_response: str = ""


@dataclass(frozen=True)
class AgentWorkflowConfig:
    state_flag: str
    agent_id: str
    extract_url: str
    action_url: str
    action_capability: str
    keywords: tuple[str, ...]
    success_summary: Callable[[Dict[str, Any]], str]
    failure_summary: Callable[[Dict[str, Any]], str]
    extract_payload_builder: Callable[[str], Dict[str, Any]]
    action_payload_builder: Callable[[Dict[str, Any]], Dict[str, Any]]
    extract_error_fallback: str


class LangGraphRuntime:
    """Rule-based orchestration workflow implemented with LangGraph."""

    _WEATHER_KEYWORDS = ("weather", "temperature", "forecast", "climate", "humidity", "rain", "snow")
    _NEWS_KEYWORDS = ("news", "headline", "headlines", "story", "stories", "article", "articles", "update")

    def __init__(
        self,
        weather_base_url: str,
        news_base_url: str,
        *,
        http_timeout: float = 15.0,
    ) -> None:
        self._weather_base = weather_base_url.rstrip("/")
        self._news_base = news_base_url.rstrip("/")
        self._weather_extract = urljoin(f"{self._weather_base}/", "extract")
        self._weather_endpoint = urljoin(f"{self._weather_base}/", "weather")
        self._news_extract = urljoin(f"{self._news_base}/", "extract")
        self._news_endpoint = urljoin(f"{self._news_base}/", "news")
        self._timeout = http_timeout
        self._graph = self._build_graph()
        self._agent_configs: Dict[str, AgentWorkflowConfig] = {
            "weather": AgentWorkflowConfig(
                state_flag="needs_weather",
                agent_id="weather-agent",
                extract_url=self._weather_extract,
                action_url=self._weather_endpoint,
                action_capability="get_weather",
                keywords=self._WEATHER_KEYWORDS,
                success_summary=self._format_weather_summary,
                failure_summary=self._build_failure_summary("Weather", "Weather agent error."),
                extract_payload_builder=self._default_extract_payload,
                action_payload_builder=self._passthrough_action_payload,
                extract_error_fallback="Failed to extract weather parameters.",
            ),
            "news": AgentWorkflowConfig(
                state_flag="needs_news",
                agent_id="news-agent",
                extract_url=self._news_extract,
                action_url=self._news_endpoint,
                action_capability="get_news",
                keywords=self._NEWS_KEYWORDS,
                success_summary=self._format_news_summary,
                failure_summary=self._build_failure_summary("News", "News agent error."),
                extract_payload_builder=self._default_extract_payload,
                action_payload_builder=self._passthrough_action_payload,
                extract_error_fallback="Failed to extract news topic.",
            ),
        }

    def _build_graph(self):
        workflow = StateGraph(GraphState)
        workflow.add_node("analyze", self._analyze)
        workflow.add_node("get_weather", self._call_weather)
        workflow.add_node("get_news", self._call_news)
        workflow.add_node("finalize", self._finalize)

        workflow.set_entry_point("analyze")

        workflow.add_conditional_edges(
            "analyze",
            self._route_after_analyze,
            {
                "weather": "get_weather",
                "news": "get_news",
                "final": "finalize",
            },
        )

        workflow.add_conditional_edges(
            "get_weather",
            self._route_after_weather,
            {
                "weather": "get_weather",
                "news": "get_news",
                "final": "finalize",
            },
        )

        workflow.add_conditional_edges(
            "get_news",
            self._route_after_news,
            {
                "weather": "get_weather",
                "news": "get_news",
                "final": "finalize",
            },
        )

        workflow.add_edge("finalize", END)

        return workflow.compile()

    async def run(
        self,
        query: str,
        *,
        history: Optional[List[Dict[str, str]]] = None,
        parent_span: Optional[Any] = None,
    ) -> Dict[str, Any]:
        initial_state = GraphState(query=query, history=history or [])
        final_state = await self._graph.ainvoke(initial_state.__dict__)

        final_response = final_state.get("final_response") or self._default_summary(final_state)

        return {
            "mode": "langgraph",
            "query": query,
            "final_response": final_response.strip(),
            "invoked_agents": final_state.get("invoked_agents", []),
            "tool_outputs": final_state.get("tool_outputs", []),
            "results": final_state.get("results", {}),
            "warnings": final_state.get("error_messages", []),
        }

    # ---------- Graph Nodes ----------

    def _analyze(self, state: Dict[str, Any]) -> Dict[str, Any]:
        history_text = " ".join(
            [msg["content"] for msg in state.get("history", []) if msg.get("role") == "user"]
        )
        text = f"{history_text} {state['query']}".lower()

        detected_flags: Dict[str, bool] = {}
        for config in self._agent_configs.values():
            detected_flags[config.state_flag] = any(keyword in text for keyword in config.keywords)

        if not any(detected_flags.values()):
            detected_flags = {flag: True for flag in detected_flags}

        return detected_flags

    def _route_after_analyze(self, state: Dict[str, Any]) -> str:
        if state.get("needs_weather"):
            return "weather"
        if state.get("needs_news"):
            return "news"
        return "final"

    def _route_after_weather(self, state: Dict[str, Any]) -> str:
        if state.get("needs_news"):
            return "news"
        if state.get("needs_weather"):
            return "weather"
        return "final"

    def _route_after_news(self, state: Dict[str, Any]) -> str:
        if state.get("needs_weather"):
            return "weather"
        if state.get("needs_news"):
            return "news"
        return "final"

    async def _call_weather(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await self._invoke_agent(state, self._agent_configs["weather"])

    async def _call_news(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await self._invoke_agent(state, self._agent_configs["news"])

    def _finalize(self, state: Dict[str, Any]) -> Dict[str, Any]:
        summary = self._default_summary(state)
        return {"final_response": summary}

    async def _invoke_agent(self, state: Dict[str, Any], config: AgentWorkflowConfig) -> Dict[str, Any]:
        query = state["query"]
        extract_payload = config.extract_payload_builder(query)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            extract_resp = await client.post(config.extract_url, json=extract_payload)
            extract_data = extract_resp.json() if extract_resp.content else {}

            tool_outputs = list(state.get("tool_outputs", []))
            tool_outputs.append(
                {
                    "agent_id": config.agent_id,
                    "capability": "extract",
                    "endpoint": config.extract_url,
                    "http_status": extract_resp.status_code,
                    "payload": extract_data,
                }
            )

            if extract_resp.status_code != 200 or extract_data.get("error"):
                error_messages = list(state.get("error_messages", []))
                error_messages.append(extract_data.get("error") or config.extract_error_fallback)
                return {
                    config.state_flag: False,
                    "tool_outputs": tool_outputs,
                    "error_messages": error_messages,
                }

            action_payload = config.action_payload_builder(extract_data)
            action_resp = await client.post(config.action_url, json=action_payload)
            action_data = action_resp.json() if action_resp.content else {}
            tool_outputs.append(
                {
                    "agent_id": config.agent_id,
                    "capability": config.action_capability,
                    "endpoint": config.action_url,
                    "http_status": action_resp.status_code,
                    "payload": action_data,
                }
            )

        summary_parts = list(state.get("summary_parts", []))
        results = dict(state.get("results", {}))
        invoked = list(state.get("invoked_agents", []))

        if action_resp.status_code == 200 and not action_data.get("error"):
            if config.agent_id not in invoked:
                invoked.append(config.agent_id)
            results[config.agent_id] = action_data
            summary_parts.append(config.success_summary(action_data))
        else:
            summary_parts.append(config.failure_summary(action_data))

        return {
            config.state_flag: False,
            "invoked_agents": invoked,
            "tool_outputs": tool_outputs,
            "results": results,
            "summary_parts": summary_parts,
        }

    @staticmethod
    def _default_summary(state: Dict[str, Any]) -> str:
        parts = state.get("summary_parts") or []
        errors = state.get("error_messages") or []
        text = "\n".join(parts)
        if errors:
            text = f"{text}\n" + "\n".join(f"⚠️ {msg}" for msg in errors if msg)
        return text.strip() or "No results available."

    @staticmethod
    def _default_extract_payload(query: str) -> Dict[str, Any]:
        return {"query": query}

    @staticmethod
    def _passthrough_action_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        return payload

    @staticmethod
    def _format_weather_summary(payload: Dict[str, Any]) -> str:
        return "Weather in {city}: {temp}°C with {desc}".format(
            city=payload.get("city", "unknown city"),
            temp=payload.get("temperature", "unknown"),
            desc=payload.get("description", "no description"),
        )

    @staticmethod
    def _format_news_summary(payload: Dict[str, Any]) -> str:
        articles = payload.get("articles") or []
        if not articles:
            return "News agent returned no articles."
        formatted = ", ".join(article.get("title", "Untitled article") for article in articles[:3])
        return f"Top headlines: {formatted}"

    @staticmethod
    def _build_failure_summary(agent_label: str, default_message: str) -> Callable[[Dict[str, Any]], str]:
        def formatter(payload: Dict[str, Any]) -> str:
            message = payload.get("error") if isinstance(payload, dict) else None
            if not message:
                message = default_message
            return f"{agent_label} agent reported: {message}"

        return formatter
