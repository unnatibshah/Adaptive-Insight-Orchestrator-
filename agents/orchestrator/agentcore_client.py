from __future__ import annotations

import asyncio
import inspect
import json
import logging
import uuid
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urljoin

import boto3
import httpx
from a2a.types import AgentCard as A2AgentCard
from botocore.config import Config as BotoConfig
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from strands import Agent, ToolContext, tool
from strands.models.bedrock import DEFAULT_BEDROCK_MODEL_ID, DEFAULT_BEDROCK_REGION, BedrockModel

from agents.shared.config import get_settings
from agents.shared.llm import llm_enabled
from .tracing import init_tracing

try:
    from .langgraph_runtime import LangGraphRuntime
except ImportError:  # pragma: no cover - optional dependency
    LangGraphRuntime = None  # type: ignore

try:
    from bedrock_agentcore.memory.integrations.strands import (
        AgentCoreMemoryConfig,
        AgentCoreMemorySessionManager,
    )
except ImportError:  # pragma: no cover - optional dependency
    AgentCoreMemoryConfig = None  # type: ignore[assignment]
    AgentCoreMemorySessionManager = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class AgentCapability:
    name: str
    description: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    endpoint: Mapping[str, Any]


@dataclass
class AgentDescriptor:
    logical_id: str
    card: A2AgentCard
    base_url: str
    description: str
    capabilities: list[AgentCapability]
    metadata: Mapping[str, Any]
    communication: Mapping[str, Any]

    @property
    def default_base_url(self) -> str:
        """Return the configured base URL for agent-to-agent communication."""
        return self.base_url.rstrip("/")


class LocalStrandsRuntime:
    """
    Local orchestrator built on the Strands SDK.

    The runtime composes agent card metadata, A2A endpoints, and a Bedrock Claude 4 model
    to provide multi-agent orchestration during development.
    """

    _tool_results: ContextVar[Optional[list[dict[str, Any]]]] = ContextVar("tool_results", default=None)

    def __init__(
        self,
        agent_cards: Mapping[str, AgentDescriptor],
        *,
        tracer: trace.Tracer,
        model_id: str,
        region: str,
        http_timeout: float,
        memory_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._agent_cards = agent_cards
        self._tracer = tracer
        self._model_id = model_id
        self._region = region
        self._http_timeout = http_timeout
        self._memory_config = dict(memory_config or {})
        self._session_manager = self._build_session_manager()
        self._agent: Optional[Agent] = None
        if llm_enabled():
            self._agent = Agent(
                model=BedrockModel(model_id=self._model_id, region_name=self._region),
                tools=self._build_tools(),
                system_prompt=self._build_system_prompt(),
                trace_attributes={
                    "deployment.mode": "local",
                    "bedrock.model_id": self._model_id,
                },
                session_manager=self._session_manager,
            )

    async def run(
        self,
        query: str,
        *,
        history: Optional[list[dict[str, str]]] = None,
        parent_span: Optional[trace.Span] = None,
    ) -> dict[str, Any]:
        """Invoke the Strands agent and capture tool outputs per request."""
        if self._agent is None:
            raise RuntimeError("LLM is required for orchestration. Please configure AWS credentials.")
        
        tool_results_token = self._tool_results.set([])
        try:
            with self._tracer.start_as_current_span(
                "strands.invoke", context=trace.set_span_in_context(parent_span) if parent_span else None
            ) as span:
                span.set_attribute("llm.model_id", self._model_id)
                span.set_attribute("request.query", query)
                
                agent_result = await self._agent.invoke_async(query)
                tool_output = self._tool_results.get() or []
                span.set_attribute("a2a.invoked_agents", ",".join(sorted({entry["agent_id"] for entry in tool_output})))
                span.set_attribute("a2a.tool_count", len(tool_output))
                
                invoked_agents = sorted({entry["agent_id"] for entry in tool_output})
                results_map = {
                    entry["agent_id"]: entry.get("payload") for entry in tool_output if "agent_id" in entry
                }

                return {
                    "mode": "strands-local",
                    "llm_model": self._model_id,
                    "query": query,
                    "final_response": str(agent_result).strip(),
                    "raw_message": agent_result.message,
                    "invoked_agents": invoked_agents,
                    "tool_outputs": tool_output,
                    "results": results_map,
                }
        finally:
            self._tool_results.reset(tool_results_token)

    def _build_system_prompt(self) -> str:
        """Compose a system prompt that wires Claude 4 to the registered tools."""
        lines = [
            "You are the A2A orchestrator for AWS Bedrock AgentCore.",
            "IMPORTANT: You MUST use the registered tools to get real-time data. Never answer from memory or assumptions.",
            "For ANY query about weather, news, or current information, you MUST call the appropriate tool.",
            "Always return a concise summary along with any structured data from tool calls.",
            "\nAvailable agents:",
        ]
        for descriptor in self._agent_cards.values():
            lines.append(f"- {descriptor.logical_id}: {descriptor.description}")
        return "\n".join(lines)

    def _build_tools(self) -> list[Any]:
        tools: list[Any] = []
        for descriptor in self._agent_cards.values():
            for capability in descriptor.capabilities:
                tools.append(self._create_tool(descriptor, capability))
        return tools

    def _create_tool(self, descriptor: AgentDescriptor, capability: AgentCapability) -> Any:
        base_url = descriptor.default_base_url
        endpoint = capability.endpoint or {}
        method = endpoint.get("method", "POST").upper()
        path = endpoint.get("path", "/")
        url = urljoin(f"{base_url}/", path.lstrip("/"))
        timeout = self._http_timeout
        tracer = self._tracer
        tool_results_var = self._tool_results

        async def tool_impl(tool_context: ToolContext | None = None, **payload: Any) -> str:
            request_payload = payload
            with tracer.start_as_current_span(f"tool.{descriptor.logical_id}") as span:
                span.set_attribute("a2a.agent_id", descriptor.logical_id)
                span.set_attribute("a2a.capability", capability.name)
                span.set_attribute("http.method", method)
                span.set_attribute("http.url", url)
                tool_use = getattr(tool_context, "tool_use", None)
                if isinstance(tool_use, dict):
                    tool_use_id = tool_use.get("toolUseId", "")
                else:
                    tool_use_id = getattr(tool_use, "toolUseId", "")
                span.set_attribute("a2a.tool_use_id", tool_use_id)

                data: Any = {}
                status_code: Optional[int] = None
                try:
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        response = await client.request(method, url, json=request_payload)
                        response.raise_for_status()
                        data = response.json()
                        status_code = response.status_code
                except Exception as exc:  # pragma: no cover - network failure path
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise

            collected = tool_results_var.get()
            if collected is None:
                collected = []
                tool_results_var.set(collected)

            collected.append(
                {
                    "agent_id": descriptor.logical_id,
                    "capability": capability.name,
                    "endpoint": url,
                    "http_status": status_code,
                    "payload": data,
                }
            )

            summary = json.dumps(data, ensure_ascii=False) if isinstance(data, (dict, list)) else str(data)
            return f"{descriptor.logical_id} responded with: {summary}"

        self._decorate_tool_signature(tool_impl, capability)
        return tool(
            name=capability.name,
            description=capability.description,
            inputSchema=capability.input_schema,
            context="tool_context",
        )(tool_impl)

    @staticmethod
    def _decorate_tool_signature(func: Any, capability: AgentCapability) -> None:
        """Attach a dynamic signature so Strands exposes accurate parameter schemas."""
        properties = capability.input_schema.get("properties", {})
        required = set(capability.input_schema.get("required", []))

        parameters = [
            inspect.Parameter(
                "tool_context",
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Optional[ToolContext],
            )
        ]

        for name, schema in properties.items():
            annotation = _schema_type_to_annotation(schema.get("type"))
            default: Any = inspect._empty
            if name not in required:
                default = schema.get("default", None)
            parameters.append(
                inspect.Parameter(
                    name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=annotation,
                )
            )

        func.__signature__ = inspect.Signature(parameters)  # type: ignore[attr-defined]
        func.__annotations__ = {param.name: param.annotation for param in parameters if param.annotation}
        func.__annotations__["return"] = str

    def _build_session_manager(self):
        if not AgentCoreMemoryConfig or not AgentCoreMemorySessionManager:
            return None

        memory_id = self._memory_config.get("memory_id")
        session_id = self._memory_config.get("session_id")
        actor_id = self._memory_config.get("actor_id")
        if not (memory_id and session_id and actor_id):
            return None

        retrieval_config = self._memory_config.get("retrieval_config")
        if isinstance(retrieval_config, str):
            try:
                retrieval_config = json.loads(retrieval_config)
            except json.JSONDecodeError:
                retrieval_config = None

        try:
            config = AgentCoreMemoryConfig(
                memory_id=memory_id,
                session_id=session_id,
                actor_id=actor_id,
                retrieval_config=retrieval_config,
            )
            return AgentCoreMemorySessionManager(
                agentcore_memory_config=config,
                region_name=self._region,
            )
        except Exception:  # pragma: no cover - best effort integration
            return None


class AgentCoreRuntime:
    """Production runtime that defers orchestration to AWS Bedrock AgentCore."""

    def __init__(
        self,
        *,
        tracer: trace.Tracer,
        agent_id: str,
        agent_alias_id: str,
        region: str,
        model_id: str,
        client: Optional[Any] = None,
    ) -> None:
        self._tracer = tracer
        self._agent_id = agent_id
        self._agent_alias_id = agent_alias_id
        self._model_id = model_id
        self._client = client or boto3.client(
            "bedrock-agent-runtime",
            region_name=region,
            config=BotoConfig(user_agent_extra="a2a-agentcore-sample"),
        )

    async def run(
        self,
        query: str,
        *,
        history: Optional[list[dict[str, str]]] = None,
        parent_span: Optional[trace.Span] = None,
    ) -> dict[str, Any]:
        """Invoke the Bedrock AgentCore runtime and gather the streamed response."""
        parent_context = trace.set_span_in_context(parent_span) if parent_span else None

        def _invoke() -> dict[str, Any]:
            token = otel_context.attach(parent_context) if parent_context else None
            try:
                with self._tracer.start_as_current_span("bedrock.invoke_agent") as span:
                    span.set_attribute("llm.model_id", self._model_id)
                    span.set_attribute("aws.agent_id", self._agent_id)
                    session_id = str(uuid.uuid4())
                    response = self._client.invoke_agent(
                        agentId=self._agent_id,
                        agentAliasId=self._agent_alias_id,
                        sessionId=session_id,
                        inputText=query,
                        enableTrace=True,
                    )

                    completion_chunks: list[str] = []
                    citations: list[Any] = []
                    trace_events: list[Any] = []

                    for event in response.get("completion", []):
                        chunk = event.get("chunk")
                        if chunk and "bytes" in chunk:
                            completion_chunks.append(chunk["bytes"].decode("utf-8"))
                        if event.get("citations"):
                            citations.extend(event["citations"])
                        if event.get("trace"):
                            trace_events.append(event["trace"])

                    span.set_attribute("response.citation_count", len(citations))
                    span.set_attribute("response.trace_events", len(trace_events))

                    return {
                        "mode": "agentcore-runtime",
                        "llm_model": self._model_id,
                        "session_id": session_id,
                        "final_response": "".join(completion_chunks).strip(),
                        "citations": citations,
                        "trace_events": trace_events,
                    }
            finally:
                if token:
                    otel_context.detach(token)

        return await asyncio.to_thread(_invoke)


class AgentCoreOrchestrator:
    """Facade that selects the correct runtime (local Strands vs. AgentCore runtime)."""

    def __init__(self) -> None:
        self._tracer = init_tracing()
        self._config = settings.orchestrator
        self._mode = self._config.agentcore_mode
        self._model_id = self._config.bedrock_model_id or DEFAULT_BEDROCK_MODEL_ID
        self._region = self._config.aws_region or DEFAULT_BEDROCK_REGION
        self._http_timeout = self._config.http_timeout_seconds
        self._cards = self._load_agent_cards()

        if self._mode == "runtime":
            agent_id = self._config.bedrock_agent_id
            agent_alias_id = self._config.bedrock_agent_alias_id
            if not agent_id or not agent_alias_id:
                raise RuntimeError("BEDROCK_AGENT_ID and BEDROCK_AGENT_ALIAS_ID must be set for runtime mode.")
            self._runtime: Any = AgentCoreRuntime(
                tracer=self._tracer,
                agent_id=agent_id,
                agent_alias_id=agent_alias_id,
                region=self._region,
                model_id=self._model_id,
            )
        elif self._mode == "langgraph":
            if LangGraphRuntime is None:
                raise RuntimeError("LangGraph runtime requested but langgraph is not installed.")
            self._runtime = LangGraphRuntime(
                weather_base_url=self._config.weather_agent_url,
                news_base_url=self._config.news_agent_url,
                http_timeout=self._http_timeout,
            )
        else:
            if not llm_enabled():
                raise RuntimeError(
                    "LLM credentials not detected. Configure AWS credentials to enable local orchestration."
                )
            self._runtime = LocalStrandsRuntime(
                self._cards,
                tracer=self._tracer,
                model_id=self._model_id,
                region=self._region,
                http_timeout=self._http_timeout,
                memory_config={
                    "memory_id": self._config.agentcore_memory_id,
                    "session_id": self._config.agentcore_memory_session_id,
                    "actor_id": self._config.agentcore_memory_actor_id,
                    "retrieval_config": self._config.agentcore_memory_retrieval_config,
                },
            )

    async def orchestrate(self, query: str, history: Optional[list[dict[str, str]]] = None) -> Dict[str, Any]:
        with self._tracer.start_as_current_span("a2a.orchestrate") as span:
            span.set_attribute("orchestrator.mode", self._mode)
            span.set_attribute("request.query", query)
            result = await self._runtime.run(query, history=history, parent_span=span)
            span.set_attribute("response.mode", result.get("mode", "unknown"))
            span.set_attribute("response.invoked_agents", ",".join(result.get("invoked_agents", [])))
            return result

    def _load_agent_cards(self) -> dict[str, AgentDescriptor]:
        import time
        import httpx
        
        agent_urls = {
            "weather-agent": self._config.weather_agent_url,
            "news-agent": self._config.news_agent_url,
        }
        
        cards: dict[str, AgentDescriptor] = {}
        max_retries = 10
        
        for agent_id, base_url in agent_urls.items():
            for attempt in range(max_retries):
                try:
                    response = httpx.get(
                        f"{base_url}/.well-known/agent.json",
                        timeout=self._http_timeout,
                    )
                    if response.status_code == 200:
                        cards[agent_id] = self._parse_agent_card(response.json(), agent_id, base_url)
                        logger.info("Discovered agent card for %s at %s", agent_id, base_url)
                        break
                    logger.warning(
                        "Agent card fetch failed for %s (status %s, attempt %s/%s)",
                        agent_id,
                        response.status_code,
                        attempt + 1,
                        max_retries,
                    )
                except Exception as exc:
                    logger.warning(
                        "Error fetching agent card for %s at %s (attempt %s/%s): %s",
                        agent_id,
                        base_url,
                        attempt + 1,
                        max_retries,
                        exc,
                    )
                if attempt < max_retries - 1:
                    time.sleep(3)
        
        if not cards:
            raise RuntimeError("No agent cards discovered. Ensure agents are running and accessible.")
        return cards
    
    def _parse_agent_card(self, raw: dict, agent_id: str, base_url: str) -> AgentDescriptor:
        validated_card = A2AgentCard.model_validate(raw)
        capability_specs = raw.get("capability_specs", [])
        communication = raw.get("a2a_communication", {})
        endpoint_map = {ep.get("capability"): ep for ep in communication.get("endpoints", [])}
        
        capabilities = [
            AgentCapability(
                name=cap["name"],
                description=cap.get("description", ""),
                input_schema=cap.get("input_schema", {}),
                output_schema=cap.get("output_schema", {}),
                endpoint=endpoint_map.get(cap["name"], {}),
            )
            for cap in capability_specs if cap.get("name")
        ]

        return AgentDescriptor(
            logical_id=raw.get("agent_id", validated_card.name),
            card=validated_card,
            base_url=base_url or validated_card.url or f"http://{agent_id}:8000",
            description=validated_card.description,
            capabilities=capabilities,
            metadata=raw.get("metadata", {}),
            communication=communication,
        )


def _schema_type_to_annotation(schema_type: Optional[str]) -> Any:
    mapping = {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    return mapping.get(schema_type, Any)
