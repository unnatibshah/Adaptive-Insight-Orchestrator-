from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Fetch an environment variable with fallback."""
    return os.getenv(name, default)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class OrchestratorConfig:
    agentcore_mode: str
    bedrock_model_id: str
    aws_region: str
    bedrock_agent_id: Optional[str]
    bedrock_agent_alias_id: Optional[str]
    weather_agent_url: str
    news_agent_url: str
    phoenix_project_name: str
    phoenix_collector_endpoint: Optional[str]
    phoenix_api_key: Optional[str]
    agentcore_memory_id: Optional[str]
    agentcore_memory_session_id: Optional[str]
    agentcore_memory_actor_id: Optional[str]
    agentcore_memory_retrieval_config: Optional[str]
    http_timeout_seconds: float

    @classmethod
    def load(cls) -> "OrchestratorConfig":
        return cls(
            agentcore_mode=_env("AGENTCORE_MODE", "local").lower(),
            bedrock_model_id=_env("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0"),
            aws_region=_env("AWS_REGION", "us-east-1"),
            bedrock_agent_id=_env("BEDROCK_AGENT_ID"),
            bedrock_agent_alias_id=_env("BEDROCK_AGENT_ALIAS_ID"),
            weather_agent_url=_env("WEATHER_AGENT_URL", "http://weather-agent:8000"),
            news_agent_url=_env("NEWS_AGENT_URL", "http://news-agent:8000"),
            phoenix_project_name=_env("PHOENIX_PROJECT_NAME", "a2a-agentcore"),
            phoenix_collector_endpoint=_env("PHOENIX_COLLECTOR_ENDPOINT"),
            phoenix_api_key=_env("PHOENIX_API_KEY"),
            agentcore_memory_id=_env("AGENTCORE_MEMORY_ID"),
            agentcore_memory_session_id=_env("AGENTCORE_MEMORY_SESSION_ID"),
            agentcore_memory_actor_id=_env("AGENTCORE_MEMORY_ACTOR_ID"),
            agentcore_memory_retrieval_config=_env("AGENTCORE_MEMORY_RETRIEVAL_CONFIG"),
            http_timeout_seconds=_env_float("ORCHESTRATOR_HTTP_TIMEOUT", 30.0),
        )


@dataclass(frozen=True)
class WeatherConfig:
    openweather_api_key: Optional[str]
    mcp_url: Optional[str]
    mcp_host: str
    mcp_port: int
    mcp_service_name: str
    mcp_transport: str

    @classmethod
    def load(cls) -> "WeatherConfig":
        return cls(
            openweather_api_key=_env("OPENWEATHER_API_KEY"),
            mcp_url=_env("WEATHER_MCP_URL"),
            mcp_host=_env("WEATHER_MCP_HOST", "0.0.0.0"),
            mcp_port=_env_int("WEATHER_MCP_PORT", 9000),
            mcp_service_name=_env("WEATHER_MCP_SERVICE_NAME", "weather-mcp"),
            mcp_transport=_env("WEATHER_MCP_TRANSPORT", "http"),
        )


@dataclass(frozen=True)
class NewsConfig:
    news_api_key: Optional[str]
    mcp_url: Optional[str]
    mcp_host: str
    mcp_port: int
    mcp_service_name: str
    mcp_transport: str

    @classmethod
    def load(cls) -> "NewsConfig":
        return cls(
            news_api_key=_env("NEWS_API_KEY"),
            mcp_url=_env("NEWS_MCP_URL"),
            mcp_host=_env("NEWS_MCP_HOST", "0.0.0.0"),
            mcp_port=_env_int("NEWS_MCP_PORT", 9000),
            mcp_service_name=_env("NEWS_MCP_SERVICE_NAME", "news-mcp"),
            mcp_transport=_env("NEWS_MCP_TRANSPORT", "http"),
        )


@dataclass(frozen=True)
class Settings:
    orchestrator: OrchestratorConfig
    weather: WeatherConfig
    news: NewsConfig


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        orchestrator=OrchestratorConfig.load(),
        weather=WeatherConfig.load(),
        news=NewsConfig.load(),
    )
