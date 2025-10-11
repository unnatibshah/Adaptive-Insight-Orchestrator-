import logging
import os
from functools import lru_cache
from typing import Dict, Optional, Tuple

try:
    from strands import Agent as StrandsAgent
    from strands.models.bedrock import (
        BedrockModel,
        DEFAULT_BEDROCK_MODEL_ID,
        DEFAULT_BEDROCK_REGION,
    )
except Exception:  # pragma: no cover - dependency may be missing
    StrandsAgent = None  # type: ignore

from agents.shared.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def llm_enabled() -> bool:
    """Returns True if AWS credentials are available."""
    return bool(os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))


@lru_cache(maxsize=1)
def _bedrock_agent() -> Optional["StrandsAgent"]:
    if StrandsAgent is None or not llm_enabled():
        return None
    orchestrator = settings.orchestrator
    return StrandsAgent(
        model=BedrockModel(
            model_id=orchestrator.bedrock_model_id or DEFAULT_BEDROCK_MODEL_ID,
            region_name=orchestrator.aws_region or DEFAULT_BEDROCK_REGION,
        )
    )


async def summarize(prompt: str) -> str:
    """Generate a short natural language summary using Bedrock."""
    agent = _bedrock_agent()
    if agent is None:
        raise RuntimeError("LLM not available - check AWS credentials")
    result = await agent.invoke_async(prompt)
    return str(result).strip()


async def extract_params(prompt: str) -> dict:
    """Extract structured parameters from query using Bedrock."""
    import json
    agent = _bedrock_agent()
    if agent is None:
        raise RuntimeError("LLM not available - check AWS credentials")
    
    text = str(await agent.invoke_async(prompt)).strip()
    if "{" not in text or "}" not in text:
        raise ValueError(f"No valid JSON found in LLM response: {text}")
    
    json_str = text[text.find("{"):text.rfind("}")+1]
    return json.loads(json_str)


async def extract_weather_params(query: str) -> dict:
    """Extract weather parameters exclusively via the Bedrock LLM."""
    prompt = (
        f'Extract city and country from: "{query}"\n'
        "If a US state is mentioned, use its capital city. If only a state/region is "
        "given, use the largest city.\nReturn ONLY valid JSON: "
        '{"city":"CityName","country_code":"XX"}'
    )
    try:
        candidate = await extract_params(prompt)
    except Exception as exc:
        logger.error("LLM weather parameter extraction failed", exc_info=exc)
        raise RuntimeError("Unable to extract weather parameters from query") from exc

    if not _is_valid_weather_payload(candidate):
        raise RuntimeError("LLM returned invalid weather parameters")

    city = str(candidate["city"]).strip()
    country_code = str(candidate["country_code"]).strip().upper()
    return {"city": city, "country_code": country_code}


async def extract_news_topic(query: str) -> dict:
    """Return the primary topic for news queries, with heuristics as a fallback."""
    prompt = (
        f'Extract the main search topic from: "{query}"\n'
        'Return ONLY valid JSON: {"query":"search topic"}'
    )
    try:
        candidate = await extract_params(prompt)
        if isinstance(candidate, dict) and candidate.get("query"):
            return {"query": str(candidate["query"]).strip()}
    except Exception as exc:
        logger.info("LLM news extraction failed: %s", exc)

    cleaned = query.strip()
    if not cleaned:
        raise RuntimeError("Query is required")
    return {"query": cleaned}


def _is_valid_weather_payload(payload: dict) -> bool:
    if not isinstance(payload, dict):
        return False
    city = payload.get("city")
    country = payload.get("country_code")
    return isinstance(city, str) and city.strip() and isinstance(country, str) and len(country.strip()) == 2
