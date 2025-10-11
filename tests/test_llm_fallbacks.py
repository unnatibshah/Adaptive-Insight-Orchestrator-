from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.shared.llm import extract_news_topic, extract_weather_params


@pytest.mark.asyncio
async def test_extract_weather_params_uses_llm(monkeypatch):
    async def fake_extract(prompt: str) -> dict:
        return {"city": "Los Angeles", "country_code": "us"}

    monkeypatch.setattr("agents.shared.llm.extract_params", fake_extract, raising=True)

    result = await extract_weather_params("What's the weather like in California?")
    assert result == {"city": "Los Angeles", "country_code": "US"}


@pytest.mark.asyncio
async def test_extract_weather_params_invalid_response(monkeypatch):
    async def fake_extract(prompt: str) -> dict:
        return {"city": "", "country_code": "USA"}

    monkeypatch.setattr("agents.shared.llm.extract_params", fake_extract, raising=True)

    with pytest.raises(RuntimeError):
        await extract_weather_params("Need the forecast for Paris, FR")


@pytest.mark.asyncio
async def test_extract_news_topic_fallback(monkeypatch):
    async def fake_extract(prompt: str) -> dict:
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr("agents.shared.llm.extract_params", fake_extract, raising=True)

    result = await extract_news_topic("Give me the latest news on renewable energy")
    assert result["query"] == "Give me the latest news on renewable energy"
