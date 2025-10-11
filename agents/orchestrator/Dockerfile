FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY agents/orchestrator/ ./agents/orchestrator/
COPY agents/weather/agent.json ./agents/weather/
COPY agents/news/agent.json ./agents/news/
COPY agents/shared/ ./agents/shared/
COPY agents/__init__.py ./agents/

EXPOSE 8000
CMD ["python", "-m", "agents.orchestrator.main"]