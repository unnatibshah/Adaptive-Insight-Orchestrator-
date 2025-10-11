FROM python:3.13-slim

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY agents/weather/ ./agents/weather/
COPY agents/shared/ ./agents/shared/
COPY agents/__init__.py ./agents/
COPY mcp-servers/weather/ ./mcp-servers/weather/

EXPOSE 8000
CMD ["python", "-m", "agents.weather.main"]