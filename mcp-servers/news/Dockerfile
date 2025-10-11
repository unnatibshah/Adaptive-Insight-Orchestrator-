FROM python:3.13-slim

WORKDIR /app

ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY mcp-servers/news/ ./mcp-servers/news/
COPY agents/shared/ ./agents/shared/
COPY agents/__init__.py ./agents/

ENV MCP_TRANSPORT=http \
    MCP_SERVER_HOST=0.0.0.0 \
    MCP_SERVER_PORT=9000

EXPOSE 9000

CMD ["python", "mcp-servers/news/server.py"]
