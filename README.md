# Adaptive Insight Orchestrator

**One question. Many agents.** A LangGraph planner reads three agents' A2A cards, calls their MCP tools in parallel, and writes an answer grounded only in what the tools returned. The UI streams every step: the plan, each tool call with its arguments and raw result, timings, and the final answer.

**[Try the live demo →](https://unnatibshah.github.io/Adaptive-Insight-Orchestrator-/)** · UI on GitHub Pages, API on Render's free tier (the first run after a quiet spell takes up to a minute while the server wakes).

```
            ┌─▶ weather-agent ── MCP get_weather ── Open-Meteo ─┐
 plan ──────┼─▶ news-agent ───── MCP get_news ───── Google News ┼──▶ synthesize ──▶ answer
(gpt-oss-20b)└─▶ knowledge-agent ─ MCP get_background ─ Wikipedia ┘   (gpt-oss-120b)
```

## How it works

| Piece | What it does |
|---|---|
| **LangGraph** (`app/graph.py`) | A `StateGraph` runs `plan → agent → synthesize`. The planner's tasks fan out with `Send`, so agents run in parallel, and results merge through a reducer. Nodes emit trace events with `get_stream_writer()`. |
| **Planner** | `openai/gpt-oss-20b` on Groq gets each agent's A2A card description plus its MCP tool schema and returns a JSON plan. The plan is validated against the schemas (unknown agents, wrong or missing args and duplicates are dropped). If the model is unavailable or the plan is invalid, keyword routing takes over. |
| **MCP** (`app/tools.py`) | Each agent owns a real MCP server (`MCPServer`). Agents call it through an in-process MCP client (`mcp.Client(server)`): the same protocol as over HTTP, without running separate services. Tool schemas are discovered with `list_tools`. |
| **A2A agent cards** | Each agent publishes a card at `/agents/<id>/.well-known/agent-card.json`; all cards are listed at `/api/agents`. |
| **Writer** | `openai/gpt-oss-120b` answers only from the tool results. If it's unavailable, a deterministic template answer is built from the same data. |
| **Data** | Keyless public sources: Open-Meteo (weather), Google News RSS (news), Wikipedia (background). |
| **Hosting** | The UI (`docs/`) is served by GitHub Pages, so it loads instantly; it pings `/health` and shows a wake-up status while the Render API (`render.yaml`, Docker) starts. CORS allows only the Pages origin. |

## Guardrails

- Input validated (2–300 characters); per-visitor rate limit (6 runs/minute, 60/day) keyed on the proxy-appended client IP, so a spoofed `X-Forwarded-For` doesn't bypass it.
- At most 3 runs in flight; a 45-second budget per run and a 10-second timeout on every tool call.
- Tool responses cached (weather 10 min, news 15 min, background 1 day) to stay well within the free APIs' limits.
- A failed agent never fails the run: the answer covers what succeeded and says what didn't.
- Planner and writer fall back to keyword routing and template answers, so the app still works with no model key at all.
- Errors returned to the browser are generic; nothing internal leaks.

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
GROQ_API_KEY=... .venv/bin/uvicorn app.main:app --reload     # http://localhost:8000
.venv/bin/python -m pytest                                    # offline tests: fake MCP servers, stubbed model
```

Or with Docker:

```bash
docker build -t orchestrator . && docker run -p 7860:7860 -e GROQ_API_KEY=... orchestrator
```

`GROQ_API_KEY` is optional: without it the planner uses keyword routing and answers come from templates. `PLANNER_MODEL` and `WRITER_MODEL` override the defaults.

## API

| Route | |
|---|---|
| `POST /api/run` `{"query": "..."}` | Runs the graph and streams events (`text/event-stream`): `node`, `plan`, `tool`, `answer`, `done`, `error`. |
| `GET /api/agents` | All A2A agent cards. |
| `GET /agents/{id}/.well-known/agent-card.json` | One agent's card. |
| `GET /api/docs` | OpenAPI docs. |

## History

v1 (tag [`v1-bedrock`](../../tree/v1-bedrock)) ran six Docker services on AWS Bedrock with the Strands SDK and paid weather and news APIs. v2 keeps the same multi-agent design (planner, A2A cards, MCP tools, tracing) in one deployable service with real LangGraph, keyless data sources and hardening for public traffic.

---

Built by [Unnati Shah](https://unnatibshah.github.io/).
