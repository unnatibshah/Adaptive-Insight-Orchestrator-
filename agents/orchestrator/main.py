from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional

from pydantic import BaseModel

from .agentcore_client import AgentCoreOrchestrator

app = FastAPI(title="Bedrock AgentCore Orchestrator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = AgentCoreOrchestrator()


class OrchestrationQuery(BaseModel):
    query: str
    history: Optional[List[Dict[str, str]]] = None


@app.post("/orchestrate")
async def orchestrate(query: OrchestrationQuery):
    try:
        history = query.history or []
        result = await orchestrator.orchestrate(query.query, history=history)
        mode = result.get("mode")
        pattern_map = {
            "agentcore-runtime": "AWS Bedrock AgentCore Runtime",
            "langgraph": "LangGraph + MCP (Local)",
            "strands-local": "Strands SDK + A2A (Local)",
        }
        pattern = pattern_map.get(mode, "Strands SDK + A2A (Local)")
        return {"query": query.query, "pattern": pattern, "orchestration_result": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "orchestrator"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
