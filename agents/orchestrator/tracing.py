from opentelemetry import trace
from phoenix.otel import register

from agents.shared.config import get_settings

settings = get_settings()

_initialized = False

def init_tracing() -> trace.Tracer:
    global _initialized
    if _initialized:
        return trace.get_tracer(settings.orchestrator.phoenix_project_name)

    register(
        project_name=settings.orchestrator.phoenix_project_name,
        endpoint=settings.orchestrator.phoenix_collector_endpoint,
        api_key=settings.orchestrator.phoenix_api_key,
        batch=True,
        verbose=False,
    )
    _initialized = True
    return trace.get_tracer(settings.orchestrator.phoenix_project_name)
