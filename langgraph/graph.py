from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

START = "__start__"
END = "__end__"


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class CompiledGraph:
    def __init__(
        self,
        nodes: Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]],
        start: str,
        edges: Dict[str, str],
        conditional_edges: Dict[str, Tuple[Callable[[Dict[str, Any]], str], Dict[str, str]]],
    ) -> None:
        self._nodes = nodes
        self._start = start
        self._edges = edges
        self._conditional_edges = conditional_edges

    async def ainvoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        current = self._start
        data = dict(state)

        while current and current != END:
            node = self._nodes[current]
            result = await _maybe_await(node(data))
            if result:
                data.update(result)

            if current in self._conditional_edges:
                router, mapping = self._conditional_edges[current]
                next_key = router(data)
                if next_key not in mapping:
                    raise ValueError(f"Unknown route '{next_key}' for node '{current}'")
                current = mapping[next_key]
            elif current in self._edges:
                current = self._edges[current]
            else:
                current = END

        return data


class StateGraph:
    def __init__(self, state_type: Any) -> None:
        self._nodes: Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = {}
        self._edges: Dict[str, str] = {}
        self._conditional_edges: Dict[str, Tuple[Callable[[Dict[str, Any]], str], Dict[str, str]]] = {}
        self._start: Optional[str] = None
        self._state_type = state_type

    def add_node(self, name: str, func: Callable[[Dict[str, Any]], Any]) -> None:
        async def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            result = func(state)
            return await _maybe_await(result)

        self._nodes[name] = wrapper

    def set_entry_point(self, name: str) -> None:
        if name not in self._nodes:
            raise ValueError(f"Node '{name}' is not defined")
        self._start = name

    def add_edge(self, source: str, target: str) -> None:
        if source not in self._nodes:
            raise ValueError(f"Node '{source}' is not defined")
        self._edges[source] = target

    def add_conditional_edges(
        self,
        source: str,
        router: Callable[[Dict[str, Any]], str],
        mapping: Dict[str, str],
    ) -> None:
        if source not in self._nodes:
            raise ValueError(f"Node '{source}' is not defined")
        self._conditional_edges[source] = (router, mapping)

    def compile(self) -> CompiledGraph:
        if self._start is None:
            raise ValueError("Entry point not set for StateGraph")
        return CompiledGraph(self._nodes, self._start, self._edges, self._conditional_edges)
