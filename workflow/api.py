"""Compatibility facade for the modular workflow package."""

from __future__ import annotations

from . import (
    amap_tools,
    graph,
    intent_parser,
    narratives,
    reservation_tools,
    route_builder,
    route_validation,
    state,
)
from . import legacy
from .state import AgentState


_MODULES = [
    state,
    amap_tools,
    intent_parser,
    route_builder,
    route_validation,
    narratives,
    reservation_tools,
    graph,
]


def init_models():
    return state.init_models()


def init_workflows():
    return graph.init_workflows()


def __getattr__(name: str):
    if name in {"plan_workflow", "book_workflow"}:
        return getattr(graph, name)
    for module in _MODULES:
        if name in getattr(module, "__all__", []):
            return getattr(module, name)
    return legacy.get(name)
