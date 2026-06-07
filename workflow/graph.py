"""LangGraph workflow assembly and runtime workflow objects."""

from __future__ import annotations

from . import legacy


__all__ = [
    "build_plan_workflow",
    "build_book_workflow",
    "init_workflows",
    "plan_workflow",
    "book_workflow",
    "run_demo",
]


def build_plan_workflow():
    return legacy.call("build_plan_workflow")


def build_book_workflow():
    return legacy.call("build_book_workflow")


def init_workflows() -> None:
    return legacy.call("init_workflows")


def run_demo():
    return legacy.call("run_demo")


def __getattr__(name: str):
    if name in {"plan_workflow", "book_workflow"}:
        return getattr(legacy.module(), name)
    if name in __all__:
        return legacy.get(name)
    raise AttributeError(name)
