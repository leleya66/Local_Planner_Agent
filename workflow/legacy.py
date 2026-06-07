"""Lazy access to the current monolithic workflow module."""

from __future__ import annotations

import importlib
from types import ModuleType


def module() -> ModuleType:
    return importlib.import_module("agent_workflow_improved")


def get(name: str):
    return getattr(module(), name)


def call(name: str, *args, **kwargs):
    return get(name)(*args, **kwargs)
