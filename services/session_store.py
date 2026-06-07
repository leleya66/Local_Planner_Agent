"""In-process runtime storage for sessions, waiters, cache, and room tasks."""

from __future__ import annotations

import asyncio
import copy
from typing import Callable, Optional


session_store: dict = {}
session_locks: dict = {}
session_waiters: dict = {}
plan_result_cache: dict = {}
room_revision_tasks: dict = {}


PRESERVED_SESSION_KEYS = [
    "__events",
    "__next_event_id",
    "room_mode",
    "room_id",
    "room_participants",
    "room_confirmations",
    "plan_revision",
    "group_discussion",
    "requirement_revision_mode",
    "requirement_change_source_text",
    "latest_requirement_explicit_fields",
    "latest_requirement_changes",
    "requirement_change_log",
    "room_revision_pending",
    "room_revision_pending_client_id",
    "room_revision_pending_speaker",
    "room_revision_pending_text",
]


RUNTIME_CACHE_KEYS = [
    "__events",
    "__next_event_id",
    "__updated_at",
    "room_mode",
    "room_id",
    "room_participants",
    "room_confirmations",
    "plan_revision",
    "group_discussion",
    "room_revision_pending",
    "room_revision_pending_client_id",
    "room_revision_pending_speaker",
    "room_revision_pending_text",
]


def cleanup_sessions(
    session_ttl_seconds: int,
    plan_cache_ttl_seconds: int,
    now_func: Callable[[], float],
) -> None:
    cutoff = now_func() - session_ttl_seconds
    expired = [
        session_id
        for session_id, state in session_store.items()
        if float((state or {}).get("__updated_at", now_func())) < cutoff
    ]
    for session_id in expired:
        session_store.pop(session_id, None)
        session_locks.pop(session_id, None)
        session_waiters.pop(session_id, None)

    cache_cutoff = now_func() - plan_cache_ttl_seconds
    for key, item in list(plan_result_cache.items()):
        if float(item.get("created_at", 0)) < cache_cutoff:
            plan_result_cache.pop(key, None)


def get_cached_plan(cache_key: str, plan_cache_ttl_seconds: int, now_func: Callable[[], float]) -> Optional[dict]:
    item = plan_result_cache.get(cache_key)
    if not item:
        return None
    if now_func() - float(item.get("created_at", 0)) > plan_cache_ttl_seconds:
        plan_result_cache.pop(cache_key, None)
        return None
    cached = copy.deepcopy(item.get("result") or {})
    cached["cache_hit"] = True
    return cached


def set_cached_plan(cache_key: str, result: dict, max_items: int, now_func: Callable[[], float]) -> None:
    if not cache_key or not result:
        return
    while len(plan_result_cache) >= max_items:
        oldest_key = min(plan_result_cache, key=lambda key: plan_result_cache[key].get("created_at", 0))
        plan_result_cache.pop(oldest_key, None)
    cached_result = copy.deepcopy(result)
    for runtime_key in RUNTIME_CACHE_KEYS:
        cached_result.pop(runtime_key, None)
    plan_result_cache[cache_key] = {
        "created_at": now_func(),
        "result": cached_result,
    }


def save_session(session_id: str, state: dict, now_func: Callable[[], float]) -> None:
    previous = session_store.get(session_id) or {}
    keep_pending_room_revision = bool(previous.get("room_revision_pending"))
    for key in PRESERVED_SESSION_KEYS:
        if key in previous and (
            key not in state
            or (keep_pending_room_revision and key.startswith("room_revision_pending"))
        ):
            state[key] = copy.deepcopy(previous[key])
    state["__updated_at"] = now_func()
    session_store[session_id] = state


def touch_session(state: dict, now_func: Callable[[], float]) -> None:
    if state is not None:
        state["__updated_at"] = now_func()


def get_session_lock(session_id: str) -> asyncio.Lock:
    lock = session_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        session_locks[session_id] = lock
    return lock


def notify_session_waiters(session_id: str) -> None:
    for queue in list(session_waiters.get(session_id) or []):
        try:
            queue.put_nowait(True)
        except asyncio.QueueFull:
            pass


def append_event(
    session_id: str,
    role: str,
    text: str,
    now_func: Callable[[], float],
    client_id: str = "",
    speaker: str = "",
    payload: Optional[dict] = None,
) -> None:
    state = session_store.get(session_id)
    if state is None:
        state = {}
    events = state.setdefault("__events", [])
    next_id = int(state.get("__next_event_id", 1) or 1)
    events.append({
        "id": next_id,
        "ts": now_func(),
        "role": role,
        "text": text or "",
        "client_id": client_id or "",
        "speaker": speaker or "",
        "payload": payload or {},
    })
    state["__next_event_id"] = next_id + 1
    state["__events"] = events[-200:]
    save_session(session_id, state, now_func=now_func)
    notify_session_waiters(session_id)
