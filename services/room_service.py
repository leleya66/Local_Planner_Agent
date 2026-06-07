"""Pure helpers for collaboration room state."""

from __future__ import annotations

import copy
import re
from typing import Optional


ROOM_RUNTIME_KEYS = [
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
]


def normalize_room_id(room_id: Optional[str]) -> Optional[str]:
    room = re.sub(r"[^a-zA-Z0-9_-]", "", str(room_id or "").strip())
    if not room:
        return None
    return f"room:{room[:40]}"


def public_room_id(session_id: str) -> str:
    return str(session_id or "").removeprefix("room:")


def room_participants(state: dict) -> list[dict]:
    participants = (state or {}).get("room_participants") or {}
    return [
        {
            "client_id": client_id,
            "speaker": str(item.get("speaker") or "群友").strip() or "群友",
        }
        for client_id, item in participants.items()
    ]


def room_confirmation_summary(state: dict) -> dict:
    revision = int((state or {}).get("plan_revision") or 0)
    confirmations = (state or {}).get("room_confirmations") or {}
    participants = room_participants(state)
    participant_ids = {item["client_id"] for item in participants}
    items = [
        {
            "client_id": client_id,
            "speaker": str(item.get("speaker") or "群友"),
            "confirmed_at": item.get("confirmed_at"),
            "plan_revision": int(item.get("plan_revision") or revision),
        }
        for client_id, item in confirmations.items()
        if client_id in participant_ids and int(item.get("plan_revision") or revision) == revision
    ]
    confirmed_ids = {item["client_id"] for item in items}
    pending = [
        item["speaker"]
        for item in participants
        if item.get("client_id") not in confirmed_ids
    ]
    return {
        "plan_revision": revision,
        "confirmed": items,
        "confirmed_count": len(items),
        "total_count": len(participants),
        "pending_speakers": pending,
        "all_confirmed": bool(participants) and len(items) >= len(participants),
    }


def inherit_room_runtime_state(result: dict, source: dict) -> dict:
    result = dict(result or {})
    if not (source or {}).get("room_mode"):
        return result
    for key in ROOM_RUNTIME_KEYS:
        if key in source:
            result[key] = copy.deepcopy(source[key])
    return result


def register_room_participant(state: dict, client_id: str, speaker: str = "", now_ts=None) -> dict:
    client_id = str(client_id or "").strip()
    if not client_id:
        return state
    participants = state.setdefault("room_participants", {})
    previous = participants.get(client_id) or {}
    joined_at = previous.get("joined_at")
    if joined_at is None and now_ts is not None:
        joined_at = now_ts()
    participants[client_id] = {
        "speaker": str(speaker or previous.get("speaker") or f"群友{len(participants) + 1}").strip(),
        "joined_at": joined_at,
    }
    state["room_participants"] = participants
    state["room_mode"] = True
    return state
