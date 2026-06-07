"""Collaboration room routes."""

from __future__ import annotations

import uuid
from typing import Callable

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from core.schemas import RoomConfirmRequest, RoomRequest


def create_room_router(
    *,
    session_store: dict,
    normalize_room_id: Callable,
    public_room_id: Callable,
    register_room_participant: Callable,
    room_participants: Callable,
    room_confirmation_summary: Callable,
    build_room_post_confirm_payload: Callable,
    save_session: Callable,
    append_event: Callable,
    local_lan_ip: Callable,
    now_ts: Callable,
    public_base_url: str,
) -> APIRouter:
    router = APIRouter()

    @router.post("/rooms")
    async def create_room(req: RoomRequest):
        room_id = uuid.uuid4().hex[:10]
        session_id = normalize_room_id(room_id)
        state = register_room_participant(
            {
                "room_mode": True,
                "room_id": room_id,
                "__events": [],
                "__next_event_id": 1,
                "plan_revision": 0,
                "room_confirmations": {},
            },
            req.client_id,
            req.speaker or "",
        )
        save_session(session_id, state)
        append_event(
            session_id,
            "bot",
            "协作房间已创建。群友可以先自由聊天；讨论结束后，任一成员发送“@Agent 开始规划”即可汇总群聊并生成方案。",
        )
        return JSONResponse({
            "status": "room_ready",
            "room_id": room_id,
            "session_id": session_id,
            "lan_ip": local_lan_ip(),
            "public_base_url": public_base_url,
            "participants": room_participants(state),
            "last_event_id": int((session_store.get(session_id) or {}).get("__next_event_id", 1)) - 1,
        })

    @router.post("/rooms/{room_id}/join")
    async def join_room(room_id: str, req: RoomRequest):
        session_id = normalize_room_id(room_id)
        state = session_store.get(session_id)
        if not state:
            return JSONResponse({"error": "协作房间不存在或已过期"}, status_code=404)
        previous = dict((state.get("room_participants") or {}).get(req.client_id) or {})
        state = register_room_participant(state, req.client_id, req.speaker or "")
        save_session(session_id, state)
        speaker = (state.get("room_participants") or {}).get(req.client_id, {}).get("speaker") or "群友"
        if not previous or str(previous.get("speaker") or "").strip() != speaker:
            append_event(session_id, "bot", f"{speaker} 已加入协作房间。")
        return JSONResponse({
            "status": "room_joined",
            "room_id": public_room_id(session_id),
            "session_id": session_id,
            "participants": room_participants(state),
            "group_discussion": state.get("group_discussion") or {},
            "last_event_id": int((session_store.get(session_id) or {}).get("__next_event_id", 1)) - 1,
        })

    @router.post("/rooms/{room_id}/confirm")
    async def confirm_room_plan(room_id: str, req: RoomConfirmRequest):
        session_id = normalize_room_id(room_id)
        state = session_store.get(session_id)
        if not state:
            return JSONResponse({"error": "协作房间不存在或已过期"}, status_code=404)
        if state.get("planning_in_progress") or not state.get("structured_plan"):
            return JSONResponse({"error": "当前还没有可确认的方案"}, status_code=409)

        state = register_room_participant(state, req.client_id, req.speaker or "")
        revision = int(state.get("plan_revision") or 0)
        if req.plan_revision is not None and int(req.plan_revision) != revision:
            return JSONResponse({"error": "方案已更新，请刷新后确认最新版本"}, status_code=409)

        speaker = (state.get("room_participants") or {}).get(req.client_id, {}).get("speaker") or "群友"
        confirmations = dict(state.get("room_confirmations") or {})
        existing = confirmations.get(req.client_id) or {}

        def confirmation_payload(summary: dict) -> dict:
            payload = {
                "status": "room_confirmation",
                "session_id": session_id,
                "room_id": public_room_id(session_id),
                "participants": room_participants(state),
                "plan_revision": revision,
                "room_confirmation": summary,
            }
            if summary.get("all_confirmed"):
                payload.update(build_room_post_confirm_payload(state))
            return payload

        if int(existing.get("plan_revision") or 0) == revision:
            summary = room_confirmation_summary(state)
            payload = confirmation_payload(summary)
            payload["last_event_id"] = int((session_store.get(session_id) or {}).get("__next_event_id", 1)) - 1
            return JSONResponse(payload)

        confirmations[req.client_id] = {
            "speaker": speaker,
            "confirmed_at": now_ts(),
            "plan_revision": revision,
        }
        state["room_confirmations"] = confirmations
        save_session(session_id, state)
        summary = room_confirmation_summary(state)
        event_payload = confirmation_payload(summary)
        append_event(
            session_id,
            "bot",
            (
                f"{speaker} 已确认第 {revision} 版方案。所有群友均已确认，可以继续执行预订/团购任务。"
                if summary.get("all_confirmed")
                else f"{speaker} 已确认第 {revision} 版方案。"
            ),
            client_id=req.client_id,
            speaker=speaker,
            payload=event_payload,
        )
        event_payload["last_event_id"] = int((session_store.get(session_id) or {}).get("__next_event_id", 1)) - 1
        return JSONResponse(event_payload)

    return router
