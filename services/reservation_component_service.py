"""Reservation and coupon response helpers."""

from __future__ import annotations

from typing import Callable


SESSION_EXPIRED_ERROR = {"error": "Session 已过期，请重新生成方案"}


def reservation_lookup(items: list[dict]) -> dict:
    allowed_lookup = {}
    for item in items or []:
        place_name = item.get("place_name")
        display_name = item.get("display_name")
        if place_name:
            allowed_lookup[place_name] = place_name
        if display_name:
            allowed_lookup[display_name] = place_name
    return allowed_lookup


def coupon_lookup(state: dict) -> dict:
    return reservation_lookup(((state or {}).get("coupon_info") or {}).get("items") or [])


def place_lookup(state: dict, build_reservation_options: Callable[[dict], list[dict]]) -> dict:
    options = (state or {}).get("reservation_options") or build_reservation_options((state or {}).get("structured_plan") or {})
    return reservation_lookup(options)


def mark_reservation_pending(
    state: dict,
    canonical_place_name: str,
    build_reservation_info: Callable[[str], dict],
) -> dict:
    info = build_reservation_info(canonical_place_name)
    state.setdefault("coupon_reservations", []).append(info)
    state["awaiting_departure_confirmation"] = True
    state["awaiting_satisfaction"] = False
    return info


def reservation_pending_payload(
    *,
    session_id: str,
    reservation: dict,
    question: str,
) -> dict:
    return {
        "status": "reservation_pending",
        "session_id": session_id,
        "reservation": reservation,
        "message": reservation.get("message", "预约信息已生成，请确认是否出发。"),
        "question": question,
    }


def cancel_payload(session_id: str) -> dict:
    return {
        "status": "cancelled",
        "session_id": session_id,
        "order_result": "已取消预订，如需重新规划请重新输入需求。",
    }


def booking_result_payload(session_id: str, result: dict) -> dict:
    return {
        "status": "ended",
        "session_id": session_id,
        "order_result": (result or {}).get("order_result", "预订失败，请重试"),
    }
