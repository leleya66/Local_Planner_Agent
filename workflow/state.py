"""Workflow state and model lifecycle entry points."""

from __future__ import annotations

from typing import Optional, TypedDict

from . import legacy


class AgentState(TypedDict):
    user_input: str
    collected_info: Optional[dict]
    info_complete: Optional[bool]
    pending_question: Optional[str]
    intent: Optional[dict]
    weather_info: Optional[dict]
    rag_context: Optional[str]
    attraction_info: Optional[str]
    ticket_info: Optional[str]
    route_plan: Optional[str]
    route_distance_info: Optional[str]
    route_map: Optional[dict]
    coupon_info: Optional[dict]
    structured_plan: Optional[dict]
    validation_report: Optional[dict]
    feasibility_report: Optional[dict]
    reservation_options: Optional[list]
    exception: Optional[str]
    final_plan: Optional[str]
    confirmed: Optional[bool]
    order_result: Optional[str]
    awaiting_satisfaction: Optional[bool]
    revision_count: Optional[int]
    group_discussion: Optional[dict]
    adjustment_mode: Optional[str]
    adjustment_modes: Optional[list]
    avoid_places: Optional[list]
    previous_plan_places: Optional[list]
    locked_places: Optional[list]
    exception_events: Optional[list]
    latest_user_input: Optional[str]
    node_timings: Optional[dict]
    requirement_revision_mode: Optional[bool]
    requirement_change_source_text: Optional[str]
    latest_requirement_explicit_fields: Optional[list]
    latest_requirement_changes: Optional[list]
    requirement_change_log: Optional[list]


def init_models():
    return legacy.call("init_models")


def refresh_place_data_if_changed(force: bool = False) -> bool:
    return legacy.call("refresh_place_data_if_changed", force)


def place_data_signature() -> str:
    return legacy.call("place_data_signature")
