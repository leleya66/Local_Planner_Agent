"""Route feasibility, availability, business-hour, and validation helpers."""

from __future__ import annotations

from . import legacy


__all__ = [
    "row_seat_count",
    "row_has_seat",
    "row_availability_kind",
    "place_price_detail",
    "estimate_schedule_budget",
    "long_route_segment_warnings",
    "place_effort_level",
    "place_is_rest_break",
    "evaluate_route_feasibility",
    "place_is_indoor",
    "place_has_coupon",
    "place_min_price",
    "estimate_queue_minutes",
    "validate_generated_plan",
    "add_reservation_consistency_checks",
    "has_blocking_distance_conflict",
    "build_distance_conflict_plan",
    "plan_has_coupon_claim",
    "remove_unsupported_coupon_claims",
    "sanitize_final_plan_text",
    "ensure_schedule_places_rendered",
    "append_canonical_schedule_table",
    "parse_start_time_minutes",
    "format_time_minutes",
    "place_duration_profile",
    "place_duration_weight",
    "allocate_dynamic_stop_minutes",
    "ordered_step_matches_place",
    "explicit_stop_minutes_for_places",
    "apply_explicit_stop_durations",
    "build_schedule_slots",
    "duration_minutes_from_slot",
    "build_availability_event",
    "event_message",
    "normalize_event_item",
    "unique_event_preserve_order",
    "sanitize_user_visible_inventory_text",
    "user_visible_availability_event",
    "should_replace_unavailable_place",
    "availability_status_for_place",
    "parse_time_token_to_minutes",
    "parse_open_intervals",
    "slot_minutes",
    "open_intervals_cover_slot",
    "best_business_poi_for_place",
    "check_place_open_for_slot",
    "apply_schedule_business_hour_checks",
    "apply_schedule_availability_checks",
    "infer_crowd_context",
    "schedule_item_need_booking",
]


def has_blocking_distance_conflict(structured_plan: dict) -> bool:
    return False


def __getattr__(name: str):
    if name in __all__:
        return legacy.get(name)
    raise AttributeError(name)
