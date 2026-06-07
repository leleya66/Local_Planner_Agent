"""Intent parsing, requirement rewriting, and anchor-change helpers."""

from __future__ import annotations

from . import legacy
from .pure_utils import _safe_int, normalize_place_text


def clamp_duration_hours(value) -> int:
    duration = _safe_int(value, 5)
    return max(2, min(12, duration))


__all__ = [
    "normalize_intent_place_type",
    "parse_people_count",
    "extract_start_time_hint_from_user_text",
    "normalize_place_text",
    "place_aliases",
    "find_explicit_place_in_user_input",
    "detect_unmatched_specific_place",
    "extract_location_hint_from_user_text",
    "clean_location_hint_candidate",
    "extract_time_period_hint_from_user_text",
    "extract_weather_hint_from_user_text",
    "split_requirement_clauses",
    "clause_has_negative_preference",
    "extract_negated_requirement_terms",
    "positive_requirement_text_for_matching",
    "heuristic_rewrite_requirement_text",
    "normalize_step_search_keyword",
    "sanitize_ordered_steps",
    "build_ordered_steps_from_keywords",
    "merge_ordered_steps_without_downgrade",
    "extract_ordered_steps_hint_from_user_text",
    "sync_ordered_steps_and_keywords",
    "ordered_search_specs_from_intent",
    "sanitize_structured_requirement_fields",
    "rewrite_user_requirement_for_pipeline",
    "effective_positive_text_from_rewrite",
    "prune_collected_soft_preferences_by_rewrite",
    "extract_meal_pref_hint_from_user_text",
    "extract_excluded_places_from_user_text",
    "is_departure_only_mention",
    "extract_departure_hint_from_user_text",
    "extract_destination_hint_from_user_text",
    "clear_saved_anchor_history_in_workflow",
    "update_locked_places_from_state",
    "is_destination_anchor_intent",
    "planning_anchor_for_intent",
    "same_route_place",
    "user_requests_departure_change",
    "user_requests_destination_change",
    "set_transient_anchor_exclusions_in_state",
    "clear_route_artifacts_after_anchor_change",
    "apply_area_anchor_change_to_state",
    "update_latest_requirement_state",
    "apply_fixed_anchor_guards",
    "same_anchor_identity",
    "move_destination_anchor_to_end",
    "move_destination_anchor_to_start",
    "reconcile_intent_with_rules",
    "clamp_duration_hours",
    "extract_transport_mode_from_user_text",
    "expand_place_match_terms",
    "resolve_generic_location",
    "extract_budget_hint_from_user_text",
    "extract_date_hint_from_user_text",
    "extract_duration_hint_from_user_text",
    "strip_ui_preference_hints",
    "extract_ui_preferences_from_text",
    "interest_keywords",
    "build_interest_match_notes",
    "extract_group_type_from_user_text",
    "extract_latest_requirement_field_hints",
    "merge_latest_requirement_changes",
    "filter_unmentioned_revision_extractions",
    "summarize_latest_requirement_changes",
    "summarize_collected_info_for_followup",
    "build_missing_followup_question",
    "build_canonical_planning_input",
    "collect_required_info_for_api",
    "fast_collect_required_info_for_api",
    "parse_intent",
    "extract_tool_place_name",
]


def __getattr__(name: str):
    if name in __all__:
        return legacy.get(name)
    raise AttributeError(name)
