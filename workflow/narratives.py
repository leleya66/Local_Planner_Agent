"""Narrative, explanation, and final-plan rendering helpers."""

from __future__ import annotations

import json
import re

from . import legacy


__all__ = [
    "truncate_text",
    "parse_json_object_from_llm_text",
    "_safe_story_value",
    "_first_place_row_value",
    "build_stop_story_payload",
    "generate_one_stop_narrative_with_llm",
    "generate_route_narratives_once_with_llm",
    "generate_stop_narratives_with_llm",
    "stop_card_narrative",
    "build_route_overview_copy",
    "_safe_display_value",
    "summarize_availability_for_explanation",
    "build_planning_explanation",
    "quick_mode_label",
    "quick_mode_default_explanation",
    "build_adjustment_summary_lines",
    "stop_vibe_text",
    "render_fast_plan_text",
    "render_schedule_narrative_plan",
    "result_formatter",
]


def truncate_text(text: str, limit: int = 500) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def parse_json_object_from_llm_text(text: str) -> dict:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def __getattr__(name: str):
    if name in __all__:
        return legacy.get(name)
    raise AttributeError(name)
