"""AMap, address, POI, distance, and static-map workflow helpers."""

from __future__ import annotations

import os
import re
from typing import Optional

from . import legacy


__all__ = [
    "get_amap_key",
    "is_amap_transient_limit",
    "amap_backoff_remaining",
    "amap_get_json",
    "resolve_weather_target_date",
    "infer_weather_city",
    "pick_weather_cast",
    "weather_period_fields",
    "query_amap_weather",
    "weather_lookup",
    "normalize_shanghai_address",
    "amap_geocode",
    "clean_amap_address_part",
    "join_amap_address_parts",
    "amap_search_place_text",
    "extract_amap_business_hours",
    "amap_search_place_business",
    "choose_best_poi_for_place",
    "amap_geocode_detail",
    "resolve_place_address",
    "build_specific_place_display_name",
    "build_place_display_detail",
    "infer_amap_search_spec",
    "amap_search_pois_near",
    "normalize_area_anchor",
    "is_shanghai_area_location",
    "is_category_like_location",
    "is_concrete_location_anchor",
    "default_amap_search_spec",
    "classify_amap_poi_spec",
    "resolve_unmatched_location_with_amap",
    "sync_resolved_poi_anchor_fields",
    "amap_driving_distance",
    "format_distance",
    "format_duration",
    "suggest_transport",
    "build_route_distance_info",
    "compute_route_segments",
    "parse_lnglat",
    "parse_polyline_coords",
    "build_route_map_info",
    "build_route_map_placeholder",
    "route_polyline_points_from_segments",
    "route_endpoint_coords_from_segments",
]


def format_distance(meters) -> str:
    if meters is None:
        return "距离未知"
    if meters >= 1000:
        return f"{meters / 1000:.1f}公里"
    return f"{int(meters)}米"


def format_duration(seconds) -> str:
    if seconds is None:
        return "时间未知"
    minutes = max(1, int(round(seconds / 60)))
    if minutes >= 60:
        return f"{minutes // 60}小时{minutes % 60}分钟"
    return f"{minutes}分钟"


def parse_lnglat(location: str) -> Optional[tuple[float, float]]:
    match = re.match(r"^\s*([0-9.\\-]+)\s*,\s*([0-9.\\-]+)\s*$", str(location or ""))
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def short_static_map_label(place: str, index: int) -> str:
    cleaned = re.sub(r"[（(].*?[）)]", "", str(place or "")).strip()
    token = cleaned[:3] if cleaned else str(index)
    return token


def static_map_marker_label(place: str, index: int) -> str:
    mode = os.getenv("ROUTE_MAP_MARKER_LABEL_MODE", "index").strip().lower()
    if mode == "name":
        return short_static_map_label(place, index)
    return str(index)


def __getattr__(name: str):
    if name in __all__:
        return legacy.get(name)
    raise AttributeError(name)
