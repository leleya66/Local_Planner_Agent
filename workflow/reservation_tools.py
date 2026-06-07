"""Workflow-level reservation, coupon, and booking helpers."""

from __future__ import annotations

from . import legacy


__all__ = [
    "build_coupon_info",
    "build_coupon_info_for_places",
    "infer_coupon_theme",
    "filter_coupon_info_for_schedule",
    "estimate_queue_minutes",
    "build_reservation_info",
    "build_reservation_options",
    "book_order_node",
    "skip_booking",
]


def __getattr__(name: str):
    if name in __all__:
        return legacy.get(name)
    raise AttributeError(name)
