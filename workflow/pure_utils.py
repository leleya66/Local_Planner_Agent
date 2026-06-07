"""Pure utility functions shared by workflow modules."""

from __future__ import annotations

import re
from typing import Optional


CHINESE_NUMERAL_MAP = {
    "一": 1,
    "二": 2,
    "两": 2,
    "俩": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def _safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            match = re.search(r"-?\d+", value)
            if not match:
                return default
            value = match.group(0)
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_place_text(text: str) -> str:
    value = str(text or "").lower()
    for ch in [" ", "\t", "\n", "，", ",", "。", ".", "、", "-", "_", "·", "（", "）", "(", ")"]:
        value = value.replace(ch, "")
    return value


def unique_preserve_order(items: list) -> list:
    seen = set()
    result = []
    for item in items or []:
        key = str(item)
        if key and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def parse_people_count(text: str, default: int = 2) -> Optional[int]:
    raw_count = text
    if isinstance(raw_count, (int, float)):
        value = int(raw_count)
        return value if value > 0 else default

    compact_text = str(text or "")
    digit_patterns = [
        r"(\d+)\s*(?:个)?人",
        r"(\d+)\s*(?:位|名)",
        r"人数\s*(\d+)",
        r"(\d+)\s*人出行",
    ]
    for pattern in digit_patterns:
        match = re.search(pattern, compact_text)
        if match:
            try:
                value = int(match.group(1))
                return value if value > 0 else default
            except (TypeError, ValueError):
                pass

    chinese_patterns = [
        r"([一二两俩三四五六七八九十])\s*(?:个)?人",
        r"([一二两俩三四五六七八九十])\s*(?:位|名)",
    ]
    for pattern in chinese_patterns:
        match = re.search(pattern, compact_text)
        if match:
            return CHINESE_NUMERAL_MAP.get(match.group(1), default)

    companion_patterns = [
        (["情侣", "约会", "两个人", "2个人"], 2),
        (["一家三口", "三口之家"], 3),
        (["亲子"], 3),
        (["四人", "四个"], 4),
    ]
    for words, value in companion_patterns:
        if any(word in compact_text for word in words):
            return value

    friend_count = next(
        (
            CHINESE_NUMERAL_MAP.get(word)
            for word in CHINESE_NUMERAL_MAP
            if f"和{word}个朋友" in compact_text or f"跟{word}个朋友" in compact_text
        ),
        None,
    )
    if friend_count:
        return friend_count + 1

    return default
