"""AMap-related helpers used at the API boundary."""

from __future__ import annotations

import os
import urllib.error
import urllib.request


def browser_key() -> str:
    return (
        os.getenv("AMAP_JS_KEY")
        or os.getenv("AMAP_WEB_KEY")
        or os.getenv("AMAP_API_KEY")
        or os.getenv("GAODE_API_KEY")
        or ""
    )


def fetch_static_map(url: str, timeout_seconds: int = 8) -> tuple[bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "LocalMate/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        content = resp.read()
        content_type = resp.headers.get("Content-Type", "image/png")
    return content, content_type
