"""Runtime configuration loaded from environment variables."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class AppConfig:
    session_ttl_seconds: int
    agent_workers: int
    plan_time_limit_seconds: float
    plan_cache_ttl_seconds: int
    plan_cache_max_items: int
    frontend_file: str
    public_base_url: str

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            session_ttl_seconds=int(os.getenv("SESSION_TTL_SECONDS", "3600")),
            agent_workers=int(os.getenv("AGENT_WORKERS", "4")),
            plan_time_limit_seconds=float(os.getenv("PLAN_TIME_LIMIT_SECONDS", "30")),
            plan_cache_ttl_seconds=int(os.getenv("PLAN_CACHE_TTL_SECONDS", "600")),
            plan_cache_max_items=int(os.getenv("PLAN_CACHE_MAX_ITEMS", "32")),
            frontend_file=os.getenv("LOCALMATE_FRONTEND_FILE", "shanghai_agent_1440_v8_live.html"),
            public_base_url=os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/"),
        )
