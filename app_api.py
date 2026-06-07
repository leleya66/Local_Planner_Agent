# app_api.py
# 后端 API 入口文件：
# - 提供前端页面、单人 / 多人协作规划接口、分享链接、SSE 事件流。
# - 复杂的路线规划逻辑不在这里写，而是调用 agent_workflow_improved.py 的 LangGraph 工作流。
# - 这里主要负责 session 状态保存、多人房间事件同步、超时/缓存控制和前端返回格式。
import asyncio
import uuid
import re
import time
import copy
import hashlib
import json
import ctypes
import urllib.error
import socket
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional
import workflow.api as agent_workflow
from core.config import AppConfig
from core.schemas import (
    ConfirmRequest,
    CouponReserveRequest,
    PlaceReserveRequest,
    PlanRequest,
    RoomConfirmRequest,
    RoomRequest,
)
from services import amap_client
from services import reservation_component_service as reservation_service
from services import room_service
from services.session_store import (
    session_store,
    session_locks,
    session_waiters,
    plan_result_cache,
    room_revision_tasks,
)
from services import session_store as session_service
from routers.room_router import create_room_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 只有 static 目录存在才挂载
import os
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ 全局 session 存储
# session_store 是当前进程内存数据库：单人会话和多人房间都存在这里。
# 注意：现在是单进程内存态；如果多机器/多进程部署，需要换成 Redis 之类共享存储。
APP_CONFIG = AppConfig.from_env()
SESSION_TTL_SECONDS = APP_CONFIG.session_ttl_seconds
GLOBAL_EXECUTOR = ThreadPoolExecutor(max_workers=APP_CONFIG.agent_workers)
PLAN_TIME_LIMIT_SECONDS = APP_CONFIG.plan_time_limit_seconds
PLAN_CACHE_TTL_SECONDS = APP_CONFIG.plan_cache_ttl_seconds
PLAN_CACHE_MAX_ITEMS = APP_CONFIG.plan_cache_max_items
FRONTEND_FILE = APP_CONFIG.frontend_file
PUBLIC_BASE_URL = APP_CONFIG.public_base_url


def now_ts() -> float:
    return time.time()


def current_process_memory_mb() -> Optional[float]:
    """Return current Python process RSS memory in MB without making psutil mandatory."""
    try:
        import psutil  # type: ignore
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        pass

    if os.name != "nt":
        return None

    try:
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle,
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            return None
        return float(counters.WorkingSetSize) / 1024 / 1024
    except Exception:
        return None


def cleanup_sessions() -> None:
    """定期清理过期 session 和方案缓存，避免长期运行时内存无限增长。"""
    session_service.cleanup_sessions(
        SESSION_TTL_SECONDS,
        PLAN_CACHE_TTL_SECONDS,
        now_func=now_ts,
    )


def make_plan_cache_key(state: dict) -> str:
    """根据会影响方案的核心字段生成缓存 key；用于避免同一需求重复跑完整规划链。"""
    relevant = {
        "user_input": state.get("user_input", ""),
        "collected_info": state.get("collected_info") or {},
        "adjustment_modes": state.get("adjustment_modes") or [],
        "locked_places": state.get("locked_places") or [],
        "fixed_departure": state.get("fixed_departure") or (state.get("collected_info") or {}).get("fixed_departure") or "",
        "fixed_destination": state.get("fixed_destination") or (state.get("collected_info") or {}).get("fixed_destination") or "",
        "center_anchor": (state.get("collected_info") or {}).get("center_anchor") or "",
        "avoid_places": state.get("avoid_places") or [],
        "route_variant_seed": state.get("route_variant_seed") or "",
    }
    raw = repr(relevant).encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()


def enforce_explicit_time_hints(state: dict, text: str = "") -> dict:
    """保护用户明确说出的出发时间，避免 LLM/后续合并把“上午8点”等改丢。"""
    state = dict(state or {})
    start_time = agent_workflow.extract_start_time_hint_from_user_text(text)
    time_period = agent_workflow.extract_time_period_hint_from_user_text(text)
    if not start_time and not time_period:
        return state

    collected = dict(state.get("collected_info") or {})
    intent = dict(state.get("intent") or {})
    if start_time:
        collected["start_time"] = start_time
        intent["start_time"] = start_time
    if time_period:
        collected["time_period"] = time_period
        intent["time_period"] = time_period
    state["collected_info"] = collected
    state["intent"] = intent
    return state


def should_vary_route(state: dict, latest_text: str = "") -> bool:
    """判断是否允许路线随机变化；明确锁定目的地时尽量稳定，开放式推荐才允许换一版。"""
    if os.getenv("ENABLE_ROUTE_VARIATION", "1") != "1":
        return False
    text = f"{state.get('user_input', '')} {latest_text or ''}"
    open_terms = [
        "你看着办",
        "随便",
        "随机",
        "都行",
        "附近",
        "周边",
        "推荐",
        "安排",
        "换一个",
        "换条路线",
        "再来一个",
        "再来一版",
        "重新生成",
        "重生成",
        "重新规划",
        "regenerate",
        "不满意",
    ]
    if any(term in text for term in open_terms):
        return True
    # If the user did not lock an explicit place, small variation is acceptable.
    return not bool(state.get("locked_places"))


def get_cached_plan(cache_key: str) -> Optional[dict]:
    return session_service.get_cached_plan(
        cache_key,
        PLAN_CACHE_TTL_SECONDS,
        now_func=now_ts,
    )


def set_cached_plan(cache_key: str, result: dict) -> None:
    session_service.set_cached_plan(
        cache_key,
        result,
        PLAN_CACHE_MAX_ITEMS,
        now_func=now_ts,
    )


def log_generation_time(session_id: str, elapsed: float, stage: str, cache_hit: bool = False) -> None:
    marker = "✅" if elapsed <= PLAN_TIME_LIMIT_SECONDS else "⚠️"
    cache_text = "，命中缓存" if cache_hit else ""
    print(f"{marker} 方案耗时统计 [{session_id}] {stage}: {elapsed:.2f}s / 限制 {PLAN_TIME_LIMIT_SECONDS:.0f}s{cache_text}")
    if elapsed > PLAN_TIME_LIMIT_SECONDS:
        print(
            "⚠️ 已超过30秒目标。优先优化项：降低LLM输出长度、减少高德逐段请求、复用RAG/高德缓存、"
            "开启本地plan_result_cache；多人/线上部署时再考虑Redis共享缓存。"
        )


def save_session(session_id: str, state: dict) -> None:
    """保存 session，并保留多人房间事件、确认状态、需求修改状态等运行时字段。"""
    session_service.save_session(session_id, state, now_func=now_ts)


def touch_session(state: dict) -> None:
    session_service.touch_session(state, now_func=now_ts)


def normalize_room_id(room_id: Optional[str]) -> Optional[str]:
    """把 URL 里的 room 参数清洗成内部 room:xxx 格式，防止非法字符进入 session_id。"""
    return room_service.normalize_room_id(room_id)


def public_room_id(session_id: str) -> str:
    """把内部 room:xxx 转回前端分享链接里使用的短 room id。"""
    return room_service.public_room_id(session_id)


def local_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"


def get_session_lock(session_id: str) -> asyncio.Lock:
    return session_service.get_session_lock(session_id)


def notify_session_waiters(session_id: str) -> None:
    """Wake SSE listeners that are waiting for this session's next event.

    This replaces the previous browser-side /session polling loop after @Agent starts.
    The function is intentionally best-effort: if a listener queue is already full,
    that listener will still read the latest state on its next wake-up/heartbeat.
    """
    session_service.notify_session_waiters(session_id)


def append_event(session_id: str, role: str, text: str, client_id: str = "", speaker: str = "", payload: Optional[dict] = None) -> None:
    session_service.append_event(
        session_id,
        role,
        text,
        now_func=now_ts,
        client_id=client_id,
        speaker=speaker,
        payload=payload,
    )


def response_text(payload: dict) -> str:
    for key in ["plan", "question", "message", "order_result", "error"]:
        if payload.get(key):
            return str(payload.get(key))
    return str(payload.get("status", ""))


def json_event_response(session_id: str, payload: dict, client_id: str = "") -> JSONResponse:
    append_event(
        session_id,
        "bot",
        response_text(payload),
        client_id=client_id,
        payload=dict(payload),
    )
    payload["last_event_id"] = int((session_store.get(session_id) or {}).get("__next_event_id", 1)) - 1
    return JSONResponse(payload)


def public_route_map(route_map: dict) -> dict:
    public = dict(route_map or {})
    public.pop("amap_url", None)
    public.pop("amap_base_url", None)
    return public


def clear_saved_anchor_history(state: dict) -> dict:
    """Do not persist old start/destination anchors across turns.

    Older versions saved replaced_destination_keys/replaced_destinations so downstream
    code could filter stale anchors. The product behavior is now simpler: keep only
    the latest fixed_departure/fixed_destination in the session. Temporary exclusion
    keys may exist during one planning run, but they are removed before saving.
    """
    state = dict(state or {})
    for key in [
        "replaced_destination_keys",
        "replaced_destinations",
        "replaced_departure_keys",
        "replaced_departures",
    ]:
        state.pop(key, None)
    collected = dict(state.get("collected_info") or {})
    for key in [
        "replaced_destination_keys",
        "replaced_destinations",
        "replaced_departure_keys",
        "replaced_departures",
    ]:
        collected.pop(key, None)
    state["collected_info"] = collected
    return state


def set_transient_anchor_exclusions(state: dict, names: list) -> dict:
    """Remember stale anchors only for the current planning call, not for the session."""
    state = dict(state or {})
    clean_names = []
    clean_keys = []
    for name in names or []:
        text = str(name or "").strip()
        if not text:
            continue
        key = agent_workflow.normalize_place_text(text)
        if key and key not in clean_keys:
            clean_keys.append(key)
            clean_names.append(text)
    if clean_names:
        state["exclude_anchor_names_once"] = clean_names
        state["exclude_anchor_keys_once"] = clean_keys
    return state


def purge_transient_anchor_exclusions(state: dict) -> dict:
    """Remove one-run exclusion/debug anchor lists before caching or returning state."""
    state = clear_saved_anchor_history(state or {})
    for key in ["exclude_anchor_names_once", "exclude_anchor_keys_once"]:
        state.pop(key, None)
    collected = dict(state.get("collected_info") or {})
    for key in ["exclude_anchor_names_once", "exclude_anchor_keys_once"]:
        collected.pop(key, None)
    state["collected_info"] = collected
    return state


def estimate_budget_from_structured_plan(result: dict) -> dict:
    """Compute a concrete, per-version budget from the final schedule."""
    structured = (result or {}).get("structured_plan") or {}
    hard = structured.get("hard_constraints") or {}
    schedule = structured.get("schedule") or []
    num_people = safe_int(hard.get("num_people"), 1)
    per_min = 0.0
    per_max = 0.0
    unknown_count = 0
    for item in schedule:
        if not isinstance(item, dict):
            continue
        price_text = str(item.get("price_text") or "")
        low = float(item.get("price_min") or 0)
        high = float(item.get("price_max") or low or 0)
        if "待核验" in price_text and low <= 0 and high <= 0:
            unknown_count += 1
            continue
        per_min += max(0.0, low)
        per_max += max(0.0, high if high >= low else low)
    if per_min <= 0 and per_max <= 0 and unknown_count:
        text = "预算待核验"
    elif int(per_min) == int(per_max):
        text = f"预计人均约{int(round(per_max))}元，总计约{int(round(per_max * num_people))}元（{num_people}人）"
    else:
        text = (
            f"预计人均约{int(round(per_min))}-{int(round(per_max))}元，"
            f"总计约{int(round(per_min * num_people))}-{int(round(per_max * num_people))}元（{num_people}人）"
        )
    if unknown_count and text != "预算待核验":
        text += f"；另有{unknown_count}个地点价格待核验"
    requested = hard.get("requested_budget") or hard.get("budget") or ((result or {}).get("collected_info") or {}).get("budget")
    return {
        "per_person_min": int(round(per_min)),
        "per_person_max": int(round(per_max)),
        "total_min": int(round(per_min * num_people)),
        "total_max": int(round(per_max * num_people)),
        "unknown_count": unknown_count,
        "num_people": num_people,
        "text": text,
        "requested_budget": requested or "",
    }


def attach_budget_estimate(result: dict) -> dict:
    """Update result.structured_plan with a budget that follows the final route."""
    if not isinstance(result, dict):
        return result
    structured = dict(result.get("structured_plan") or {})
    if not structured:
        return result
    hard = dict(structured.get("hard_constraints") or {})
    requested = hard.get("requested_budget") or hard.get("budget") or ((result.get("collected_info") or {}).get("budget")) or ""
    estimate = estimate_budget_from_structured_plan({**result, "structured_plan": {**structured, "hard_constraints": hard}})
    hard["requested_budget"] = requested
    hard["budget_estimate"] = estimate
    hard["budget_estimate_text"] = estimate.get("text") or "预算待核验"
    hard["budget"] = requested
    structured["hard_constraints"] = hard
    structured["budget_estimate"] = estimate
    result["structured_plan"] = structured
    return result


def is_regenerate_request(text: str) -> bool:
    normalized = str(text or "").strip().lower().replace(" ", "")
    patterns = [
        "regenerate", "重新生成", "重生成", "再生成", "再来一版", "再来一个",
        "换条路线", "换一条路线", "重新规划", "按当前偏好重新生成", "不变",
    ]
    return any(pattern in normalized for pattern in patterns)


def build_frontend_meta(result: dict) -> dict:
    """Small, stable UI summary so the conversational HTML and backend stay aligned.

    The frontend still renders from structured_plan.schedule as the source of truth;
    this summary is only a convenience layer for title/meta chips and does not remove
    any existing response fields.
    """
    structured = (result or {}).get("structured_plan") or {}
    hard = structured.get("hard_constraints") or {}
    schedule = structured.get("schedule") or []
    names = []
    for item in schedule:
        if isinstance(item, dict):
            names.append(str(item.get("display_name") or item.get("place") or "").strip())
    names = [n for n in names if n]
    return {
        "style": "conversational_tabs_v9",
        "title": " + ".join(names[:2]) if names else "上海周末出行规划助手",
        "date": hard.get("date") or "本周末",
        "time_period": hard.get("time_period") or "半日",
        "stops": len(schedule),
        "duration_hours": hard.get("duration_hours") or 5,
        "budget": hard.get("budget") or hard.get("budget_estimate_text") or "预算待定",
        "requested_budget": hard.get("requested_budget") or "",
        "budget_estimate": hard.get("budget_estimate") or {},
        "route_logic_mode": hard.get("route_logic_mode") or "",
        "adjustment_modes": hard.get("adjustment_modes") or [],
        "amap_js_key": amap_client.browser_key(),
    }


def build_room_post_confirm_payload(state: dict) -> dict:
    """全员确认后给前端补齐可执行任务所需的数据。"""
    state = state or {}
    structured_plan = state.get("structured_plan") or {}
    return {
        "plan": state.get("final_plan") or "",
        "final_plan": state.get("final_plan") or "",
        "distance_info": state.get("route_distance_info") or "",
        "structured_plan": structured_plan,
        "validation_report": state.get("validation_report") or {},
        "feasibility_report": state.get("feasibility_report") or (structured_plan.get("feasibility_report") or {}),
        "coupon_info": state.get("coupon_info") or {},
        "reservation_options": state.get("reservation_options") or [],
        "route_map": public_route_map(state.get("route_map") or {}),
        "generation_time_seconds": state.get("generation_time_seconds"),
        "generation_time_actual_seconds": state.get("generation_time_actual_seconds"),
        "generation_time_excluded_seconds": state.get("generation_time_excluded_seconds"),
        "amap_route_generation_seconds": state.get("amap_route_generation_seconds"),
        "route_segments_generation_seconds": state.get("route_segments_generation_seconds"),
        "route_map_generation_seconds": state.get("route_map_generation_seconds"),
        "generation_time_over_limit": state.get("generation_time_over_limit"),
        "frontend_meta": build_frontend_meta(state),
        "group_considerations": structured_plan.get("group_considerations") or state.get("group_considerations") or [],
        "exception_events": state.get("exception_events") or ((structured_plan.get("route_logic_validation") or {}).get("exception_events") or []),
    }


def _feedback_plain_text(text: str) -> str:
    """去掉前端自动拼接的“兴趣偏好/节奏偏好”，避免“偏好”里的“好”被误判为满意。"""
    raw = str(text or "").strip()
    try:
        raw = agent_workflow.strip_ui_preference_hints(raw)
    except Exception:
        raw = re.sub(r"[（(]\s*(?:兴趣偏好|Interests?|节奏偏好|Pace)[:：][^）)]*[）)]", "", raw, flags=re.I)
        raw = re.sub(r"(?:兴趣偏好|Interests?)[:：][^；;。\n]*", "", raw, flags=re.I)
        raw = re.sub(r"(?:节奏偏好|Pace)[:：][^；;。\n]*", "", raw, flags=re.I)
    raw = re.sub(r"\s+", "", raw)
    return raw.strip("，,。；;！!？?｜|")


def _is_clean_positive_feedback(clean_text: str) -> bool:
    """只接受完整短句，不做“好/行/ok”的任意子串匹配。"""
    compact = re.sub(r"\s+", "", str(clean_text or "").strip().lower())
    if not compact:
        return False
    exact_positive = {
        "满意", "可以", "可", "行", "好", "好的", "好呀", "好啊", "ok", "okay",
        "没问题", "就这个", "就这样", "确认", "不错", "挺好", "很好", "可以了",
        "就这个吧", "就这样吧", "方案可以", "方案不错", "我满意", "我觉得可以",
    }
    if compact in exact_positive:
        return True
    positive_patterns = [
        r"^(这个|这版|方案)?(很|挺|蛮|还)?(好|不错|可以)(的|了|呀|啊|哦|啦|吧)?$",
        r"^(我)?(满意|确认)(了|呀|啊|哦|啦|吧)?$",
        r"^就(这个|这版|这样)(了|呀|啊|哦|啦|吧)?$",
        r"^(没问题|可以了|ok|okay)(呀|啊|哦|啦|吧)?$",
    ]
    return any(re.fullmatch(pattern, compact, flags=re.I) for pattern in positive_patterns)


def is_revision_feedback(text: str) -> bool:
    """判断用户是否是在改方案，而不是满意确认。"""
    clean_text = _feedback_plain_text(text)
    compact = clean_text.lower()

    if not compact:
        raw = str(text or "")
        try:
            ui_prefs = agent_workflow.extract_ui_preferences_from_text(raw)
        except Exception:
            ui_prefs = {}
        # 只有用户显式选了兴趣或非默认节奏时，才把纯偏好输入当成调整；默认 Balanced 不算。
        return bool(ui_prefs.get("interests") or ui_prefs.get("pace") in {"Relaxed", "Packed"})

    # 明确满意短句优先排除，防止“可以/就这个”被“可以再”这类规则误伤。
    if _is_clean_positive_feedback(compact):
        return False

    revision_keywords = [
        "不满意", "不太满意", "不行", "不好", "不喜欢", "不合适",
        "改", "调整", "修改", "换", "重新", "再来", "再生成", "重新生成", "重新规划",
        "想", "希望", "能不能", "可以再", "麻烦", "帮我", "请",
        "不要", "别", "太远", "太贵", "太累", "更近", "便宜", "室内", "团购",
        "预算", "人均", "总预算", "人数", "几个人", "节奏", "偏好", "兴趣",
        "出发地", "起点", "目的地", "终点", "附近", "周边",
    ]
    if any(word in compact for word in revision_keywords):
        return True

    # 字段级规则：只要抽到新的时间/日期/预算/交通等，就是修改需求。
    extractor_names = [
        "extract_start_time_hint_from_user_text",
        "extract_time_period_hint_from_user_text",
        "extract_date_hint_from_user_text",
        "extract_duration_hint_from_user_text",
        "extract_departure_hint_from_user_text",
        "extract_destination_hint_from_user_text",
        "extract_transport_mode_from_user_text",
        "extract_meal_pref_hint_from_user_text",
        "extract_place_type_hint_from_user_text",
        "extract_excluded_places_from_user_text",
    ]
    for name in extractor_names:
        func = getattr(agent_workflow, name, None)
        if not callable(func):
            continue
        try:
            if func(clean_text):
                return True
        except Exception:
            continue

    if re.search(r"(上午|早上|下午|晚上|中午|夜里|傍晚)?\d{1,2}([:：点]\d{0,2})?(分)?(出发|开始|到|去|走)", compact):
        return True

    return False


def is_satisfied_feedback(text: str) -> bool:
    """只在用户明确短句确认满意时返回 True；含时间/地点/预算/偏好等修改诉求时一律返回 False。"""
    clean_text = _feedback_plain_text(text)
    compact = clean_text.lower()
    negative_patterns = [
        "不满意", "不太满意", "不行", "不好", "不喜欢", "不合适",
        "不要", "但是", "还有", "太远", "太贵", "太累",
    ]
    if any(p in compact for p in negative_patterns):
        return False
    if is_revision_feedback(text):
        return False
    return _is_clean_positive_feedback(compact)


def is_departure_confirm_text(text: str) -> bool:
    normalized = (text or "").strip().lower().replace(" ", "")
    patterns = ["那就出发", "就出发", "出发", "就这样", "就这样吧", "可以了", "确认出发", "冲", "走吧"]
    return any(pattern in normalized for pattern in patterns)


def build_revision_input(previous_input: str, feedback: str) -> str:
    return (
        f"{previous_input}\n"
        f"用户对上一版方案不满意，新的反馈或补充需求是：{feedback}\n"
        "请根据用户反馈重新调整方案，避免重复上一版不满意的点。"
    )


def structured_schedule_places(state: dict) -> list[str]:
    structured = state.get("structured_plan") or {}
    places = []
    for item in structured.get("schedule") or []:
        place = str(item.get("place") or "").strip()
        if place:
            places.append(place)
    for place in structured.get("places") or []:
        if isinstance(place, str) and place.strip():
            places.append(place.strip())
    return list(dict.fromkeys(places))


def update_locked_places_from_state(state: dict, latest_text: str = "") -> dict:
    """缓存用户明确点名的核心地点，并锁定/覆盖明确起点和终点。

    关键规则：
    - 后续“换近一点/换便宜一点/换室内/重新生成”不改出发地、目的地、单中心锚点。
    - 后续“目的地换成X/更换目的地为X/终点改为X”只覆盖目的地；旧目的地从 locked_places 中移除。
    - 后续“出发地换成X/起点改为X”只覆盖出发地。
    """
    state = clear_saved_anchor_history(state)
    locked = list(state.get("locked_places") or [])
    intent = dict(state.get("intent") or {})
    collected = dict(state.get("collected_info") or {})
    text = str(latest_text or "")
    # 重要：用于判断本轮是否明确点名，只看 latest_text，不能把历史 user_input 拼进来，
    # 否则旧目的地会因为历史文本仍然存在而被再次锁回去。
    user_text_for_this_turn = text

    old_fixed_departure = str(
        state.get("fixed_departure") or collected.get("fixed_departure") or collected.get("departure") or ""
    ).strip()
    old_fixed_destination = str(
        state.get("fixed_destination") or collected.get("fixed_destination") or collected.get("location") or ""
    ).strip()

    latest_departure_hint = agent_workflow.extract_departure_hint_from_user_text(text)
    latest_destination_hint = agent_workflow.extract_destination_hint_from_user_text(text)

    if agent_workflow.is_departure_edit_value(old_fixed_destination, text, latest_departure_hint or ""):
        # 兜底清理：上一轮若已把“出发地换为X”误存成目的地，本轮不能继续保留。
        old_fixed_destination = ""
        for key in ["fixed_destination", "active_destination_anchor"]:
            state.pop(key, None)
        for key in ["fixed_destination", "active_destination_anchor", "location"]:
            collected.pop(key, None)
        collected["_location_explicit"] = False
        intent_location = str(intent.get("location") or "").strip()
        if agent_workflow.is_departure_edit_value(intent_location, text, latest_departure_hint or ""):
            intent.pop("location", None)

    departure_change = bool(latest_departure_hint) and agent_workflow.user_requests_departure_change(text, old_fixed_departure)
    destination_change = bool(latest_destination_hint) and agent_workflow.user_requests_destination_change(text, old_fixed_destination)

    # 先处理显式起点修改；没有改出发地时必须保留旧出发地。
    if departure_change:
        new_departure = str(latest_departure_hint).strip()
        state["fixed_departure"] = new_departure
        collected["fixed_departure"] = new_departure
        collected["departure"] = new_departure
        collected["_departure_explicit"] = True
    elif old_fixed_departure:
        state["fixed_departure"] = old_fixed_departure
        collected["fixed_departure"] = old_fixed_departure
        collected["departure"] = old_fixed_departure
        collected["_departure_explicit"] = True

    # 目的地明确修改：新目的地覆盖旧目的地；旧目的地和更早历史目的地不能继续锁定。
    if destination_change:
        new_destination = str(latest_destination_hint).strip()
        previous_destination_candidates = [
            old_fixed_destination,
            state.get("fixed_destination"),
            collected.get("fixed_destination"),
            collected.get("location"),
            intent.get("location"),
        ]
        # 如果 locked_places 里有非出发地的旧锚点，通常就是旧目的地或旧路线核心点；换目的地时清掉，
        # 避免“陆家嘴→迪士尼→上海动物园”时陆家嘴/迪士尼继续混入新方案。
        for place in locked:
            if not place:
                continue
            if old_fixed_departure and agent_workflow.same_route_place(str(place), old_fixed_departure):
                continue
            if agent_workflow.same_route_place(str(place), new_destination):
                continue
            previous_destination_candidates.append(place)

        stale_anchor_names = []
        for place in previous_destination_candidates:
            name = str(place or "").strip()
            if not name:
                continue
            if agent_workflow.same_route_place(name, new_destination):
                continue
            if old_fixed_departure and agent_workflow.same_route_place(name, old_fixed_departure):
                continue
            stale_anchor_names.append(name)
        stale_anchor_names = list(dict.fromkeys(stale_anchor_names))
        # 旧目的地只用于本轮过滤，不再持久化到 session。
        state = set_transient_anchor_exclusions(state, stale_anchor_names)

        # 只保留原出发地和最新目的地；不要保留旧目的地。
        kept_locked = []
        for place in locked:
            if old_fixed_departure and agent_workflow.same_route_place(str(place), old_fixed_departure):
                kept_locked.append(place)
        locked = kept_locked
        state["fixed_destination"] = new_destination
        state["active_destination_anchor"] = new_destination
        collected["fixed_destination"] = new_destination
        collected["active_destination_anchor"] = new_destination
        collected["location"] = new_destination
        collected["_location_explicit"] = True
        collected["center_anchor"] = new_destination
        intent["location"] = new_destination
        # 清掉容易把旧目的地带回来的历史候选；只使用本轮 stale_anchor_names 临时判断，不写入 session。
        def _is_stale_anchor(value: str) -> bool:
            return any(agent_workflow.same_route_place(str(value or ""), old) for old in stale_anchor_names)
        state["avoid_places"] = [p for p in (state.get("avoid_places") or []) if not _is_stale_anchor(str(p))]
        cleaned_previous = []
        for route in (state.get("previous_plan_places") or []):
            cleaned_previous.append([p for p in route if not _is_stale_anchor(str(p))])
        state["previous_plan_places"] = cleaned_previous
        if new_destination not in locked:
            locked.append(new_destination)
    elif old_fixed_destination and agent_workflow.is_concrete_location_anchor(old_fixed_destination):
        state["fixed_destination"] = old_fixed_destination
        state["active_destination_anchor"] = old_fixed_destination
        collected["fixed_destination"] = old_fixed_destination
        collected["active_destination_anchor"] = old_fixed_destination
        collected["location"] = old_fixed_destination
        collected["_location_explicit"] = True
        if old_fixed_destination not in locked:
            locked.append(old_fixed_destination)

    if collected.get("_area_anchor_explicit"):
        collected.pop("fixed_destination", None)
        collected.pop("active_destination_anchor", None)
        collected.pop("center_anchor", None)
        state.pop("fixed_destination", None)
        state.pop("active_destination_anchor", None)
        state.pop("center_anchor", None)

    # 首轮或非显式修改时，从已收集信息中建立锁定。
    departure = str(collected.get("departure") or intent.get("departure") or "").strip()
    if departure and collected.get("_departure_explicit"):
        state["fixed_departure"] = departure
        collected["fixed_departure"] = departure
        collected["center_anchor"] = collected.get("center_anchor") or departure

    destination = str(collected.get("location") or intent.get("location") or "").strip()
    if destination and collected.get("_location_explicit") and agent_workflow.is_concrete_location_anchor(destination):
        state["fixed_destination"] = destination
        state["active_destination_anchor"] = destination
        collected["fixed_destination"] = destination
        collected["active_destination_anchor"] = destination
        collected["center_anchor"] = destination
        if destination not in locked:
            locked.append(destination)

    # 通用显式地点锁定：换目的地的这一轮不能用旧 intent/location 从历史里再锁回旧目的地。
    if destination_change:
        candidate = str(collected.get("location") or "").strip()
    elif state.get("fixed_destination") or collected.get("fixed_destination"):
        candidate = str(collected.get("fixed_destination") or state.get("fixed_destination") or collected.get("location") or "").strip()
    else:
        candidate = str(intent.get("location") or collected.get("location") or "").strip()
    user_mentioned_candidate = bool(candidate) and agent_workflow.place_matches_text(candidate, user_text_for_this_turn)
    if (
        candidate
        and candidate != "待确认地点"
        and (intent.get("explicit_place_match") is True or collected.get("_location_explicit"))
        and user_mentioned_candidate
    ):
        if candidate not in locked:
            locked.append(candidate)

    persistent_anchor_keys = {agent_workflow.normalize_place_text(p) for p in [
        collected.get("fixed_departure"), collected.get("fixed_destination"), collected.get("center_anchor")
    ] if p}
    replaced_keys = set(state.get("exclude_anchor_keys_once") or [])
    current_destination_key = agent_workflow.normalize_place_text(collected.get("fixed_destination") or "")
    kept = []
    for place in locked:
        place_key = agent_workflow.normalize_place_text(place)
        if place_key in replaced_keys and place_key != current_destination_key:
            continue
        if place_key not in persistent_anchor_keys and not agent_workflow.place_matches_text(place, user_text_for_this_turn) and not agent_workflow.same_route_place(place, destination):
            continue
        revoke_patterns = [
            f"不想去{place}", f"不要{place}", f"换掉{place}", f"不去{place}", f"{place}不要了",
            f"{place}换掉", f"{place}不去了",
        ]
        if any(pattern in text for pattern in revoke_patterns):
            continue
        kept.append(place)

    # 如果本轮明确换目的地，旧目的地绝不能继续留在 locked_places。
    if destination_change:
        current_destination = str(collected.get("fixed_destination") or "").strip()
        # 只保留原出发地和最新目的地。任何旧目的地/旧路线核心点都不能继续锁定。
        strict_kept = []
        for p in kept:
            if old_fixed_departure and agent_workflow.same_route_place(str(p), old_fixed_departure):
                strict_kept.append(p)
            elif current_destination and agent_workflow.same_route_place(str(p), current_destination):
                strict_kept.append(current_destination)
        kept = strict_kept or ([current_destination] if current_destination else [])

    state["intent"] = intent
    state["locked_places"] = list(dict.fromkeys(kept))
    state["collected_info"] = collected
    return state

QUICK_ADJUSTMENT_MAP = {
    "换近一点": "nearer",
    "换便宜一点": "cheaper",
    "换成室内": "indoor",
    "换室内": "indoor",
    "优先有团购": "coupon",
    "少走路": "less_walk",
}


def detect_quick_adjustments(text: str) -> list[str]:
    """识别前端快捷按钮或自然语言里的调整指令，如更近、更便宜、室内、有团购、少走路。"""
    normalized = (text or "").strip().replace(" ", "")
    modes = []
    for label, mode in QUICK_ADJUSTMENT_MAP.items():
        if label.replace(" ", "") in normalized and mode not in modes:
            modes.append(mode)
    synonym_rules = [
        ("nearer", ["近一点", "距离近", "别太远", "更近", "附近"]),
        ("cheaper", ["便宜", "预算低", "省钱", "低价", "更划算"]),
        ("indoor", ["室内", "不要露天", "别晒", "避雨", "商场"]),
        ("coupon", ["团购", "优惠券", "有券", "满减"]),
        ("less_walk", ["少走路", "不想走", "别走太多", "少步行"]),
    ]
    for mode, words in synonym_rules:
        if mode not in modes and any(word in normalized for word in words):
            modes.append(mode)
    return modes


def build_quick_revision_input(previous_input: str, modes: list[str]) -> str:
    descriptions = {
        "nearer": "请换近一点：所有相邻地点之间的距离都要缩短，优先选择同一区域或高德附近 POI，不只改一个点。",
        "cheaper": "请换便宜一点：降低门票/活动/餐饮预算，必要时替换为更便宜的饭店或活动，并同时考虑距离。",
        "indoor": "请换成室内：整条路线所有地点都必须是室内，剔除公园、街道、滨江步道等露天地点。",
        "coupon": "请优先有团购：吃饭或活动地点必须优先选择有团购券的地点，没有券的地点要替换；如果确实没有可用券，明确说明。",
        "less_walk": "请少走路：尽量减少步行，控制站点数量，交通建议优先地铁、骑行、打车，不安排长距离步行串联。",
    }
    requirement_text = "\n".join(f"- {descriptions.get(mode, mode)}" for mode in modes)
    return (
        f"{previous_input}\n"
        f"用户提出了以下快捷调整要求：\n{requirement_text}\n"
        "请基于上一版方案重新生成，必须同时满足这些调整，不要只处理其中一个，也不要原样复述上一版。"
    )


def safe_int(value, default: int = 1) -> int:
    chinese_numbers = {"一": 1, "二": 2, "两": 2, "俩": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    try:
        return int(value)
    except (TypeError, ValueError):
        text = str(value or "")
        match = re.search(r"\d+", text)
        if match:
            return int(match.group())
        normalized = re.sub(r"\s+", "", text)
        if any(word in normalized for word in ["父母", "爸妈", "爸爸妈妈", "爹妈"]) and re.search(r"(?:我|我们)?(?:和|跟|带|陪)", normalized):
            return 3
        match = re.search(r"一家\s*([一二两俩三四五六七八九十])\s*口", normalized)
        if match:
            return chinese_numbers.get(match.group(1), default)
        match = re.search(r"([一二两俩三四五六七八九十])\s*(?:个)?人?", normalized)
        if match:
            return chinese_numbers.get(match.group(1), default)
        return default


def room_participants(state: dict) -> list[dict]:
    """把房间参与者从内部 dict 转成前端可直接渲染的列表。"""
    participants = (state or {}).get("room_participants") or {}
    return [
        {"client_id": client_id, "speaker": str(item.get("speaker") or "群友").strip() or "群友"}
        for client_id, item in participants.items()
    ]


def room_confirmation_summary(state: dict) -> dict:
    """汇总当前版本方案的逐人确认状态，前端据此显示“xx已确认/待确认”。"""
    revision = int((state or {}).get("plan_revision") or 0)
    confirmations = (state or {}).get("room_confirmations") or {}
    participants = room_participants(state)
    participant_ids = {item["client_id"] for item in participants}
    items = [
        {
            "client_id": client_id,
            "speaker": str(item.get("speaker") or "群友"),
            "confirmed_at": item.get("confirmed_at"),
            "plan_revision": int(item.get("plan_revision") or revision),
        }
        for client_id, item in confirmations.items()
        if client_id in participant_ids and int(item.get("plan_revision") or revision) == revision
    ]
    confirmed_ids = {item["client_id"] for item in items}
    pending = [
        item["speaker"]
        for item in participants
        if item.get("client_id") not in confirmed_ids
    ]
    return {
        "plan_revision": revision,
        "confirmed": items,
        "confirmed_count": len(items),
        "total_count": len(participants),
        "pending_speakers": pending,
        "all_confirmed": bool(participants) and len(items) >= len(participants),
    }


def inherit_room_runtime_state(result: dict, source: dict) -> dict:
    """把多人房间运行时字段从旧 state 继承到规划结果里，避免工作流输出覆盖协作状态。"""
    result = dict(result or {})
    if not (source or {}).get("room_mode"):
        return result
    for key in [
        "room_mode", "room_id", "room_participants", "room_confirmations",
        "plan_revision", "group_discussion", "requirement_revision_mode",
        "requirement_change_source_text", "latest_requirement_explicit_fields",
        "latest_requirement_changes", "requirement_change_log",
    ]:
        if key in source:
            result[key] = copy.deepcopy(source[key])
    return result


def register_room_participant(state: dict, client_id: str, speaker: str = "") -> dict:
    """登记房间成员；同一个浏览器 client_id 重复加入时更新昵称，不重复计数。"""
    client_id = str(client_id or "").strip()
    if not client_id:
        return state
    participants = state.setdefault("room_participants", {})
    previous = participants.get(client_id) or {}
    participants[client_id] = {
        "speaker": str(speaker or previous.get("speaker") or f"群友{len(participants) + 1}").strip(),
        "joined_at": previous.get("joined_at") or now_ts(),
    }
    state["room_participants"] = participants
    state["room_mode"] = True
    return state


room_participants = room_service.room_participants
room_confirmation_summary = room_service.room_confirmation_summary
inherit_room_runtime_state = room_service.inherit_room_runtime_state


def register_room_participant(state: dict, client_id: str, speaker: str = "") -> dict:
    return room_service.register_room_participant(state, client_id, speaker, now_ts=now_ts)


def is_agent_mention(text: str) -> bool:
    return bool(re.search(r"@(?:agent|ai|助手|规划助手)", str(text or ""), flags=re.IGNORECASE))


def strip_agent_mention(text: str) -> str:
    return re.sub(r"@(?:agent|ai|助手|规划助手)", "", str(text or ""), flags=re.IGNORECASE).strip()


def build_group_discussion_summary(discussion: dict) -> str:
    members = discussion.get("members") or []
    notes = discussion.get("notes") or []
    lines = ["多人讨论记录（按聊天时间顺序）："]
    if notes:
        for note in notes:
            lines.append(f"- {note}")
    else:
        for member in members:
            lines.append(f"- {member.get('label')}: {member.get('preference')}")
    lines.append("请综合所有成员在上方表达的需求生成方案。")
    return "\n".join(lines)


def room_discussion_from_events(state: dict) -> dict:
    """Collect room messages in chronological order and group them by speaker."""
    participant_names = [item["speaker"] for item in room_participants(state)]
    messages_by_speaker: dict[str, list[str]] = {}
    speaker_order = []
    notes = []
    messages = []
    for event in (state or {}).get("__events") or []:
        if event.get("role") != "user":
            continue
        raw_text = str(event.get("text") or "")
        if is_agent_mention(raw_text):
            continue
        text = raw_text.strip()
        if not text:
            continue
        speaker = str(event.get("speaker") or "群友").strip() or "群友"
        if speaker not in messages_by_speaker:
            messages_by_speaker[speaker] = []
            speaker_order.append(speaker)
        messages_by_speaker[speaker].append(text)
        notes.append(f"{speaker}: {text}")
        messages.append({"speaker": speaker, "text": text})

    ordered_speakers = list(dict.fromkeys(participant_names + speaker_order))
    members = [
        {"label": speaker, "preference": "；".join(messages_by_speaker.get(speaker) or [])}
        for speaker in ordered_speakers
        if messages_by_speaker.get(speaker)
    ]
    return {
        "active": False,
        "complete": bool(members),
        "required_count": len(ordered_speakers),
        "current_index": len(members),
        "members": members,
        "notes": notes,
        "messages": messages,
        "member_options": ordered_speakers,
        "source": "room_chat_history",
    }


def group_consideration_text(preference: str) -> str:
    raw_text = str(preference or "")
    # 多人讨论里也要避免把“我不想喝咖啡”展示成“安排咖啡偏好”。
    try:
        text = agent_workflow.positive_requirement_text_for_matching(raw_text) or raw_text
        negated_terms = agent_workflow.extract_negated_requirement_terms(raw_text)
    except Exception:
        text = raw_text
        negated_terms = []
    considerations = []
    if negated_terms:
        considerations.append("避开明确否定的地点或品类")
    if any(word in text for word in ["预算", "便宜", "人均", "团购"]):
        considerations.append("控制预算并优先核验团购")
    if any(word in text for word in ["少走", "附近", "近一点", "不想走", "距离"]):
        considerations.append("缩短转场并减少步行")
    if any(word in text for word in ["室内", "下雨", "怕晒", "空调"]):
        considerations.append("优先安排室内或天气友好地点")
    if any(word in text for word in ["吃", "餐", "火锅", "咖啡", "甜品"]):
        considerations.append("在路线中安排对应餐饮偏好")
    if any(word in text for word in ["拍照", "展", "公园", "散步", "亲子", "运动", "购物"]):
        considerations.append("将兴趣偏好纳入地点筛选")
    if not considerations:
        considerations.append("纳入地点筛选、节奏安排和路线取舍")
    return "；".join(dict.fromkeys(considerations)) + "。"


def attach_group_considerations(result: dict) -> dict:
    result = dict(result or {})
    discussion = result.get("group_discussion") or {}
    members = discussion.get("members") or []
    if not members:
        return result
    considerations = [
        {
            "member": str(member.get("label") or "群友"),
            "preference": str(member.get("preference") or "").strip(),
            "consideration": group_consideration_text(member.get("preference") or ""),
        }
        for member in members
        if str(member.get("preference") or "").strip()
    ]
    structured = dict(result.get("structured_plan") or {})
    structured["group_considerations"] = considerations
    hard = dict(structured.get("hard_constraints") or {})
    hard["group_considerations"] = considerations
    structured["hard_constraints"] = hard
    result["structured_plan"] = structured
    result["group_considerations"] = considerations
    return result


@app.on_event("startup")
async def startup():
    agent_workflow.init_models()
    agent_workflow.init_workflows()


@app.on_event("shutdown")
async def shutdown():
    GLOBAL_EXECUTOR.shutdown(wait=False, cancel_futures=True)


# ✅ 首页直接返回新版 V8 live 前端
@app.get("/")
async def index():
    return FileResponse(
        FRONTEND_FILE,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/classic")
async def index_classic():
    """保留旧版聊天前端入口，避免原页面功能丢失。"""
    return FileResponse(
        "index.html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/v8")
async def index_v8():
    """兼容旧链接：新版 V8 风格前端。"""
    return FileResponse(
        FRONTEND_FILE,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/session/{session_id}")
async def get_session_state(session_id: str):
    """给多端共享链接做轻量状态读取；分享页恢复、SSE 兜底同步都走这里。"""
    cleanup_sessions()
    state = session_store.get(session_id)
    if not state:
        return JSONResponse({"exists": False, "session_id": session_id})
    touch_session(state)
    return JSONResponse({
        "exists": True,
        "status": (
            "planning_failed" if state.get("planning_failed")
            else "planning" if state.get("planning_in_progress")
            else "plan_ready" if state.get("structured_plan")
            else "session_ready"
        ),
        "session_id": session_id,
        "last_event_id": int(state.get("__next_event_id", 1)) - 1,
        "ended": bool(state.get("__ended")),
        "awaiting_satisfaction": bool(state.get("awaiting_satisfaction")),
        "planning_in_progress": bool(state.get("planning_in_progress")),
        "planning_failed": bool(state.get("planning_failed")),
        "error": state.get("planning_error") or "",
        "group_discussion": state.get("group_discussion") or {},
        "participants": room_participants(state),
        "plan_revision": int(state.get("plan_revision") or 0),
        "room_confirmation": room_confirmation_summary(state),
        "group_considerations": (state.get("structured_plan") or {}).get("group_considerations") or state.get("group_considerations") or [],
        "plan": state.get("final_plan") or "",
        "distance_info": state.get("route_distance_info") or "",
        "structured_plan": state.get("structured_plan") or {},
        "validation_report": state.get("validation_report") or {},
        "feasibility_report": state.get("feasibility_report") or ((state.get("structured_plan") or {}).get("feasibility_report") or {}),
        "coupon_info": state.get("coupon_info") or {},
        "reservation_options": state.get("reservation_options") or [],
        "route_map": public_route_map(state.get("route_map") or {}),
        "generation_time_seconds": state.get("generation_time_seconds"),
        "generation_time_actual_seconds": state.get("generation_time_actual_seconds"),
        "generation_time_excluded_seconds": state.get("generation_time_excluded_seconds"),
        "amap_route_generation_seconds": state.get("amap_route_generation_seconds"),
        "route_segments_generation_seconds": state.get("route_segments_generation_seconds"),
        "route_map_generation_seconds": state.get("route_map_generation_seconds"),
        "generation_time_over_limit": state.get("generation_time_over_limit"),
        "frontend_meta": build_frontend_meta(state),
        "exception": state.get("exception"),
        "exception_events": state.get("exception_events") or ((state.get("structured_plan") or {}).get("route_logic_validation") or {}).get("exception_events") or [],
        "final_plan": state.get("final_plan") or "",
    })


@app.get("/events/{session_id}")
async def get_session_events(session_id: str, since: int = 0):
    """多端协作短轮询事件流；客户端按 last_event_id 拉取新增群聊/规划/确认事件。"""
    cleanup_sessions()
    state = session_store.get(session_id)
    if not state:
        return JSONResponse({"exists": False, "session_id": session_id, "events": [], "last_event_id": since})
    touch_session(state)
    events = [
        event for event in (state.get("__events") or [])
        if int(event.get("id", 0)) > int(since or 0)
    ]
    last_event_id = int(state.get("__next_event_id", 1)) - 1
    return JSONResponse({
        "exists": True,
        "session_id": session_id,
        "ended": bool(state.get("__ended")),
        "planning_in_progress": bool(state.get("planning_in_progress")),
        "planning_failed": bool(state.get("planning_failed")),
        "error": state.get("planning_error") or "",
        "participants": room_participants(state),
        "plan_revision": int(state.get("plan_revision") or 0),
        "room_confirmation": room_confirmation_summary(state),
        "events": events,
        "last_event_id": last_event_id,
    })


@app.get("/events_stream/{session_id}")
async def stream_session_events(session_id: str, since: int = 0, continuous: bool = False):
    """多人协作的主同步通道（SSE）。

    - 分享页等待 plan_ready / planning_failed 后可以关闭。
    - 协作房间传 continuous=true，保持一条 EventSource 长连接。
    - 群聊消息、@Agent 开始规划、最终方案、逐人确认都会通过这里实时推送。
    """

    def sse_payload(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"

    async def event_generator():
        last_seen = int(since or 0)
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        waiters = session_waiters.setdefault(session_id, set())
        waiters.add(queue)
        try:
            while True:
                cleanup_sessions()
                state = session_store.get(session_id)
                if not state:
                    yield sse_payload({
                        "exists": False,
                        "session_id": session_id,
                        "last_event_id": last_seen,
                        "status": "missing",
                    })
                    break
                touch_session(state)

                events = [
                    event for event in (state.get("__events") or [])
                    if int(event.get("id", 0)) > last_seen
                ]
                if events:
                    should_close = False
                    for event in events:
                        last_seen = max(last_seen, int(event.get("id", 0) or 0))
                        payload = event.get("payload") or {}
                        yield sse_payload({
                            "exists": True,
                            "session_id": session_id,
                            "event": event,
                            "last_event_id": last_seen,
                            "planning_in_progress": bool(state.get("planning_in_progress")),
                            "planning_failed": bool(state.get("planning_failed")),
                            "error": state.get("planning_error") or "",
                            "participants": room_participants(state),
                            "plan_revision": int(state.get("plan_revision") or 0),
                            "room_confirmation": room_confirmation_summary(state),
                        })
                        if not continuous and payload.get("status") in {"plan_ready", "planning_failed"}:
                            should_close = True
                    if should_close:
                        break

                if bool(state.get("__ended")):
                    yield sse_payload({
                        "exists": True,
                        "session_id": session_id,
                        "last_event_id": last_seen,
                        "status": "ended",
                    })
                    break

                try:
                    await asyncio.wait_for(queue.get(), timeout=20)
                except asyncio.TimeoutError:
                    # SSE heartbeat. This keeps proxies/ngrok from closing the connection
                    # but does not create repeated HTTP requests.
                    yield ": keepalive\n\n"
        finally:
            waiters = session_waiters.get(session_id)
            if waiters is not None:
                waiters.discard(queue)
                if not waiters:
                    session_waiters.pop(session_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


app.include_router(create_room_router(
    session_store=session_store,
    normalize_room_id=normalize_room_id,
    public_room_id=public_room_id,
    register_room_participant=register_room_participant,
    room_participants=room_participants,
    room_confirmation_summary=room_confirmation_summary,
    build_room_post_confirm_payload=build_room_post_confirm_payload,
    save_session=save_session,
    append_event=append_event,
    local_lan_ip=local_lan_ip,
    now_ts=now_ts,
    public_base_url=PUBLIC_BASE_URL,
))

@app.get("/route_map/{session_id}")
async def route_map(session_id: str):
    """Proxy Amap static map image so the frontend does not expose the API key."""
    started = time.perf_counter()
    cleanup_sessions()
    state = session_store.get(session_id)
    route_map_info = (state or {}).get("route_map") or {}
    if route_map_info.get("available") and (
        not route_map_info.get("amap_url")
        or not route_map_info.get("amap_base_url")
        or not route_map_info.get("center")
        or not route_map_info.get("path_points")
    ):
        try:
            route_map_info = agent_workflow.build_route_map_info((state or {}).get("structured_plan") or {})
        except Exception as exc:
            print(f"⚠️ 路线地图信息生成失败 [{session_id}]: {exc}")
            route_map_info = {
                "available": False,
                "reason": f"路线地图信息生成失败: {exc}",
                "markers": route_map_info.get("markers") or [],
            }
        if state is not None:
            state["route_map"] = route_map_info
            save_session(session_id, state)
    amap_url = route_map_info.get("amap_url")
    if not amap_url:
        return JSONResponse({"error": route_map_info.get("reason") or "当前会话没有可用路线地图"}, status_code=404)
    try:
        content, content_type = amap_client.fetch_static_map(amap_url, timeout_seconds=10)
        if not str(content_type).lower().startswith("image/"):
            preview = content[:300].decode("utf-8", errors="replace")
            print(f"⚠️ 高德静态地图返回非图片 [{session_id}]: {content_type} / {preview}")
            return JSONResponse({"error": f"高德静态地图返回非图片: {preview[:180]}"}, status_code=502)
        print(f"⏱️ 地图接口耗时 [{session_id}]: {time.perf_counter() - started:.2f}s")
        return Response(content=content, media_type=content_type)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return JSONResponse({"error": f"路线地图获取失败: {exc}"}, status_code=502)


async def run_pending_room_revision(session_id: str) -> None:
    """规划中又收到群友修改意见时，当前版本完成后自动合并生成下一版。"""
    await asyncio.sleep(0.05)
    state = session_store.get(session_id) or {}
    if state.get("planning_in_progress") or not state.get("room_revision_pending"):
        return
    client_id = str(state.get("room_revision_pending_client_id") or "")
    speaker = str(state.get("room_revision_pending_speaker") or "群友")
    pending_text = str(state.get("room_revision_pending_text") or "")
    state["room_revision_pending"] = False
    state.pop("room_revision_pending_client_id", None)
    state.pop("room_revision_pending_speaker", None)
    state.pop("room_revision_pending_text", None)
    save_session(session_id, state)
    await plan(PlanRequest(
        user_input="@Agent 应用规划期间收到的新增修改意见，继续生成下一版方案",
        session_id=session_id,
        client_id=client_id,
        speaker=speaker,
        internal_room_replan=True,
        latest_requirement_input=pending_text,
    ))


def schedule_pending_room_revision(session_id: str) -> None:
    """合并规划期间的多条反馈，避免每条消息都触发一次完整重算。"""
    if not (session_store.get(session_id) or {}).get("room_revision_pending"):
        return
    existing = room_revision_tasks.get(session_id)
    if existing and not existing.done():
        return
    task = asyncio.create_task(run_pending_room_revision(session_id))
    room_revision_tasks[session_id] = task
    task.add_done_callback(lambda _task: room_revision_tasks.pop(session_id, None))


@app.post("/plan")
async def plan(req: PlanRequest):
    """主规划入口。

    单人模式直接进入 _plan_impl；多人模式先写房间事件，
    只有 @Agent 或首版后新增修改意见才会触发完整规划。
    """
    cleanup_sessions()
    session_id = req.session_id or normalize_room_id(req.room_id) or str(uuid.uuid4())
    if str(session_id).startswith("room:"):
        state = session_store.get(session_id) or {"room_mode": True, "room_id": public_room_id(session_id)}
        state = register_room_participant(state, req.client_id or "", req.speaker or "")
        save_session(session_id, state)
        queued_revision_requested = bool(req.internal_room_replan)
        agent_mentioned = queued_revision_requested or is_agent_mention(req.user_input)
        has_generated_plan = bool(state.get("structured_plan") or state.get("final_plan"))
        if not queued_revision_requested:
            append_event(session_id, "user", req.user_input, client_id=req.client_id or "", speaker=req.speaker or "")

        if state.get("planning_in_progress"):
            if not agent_mentioned:
                state["room_revision_pending"] = True
                state["room_revision_pending_client_id"] = req.client_id or ""
                state["room_revision_pending_speaker"] = req.speaker or "群友"
                pending_text = str(state.get("room_revision_pending_text") or "").strip()
                new_text = str(req.user_input or "").strip()
                state["room_revision_pending_text"] = "\n".join(
                    dict.fromkeys([text for text in [pending_text, new_text] if text])
                )
                save_session(session_id, state)
            return JSONResponse({
                "status": "planning",
                "session_id": session_id,
                "room_id": public_room_id(session_id),
                "participants": room_participants(state),
                "message": (
                    "助手已经在生成方案中。本条意见已记录，当前版本完成后会自动继续生成下一版。"
                    if not agent_mentioned
                    else "助手已经在生成方案中，请等待当前版本完成。"
                ),
                "last_event_id": int((session_store.get(session_id) or {}).get("__next_event_id", 1)) - 1,
            })

        # 首版方案生成前保留自由讨论；首版方案生成后，新的群聊意见直接进入下一版规划。
        # 明确的满意短句不触发重算，成员仍可使用方案卡片中的确认按钮完成逐人确认。
        room_revision_requested = has_generated_plan and not agent_mentioned and not is_satisfied_feedback(req.user_input)
        if not agent_mentioned and not room_revision_requested:
            return JSONResponse({
                "status": "group_chat",
                "session_id": session_id,
                "room_id": public_room_id(session_id),
                "participants": room_participants(state),
                "last_event_id": int((session_store.get(session_id) or {}).get("__next_event_id", 1)) - 1,
            })

        if agent_mentioned or room_revision_requested:
            discussion = room_discussion_from_events(state)
            if not discussion.get("members"):
                return json_event_response(session_id, {
                    "status": "group_chat",
                    "session_id": session_id,
                    "room_id": public_room_id(session_id),
                    "participants": room_participants(state),
                    "message": "还没有可汇总的群友需求。请大家先自由讨论，表达想去的地点、预算、时间、交通和限制；讨论结束后再发送“@Agent 开始规划”。",
                }, req.client_id or "")
            summary_input = (
                f"{build_group_discussion_summary(discussion)}\n"
                "群友讨论已结束，请基于以上按聊天顺序汇总的需求生成方案。"
            )
            discussion_messages = discussion.get("messages") or []
            if not has_generated_plan:
                # First collaboration plan: apply each room message in chronological
                # order so later clarifications replace earlier values field by field.
                for item in discussion_messages:
                    state = agent_workflow.merge_latest_requirement_changes(state, str(item.get("text") or ""))
            state["planning_in_progress"] = True
            state["planning_failed"] = False
            state["planning_error"] = ""
            plan_revision = int(state.get("plan_revision") or 0) + 1
            state["plan_revision"] = plan_revision
            state["room_confirmations"] = {}
            state["force_regenerate"] = plan_revision > 1
            state["requirement_revision_mode"] = plan_revision > 1
            state["route_variant_seed"] = f"{session_id}:revision:{plan_revision}:{time.time_ns()}"
            if plan_revision > 1:
                summary_input += (
                    f"\n这是协作房间第 {plan_revision} 版方案。请优先吸收上一版方案之后新增的群聊修改意见，"
                    "并生成更新后的路线，不要直接复用上一版结果。"
                )
            latest_revision_input = str(req.latest_requirement_input or req.user_input or "").strip()
            if room_revision_requested or queued_revision_requested:
                summary_input += (
                    f"\n本轮由群友“{req.speaker or '群友'}”的新意见触发。"
                    + (f"最新修改意见是：{latest_revision_input}" if latest_revision_input else "请合并规划期间收到的新增群聊意见。")
                )
            for runtime_key in [
                "final_plan", "structured_plan", "route_plan", "route_distance_info",
                "route_map", "coupon_info", "reservation_options", "validation_report",
                "feasibility_report", "exception", "exception_events",
            ]:
                state[runtime_key] = None
            state["awaiting_satisfaction"] = False
            state["room_revision_pending"] = False
            state.pop("room_revision_pending_client_id", None)
            state.pop("room_revision_pending_speaker", None)
            state.pop("room_revision_pending_text", None)
            state["group_discussion"] = discussion
            state["user_input"] = summary_input
            latest_requirement_input = (
                latest_revision_input
                if room_revision_requested or queued_revision_requested
                else str((discussion_messages[-1] if discussion_messages else {}).get("text") or summary_input)
            )
            state["latest_user_input"] = latest_requirement_input
            existing_people = safe_int((state.get("collected_info") or {}).get("num_people"), 0)
            state["collected_info"] = {
                **(state.get("collected_info") or {}),
                "num_people": max(2, len(room_participants(state)), existing_people),
            }
            state["info_followup_asked"] = True
            state["suppress_info_followup"] = True
            state = agent_workflow.merge_latest_requirement_changes(state, latest_requirement_input)
            state = enforce_explicit_time_hints(state, latest_requirement_input)
            save_session(session_id, state)
            append_event(
                session_id,
                "bot",
                (
                    f"已收到新的群聊修改意见。正在重新识别需求并生成第 {plan_revision} 版方案。"
                    if room_revision_requested or queued_revision_requested
                    else "已收到 @Agent 指令。我会按聊天顺序汇总每位群友的需求，并开始生成方案。"
                ),
                payload={
                    "status": "planning_started",
                    "participants": room_participants(state),
                    "plan_revision": plan_revision,
                    "trigger": "room_revision" if room_revision_requested or queued_revision_requested else "agent_mention",
                },
            )
            planning_req = req.model_copy(update={
                "user_input": summary_input,
                "latest_requirement_input": latest_requirement_input,
            })
            async with get_session_lock(session_id):
                response = await _plan_impl(planning_req, session_id, append_user_event=False)
            schedule_pending_room_revision(session_id)
            return response
    async with get_session_lock(session_id):
        return await _plan_impl(req, session_id)


async def _plan_impl(req: PlanRequest, session_id: str, append_user_event: bool = True):
    """
    真正执行规划的内部函数，支持多轮信息收集：
    - status="plan_ready" → 规划完成，前端展示 plan
    - 信息收集、意图识别和方案生成最终都交给 agent_workflow_improved.py
    当前产品策略：用户未提供的信息由默认值补齐，正常不再返回 need_info。
    """
    cleanup_sessions()
    request_started_at = time.perf_counter()
    # ── Step 1: 恢复或初始化 session ──────────────────────────

    if session_id and session_id in session_store:
        # 多轮对话：追加用户新输入到历史输入
        state = dict(session_store[session_id])
        latest_requirement_input = str(req.latest_requirement_input or req.user_input or "").strip()
        if state.get("awaiting_satisfaction") or int(state.get("plan_revision") or 0) > 1:
            state["requirement_revision_mode"] = True
        state["latest_user_input"] = latest_requirement_input
        state = agent_workflow.merge_latest_requirement_changes(state, latest_requirement_input)
        save_session(session_id, purge_transient_anchor_exclusions(dict(state)))
        if append_user_event:
            append_event(session_id, "user", req.user_input, client_id=req.client_id or "", speaker=req.speaker or "")
        print(f"🔄 Session [{session_id}] 继续对话")

        if state.get("awaiting_departure_confirmation"):
            if is_departure_confirm_text(req.user_input):
                state["__ended"] = True
                state["awaiting_departure_confirmation"] = False
                save_session(session_id, state)
                return json_event_response(session_id, {
                    "status": "trip_ready",
                    "session_id": session_id,
                    "message": "已确认预约信息。祝你们出发顺利，记得出行前再核验一次实时营业、余位和天气。"
                }, req.client_id or "")
            save_session(session_id, state)
            return json_event_response(session_id, {
                "status": "reservation_pending",
                "session_id": session_id,
                "message": (
                    f"已收到你的补充：{req.user_input}\n"
                    "如果需要调整方案，请告诉我具体要改哪里；如果确认出发，请回复“那就出发”或“就这样”。"
                )
            }, req.client_id or "")

        # 方案已生成后，下一轮输入先视为满意度反馈。
        if state.get("awaiting_satisfaction"):
            if is_satisfied_feedback(latest_requirement_input):
                state["awaiting_satisfaction"] = False
                state["confirmed"] = None
                save_session(session_id, state)
                return json_event_response(session_id, {
                    "status": "satisfied",
                    "session_id": session_id,
                    "message": "好的，已记录你对当前方案满意。如需预订，请点击方案下方具体地点的预订按钮。"
                }, req.client_id or "")

            previous_places = structured_schedule_places(state)
            if previous_places:
                locked_keys = {
                    agent_workflow.normalize_place_text(place)
                    for place in (state.get("locked_places") or [])
                }
                avoid_candidates = [
                    place for place in previous_places
                    if agent_workflow.normalize_place_text(place) not in locked_keys
                ]
                state["avoid_places"] = list(dict.fromkeys((state.get("avoid_places") or []) + avoid_candidates))
                state.setdefault("previous_plan_places", []).append(previous_places)
            ###
            quick_modes = detect_quick_adjustments(latest_requirement_input)
            regenerate_requested = is_regenerate_request(latest_requirement_input)

            state["latest_user_input"] = latest_requirement_input

            if quick_modes:
                state["adjustment_modes"] = quick_modes
                state["adjustment_mode"] = quick_modes[0]
                state["user_input"] = latest_requirement_input
            elif regenerate_requested:
                state["adjustment_modes"] = ["regenerate"]
                state["adjustment_mode"] = "regenerate"
                state["force_regenerate"] = True
                state["user_input"] = (
                    "按当前已收集的最新出发地、目的地/区域锚点、人数、时间、预算和偏好重新生成一版不同路线；"
                    "不要原样复用上一版非锁定地点。"
                )
            else:
                state["adjustment_modes"] = []
                state["adjustment_mode"] = None
                state.pop("force_regenerate", None)
                state["user_input"] = latest_requirement_input

            state["final_plan"] = None
            state["structured_plan"] = None
            state["weather_info"] = None
            state["route_distance_info"] = None
            state["route_map"] = None
            state["coupon_info"] = None
            state["reservation_options"] = None
            state["awaiting_satisfaction"] = False
            state["revision_count"] = int(state.get("revision_count") or 0) + 1
            # 当前产品不做缺项追问；后续修改缺的软字段直接沿用旧值或默认值。
            state["suppress_info_followup"] = True
            state["info_followup_asked"] = True
            print(f"🔁 用户不满意，开始第 {state['revision_count']} 次调整方案")
        else:
            state["latest_user_input"] = latest_requirement_input
            state["user_input"] = req.user_input
    else:
        # 首次请求：新建 session
        state = agent_workflow.AgentState(
            user_input=req.user_input,
            collected_info={},
            info_complete=False,
            pending_question=None,
            intent=None,
            weather_info=None,
            rag_context=None,
            attraction_info=None,
            ticket_info=None,
            route_plan=None,
            route_distance_info=None,
            route_map=None,
            coupon_info=None,
            structured_plan=None,
            validation_report=None,
            reservation_options=None,
            exception=None,
            final_plan=None,
            confirmed=None,
            order_result=None,
            awaiting_satisfaction=False,
            revision_count=0,
            group_discussion=None,
            adjustment_mode=None,
            adjustment_modes=[],
            avoid_places=[],
            previous_plan_places=[],
            locked_places=[],
            exception_events=[],
            latest_user_input=req.latest_requirement_input or req.user_input,
            fixed_departure="",
            fixed_destination="",
            active_destination_anchor="",
            node_timings={},
            awaiting_departure_confirmation=False,
            info_followup_asked=False,
            suppress_info_followup=False,
            coupon_reservations=[]
        )
        save_session(session_id, state)
        append_event(session_id, "user", req.user_input, client_id=req.client_id or "", speaker=req.speaker or "")
        print(f"🆕 新建 Session [{session_id}]")

    # ── Step 2: 执行信息收集（异步非阻塞）─────────────────────
    loop = asyncio.get_event_loop()
    try:
        info_started = time.perf_counter()
        # 30s 版本默认使用 fast_collect：只做必要字段和“餐饮/景点”二分类；
        # 复杂语义可通过 FAST_COLLECT_REQUIRED_INFO=0 切回原 LLM 收集链。
        collector = (
            agent_workflow.fast_collect_required_info_for_api
            if os.getenv("FAST_COLLECT_REQUIRED_INFO", "1") == "1"
            and hasattr(agent_workflow, "fast_collect_required_info_for_api")
            else agent_workflow.collect_required_info_for_api
        )
        new_state = await loop.run_in_executor(
            GLOBAL_EXECUTOR,
            collector,
            state
        )
        info_elapsed = time.perf_counter() - info_started
        print(f"⏱️ 工具调用耗时 [{collector.__name__}]: {info_elapsed:.2f}s")
        latest_requirement_input = str(req.latest_requirement_input or req.user_input or "").strip()
        new_state["latest_user_input"] = latest_requirement_input
        if new_state.get("_latest_requirement_merged_text") != latest_requirement_input:
            new_state = agent_workflow.merge_latest_requirement_changes(new_state, latest_requirement_input)
        collected_after_merge = dict(new_state.get("collected_info") or {})
        if not collected_after_merge.get("ordered_steps"):
            ordered_hint = agent_workflow.extract_ordered_steps_hint_from_user_text(latest_requirement_input)
            if ordered_hint:
                collected_after_merge["ordered_steps"] = ordered_hint
                collected_after_merge["place_keywords"] = [
                    step.get("keyword") for step in ordered_hint if isinstance(step, dict) and step.get("keyword")
                ]
                new_state["collected_info"] = collected_after_merge
        new_state = enforce_explicit_time_hints(new_state, latest_requirement_input)
    except Exception as e:
        error_text = f"信息收集失败: {str(e)}"
        failed_state = dict(session_store.get(session_id) or state or {})
        failed_state["planning_in_progress"] = False
        failed_state["planning_failed"] = True
        failed_state["planning_error"] = error_text
        save_session(session_id, failed_state)
        if str(session_id).startswith("room:"):
            append_event(session_id, "bot", f"规划失败：{error_text}", payload={"status": "planning_failed", "error": error_text})
        return JSONResponse({"error": error_text}, status_code=500)

    # ── Step 3: 信息不齐全 → 保存状态，返回追问给前端 ─────────
    if not new_state.get("info_complete"):
        new_state["planning_in_progress"] = False
        new_state = purge_transient_anchor_exclusions(new_state)
        save_session(session_id, new_state)
        log_generation_time(session_id, time.perf_counter() - request_started_at, "need_info")
        return json_event_response(session_id, {
            "status": "need_info",
            "session_id": session_id,
            "question": new_state.get(
                "pending_question",
                "请补充出发地点、人数、时间和预算信息～😊"
            )
        }, req.client_id or "")

    # ── Step 4: 信息齐全 → 执行完整规划流程 ───────────────────
    if bool(new_state.get("force_regenerate")) or should_vary_route(new_state, req.user_input):
        new_state["route_variant_seed"] = f"{session_id}:{time.time_ns()}:{uuid.uuid4().hex[:8]}"
    else:
        new_state.pop("route_variant_seed", None)

    cache_key = make_plan_cache_key(new_state)
    result = get_cached_plan(cache_key)
    cache_hit = result is not None
    try:
        if result is None:
            result = await loop.run_in_executor(
                GLOBAL_EXECUTOR,
                agent_workflow.plan_workflow.invoke,
                new_state
            )
            result = inherit_room_runtime_state(result, new_state)
            result = attach_budget_estimate(result)
            result = attach_group_considerations(result)
            result = purge_transient_anchor_exclusions(result)
            set_cached_plan(cache_key, result)
        else:
            result = inherit_room_runtime_state(result, new_state)
            result = attach_budget_estimate(result)
            result = attach_group_considerations(result)
            result = purge_transient_anchor_exclusions(result)
    except Exception as e:
        error_text = f"规划失败: {str(e)}"
        failed_state = dict(session_store.get(session_id) or new_state or {})
        failed_state["planning_in_progress"] = False
        failed_state["planning_failed"] = True
        failed_state["planning_error"] = error_text
        save_session(session_id, failed_state)
        if str(session_id).startswith("room:"):
            append_event(session_id, "bot", f"规划失败：{error_text}", payload={"status": "planning_failed", "error": error_text})
        return JSONResponse({"error": error_text}, status_code=500)

    
    actual_generation_seconds = time.perf_counter() - request_started_at

    # 只改统计口径：方案生成展示时间不计入高德路线/地图耗时。
    # 注意：缓存命中时，本轮没有重新调用高德路线/地图，所以不能扣历史缓存里的高德耗时。
    excluded_amap_route_seconds = 0.0
    raw_route_segments_seconds = 0.0
    raw_route_map_seconds = 0.0

    if not cache_hit:
        try:
            raw_route_segments_seconds = float(result.get("route_segments_generation_seconds") or 0)
        except (TypeError, ValueError):
            raw_route_segments_seconds = 0.0

        try:
            raw_route_map_seconds = float(
                result.get("route_map_generation_seconds")
                or result.get("map_generation_seconds")
                or 0
            )
        except (TypeError, ValueError):
            raw_route_map_seconds = 0.0

        try:
            # 新字段优先：它已经等于“高德路线 + 地图”总耗时，避免重复相加。
            excluded_amap_route_seconds = float(
                result.get("amap_route_generation_seconds")
                or (raw_route_segments_seconds + raw_route_map_seconds)
                or 0
            )
        except (TypeError, ValueError):
            excluded_amap_route_seconds = 0.0

    generation_seconds = max(0.0, actual_generation_seconds - excluded_amap_route_seconds)

    log_generation_time(session_id, generation_seconds, "plan_ready", cache_hit=cache_hit)

    if excluded_amap_route_seconds > 0:
        print(
            f"🗺️ 方案耗时统计已排除高德路线/地图生成: "
            f"实际 {actual_generation_seconds:.2f}s - 高德路线/地图 {excluded_amap_route_seconds:.2f}s "
            f"= 展示 {generation_seconds:.2f}s"
        )
        print(f"   ├─ 高德路线距离计算: {raw_route_segments_seconds:.2f}s")
        print(f"   └─ 高德地图数据生成: {raw_route_map_seconds:.2f}s")

    result["awaiting_satisfaction"] = True
    result["revision_count"] = int(result.get("revision_count") or new_state.get("revision_count") or 0)

    # 前端展示用：扣除高德路线/地图后的方案生成时间
    result["generation_time_seconds"] = round(generation_seconds, 2)

    # 调试用：真实请求总耗时和被排除耗时
    result["generation_time_actual_seconds"] = round(actual_generation_seconds, 2)
    result["generation_time_excluded_seconds"] = round(excluded_amap_route_seconds, 2)
    result["amap_route_generation_seconds"] = round(excluded_amap_route_seconds, 2)
    result["route_segments_generation_seconds"] = round(raw_route_segments_seconds, 2)
    result["route_map_generation_seconds"] = round(raw_route_map_seconds, 2)

    result["generation_time_over_limit"] = generation_seconds > PLAN_TIME_LIMIT_SECONDS
    
    
    result["planning_in_progress"] = False
    result["planning_failed"] = False
    result["planning_error"] = ""
    result = inherit_room_runtime_state(result, new_state)
    result.pop("force_regenerate", None)
    result = purge_transient_anchor_exclusions(result)
    memory_mb = current_process_memory_mb()
    if memory_mb is not None:
        result["process_memory_mb"] = round(memory_mb, 2)
        print(f"🧠 方案生成后 Python 进程内存 [{session_id}]: {memory_mb:.2f} MB")
    else:
        print(f"🧠 方案生成后 Python 进程内存 [{session_id}]: 当前环境无法读取")
    save_session(session_id, result)

    return json_event_response(session_id, {
        "status": "plan_ready",
        "session_id": session_id,
        "plan": result.get("final_plan", ""),
        "distance_info": result.get("route_distance_info", ""),
        "structured_plan": result.get("structured_plan", {}),
        "validation_report": result.get("validation_report", {}),
        "feasibility_report": result.get("feasibility_report") or ((result.get("structured_plan") or {}).get("feasibility_report") or {}),
        "coupon_info": result.get("coupon_info", {}),
        "reservation_options": result.get("reservation_options", []),
        "route_map": public_route_map(result.get("route_map", {})),
        "frontend_meta": build_frontend_meta(result),
        "group_considerations": result.get("group_considerations", []),
        "node_timings": result.get("node_timings", {}),
        "generation_time_seconds": result.get("generation_time_seconds"),
        "generation_time_actual_seconds": result.get("generation_time_actual_seconds"),
        "generation_time_excluded_seconds": result.get("generation_time_excluded_seconds"),
        "amap_route_generation_seconds": result.get("amap_route_generation_seconds"),
        "route_segments_generation_seconds": result.get("route_segments_generation_seconds"),
        "route_map_generation_seconds": result.get("route_map_generation_seconds"),
        "generation_time_over_limit": result.get("generation_time_over_limit"),
        "cache_hit": cache_hit,
        "planning_in_progress": False,
        "planning_failed": False,
        "plan_revision": int(result.get("plan_revision") or 0),
        "participants": room_participants(result),
        "room_confirmation": room_confirmation_summary(result),
        "exception": result.get("exception"),
        "exception_events": result.get("exception_events") or ((result.get("structured_plan") or {}).get("route_logic_validation") or {}).get("exception_events") or [],
        "adjustment_conflicts": ((result.get("structured_plan") or {}).get("route_logic_validation") or {}).get("adjustment_conflicts") or [],
        "satisfaction_question": "你对这个方案满意吗？如果满意请回复“满意/可以/就这个”，如果不满意请直接告诉我想怎么改，我会继续调整。"
    }, req.client_id or "")


@app.post("/confirm")
async def confirm(req: ConfirmRequest):
    """预订确认接口"""
    cleanup_sessions()
    state = session_store.get(req.session_id)
    if not state:
        return JSONResponse(
            reservation_service.SESSION_EXPIRED_ERROR,
            status_code=400
        )

    # ✅ 用户取消预订
    if not req.confirmed:
        state["__ended"] = True
        save_session(req.session_id, state)
        return json_event_response(req.session_id, reservation_service.cancel_payload(req.session_id))

    # ✅ 用户确认预订 → 执行下单
    state["confirmed"] = True
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            GLOBAL_EXECUTOR,
            agent_workflow.book_workflow.invoke,
            state
        )
    except Exception as e:
        return JSONResponse({"error": f"预订失败: {str(e)}"}, status_code=500)

    result["__ended"] = True
    save_session(req.session_id, result)
    return json_event_response(req.session_id, reservation_service.booking_result_payload(req.session_id, result))


@app.post("/reserve_coupon")
async def reserve_coupon(req: CouponReserveRequest):
    """点击某张团购券后，生成该店铺/地点的详细预约信息。"""
    cleanup_sessions()
    state = session_store.get(req.session_id)
    if not state:
        return JSONResponse(reservation_service.SESSION_EXPIRED_ERROR, status_code=400)

    allowed_lookup = reservation_service.coupon_lookup(state)
    canonical_place_name = allowed_lookup.get(req.place_name)
    if not canonical_place_name:
        return JSONResponse({
            "error": "该团购券不在当前最终方案中，不能预约。请重新生成方案或选择页面展示的券。"
        }, status_code=400)

    info = reservation_service.mark_reservation_pending(
        state,
        canonical_place_name,
        agent_workflow.build_reservation_info,
    )
    save_session(req.session_id, state)

    return JSONResponse(reservation_service.reservation_pending_payload(
        session_id=req.session_id,
        reservation=info,
        question="请确认预约信息是否满意。满意请回复“那就出发”或“就这样”；不满意请说明要调整哪里。",
    ))


@app.post("/reserve_place")
async def reserve_place(req: PlaceReserveRequest):
    """按最终 structured_plan 中的具体地点生成预约信息。"""
    cleanup_sessions()
    state = session_store.get(req.session_id)
    if not state:
        return JSONResponse(reservation_service.SESSION_EXPIRED_ERROR, status_code=400)

    allowed_lookup = reservation_service.place_lookup(state, agent_workflow.build_reservation_options)
    canonical_place_name = allowed_lookup.get(req.place_name)
    if not canonical_place_name:
        return JSONResponse({
            "error": "该地点不在当前最终方案的可预订地点中，不能预约。请重新生成方案或选择页面展示的预订按钮。"
        }, status_code=400)

    info = reservation_service.mark_reservation_pending(
        state,
        canonical_place_name,
        agent_workflow.build_reservation_info,
    )
    save_session(req.session_id, state)

    return JSONResponse(reservation_service.reservation_pending_payload(
        session_id=req.session_id,
        reservation=info,
        question="请确认以上具体店铺/地点的预约信息是否满意。满意请回复“那就出发”或“就这样”；不满意请说明要调整哪里。",
    ))


@app.get("/ping")
def ping():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app_api:app", host="0.0.0.0", port=int(os.getenv("LOCALMATE_PORT", "8041")), reload=False)
