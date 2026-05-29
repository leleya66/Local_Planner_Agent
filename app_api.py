# app_api.py
import asyncio
import uuid
import re
import time
import copy
import hashlib
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import agent_workflow_improved as agent_workflow

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
session_store: dict = {}
session_locks: dict = {}
plan_result_cache: dict = {}
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
GLOBAL_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.getenv("AGENT_WORKERS", "4")))
PLAN_TIME_LIMIT_SECONDS = float(os.getenv("PLAN_TIME_LIMIT_SECONDS", "30"))
PLAN_CACHE_TTL_SECONDS = int(os.getenv("PLAN_CACHE_TTL_SECONDS", "600"))
PLAN_CACHE_MAX_ITEMS = int(os.getenv("PLAN_CACHE_MAX_ITEMS", "32"))
FRONTEND_FILE = os.getenv("LOCALMATE_FRONTEND_FILE", "shanghai_agent_1440_v8_live.html")


def now_ts() -> float:
    return time.time()


def cleanup_sessions() -> None:
    cutoff = now_ts() - SESSION_TTL_SECONDS
    expired = [
        session_id for session_id, state in session_store.items()
        if float((state or {}).get("__updated_at", now_ts())) < cutoff
    ]
    for session_id in expired:
        session_store.pop(session_id, None)
        session_locks.pop(session_id, None)

    cache_cutoff = now_ts() - PLAN_CACHE_TTL_SECONDS
    for key, item in list(plan_result_cache.items()):
        if float(item.get("created_at", 0)) < cache_cutoff:
            plan_result_cache.pop(key, None)


def make_plan_cache_key(state: dict) -> str:
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


def should_vary_route(state: dict, latest_text: str = "") -> bool:
    """Enable controlled route variation only for open-ended route requests."""
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
    item = plan_result_cache.get(cache_key)
    if not item:
        return None
    if now_ts() - float(item.get("created_at", 0)) > PLAN_CACHE_TTL_SECONDS:
        plan_result_cache.pop(cache_key, None)
        return None
    cached = copy.deepcopy(item.get("result") or {})
    cached["cache_hit"] = True
    return cached


def set_cached_plan(cache_key: str, result: dict) -> None:
    if not cache_key or not result:
        return
    while len(plan_result_cache) >= PLAN_CACHE_MAX_ITEMS:
        oldest_key = min(plan_result_cache, key=lambda key: plan_result_cache[key].get("created_at", 0))
        plan_result_cache.pop(oldest_key, None)
    cached_result = copy.deepcopy(result)
    for runtime_key in ["__events", "__next_event_id", "__updated_at"]:
        cached_result.pop(runtime_key, None)
    plan_result_cache[cache_key] = {
        "created_at": now_ts(),
        "result": cached_result,
    }


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
    previous = session_store.get(session_id) or {}
    if "__events" in previous and "__events" not in state:
        state["__events"] = previous.get("__events", [])
    if "__next_event_id" in previous and "__next_event_id" not in state:
        state["__next_event_id"] = previous.get("__next_event_id", 1)
    state["__updated_at"] = now_ts()
    session_store[session_id] = state


def normalize_room_id(room_id: Optional[str]) -> Optional[str]:
    room = re.sub(r"[^a-zA-Z0-9_-]", "", str(room_id or "").strip())
    if not room:
        return None
    return f"room:{room[:40]}"


def get_session_lock(session_id: str) -> asyncio.Lock:
    lock = session_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        session_locks[session_id] = lock
    return lock


def append_event(session_id: str, role: str, text: str, client_id: str = "", speaker: str = "", payload: Optional[dict] = None) -> None:
    state = session_store.get(session_id)
    if state is None:
        state = {}
    events = state.setdefault("__events", [])
    next_id = int(state.get("__next_event_id", 1) or 1)
    events.append({
        "id": next_id,
        "ts": now_ts(),
        "role": role,
        "text": text or "",
        "client_id": client_id or "",
        "speaker": speaker or "",
        "payload": payload or {},
    })
    state["__next_event_id"] = next_id + 1
    state["__events"] = events[-200:]
    save_session(session_id, state)


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
        payload={k: v for k, v in payload.items() if k not in {"plan", "route_map"}},
    )
    payload["last_event_id"] = int((session_store.get(session_id) or {}).get("__next_event_id", 1)) - 1
    return JSONResponse(payload)


def public_route_map(route_map: dict) -> dict:
    public = dict(route_map or {})
    public.pop("amap_url", None)
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
    hard["budget"] = hard["budget_estimate_text"]
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
        "title": " + ".join(names[:2]) if names else "上海周末出行规划 Agent",
        "date": hard.get("date") or "本周末",
        "time_period": hard.get("time_period") or "半日",
        "stops": len(schedule),
        "duration_hours": hard.get("duration_hours") or 5,
        "budget": hard.get("budget_estimate_text") or hard.get("budget") or "预算待定",
        "requested_budget": hard.get("requested_budget") or "",
        "budget_estimate": hard.get("budget_estimate") or {},
        "route_logic_mode": hard.get("route_logic_mode") or "",
        "adjustment_modes": hard.get("adjustment_modes") or [],
    }


class PlanRequest(BaseModel):
    user_input: str
    session_id: Optional[str] = None
    room_id: Optional[str] = None
    client_id: Optional[str] = None
    speaker: Optional[str] = None
    enable_group_discussion: Optional[bool] = False


class ConfirmRequest(BaseModel):
    session_id: str
    confirmed: bool


class CouponReserveRequest(BaseModel):
    session_id: str
    place_name: str


class PlaceReserveRequest(BaseModel):
    session_id: str
    place_name: str


def is_satisfied_feedback(text: str) -> bool:
    """判断用户是否明确满意。保守处理：包含不满意/修改诉求时不算满意。"""
    normalized = (text or "").strip().lower()
    negative_patterns = [
        "不满意", "不太满意", "不行", "不好", "不喜欢", "换", "重新",
        "调整", "修改", "太远", "太贵", "太累", "不要", "还有", "但是",
        "能不能", "可以再", "希望", "想要", "不合适"
    ]
    positive_patterns = [
        "满意", "可以", "挺好", "很好", "好", "ok", "okay", "没问题",
        "就这个", "确认", "不错", "行"
    ]
    has_negative = any(p in normalized for p in negative_patterns)
    has_positive = any(p in normalized for p in positive_patterns)
    return has_positive and not has_negative


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


def wants_group_discussion(text: str) -> bool:
    normalized = (text or "").strip().replace(" ", "")
    patterns = ["开启多人", "多人偏好", "多人讨论", "一起讨论", "让大家说", "收集大家", "收集偏好"]
    return any(pattern in normalized for pattern in patterns)


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


def discussion_member_label(index: int) -> str:
    labels = ["A", "B", "C", "D", "E", "F"]
    return labels[index] if index < len(labels) else f"成员{index + 1}"


def discussion_member_options(count: int) -> list[str]:
    count = max(2, min(int(count or 2), 5))
    return [discussion_member_label(index) for index in range(count)]


def need_group_discussion(state: dict, enabled: bool = False) -> bool:
    """用户勾选多人讨论时，两人及以上启动 A/B/C/D/E 身份收集。"""
    if not enabled:
        return False
    collected = state.get("collected_info") or {}
    return safe_int(collected.get("num_people"), 1) >= 2


def is_group_decision_text(text: str) -> bool:
    normalized = (text or "").strip().lower().replace(" ", "")
    decision_patterns = [
        "让agent决定", "让ai决定", "让你决定", "你来决定", "你决定吧",
        "agent决定", "就这样吧", "就这样", "可以生成", "开始生成",
        "决定吧", "定了", "按这个来", "让系统决定"
    ]
    return any(pattern in normalized for pattern in decision_patterns)


def build_group_discussion_summary(discussion: dict) -> str:
    members = discussion.get("members") or []
    notes = discussion.get("notes") or []
    lines = ["多人讨论记录："]
    for member in members:
        lines.append(f"- {member.get('label')}: {member.get('preference')}")
    for note in notes:
        lines.append(f"- 补充讨论: {note}")
    lines.append("请综合所有成员需求，兼顾预算、4-6小时总时长、路线距离、天气和团购券。")
    return "\n".join(lines)


def start_group_discussion_if_needed(state: dict, session_id: str, enabled: bool = False, force_new: bool = False, client_id: str = ""):
    if not need_group_discussion(state, enabled):
        return None
    discussion = state.get("group_discussion") or {}
    if discussion.get("complete") and not force_new:
        return None
    if discussion.get("active"):
        return None

    count = max(2, min(safe_int((state.get("collected_info") or {}).get("num_people"), 2), 5))
    discussion = {
        "active": True,
        "complete": False,
        "required_count": count,
        "current_index": 0,
        "members": [],
        "notes": [],
    }
    state["group_discussion"] = discussion
    save_session(session_id, state)
    label = discussion_member_label(0)
    payload = {
        "status": "need_group_discussion",
        "session_id": session_id,
        "member_options": discussion_member_options(count),
        "question": (
            f"已进入多人讨论模式。请先在输入框左侧选择{label}身份，然后说一下自己的偏好：想玩什么、不能接受什么、"
            "预算或距离上有什么要求？"
        )
    }
    return json_event_response(session_id, payload, client_id)


def next_missing_member_label(discussion: dict) -> Optional[str]:
    required_count = int(discussion.get("required_count") or 0)
    existing = {member.get("label") for member in discussion.get("members") or []}
    for index in range(required_count):
        label = discussion_member_label(index)
        if label not in existing:
            return label
    return None


def upsert_member_preference(discussion: dict, label: str, text: str):
    members = discussion.setdefault("members", [])
    for member in members:
        if member.get("label") == label:
            member["preference"] = text.strip()
            return
    members.append({"label": label, "preference": text.strip()})


def handle_group_discussion_input(state: dict, session_id: str, user_text: str, speaker: Optional[str] = None, client_id: str = ""):
    discussion = state.get("group_discussion") or {}
    if not discussion.get("active") or discussion.get("complete"):
        return None

    required_count = int(discussion.get("required_count") or 0)
    valid_labels = {discussion_member_label(i) for i in range(required_count)}
    speaker = (speaker or "").strip().upper()

    if len(discussion.get("members") or []) < required_count:
        label = speaker if speaker in valid_labels else next_missing_member_label(discussion)
        if not label:
            label = discussion_member_label(len(discussion.get("members") or []))
        upsert_member_preference(discussion, label, user_text)
        discussion["current_index"] = min(len(discussion.get("members") or []), required_count)
        state["group_discussion"] = discussion

        next_label = next_missing_member_label(discussion)
        if next_label:
            save_session(session_id, state)
            payload = {
                "status": "need_group_discussion",
                "session_id": session_id,
                "member_options": discussion_member_options(required_count),
                "question": f"已记录{label}的需求。现在请{next_label}选择自己的身份并说一下偏好和限制。"
            }
            return json_event_response(session_id, payload, client_id)

        save_session(session_id, state)
        payload = {
            "status": "need_group_discussion",
            "session_id": session_id,
            "member_options": discussion_member_options(required_count),
            "question": (
                "所有成员的基础偏好都记录好了。你们可以继续补充讨论，例如谁更看重距离、预算、室内外、吃饭。"
                "如果讨论结束，请回复“让Agent决定”或“就这样吧”，我再综合大家需求生成方案。"
            )
        }
        return json_event_response(session_id, payload, client_id)

    if is_group_decision_text(user_text):
        discussion["complete"] = True
        discussion["active"] = False
        state["group_discussion"] = discussion
        state["user_input"] = (
            f"{state.get('user_input', '')}\n"
            f"{build_group_discussion_summary(discussion)}\n"
            f"用户已确认讨论结束：{user_text}"
        )
        return None

    discussion.setdefault("notes", []).append(user_text.strip())
    state["group_discussion"] = discussion
    save_session(session_id, state)
    payload = {
        "status": "need_group_discussion",
        "session_id": session_id,
        "member_options": discussion_member_options(required_count),
        "question": (
            "已记录这条补充意见。还可以继续讨论；如果希望我开始权衡大家需求并生成方案，"
            "请回复“让Agent决定”或“就这样吧”。"
        )
    }
    return json_event_response(session_id, payload, client_id)


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
    """给多端共享链接做轻量状态读取；不包含完整内部链路。"""
    cleanup_sessions()
    state = session_store.get(session_id)
    if not state:
        return JSONResponse({"exists": False, "session_id": session_id})
    return JSONResponse({
        "exists": True,
        "session_id": session_id,
        "ended": bool(state.get("__ended")),
        "awaiting_satisfaction": bool(state.get("awaiting_satisfaction")),
        "group_discussion": state.get("group_discussion") or {},
        "structured_plan": state.get("structured_plan") or {},
        "validation_report": state.get("validation_report") or {},
        "feasibility_report": state.get("feasibility_report") or ((state.get("structured_plan") or {}).get("feasibility_report") or {}),
        "final_plan": state.get("final_plan") or "",
    })


@app.get("/events/{session_id}")
async def get_session_events(session_id: str, since: int = 0):
    """多端协作轮询事件流。客户端每隔1-2秒按 last_event_id 拉取增量。"""
    cleanup_sessions()
    state = session_store.get(session_id)
    if not state:
        return JSONResponse({"exists": False, "session_id": session_id, "events": [], "last_event_id": since})
    events = [
        event for event in (state.get("__events") or [])
        if int(event.get("id", 0)) > int(since or 0)
    ]
    last_event_id = int(state.get("__next_event_id", 1)) - 1
    return JSONResponse({
        "exists": True,
        "session_id": session_id,
        "ended": bool(state.get("__ended")),
        "events": events,
        "last_event_id": last_event_id,
    })


@app.get("/route_map/{session_id}")
async def route_map(session_id: str):
    """Proxy Amap static map image so the frontend does not expose the API key."""
    started = time.perf_counter()
    cleanup_sessions()
    state = session_store.get(session_id)
    route_map_info = (state or {}).get("route_map") or {}
    if route_map_info.get("available") and not route_map_info.get("amap_url"):
        route_map_info = agent_workflow.build_route_map_info((state or {}).get("structured_plan") or {})
        if state is not None:
            state["route_map"] = route_map_info
            save_session(session_id, state)
    amap_url = route_map_info.get("amap_url")
    if not amap_url:
        return JSONResponse({"error": route_map_info.get("reason") or "当前会话没有可用路线地图"}, status_code=404)
    try:
        with urllib.request.urlopen(amap_url, timeout=10) as resp:
            content = resp.read()
            content_type = resp.headers.get("Content-Type") or "image/png"
        print(f"⏱️ 地图接口耗时 [{session_id}]: {time.perf_counter() - started:.2f}s")
        return Response(content=content, media_type=content_type)
    except (urllib.error.URLError, TimeoutError) as exc:
        return JSONResponse({"error": f"路线地图获取失败: {exc}"}, status_code=502)


@app.post("/plan")
async def plan(req: PlanRequest):
    cleanup_sessions()
    session_id = req.session_id or normalize_room_id(req.room_id) or str(uuid.uuid4())
    async with get_session_lock(session_id):
        return await _plan_impl(req, session_id)


async def _plan_impl(req: PlanRequest, session_id: str):
    """
    规划接口，支持多轮信息收集：
    - status="need_info"  → 信息不完整，前端展示 question 等待用户补充
    - status="plan_ready" → 规划完成，前端展示 plan
    """
    cleanup_sessions()
    request_started_at = time.perf_counter()
    # ── Step 1: 恢复或初始化 session ──────────────────────────

    if session_id and session_id in session_store:
        # 多轮对话：追加用户新输入到历史输入
        state = dict(session_store[session_id])
        state["latest_user_input"] = req.user_input
        state = update_locked_places_from_state(state, req.user_input)
        save_session(session_id, purge_transient_anchor_exclusions(dict(state)))
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
            if bool(req.enable_group_discussion) or wants_group_discussion(req.user_input):
                state["awaiting_departure_confirmation"] = False
                discussion_response = start_group_discussion_if_needed(state, session_id, True, force_new=True, client_id=req.client_id or "")
                if discussion_response is not None:
                    return discussion_response
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
            if bool(req.enable_group_discussion) or wants_group_discussion(req.user_input):
                state["awaiting_satisfaction"] = False
                discussion_response = start_group_discussion_if_needed(state, session_id, True, force_new=True, client_id=req.client_id or "")
                if discussion_response is not None:
                    return discussion_response

            if is_satisfied_feedback(req.user_input):
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

            quick_modes = detect_quick_adjustments(req.user_input)
            regenerate_requested = is_regenerate_request(req.user_input)
            if quick_modes:
                state["adjustment_modes"] = quick_modes
                state["adjustment_mode"] = quick_modes[0]
                state["user_input"] = build_quick_revision_input(state.get("user_input", ""), quick_modes)
            elif regenerate_requested:
                state["adjustment_modes"] = ["regenerate"]
                state["adjustment_mode"] = "regenerate"
                state["force_regenerate"] = True
                state["user_input"] = build_revision_input(
                    state.get("user_input", ""),
                    "按当前固定的最新出发地、目的地、人数、时间和预算重新生成一版不同路线；不要原样复用上一版非锁定地点。"
                )
            else:
                state["adjustment_modes"] = []
                state["adjustment_mode"] = None
                state.pop("force_regenerate", None)
                state["user_input"] = build_revision_input(state.get("user_input", ""), req.user_input)
            state["final_plan"] = None
            state["structured_plan"] = None
            state["weather_info"] = None
            state["route_distance_info"] = None
            state["route_map"] = None
            state["coupon_info"] = None
            state["reservation_options"] = None
            state["awaiting_satisfaction"] = False
            state["revision_count"] = int(state.get("revision_count") or 0) + 1
            print(f"🔁 用户不满意，开始第 {state['revision_count']} 次调整方案")
        else:
            discussion_response = handle_group_discussion_input(state, session_id, req.user_input, req.speaker, client_id=req.client_id or "")
            if discussion_response is not None:
                return discussion_response
            if bool(req.enable_group_discussion) or wants_group_discussion(req.user_input):
                discussion_response = start_group_discussion_if_needed(state, session_id, True, force_new=True, client_id=req.client_id or "")
                if discussion_response is not None:
                    return discussion_response
            state["user_input"] = f"{state['user_input']} {req.user_input}"
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
            latest_user_input=req.user_input,
            fixed_departure="",
            fixed_destination="",
            active_destination_anchor="",
            node_timings={},
            awaiting_departure_confirmation=False,
            info_followup_asked=False,
            coupon_reservations=[]
        )
        save_session(session_id, state)
        append_event(session_id, "user", req.user_input, client_id=req.client_id or "", speaker=req.speaker or "")
        print(f"🆕 新建 Session [{session_id}]")

    # ── Step 2: 执行信息收集（异步非阻塞）─────────────────────
    loop = asyncio.get_event_loop()
    try:
        info_started = time.perf_counter()
        new_state = await loop.run_in_executor(
            GLOBAL_EXECUTOR,
            agent_workflow.collect_required_info_for_api,
            state
        )
        info_elapsed = time.perf_counter() - info_started
        print(f"⏱️ 工具调用耗时 [collect_required_info_for_api]: {info_elapsed:.2f}s")
        new_state["latest_user_input"] = req.user_input
        new_state = update_locked_places_from_state(new_state, req.user_input)
    except Exception as e:
        return JSONResponse({"error": f"信息收集失败: {str(e)}"}, status_code=500)

    # ── Step 3: 信息不齐全 → 保存状态，返回追问给前端 ─────────
    if not new_state.get("info_complete"):
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

    discussion_response = start_group_discussion_if_needed(new_state, session_id, bool(req.enable_group_discussion), client_id=req.client_id or "")
    if discussion_response is not None:
        return discussion_response

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
            result = attach_budget_estimate(result)
            result = purge_transient_anchor_exclusions(result)
            set_cached_plan(cache_key, result)
        else:
            result = attach_budget_estimate(result)
            result = purge_transient_anchor_exclusions(result)
    except Exception as e:
        return JSONResponse({"error": f"规划失败: {str(e)}"}, status_code=500)

    generation_seconds = time.perf_counter() - request_started_at
    log_generation_time(session_id, generation_seconds, "plan_ready", cache_hit=cache_hit)
    result["awaiting_satisfaction"] = True
    result["revision_count"] = int(result.get("revision_count") or new_state.get("revision_count") or 0)
    result["generation_time_seconds"] = round(generation_seconds, 2)
    result["generation_time_over_limit"] = generation_seconds > PLAN_TIME_LIMIT_SECONDS
    result.pop("force_regenerate", None)
    result = purge_transient_anchor_exclusions(result)
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
        "node_timings": result.get("node_timings", {}),
        "generation_time_seconds": result.get("generation_time_seconds"),
        "generation_time_over_limit": result.get("generation_time_over_limit"),
        "cache_hit": cache_hit,
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
            {"error": "Session 已过期，请重新生成方案"},
            status_code=400
        )

    # ✅ 用户取消预订
    if not req.confirmed:
        state["__ended"] = True
        save_session(req.session_id, state)
        return json_event_response(req.session_id, {
            "status": "cancelled",
            "session_id": req.session_id,
            "order_result": "已取消预订，如需重新规划请重新输入需求。"
        })

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
    return json_event_response(req.session_id, {
        "status": "ended",
        "session_id": req.session_id,
        "order_result": result.get("order_result", "预订失败，请重试")
    })


@app.post("/reserve_coupon")
async def reserve_coupon(req: CouponReserveRequest):
    """点击某张团购券后，生成该店铺/地点的详细预约信息。"""
    cleanup_sessions()
    state = session_store.get(req.session_id)
    if not state:
        return JSONResponse({"error": "Session 已过期，请重新生成方案"}, status_code=400)

    allowed_items = ((state.get("coupon_info") or {}).get("items") or [])
    allowed_lookup = {}
    for item in allowed_items:
        place_name = item.get("place_name")
        display_name = item.get("display_name")
        if place_name:
            allowed_lookup[place_name] = place_name
        if display_name:
            allowed_lookup[display_name] = place_name
    canonical_place_name = allowed_lookup.get(req.place_name)
    if not canonical_place_name:
        return JSONResponse({
            "error": "该团购券不在当前最终方案中，不能预约。请重新生成方案或选择页面展示的券。"
        }, status_code=400)

    info = agent_workflow.build_reservation_info(canonical_place_name)
    state.setdefault("coupon_reservations", []).append(info)
    state["awaiting_departure_confirmation"] = True
    state["awaiting_satisfaction"] = False
    save_session(req.session_id, state)

    return JSONResponse({
        "status": "reservation_pending",
        "session_id": req.session_id,
        "reservation": info,
        "message": info.get("message", "预约信息已生成，请确认是否出发。"),
        "question": "请确认预约信息是否满意。满意请回复“那就出发”或“就这样”；不满意请说明要调整哪里。"
    })


@app.post("/reserve_place")
async def reserve_place(req: PlaceReserveRequest):
    """按最终 structured_plan 中的具体地点生成预约信息。"""
    cleanup_sessions()
    state = session_store.get(req.session_id)
    if not state:
        return JSONResponse({"error": "Session 已过期，请重新生成方案"}, status_code=400)

    options = state.get("reservation_options") or agent_workflow.build_reservation_options(state.get("structured_plan") or {})
    allowed_lookup = {}
    for item in options:
        place_name = item.get("place_name")
        display_name = item.get("display_name")
        if place_name:
            allowed_lookup[place_name] = place_name
        if display_name:
            allowed_lookup[display_name] = place_name
    canonical_place_name = allowed_lookup.get(req.place_name)
    if not canonical_place_name:
        return JSONResponse({
            "error": "该地点不在当前最终方案的可预订地点中，不能预约。请重新生成方案或选择页面展示的预订按钮。"
        }, status_code=400)

    info = agent_workflow.build_reservation_info(canonical_place_name)
    state.setdefault("coupon_reservations", []).append(info)
    state["awaiting_departure_confirmation"] = True
    state["awaiting_satisfaction"] = False
    save_session(req.session_id, state)

    return JSONResponse({
        "status": "reservation_pending",
        "session_id": req.session_id,
        "reservation": info,
        "message": info.get("message", "预约信息已生成，请确认是否出发。"),
        "question": "请确认以上具体店铺/地点的预约信息是否满意。满意请回复“那就出发”或“就这样”；不满意请说明要调整哪里。"
    })


@app.get("/ping")
def ping():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app_api:app", host="127.0.0.1", port=8041, reload=False)
