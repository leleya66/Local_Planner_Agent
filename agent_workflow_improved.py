import sys

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

# agent_workflow.py
import os
import re
import json
import urllib.parse
import urllib.request
import urllib.error
import time
import random
import hashlib
from datetime import date, datetime, timedelta
from typing import TypedDict, Optional
from dotenv import load_dotenv
import dashscope

from langchain_chroma import Chroma
from langchain_dashscope import DashScopeEmbeddings, ChatDashScope
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from mock_api_improved import search_attraction, check_ticket, plan_route, book_order, _df, add_new_place, _find_place

load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

if not os.getenv("DASHSCOPE_API_KEY"):
    raise ValueError("⚠️ 未找到 DASHSCOPE_API_KEY，请检查 .env 文件")

# ==========================================
# 1. 连接已有 RAG 知识库（延迟初始化）
# ==========================================
embeddings = None
vectorstore = None
retriever = None
llm = None


def init_models():
    global embeddings, vectorstore, retriever, llm
    print("🔄 正在连接本地向量数据库...")
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")
    vectorstore = Chroma(
        collection_name=os.getenv("CHROMA_COLLECTION", "localmate_planning_cases"),
        persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        embedding_function=embeddings
    )
    # MMR：先取更多候选，再挑选多样且相关的 chunk，减少重复片段和噪声。
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": int(os.getenv("RETRIEVER_K", "8")),
            "fetch_k": int(os.getenv("RETRIEVER_FETCH_K", "30")),
            "lambda_mult": 0.45,
        },
    )
    # 规划类输出不需要太高随机性，否则容易编造细节。
    llm = ChatDashScope(model="qwen-plus", temperature=0.35)
    llm.client = dashscope.Generation
    print("✅ 向量数据库连接成功")


# ==========================================
# 2. Agent 状态定义
# ==========================================
class AgentState(TypedDict):
    user_input: str
    collected_info: Optional[dict]  # ✅ 新增
    info_complete: Optional[bool]  # ✅ 新增
    pending_question: Optional[str]  # ✅ 新增
    intent: Optional[dict]
    weather_info: Optional[dict]
    rag_context: Optional[str]
    attraction_info: Optional[str]
    ticket_info: Optional[str]
    route_plan: Optional[str]
    route_distance_info: Optional[str]
    route_map: Optional[dict]
    coupon_info: Optional[dict]
    structured_plan: Optional[dict]
    validation_report: Optional[dict]
    feasibility_report: Optional[dict]
    reservation_options: Optional[list]
    exception: Optional[str]
    final_plan: Optional[str]
    confirmed: Optional[bool]
    order_result: Optional[str]
    awaiting_satisfaction: Optional[bool]
    revision_count: Optional[int]
    group_discussion: Optional[dict]
    adjustment_mode: Optional[str]
    adjustment_modes: Optional[list]
    avoid_places: Optional[list]
    previous_plan_places: Optional[list]
    locked_places: Optional[list]
    exception_events: Optional[list]
    latest_user_input: Optional[str]
    node_timings: Optional[dict]


CANONICAL_PLACE_TYPES = {"attraction", "restaurant", "activity", "leisure", "sports"}

PLACE_TYPE_KEYWORDS = {
    "restaurant": ["餐厅", "美食", "饭店", "咖啡", "咖啡馆", "下午茶", "小吃", "吃饭", "火锅", "海底捞", "小笼包",
                   "小笼", "生煎", "面馆", "吃面", "韩料", "韩国料理", "江浙菜", "本帮菜", "面包", "甜品", "cafe",
                   "coffee"],
    "activity": ["活动", "体验", "露营", "团建", "亲子", "电影", "影院", "影城", "手作", "展览", "看展", "艺术展",
                 "二次元", "动漫", "泡汤", "汤泉", "温泉", "camping"],
    "sports": ["运动", "徒步", "骑行", "羽毛球", "保龄球", "射箭", "健身"],
    "leisure": ["郊区", "近郊", "远郊", "户外", "踏青", "散步", "遛弯", "放松", "休闲", "江边", "滨江", "步道", "街道",
                "大学路", "citywalk", "城市漫步", "田园", "outdoor", "suburban", "suburb", "nature"],
    "attraction": ["景点", "景区", "公园", "博物馆", "美术馆", "展馆", "古镇", "街区", "商圈", "寺庙", "寺", "迪士尼",
                   "乐园", "park"],
}

GENERIC_LOCATION_TERMS = {
    "景区", "景点", "地点", "活动", "室内", "上海", "周末", "附近", "郊区", "近郊", "远郊",
    "户外", "踏青", "公园", "散步", "遛弯", "休闲", "放松", "朋友", "情侣", "家庭", "独行", "未明确", "",
    "上海周末休闲活动", "周末休闲活动", "你看着办", "随便", "随机", "都行",
    "逛吃", "吃喝", "吃喝玩乐", "玩乐", "轻松逛吃", "休闲娱乐",
}

SHANGHAI_DISTRICT_TERMS = {
    "黄浦", "黄浦区", "徐汇", "徐汇区", "长宁", "长宁区", "静安", "静安区",
    "普陀", "普陀区", "虹口", "虹口区", "杨浦", "杨浦区", "浦东", "浦东新区",
    "闵行", "闵行区", "宝山", "宝山区", "嘉定", "嘉定区", "金山", "金山区",
    "松江", "松江区", "青浦", "青浦区", "奉贤", "奉贤区", "崇明", "崇明区",
}

CHINESE_NUMERAL_MAP = {
    "一": 1, "二": 2, "两": 2, "俩": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}

_geocode_cache = {}
_geocode_detail_cache = {}
_amap_text_poi_cache = {}
_route_distance_cache = {}
_amap_poi_cache = {}
_last_amap_request_at = 0.0
_amap_rate_limited_until = 0.0

AMAP_TRANSIENT_LIMIT_INFOS = {
    "CUQPS_HAS_EXCEEDED_THE_LIMIT",
    "QPS_HAS_EXCEEDED_THE_LIMIT",
    "LOCAL_RATE_LIMIT_BACKOFF",
}


def get_amap_key() -> str:
    """读取高德 Web 服务 API Key。兼容 AMAP_API_KEY 和 GAODE_API_KEY 两种命名。"""
    return (os.getenv("AMAP_API_KEY") or os.getenv("GAODE_API_KEY") or "").strip()


def is_amap_transient_limit(info: str) -> bool:
    return str(info or "").strip() in AMAP_TRANSIENT_LIMIT_INFOS


def amap_get_json(url: str, params: dict, timeout: int = 8) -> dict:
    """带本地节流的高德请求，避免一次规划里连续请求过快触发 QPS 限制。"""
    global _last_amap_request_at, _amap_rate_limited_until

    now = time.time()
    if now < _amap_rate_limited_until:
        return {"status": "0", "info": "LOCAL_RATE_LIMIT_BACKOFF"}

    min_interval = float(os.getenv("AMAP_REQUEST_INTERVAL_SECONDS", "0.25"))
    elapsed = now - _last_amap_request_at
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)

    query = urllib.parse.urlencode(params, safe=",|")
    with urllib.request.urlopen(f"{url}?{query}", timeout=timeout) as response:
        _last_amap_request_at = time.time()
        data = json.loads(response.read().decode("utf-8"))

    if is_amap_transient_limit(data.get("info")):
        backoff_seconds = float(os.getenv("AMAP_QPS_BACKOFF_SECONDS", "8"))
        _amap_rate_limited_until = time.time() + backoff_seconds
    return data


# --获取天气信息
def resolve_weather_target_date(date_text: str) -> str:
    """Resolve common Chinese date expressions to YYYY-MM-DD for weather lookup."""
    text = str(date_text or "").strip()
    today = date.today()

    if not text:
        return today.isoformat()
    if "后天" in text:
        return (today + timedelta(days=2)).isoformat()
    if "明天" in text:
        return (today + timedelta(days=1)).isoformat()
    if "今天" in text or "今晚" in text:
        return today.isoformat()

    iso_match = re.search(r"(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})", text)
    if iso_match:
        y, m, d = map(int, iso_match.groups())
        return date(y, m, d).isoformat()

    md_match = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?", text)
    if md_match:
        m, d = map(int, md_match.groups())
        target = date(today.year, m, d)
        if target < today - timedelta(days=1):
            target = date(today.year + 1, m, d)
        return target.isoformat()

    weekend_target = 5  # Saturday, Python weekday: Monday=0
    weekday_map = {
        "周一": 0, "星期一": 0, "礼拜一": 0,
        "周二": 1, "星期二": 1, "礼拜二": 1,
        "周三": 2, "星期三": 2, "礼拜三": 2,
        "周四": 3, "星期四": 3, "礼拜四": 3,
        "周五": 4, "星期五": 4, "礼拜五": 4,
        "周六": 5, "星期六": 5, "礼拜六": 5,
        "周日": 6, "周天": 6, "星期日": 6, "星期天": 6, "礼拜天": 6,
        "周末": weekend_target, "本周末": weekend_target, "这周末": weekend_target,
    }
    target_weekday = None
    for token, weekday in weekday_map.items():
        if token in text:
            target_weekday = weekday
            break
    if target_weekday is not None:
        delta = (target_weekday - today.weekday()) % 7
        return (today + timedelta(days=delta)).isoformat()

    return today.isoformat()


def infer_weather_city(intent: dict, collected: dict) -> str:
    """Infer weather city. LocalMate is Shanghai-first, so districts map to Shanghai."""
    text = " ".join([
        str((intent or {}).get("departure") or ""),
        str((intent or {}).get("location") or ""),
        str((collected or {}).get("departure") or ""),
        str((collected or {}).get("location") or ""),
    ])
    city_match = re.search(r"([\u4e00-\u9fff]{2,12}市)", text)
    if city_match:
        return city_match.group(1)
    return "上海"


def pick_weather_cast(casts: list, target_date: str) -> tuple:
    if not casts:
        return None, "no_casts"
    for cast in casts:
        if str(cast.get("date") or "") == target_date:
            return cast, "exact"
    try:
        target = datetime.strptime(target_date, "%Y-%m-%d").date()
        sorted_casts = sorted(
            casts,
            key=lambda item: abs((datetime.strptime(str(item.get("date")), "%Y-%m-%d").date() - target).days)
        )
        return sorted_casts[0], "nearest"
    except Exception:
        return casts[0], "fallback"


def weather_period_fields(cast: dict, time_period: str) -> dict:
    period = str(time_period or "")
    use_night = any(token in period for token in ["晚", "夜"])
    if use_night:
        return {
            "period": "晚上",
            "weather": cast.get("nightweather") or cast.get("dayweather") or "",
            "temp": cast.get("nighttemp") or cast.get("daytemp") or "",
            "wind": cast.get("nightwind") or cast.get("daywind") or "",
            "power": cast.get("nightpower") or cast.get("daypower") or "",
        }
    return {
        "period": "白天",
        "weather": cast.get("dayweather") or cast.get("nightweather") or "",
        "temp": cast.get("daytemp") or cast.get("nighttemp") or "",
        "wind": cast.get("daywind") or cast.get("nightwind") or "",
        "power": cast.get("daypower") or cast.get("nightpower") or "",
    }


def query_amap_weather(city: str, target_date: str, time_period: str) -> dict:
    key = get_amap_key()
    if not key:
        return {
            "ok": False,
            "source": "amap_weather",
            "city": city,
            "target_date": target_date,
            "time_period": time_period,
            "summary": "未配置 AMAP_API_KEY/GAODE_API_KEY，天气按用户输入或默认偏好处理。",
        }

    try:
        data = amap_get_json(
            "https://restapi.amap.com/v3/weather/weatherInfo",
            {"key": key, "city": city or "上海", "extensions": "all", "output": "JSON"},
            timeout=8,
        )
    except Exception as exc:
        return {
            "ok": False,
            "source": "amap_weather",
            "city": city,
            "target_date": target_date,
            "time_period": time_period,
            "summary": f"高德天气查询失败：{exc}；方案仍按用户天气偏好生成，出行前需二次核验。",
        }

    if str(data.get("status")) != "1":
        info = data.get("info") or data.get("infocode") or "unknown_error"
        return {
            "ok": False,
            "source": "amap_weather",
            "city": city,
            "target_date": target_date,
            "time_period": time_period,
            "raw_info": info,
            "summary": f"高德天气查询未成功：{info}；方案仍按用户天气偏好生成，出行前需二次核验。",
        }

    forecasts = data.get("forecasts") or []
    forecast = forecasts[0] if forecasts else {}
    casts = forecast.get("casts") or []
    cast, match_type = pick_weather_cast(casts, target_date)
    if not cast:
        return {
            "ok": False,
            "source": "amap_weather",
            "city": city,
            "target_date": target_date,
            "time_period": time_period,
            "summary": "高德天气返回为空；方案仍按用户天气偏好生成，出行前需二次核验。",
        }

    fields = weather_period_fields(cast, time_period)
    city_name = forecast.get("city") or city
    cast_date = cast.get("date") or target_date
    summary = (
        f"{city_name}{cast_date}{fields['period']}天气：{fields['weather']}，"
        f"约{fields['temp']}℃，{fields['wind']}风{fields['power']}级。"
    )
    if match_type != "exact":
        summary += f" 目标日期 {target_date} 未命中精确预报，已使用最接近日期 {cast_date} 的预报作为参考。"

    return {
        "ok": True,
        "source": "amap_weather",
        "city": city_name,
        "adcode": forecast.get("adcode"),
        "target_date": target_date,
        "forecast_date": cast_date,
        "time_period": fields["period"],
        "weather": fields["weather"],
        "temperature": fields["temp"],
        "wind": fields["wind"],
        "wind_power": fields["power"],
        "match_type": match_type,
        "summary": summary,
        "raw_cast": cast,
    }


def weather_lookup(state: AgentState) -> AgentState:
    """Query weather from Amap based on parsed date/time and inject it into planning facts."""
    intent = dict(state.get("intent") or {})
    collected = state.get("collected_info", {}) or {}
    date_text = intent.get("date") or collected.get("date") or "本周末"
    time_period = intent.get("time_period") or collected.get("time_period") or "下午"
    target_date = resolve_weather_target_date(date_text)
    city = infer_weather_city(intent, collected)
    weather_info = query_amap_weather(city, target_date, time_period)

    if weather_info.get("weather"):
        intent["weather"] = weather_info["weather"]
        intent["weather_source"] = "amap_weather"
    print(f"🌦️ 天气查询结果: {weather_info.get('summary')}")
    return {**state, "intent": intent, "weather_info": weather_info}


def normalize_shanghai_address(address: str) -> str:
    address = str(address or "").strip()
    if not address:
        return address
    if any(token in address for token in ["上海", "上海市"]):
        return address
    return f"上海市{address}"


def amap_geocode(address: str, city: str = "上海") -> Optional[str]:
    """把地点名转换为高德经纬度字符串：lon,lat。失败时返回 None。"""
    key = get_amap_key()
    if not key:
        return None

    area_like = is_shanghai_area_location(address)
    normalized = normalize_area_anchor(address) if area_like else normalize_shanghai_address(address)
    if not normalized:
        return None
    cache_key = (normalized, city)
    if cache_key in _geocode_cache:
        return _geocode_cache[cache_key]

    try:
        # 行政区/区域名必须走高德区域地理编码，不能先被本地表里“松江万达/松江大学城”等地点误命中。
        row = None if area_like else _find_place(address)
        if row is not None:
            saved_coord = str(row.get("amap_location", "") or "").strip()
            if re.match(r"^-?\d+(\.\d+)?,-?\d+(\.\d+)?$", saved_coord):
                return saved_coord
    except Exception:
        pass

    poi = None if area_like else choose_best_poi_for_place(address, amap_search_place_text(address, city=city, limit=3))
    if poi and poi.get("location"):
        coord = str(poi.get("location") or "").strip()
        _geocode_cache[cache_key] = coord
        return coord

    try:
        data = amap_get_json(
            "https://restapi.amap.com/v3/geocode/geo",
            {
                "key": key,
                "address": normalized,
                "city": city,
                "output": "JSON",
            },
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"⚠️ 高德地理编码失败: {normalized} / {e}")
        _geocode_cache[cache_key] = None
        return None

    if data.get("status") != "1" or not data.get("geocodes"):
        info = data.get("info")
        print(f"⚠️ 高德地理编码无结果: {normalized} / {info}")
        if not is_amap_transient_limit(info):
            _geocode_cache[cache_key] = None
        return None

    coord = data["geocodes"][0].get("location")
    _geocode_cache[cache_key] = coord
    return coord


def clean_amap_address_part(value) -> str:
    if isinstance(value, list):
        value = "".join(str(item) for item in value if item)
    text = str(value or "").strip()
    if text.lower() in {"nan", "none", "null", "[]"}:
        return ""
    return text


def join_amap_address_parts(*parts) -> str:
    result = ""
    for part in parts:
        text = clean_amap_address_part(part)
        if not text or text == "[]":
            continue
        if text in {"上海城区"} and "上海" in result:
            continue
        if result and text in result:
            continue
        if result and result in text:
            result = text
            continue
        result += text
    return result


def amap_search_place_text(place_name: str, city: str = "上海", limit: int = 5) -> list[dict]:
    """用高德 POI 关键字搜索查真实地点详情，比 geocode 更适合补门牌号地址。"""
    key = get_amap_key()
    keyword = str(place_name or "").strip()
    if not key or not keyword:
        return []

    cache_key = (keyword, city, int(limit or 5))
    if cache_key in _amap_text_poi_cache:
        return _amap_text_poi_cache[cache_key]

    try:
        data = amap_get_json(
            "https://restapi.amap.com/v3/place/text",
            {
                "key": key,
                "keywords": keyword,
                "city": city,
                "citylimit": "true",
                "offset": str(max(1, min(int(limit or 5), 10))),
                "page": "1",
                "extensions": "base",
                "output": "JSON",
            },
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as e:
        print(f"⚠️ 高德POI地址搜索失败: {keyword} / {e}")
        _amap_text_poi_cache[cache_key] = []
        return []

    if data.get("status") != "1":
        info = data.get("info")
        print(f"⚠️ 高德POI地址搜索无结果: {keyword} / {info}")
        if not is_amap_transient_limit(info):
            _amap_text_poi_cache[cache_key] = []
        return []

    pois = []
    for poi in data.get("pois", []) or []:
        name = clean_amap_address_part(poi.get("name"))
        location = clean_amap_address_part(poi.get("location"))
        if not name or not location:
            continue
        full_address = join_amap_address_parts(
            poi.get("pname"),
            poi.get("cityname"),
            poi.get("adname"),
            poi.get("address"),
        )
        pois.append({
            "name": name,
            "formatted_address": full_address,
            "address": clean_amap_address_part(poi.get("address")),
            "province": clean_amap_address_part(poi.get("pname")),
            "city": clean_amap_address_part(poi.get("cityname")),
            "district": clean_amap_address_part(poi.get("adname")),
            "location": location,
            "type": clean_amap_address_part(poi.get("type")),
            "source": "amap_place_text",
        })

    _amap_text_poi_cache[cache_key] = pois
    return pois


def choose_best_poi_for_place(place_name: str, pois: list[dict]) -> Optional[dict]:
    key = normalize_place_text(place_name)
    if not key or not pois:
        return None
    scored = []
    for poi in pois:
        poi_name = str(poi.get("name") or "")
        poi_key = normalize_place_text(poi_name)
        score = 0
        if poi_key == key:
            score += 100
        elif key in poi_key or poi_key in key:
            score += 70
        tokens = significant_place_tokens(place_name)
        if tokens and all(token in poi_key for token in tokens[:2]):
            score += 40
        address = str(poi.get("formatted_address") or "")
        if address and any(token in address for token in ["路", "街", "号", "弄", "广场", "中心", "店"]):
            score += 10
        scored.append((score, poi))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1] if scored and scored[0][0] > 0 else pois[0]


def amap_geocode_detail(address: str, city: str = "上海") -> dict:
    """返回高德地理编码详情，用于补充结构化方案里的具体地址。"""
    key = get_amap_key()
    if not key or not address:
        return {}

    try:
        row = _find_place(address)
        if row is not None:
            saved_address = clean_amap_address_part(row.get("amap_address", ""))
            saved_coord = clean_amap_address_part(row.get("amap_location", ""))
            if saved_address:
                return {
                    "formatted_address": normalize_shanghai_address(saved_address),
                    "location": saved_coord,
                    "source": "mock_table",
                }
    except Exception:
        pass

    normalized = normalize_shanghai_address(address)
    cache_key = (normalized, city)
    if cache_key in _geocode_detail_cache:
        return _geocode_detail_cache[cache_key]

    poi = choose_best_poi_for_place(address, amap_search_place_text(address, city=city, limit=5))
    if poi and poi.get("formatted_address"):
        detail = {
            "formatted_address": poi.get("formatted_address", ""),
            "province": poi.get("province", ""),
            "city": poi.get("city", ""),
            "district": poi.get("district", ""),
            "location": poi.get("location", ""),
            "level": "POI",
            "poi_name": poi.get("name", ""),
            "poi_type": poi.get("type", ""),
            "source": "amap_place_text",
        }
        _geocode_detail_cache[cache_key] = detail
        return detail

    try:
        data = amap_get_json(
            "https://restapi.amap.com/v3/geocode/geo",
            {
                "key": key,
                "address": normalized,
                "city": city,
                "output": "JSON",
            },
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"⚠️ 高德地址详情查询失败: {normalized} / {e}")
        _geocode_detail_cache[cache_key] = {}
        return {}

    if data.get("status") != "1" or not data.get("geocodes"):
        info = data.get("info")
        print(f"⚠️ 高德地址详情无结果: {normalized} / {info}")
        if not is_amap_transient_limit(info):
            _geocode_detail_cache[cache_key] = {}
        return {}

    geocode = data["geocodes"][0] or {}
    detail = {
        "formatted_address": str(geocode.get("formatted_address") or normalized).strip(),
        "province": str(geocode.get("province") or "").strip(),
        "city": str(geocode.get("city") or "").strip(),
        "district": str(geocode.get("district") or "").strip(),
        "location": str(geocode.get("location") or "").strip(),
        "level": str(geocode.get("level") or "").strip(),
        "source": "amap_geocode",
    }
    _geocode_detail_cache[cache_key] = detail
    return detail


def resolve_place_address(place_name: str) -> dict:
    """为地点补一个可展示地址；失败时明确标记为需核验。"""
    name = str(place_name or "").strip()
    if not name or name == "待确认地点":
        return {"address": "", "display_name": name, "resolved_place_name": "", "address_source": "empty",
                "address_note": "地点待确认"}
    detail = amap_geocode_detail(name)
    address = str(detail.get("formatted_address") or "").strip()
    poi_name = str(detail.get("poi_name") or "").strip()
    if address:
        display_name = build_specific_place_display_name(name, poi_name, address)
        return {
            "address": address,
            "display_name": display_name,
            "resolved_place_name": poi_name,
            "address_source": detail.get("source", "amap_geocode"),
            "amap_location": detail.get("location", ""),
            "address_note": "高德地图地址，出行前建议二次核验",
        }
    return {
        "address": "",
        "display_name": name,
        "resolved_place_name": "",
        "address_source": "not_found",
        "amap_location": "",
        "address_note": "高德暂未补到具体地址，出行前需核验",
    }


def build_specific_place_display_name(place_name: str, poi_name: str = "", address: str = "") -> str:
    """券和预订卡片只展示地点/分店名，不把门牌地址拼进名称。"""
    base = str(place_name or "").strip()
    poi = str(poi_name or "").strip()
    if poi and normalize_place_text(base) in normalize_place_text(poi):
        if re.search(r"[（(].*(店|中心|商场|广场|馆|园区|校区|院区|分店).*[）)]", poi):
            if not any(token in poi for token in ["暂停", "关闭", "歇业", "停业"]):
                return poi
    if poi and not place_matches_text(poi, base) and not place_matches_text(base, poi):
        return f"{base}（{poi}）"
    return base


def build_place_display_detail(place_name: str) -> dict:
    detail = resolve_place_address(place_name)
    display_name = detail.get("display_name") or place_name
    return {
        "display_name": display_name,
        "resolved_place_name": detail.get("resolved_place_name", ""),
        "address": detail.get("address", ""),
        "address_source": detail.get("address_source", ""),
        "amap_location": detail.get("amap_location", ""),
        "address_note": detail.get("address_note", ""),
    }


PLACE_SPECIFIC_SCHEDULE_FIELDS = {
    "display_name",
    "resolved_place_name",
    "address",
    "address_source",
    "amap_location",
    "address_note",
    "transport_from_previous",
}


def reset_schedule_place_fields(item: dict) -> dict:
    """Remove cached place/address/transport fields after a schedule place changes."""
    copied = dict(item or {})
    for field in PLACE_SPECIFIC_SCHEDULE_FIELDS:
        copied.pop(field, None)
    return copied


def enrich_schedule_addresses(schedule: list[dict]) -> list[dict]:
    enriched = []
    for item in schedule or []:
        if not isinstance(item, dict):
            continue
        copied = dict(item)
        if not copied.get("address"):
            copied.update(resolve_place_address(copied.get("place", "")))
        enriched.append(copied)
    return enriched


def infer_amap_search_spec(intent: dict, user_input: str) -> Optional[dict]:
    """从用户需求中抽取适合交给高德周边搜索的关键词。"""
    pieces = [
        user_input,
        str((intent or {}).get("location") or ""),
        str((intent or {}).get("meal_pref") or ""),
        " ".join((intent or {}).get("place_keywords", []) or []),
    ]
    text = " ".join(pieces).lower()
    rules = [
        (["海底捞"], "海底捞", "restaurant", "hotpot"),
        (["火锅", "涮锅", "锅底"], "火锅", "restaurant", "hotpot"),
        (["咖啡", "coffee", "星巴克", "下午茶"], "咖啡", "restaurant", "cafe"),
        (["小笼包", "小笼", "汤包"], "小笼包", "restaurant", "xiaolongbao"),
        (["生煎"], "生煎", "restaurant", "shengjian"),
        (["面馆", "吃面", "汤面", "拉面"], "面馆", "restaurant", "noodle"),
        (["韩料", "韩国料理", "韩式"], "韩国料理", "restaurant", "korean_cuisine"),
        (["江浙菜", "本帮菜", "上海菜"], "江浙菜", "restaurant", "jiangzhe_cuisine"),
        (["面包", "烘焙", "甜品"], "面包店", "restaurant", "bakery"),
        (["看展", "艺术展", "美术馆", "画廊"], "美术馆", "attraction", "art_exhibition"),
        (["博物馆"], "博物馆", "attraction", "museum"),
        (["散步", "遛弯", "踏青", "公园", "户外"], "公园", "leisure", "park"),
        (["寺庙", "寺"], "寺庙", "attraction", "temple"),
        (["电影", "影院", "影城"], "电影院", "activity", "cinema"),
        (["泡汤", "汤泉", "温泉"], "汤泉", "activity", "spa_relax"),
    ]
    for triggers, keyword, place_type, sub_type in rules:
        if any(trigger.lower() in text for trigger in triggers):
            return {"keyword": keyword, "place_type": place_type, "sub_type": sub_type}
    return None


def amap_search_pois_near(departure: str, keyword: str, city: str = "上海", radius: Optional[int] = None,
                          limit: int = 5) -> list[dict]:
    """调用高德周边搜索，把出发地附近的真实 POI 作为候选补充。"""
    key = get_amap_key()
    if not key or not departure or not keyword:
        return []

    radius = radius or _safe_int(os.getenv("AMAP_POI_RADIUS_METERS", "12000"), 12000)
    limit = max(1, min(_safe_int(limit, 5), 10))
    cache_key = (str(departure), str(keyword), city, radius, limit)
    if cache_key in _amap_poi_cache:
        return _amap_poi_cache[cache_key]

    origin_coord = amap_geocode(departure, city=city)
    if not origin_coord:
        _amap_poi_cache[cache_key] = []
        return []

    try:
        data = amap_get_json(
            "https://restapi.amap.com/v3/place/around",
            {
                "key": key,
                "location": origin_coord,
                "keywords": keyword,
                "city": city,
                "radius": str(radius),
                "sortrule": "distance",
                "offset": str(limit),
                "page": "1",
                "extensions": "base",
                "output": "JSON",
            },
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"⚠️ 高德周边 POI 搜索失败: {departure} / {keyword} / {e}")
        _amap_poi_cache[cache_key] = []
        return []

    if data.get("status") != "1":
        print(f"⚠️ 高德周边 POI 搜索无结果: {departure} / {keyword} / {data.get('info')}")
        _amap_poi_cache[cache_key] = []
        return []

    pois = []
    for poi in data.get("pois", []) or []:
        name = str(poi.get("name") or "").strip()
        location = str(poi.get("location") or "").strip()
        if not name or not location:
            continue
        distance_m = _safe_int(poi.get("distance"), 0)
        pois.append({
            "name": name,
            "address": str(poi.get("address") or "").strip(),
            "location": location,
            "type": str(poi.get("type") or "").strip(),
            "distance_m": distance_m,
            "source": "amap_poi",
            "keyword": keyword,
            "departure": departure,
        })

    _amap_poi_cache[cache_key] = pois
    return pois


def normalize_area_anchor(location: str) -> str:
    """把“松江/嘉定”这类区域词转成高德更容易识别的上海区域锚点。"""
    text = str(location or "").strip()
    text = re.sub(r"(附近|周边|一带|那边|这边)$", "", text).strip()
    if not text:
        return ""
    if text in SHANGHAI_DISTRICT_TERMS:
        if text.endswith("新区") or text.endswith("区"):
            return f"上海市{text}"
        return f"上海市{text}区"
    return text


def is_shanghai_area_location(location: str) -> bool:
    text = str(location or "").strip()
    if not text:
        return False
    normalized = normalize_area_anchor(text)
    short = normalized.replace("上海市", "")
    return text in SHANGHAI_DISTRICT_TERMS or short in SHANGHAI_DISTRICT_TERMS


def is_category_like_location(location: str) -> bool:
    """识别“火锅/咖啡/公园”这类类型词，避免把它当成唯一具体地点。"""
    text = normalize_place_text(location)
    if not text:
        return False
    category_terms = {
        "火锅", "咖啡", "咖啡馆", "餐厅", "饭店", "美食", "吃饭", "小吃",
        "公园", "景点", "景区", "商场", "商圈", "看展", "展览", "电影",
        "影院", "面包", "甜品", "小笼", "小笼包", "生煎", "面馆",
        "海底捞", "星巴克", "manner", "seesaw", "% arabica",
        "逛吃", "吃喝", "吃喝玩乐", "玩乐", "轻松逛吃", "休闲娱乐",
    }
    category_keys = {normalize_place_text(term) for term in category_terms}
    if text in category_keys:
        return True
    soft_preference_terms = {"逛吃", "吃喝", "吃喝玩乐", "玩乐", "轻松", "休闲"}
    return bool(text) and any(term in text for term in soft_preference_terms) and not is_shanghai_area_location(location)


def is_concrete_location_anchor(location: str) -> bool:
    """Return True when a location can be treated as a route anchor instead of only a preference."""
    text = str(location or "").strip()
    if not text:
        return False
    return text not in GENERIC_LOCATION_TERMS and not is_category_like_location(text)


def default_amap_search_spec(intent: dict, user_input: str = "") -> dict:
    """当用户只给区域名时，给高德周边搜索一个合理的默认核心类型。"""
    explicit = infer_amap_search_spec(intent, user_input)
    if explicit:
        return explicit

    place_type = str((intent or {}).get("place_type") or "").lower()
    text = f"{user_input} {(intent or {}).get('location', '')} {' '.join((intent or {}).get('place_keywords', []) or [])}"
    if place_type == "restaurant" or any(term in text for term in ["吃", "饭", "美食", "餐厅"]):
        return {"keyword": "餐厅", "place_type": "restaurant", "sub_type": "restaurant"}
    if place_type == "activity" or any(term in text for term in ["看展", "电影", "体验", "室内"]):
        return {"keyword": "休闲娱乐", "place_type": "activity", "sub_type": "general_activity"}
    if any(term in text for term in ["踏青", "散步", "户外", "公园", "植物园"]):
        return {"keyword": "公园", "place_type": "leisure", "sub_type": "park"}
    return {"keyword": "景点", "place_type": "attraction", "sub_type": "scenic"}


def classify_amap_poi_spec(poi: dict, intent: dict, user_input: str = "") -> dict:
    """根据高德 POI 的类型/名称给 mock 表补一个可用类型。"""
    explicit = infer_amap_search_spec(intent, user_input)
    if explicit:
        return explicit

    name = str(poi.get("name") or "")
    poi_type = str(poi.get("type") or "")
    text = f"{name} {poi_type}"
    if any(term in text for term in ["餐饮", "火锅", "餐厅", "饭店", "咖啡", "甜品", "面包"]):
        if any(term in text for term in ["咖啡", "Coffee", "COFFEE", "甜品", "面包"]):
            return {"keyword": name, "place_type": "restaurant", "sub_type": "cafe"}
        return {"keyword": name, "place_type": "restaurant", "sub_type": "restaurant"}
    if any(term in text for term in ["公园", "植物园", "风景名胜", "景点", "旅游"]):
        return {"keyword": name, "place_type": "leisure", "sub_type": "park"}
    if any(term in text for term in ["博物馆", "美术馆", "展览", "文化"]):
        return {"keyword": name, "place_type": "attraction", "sub_type": "museum"}
    return {"keyword": name, "place_type": str((intent or {}).get("place_type") or "attraction"),
            "sub_type": "amap_poi"}


def persist_text_poi_with_mock(poi: dict, spec: dict) -> str:
    """把 place/text 返回的 POI 转成周边 POI 结构后入库。"""
    converted = {
        "name": poi.get("name", ""),
        "address": poi.get("formatted_address") or poi.get("address", ""),
        "location": poi.get("location", ""),
        "type": poi.get("type", ""),
        "distance_m": 0,
        "source": "amap_place_text",
        "keyword": spec.get("keyword", ""),
        "departure": "",
    }
    return persist_amap_poi_with_mock(converted, spec)


def resolve_unmatched_location_with_amap(intent: dict, state: AgentState) -> dict:
    """用户明确说了库外区域/地点时，先用高德确定锚点，避免退回旧案例路线。"""
    if os.getenv("ENABLE_AMAP_POI_SEARCH", "1") != "1" or not get_amap_key():
        return intent

    fixed = dict(intent or {})
    user_input = state.get("user_input", "")
    location = str(fixed.get("location") or "").strip()
    if not location or location in GENERIC_LOCATION_TERMS:
        return fixed
    area_like = is_shanghai_area_location(location)
    if not area_like and _find_place(location) is not None:
        return fixed
    if is_category_like_location(location):
        return fixed

    if area_like:
        anchor = normalize_area_anchor(location)
        spec = default_amap_search_spec(fixed, user_input)
        pois = amap_search_pois_near(
            departure=anchor,
            keyword=spec["keyword"],
            radius=_safe_int(os.getenv("AMAP_AREA_POI_RADIUS_METERS", "8000"), 8000),
            limit=_safe_int(os.getenv("AMAP_POI_LIMIT", "5"), 5),
        )
        if not pois:
            fixed["area_anchor"] = anchor
            fixed[
                "resolved_location_note"] = f"用户指定区域“{location}”，但高德周边暂未返回可用 POI；后续将尽量保留该区域约束。"
            return fixed
        chosen = pois[0]
        chosen_name = str(chosen.get("name") or "").strip()
        if chosen_name:
            try:
                persist_amap_poi_with_mock(chosen, spec)
            except Exception as e:
                print(f"⚠️ 区域锚点 POI 入库失败，仅作为本次候选使用: {chosen_name} / {e}")
            fixed["location"] = chosen_name
            fixed["place_type"] = spec["place_type"]
            fixed["explicit_place_match"] = True
            fixed["area_anchor"] = anchor
            fixed["amap_anchor_type"] = "area"
            note = (
                f"用户指定区域“{location}”，系统已用高德解析为区域锚点“{anchor}”，"
                f"并在该区域附近选择核心地点“{chosen_name}”。"
            )
            fixed["resolved_location_note"] = note
            fixed["explicit_place_note"] = note
        return fixed

    pois = amap_search_place_text(location, city="上海", limit=5)
    poi = choose_best_poi_for_place(location, pois)
    if not poi:
        return fixed
    spec = classify_amap_poi_spec(poi, fixed, user_input)
    chosen_name = str(poi.get("name") or location).strip()
    if chosen_name:
        try:
            persist_text_poi_with_mock(poi, spec)
        except Exception as e:
            print(f"⚠️ 用户点名高德 POI 入库失败，仅作为本次候选使用: {chosen_name} / {e}")
        fixed["location"] = chosen_name
        fixed["place_type"] = spec["place_type"]
        fixed["explicit_place_match"] = True
        fixed["amap_anchor_type"] = "explicit_poi"
        note = (
            f"用户点名“{location}”，mock 库未命中；系统已通过高德解析为“{chosen_name}”，"
            "并把它作为本次 structured_plan 的核心地点。"
        )
        fixed["resolved_location_note"] = note
        fixed["explicit_place_note"] = note
    return fixed


def estimate_price_by_sub_type(sub_type: str, place_type: str) -> tuple[int, int]:
    sub_type = str(sub_type or "")
    if place_type in {"leisure", "attraction"} and sub_type in {"park", "street_walk", "temple"}:
        return 0, 30
    price_map = {
        "hotpot": (80, 180),
        "cafe": (25, 60),
        "xiaolongbao": (20, 60),
        "shengjian": (15, 45),
        "noodle": (20, 60),
        "korean_cuisine": (60, 150),
        "jiangzhe_cuisine": (60, 140),
        "bakery": (20, 80),
        "cinema": (40, 90),
        "spa_relax": (120, 260),
        "art_exhibition": (0, 120),
        "museum": (0, 80),
    }
    return price_map.get(sub_type, (30, 120) if place_type == "restaurant" else (0, 100))


def build_amap_place_record(poi: dict, spec: dict) -> dict:
    """把高德 POI 转成当前 mock 表可用的运行时地点记录。"""
    place_type = spec.get("place_type", "leisure")
    sub_type = spec.get("sub_type", "amap_poi")
    low, high = estimate_price_by_sub_type(sub_type, place_type)
    keyword = spec.get("keyword", "")
    tags = "、".join([
        "高德POI",
        "附近搜索",
        str(keyword),
        str(place_type),
        str(sub_type),
        "真实地图候选",
    ])
    return {
        "placeName": poi["name"],
        "是否可以预约": False,
        "是否需要预约": False,
        "是否有余位": True,
        "余位信息": _safe_int(os.getenv("AMAP_POI_DEFAULT_SEATS", "80"), 80),
        "是否有团购": False,
        "最低价格": low,
        "最高价格": high,
        "地点类型": place_type,
        "primary_type": place_type,
        "sub_type": sub_type,
        "search_tags": tags,
        "source_note": f"高德POI搜索候选：{poi.get('address', '')}；Demo 模拟预约/余位状态，出行前需核验。",
        "amap_location": poi.get("location", ""),
        "amap_address": poi.get("address", ""),
        "amap_distance_from_query_m": _safe_int(poi.get("distance_m"), 0),
    }


def persist_amap_poi_with_mock(poi: dict, spec: dict, force_coupon: bool = False) -> str:
    """写入高德真实 POI；余位/团购等仍作为 demo mock 字段。"""
    existing = _find_place(poi["name"])
    if existing is not None:
        return poi["name"]

    record = build_amap_place_record(poi, spec)
    poi_name = str(poi.get("name") or "").strip()
    poi_location = str(poi.get("location") or "").strip()
    if poi_name and poi_location:
        _geocode_cache[(normalize_shanghai_address(poi_name), "上海")] = poi_location
        _geocode_detail_cache[(normalize_shanghai_address(poi_name), "上海")] = {
            "formatted_address": normalize_shanghai_address(poi.get("address", "")),
            "location": poi_location,
            "poi_name": poi_name,
            "source": poi.get("source", "amap_poi"),
        }
    if force_coupon:
        record["是否有团购"] = True
        record["source_note"] = str(record.get("source_note", "")) + "；为满足“优先有团购”要求，团购状态为 demo mock。"
    target_file = os.getenv("AMAP_POI_TARGET_FILE", "all_place_mock.xlsx")
    add_new_place(record, target_file=target_file)
    import mock_api_improved as mock_api_module
    globals()["_df"] = mock_api_module._df
    return poi["name"]


def maybe_add_nearby_amap_candidate(intent: dict, state: AgentState) -> dict:
    """必要时用高德附近 POI 补充一个距离更近的候选，并写入地点表。"""
    if os.getenv("ENABLE_AMAP_POI_SEARCH", "1") != "1":
        return intent
    fixed = dict(intent or {})
    collected = state.get("collected_info", {}) or {}
    departure = collected.get("departure") or fixed.get("departure") or ""
    if not departure or not get_amap_key():
        return fixed

    spec = infer_amap_search_spec(fixed, state.get("user_input", ""))
    if not spec:
        return fixed
    explicit_text = f"{fixed.get('location', '')} {fixed.get('meal_pref', '')} {state.get('user_input', '')}".lower()
    branch_or_category_terms = ["海底捞", "火锅", "咖啡", "星巴克", "小笼", "生煎", "面馆", "韩国料理", "韩料",
                                "江浙菜", "面包", "甜品"]
    if fixed.get("explicit_place_match") is True and not any(
            term.lower() in explicit_text for term in branch_or_category_terms):
        return fixed

    pois = amap_search_pois_near(
        departure=departure,
        keyword=spec["keyword"],
        radius=_safe_int(os.getenv("AMAP_POI_RADIUS_METERS", "12000"), 12000),
        limit=_safe_int(os.getenv("AMAP_POI_LIMIT", "5"), 5),
    )
    if not pois:
        return fixed

    accept_radius = _safe_int(os.getenv("AMAP_POI_ACCEPT_RADIUS_METERS", "10000"), 10000)
    chosen = None
    for poi in pois:
        distance_m = _safe_int(poi.get("distance_m"), 0)
        if distance_m <= accept_radius or chosen is None:
            chosen = poi
        if distance_m and distance_m <= accept_radius:
            break

    if not chosen:
        return fixed

    existing = _find_place(chosen["name"])
    if existing is None:
        target_file = os.getenv("AMAP_POI_TARGET_FILE", "all_place_mock.xlsx")
        try:
            persist_amap_poi_with_mock(chosen, spec)
        except Exception as e:
            print(f"⚠️ 高德 POI 入库失败，仅作为本次运行候选使用: {chosen['name']} / {e}")

    fixed["location"] = chosen["name"]
    fixed["place_type"] = spec["place_type"]
    fixed["explicit_place_match"] = True
    fixed["amap_poi_candidate"] = chosen
    fixed["resolved_location_note"] = (
        f"已通过高德在“{departure}”附近搜索“{spec['keyword']}”，"
        f"选择较近候选：{chosen['name']}（约 {format_distance(_safe_int(chosen.get('distance_m'), 0))}）。"
    )
    if spec["place_type"] == "restaurant":
        fixed["meal_pref"] = chosen["name"]
    return fixed


def amap_driving_distance(origin_coord: str, dest_coord: str) -> Optional[dict]:
    """调用高德驾车路径规划，返回距离/时间。strategy=2 表示距离优先。"""
    key = get_amap_key()
    if not key or not origin_coord or not dest_coord:
        return None

    cache_key = (origin_coord, dest_coord)
    if cache_key in _route_distance_cache:
        return _route_distance_cache[cache_key]

    try:
        data = amap_get_json(
            "https://restapi.amap.com/v3/direction/driving",
            {
                "key": key,
                "origin": origin_coord,
                "destination": dest_coord,
                "strategy": "2",
                "extensions": "base",
                "output": "JSON",
            },
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"⚠️ 高德路径规划失败: {origin_coord} -> {dest_coord} / {e}")
        _route_distance_cache[cache_key] = None
        return None

    paths = ((data.get("route") or {}).get("paths") or [])
    if data.get("status") != "1" or not paths:
        info = data.get("info")
        print(f"⚠️ 高德路径规划无结果: {origin_coord} -> {dest_coord} / {info}")
        if not is_amap_transient_limit(info):
            _route_distance_cache[cache_key] = None
        return None

    path = paths[0]
    result = {
        "distance_m": _safe_int(path.get("distance"), 0),
        "duration_s": _safe_int(path.get("duration"), 0),
        "strategy": "driving_distance_first",
    }
    _route_distance_cache[cache_key] = result
    return result


def format_distance(meters: int) -> str:
    if meters <= 0:
        return "未知"
    if meters < 1000:
        return f"{meters}米"
    return f"{meters / 1000:.1f}公里"


def format_duration(seconds: int) -> str:
    if seconds <= 0:
        return "未知"
    minutes = max(1, round(seconds / 60))
    if minutes < 60:
        return f"{minutes}分钟"
    return f"{minutes // 60}小时{minutes % 60}分钟"


def suggest_transport(distance_m: int) -> str:
    if distance_m <= 0:
        return "建议出行前用地图二次核验"
    if distance_m <= 1200:
        return "步行优先"
    if distance_m <= 4000:
        return "骑行/打车均可"
    if distance_m <= 12000:
        return "地铁优先，赶时间可打车"
    return "地铁或打车，需预留转场时间"


def extract_meal_candidates(route_plan_text: str) -> list:
    text = str(route_plan_text or "")
    match = re.search(r"用餐\((.*)\)\s*(?:→|,|，|$)", text)
    if not match:
        match = re.search(r"用餐\((.*)\)", text)
    if not match:
        return []
    names = re.split(r"[、,，/|]+", match.group(1))
    return [name.strip() for name in names if name.strip()]


def rebuild_schedule_with_places(old_schedule: list[dict], places: list[str]) -> list[dict]:
    """距离修复后按新地点列表重建 schedule，保留原时间槽。"""
    if not places:
        return []
    rebuilt = []
    for index, place in enumerate(places[:3]):
        template = old_schedule[min(index, len(old_schedule) - 1)] if old_schedule else {}
        role = place_role(place)
        if role == "meal":
            purpose = "正餐/核心用餐"
        elif role == "light_food":
            purpose = "轻量补充/咖啡休息"
        else:
            purpose = "顺路游玩/散步体验" if index else "核心游玩"
        rebuilt.append({
            "time": template.get("time", ""),
            "place": place,
            "place_role": role,
            "purpose": purpose,
        })
    return rebuilt


def attach_route_segments_to_structured_plan(structured_plan: dict, segments: list[dict]) -> dict:
    """把每段交通写入对应站点，供最终文案逐站渲染。"""
    schedule = [
        reset_schedule_place_fields(item) if isinstance(item, dict) else item
        for item in (structured_plan.get("schedule") or [])
    ]
    by_to = {segment.get("to"): segment for segment in segments}
    for item in schedule:
        if not isinstance(item, dict):
            continue
        segment = by_to.get(item.get("place"))
        if segment:
            item["transport_from_previous"] = segment
    structured_plan["schedule"] = enrich_schedule_addresses(schedule)
    structured_plan["route_segments"] = segments
    return structured_plan


def compute_route_segments(departure: str, stops: list[str]) -> tuple[str, list[dict], list[str]]:
    """返回 route_distance_info 文本和可写入 structured_plan.schedule 的分段交通数据。"""
    if not get_amap_key():
        return (
            "未配置 AMAP_API_KEY 或 GAODE_API_KEY，当前无法调用高德地图计算真实转场距离；禁止在方案中输出任何具体距离、分钟数或费用估算，只能提示用户配置 Key 后再计算。",
            [],
            [],
        )

    route_stops = unique_preserve_order([departure] + [stop for stop in stops if stop])
    route_stops = [stop for stop in route_stops if str(stop or "").strip()]
    if len(route_stops) < 2:
        return "高德距离计算失败：缺少出发地或结构化路线地点。", [], []

    lines = ["高德真实距离参考（按 structured_plan.schedule 的地点顺序计算）："]
    total_m = 0
    failed = []
    segments = []
    for idx in range(len(route_stops) - 1):
        start, end = route_stops[idx], route_stops[idx + 1]
        start_coord = amap_geocode(start)
        end_coord = amap_geocode(end)
        if not start_coord or not end_coord:
            failed.append(f"{start} -> {end} 地理编码失败")
            continue
        route = amap_driving_distance(start_coord, end_coord)
        if not route:
            failed.append(f"{start} -> {end} 路径规划失败")
            continue
        total_m += route["distance_m"]
        segment = {
            "from": start,
            "to": end,
            "distance_m": route["distance_m"],
            "distance_text": format_distance(route["distance_m"]),
            "duration_s": route["duration_s"],
            "duration_text": format_duration(route["duration_s"]),
            "transport": suggest_transport(route["distance_m"]),
        }
        segment["summary"] = (
            f"从{start}到{end}：{segment['distance_text']}，约{segment['duration_text']}，"
            f"建议{segment['transport']}。"
        )
        segments.append(segment)
        lines.append(
            f"{idx + 1}. {start} -> {end}: "
            f"{format_distance(route['distance_m'])}，约{format_duration(route['duration_s'])}，"
            f"建议：{suggest_transport(route['distance_m'])}"
        )

    if total_m:
        lines.append(f"当前 structured_plan 核心转场合计约 {format_distance(total_m)}。")
    if failed:
        lines.append("未成功计算：" + "；".join(failed))
    return "\n".join(lines), segments, failed


def parse_lnglat(location: str) -> Optional[tuple[float, float]]:
    match = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$", str(location or ""))
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def estimate_static_map_zoom(coords: list[tuple[float, float]]) -> int:
    if len(coords) < 2:
        return 16
    lngs = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    span = max(max(lngs) - min(lngs), max(lats) - min(lats))
    if span <= 0.006:
        return 17
    if span <= 0.015:
        return 16
    if span <= 0.035:
        return 15
    if span <= 0.07:
        return 14
    if span <= 0.15:
        return 13
    if span <= 0.30:
        return 12
    if span <= 0.60:
        return 10
    if span <= 1.20:
        return 9
    return 8


def estimate_route_span_km(coords: list[tuple[float, float]]) -> float:
    if len(coords) < 2:
        return 0.0
    lngs = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    # 上海附近 1 纬度约111km，1经度约95km；用于选择静态图缩放说明。
    width_km = (max(lngs) - min(lngs)) * 95
    height_km = (max(lats) - min(lats)) * 111
    return max(width_km, height_km)


def estimate_static_map_size(coords: list[tuple[float, float]]) -> str:
    explicit = os.getenv("ROUTE_MAP_SIZE", "").strip()
    if explicit:
        return explicit
    span_km = estimate_route_span_km(coords)
    if span_km <= 3:
        return "900*520"
    if span_km <= 10:
        return "960*560"
    return "1024*640"


def build_route_map_info(structured_plan: dict) -> dict:
    """Build a proxied Amap static map descriptor from final structured_plan.schedule."""
    if not get_amap_key():
        return {"available": False, "reason": "未配置高德 Web 服务 API Key，无法生成路线地图。"}

    schedule = (structured_plan or {}).get("schedule") or []
    markers = []
    coords = []
    for index, item in enumerate(schedule, start=1):
        if not isinstance(item, dict):
            continue
        place = str(item.get("place") or "").strip()
        coord = item.get("amap_location") or amap_geocode(place)
        parsed = parse_lnglat(coord)
        if not place or not parsed:
            continue
        coords.append(parsed)
        markers.append({
            "index": index,
            "place": place,
            "coord": f"{parsed[0]:.6f},{parsed[1]:.6f}",
            "time": item.get("time", ""),
        })

    if len(markers) < 2:
        return {"available": False, "reason": "结构化方案中可定位地点少于2个，暂不生成路线地图。", "markers": markers}

    center_lng = sum(c[0] for c in coords) / len(coords)
    center_lat = sum(c[1] for c in coords) / len(coords)
    marker_param = "|".join(
        f"large,0x1E63FF,{m['index']}:{m['coord']}"
        for m in markers
    )
    path_param = "7,0xD946EF,0.82,,:{}".format(";".join(m["coord"] for m in markers))
    zoom = estimate_static_map_zoom(coords)
    size = estimate_static_map_size(coords)
    span_km = estimate_route_span_km(coords)
    params = {
        "key": get_amap_key(),
        "location": f"{center_lng:.6f},{center_lat:.6f}",
        "zoom": str(zoom),
        "size": size,
        "scale": "2",
        "markers": marker_param,
        "paths": path_param,
    }
    amap_url = "https://restapi.amap.com/v3/staticmap?" + urllib.parse.urlencode(params, safe=",|:*;")
    return {
        "available": True,
        "provider": "amap_staticmap",
        "amap_url": amap_url,
        "markers": markers,
        "zoom": zoom,
        "size": size,
        "span_km": round(span_km, 2),
        "note": "高德静态地图按 structured_plan.schedule 的地点顺序生成，编号1/2/3表示游玩顺序；紫色路线线段表示转场顺序。",
    }


def build_route_map_placeholder(structured_plan: dict) -> dict:
    """Cheap map descriptor for plan response; real static map is generated by /route_map lazily."""
    schedule = (structured_plan or {}).get("schedule") or []
    markers = []
    for index, item in enumerate(schedule, start=1):
        if isinstance(item, dict) and item.get("place"):
            markers.append({
                "index": index,
                "place": item.get("place", ""),
                "time": item.get("time", ""),
            })
    if len(markers) < 2:
        return {"available": False, "reason": "结构化方案中可展示地点少于2个，暂不生成路线地图。", "markers": markers}
    return {
        "available": True,
        "lazy": True,
        "provider": "amap_staticmap",
        "markers": markers,
        "note": "路线小地图将在前端请求 /route_map 时生成，避免计入方案生成耗时。",
    }


def unique_preserve_order(items: list) -> list:
    seen = set()
    result = []
    for item in items:
        key = str(item or "").strip()
        if key and key not in seen:
            result.append(key)
            seen.add(key)
    return result


def route_variant_seed(context: dict, salt: str = "") -> int:
    seed = str((context or {}).get("route_variant_seed") or "").strip()
    if not seed:
        return 0
    digest = hashlib.sha256(f"{seed}:{salt}".encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:12], 16)


def vary_candidates(items: list, context: dict, salt: str, pool_size: Optional[int] = None) -> list:
    """Shuffle only the top candidate pool; hard constraints still run after this."""
    if not items or os.getenv("ENABLE_ROUTE_VARIATION", "1") != "1":
        return items
    seed = route_variant_seed(context, salt)
    if not seed:
        return items
    size = pool_size or _safe_int(os.getenv("ROUTE_VARIATION_POOL_SIZE", "6"), 6)
    size = max(1, min(size, len(items)))
    head = list(items[:size])
    tail = list(items[size:])
    random.Random(seed).shuffle(head)
    return head + tail


def significant_place_tokens(place_name: str) -> list[str]:
    """提取地点校验用的关键 token，避免“海底捞A店”误配到“海底捞B店”。"""
    raw = str(place_name or "").strip()
    if not raw:
        return []
    text = re.sub(r"(上海市?|火锅|餐厅|饭店|咖啡馆|咖啡|店铺|门店)", "", raw)
    parts = re.split(r"[（）()·•\s\-_/【】\[\]《》,，、]+", text)
    tokens = []
    for part in parts:
        key = normalize_place_text(part)
        if len(key) >= 2 and key not in {"上海", "火锅", "餐厅", "饭店", "咖啡", "店"}:
            tokens.append(key)
    brand_tokens = ["海底捞", "星巴克", "manner", "seesaw", "arabica", "coffee"]
    lowered = raw.lower()
    for brand in brand_tokens:
        if brand in lowered or brand in raw:
            tokens.append(normalize_place_text(brand))
    return unique_preserve_order(tokens)


def place_matches_text(place_name: str, text: str) -> bool:
    """判断正文是否真正写到了某个具体地点，支持轻微别名，但不允许只命中品牌词。"""
    name_key = normalize_place_text(place_name)
    text_key = normalize_place_text(text)
    if not name_key or not text_key:
        return False
    if name_key in text_key:
        return True

    raw_name = str(place_name or "")
    short = normalize_place_text(re.split(r"[（(]", raw_name)[0])
    branch_marked = bool(re.search(r"[（(].+?[）)]", raw_name))
    unsafe_short = {"海底捞", "海底捞火锅", "火锅", "咖啡", "星巴克", "餐厅", "饭店"}
    if not branch_marked and len(short) >= 4 and short not in unsafe_short and short in text_key:
        return True

    tokens = significant_place_tokens(place_name)
    if not tokens:
        return False
    brand_like = {"海底捞", "火锅", "咖啡", "coffee", "星巴克", "餐厅", "饭店"}
    meaningful = [token for token in tokens if token not in brand_like]
    if meaningful:
        return all(token in text_key for token in meaningful[:2])
    return all(token in text_key for token in tokens)


def find_place_exact_for_route(name: str):
    """路线/团购券使用严格地点匹配；宁可不显示券，也不串到另一家分店。"""
    raw_key = normalize_place_text(name)
    if not raw_key:
        return None

    for _, row in _df.iterrows():
        place_name = str(row.get("placeName", "")).strip()
        place_key = normalize_place_text(place_name)
        if place_key and place_key == raw_key:
            return row
    return None


def build_coupon_info_for_places(names: list[str]) -> dict:
    """只从结构化路线实际包含的地点里读取团购券信息。"""
    names = unique_preserve_order(names)
    coupons = []
    checked = []
    seen_place_names = set()

    for name in names:
        row = find_place_exact_for_route(name)
        if row is None:
            checked.append(f"{name}: 未入库")
            continue

        place_name = str(row.get("placeName", name)).strip()
        if place_name in seen_place_names:
            continue
        seen_place_names.add(place_name)
        place_type = str(row.get("地点类型", "unknown")).strip()
        has_coupon = bool(row.get("是否有团购", False))
        checked.append(f"{place_name}: {'有团购' if has_coupon else '无团购'}")
        if not has_coupon:
            continue

        display_detail = build_place_display_detail(place_name)
        display_name = display_detail.get("display_name") or place_name
        low = float(row.get("最低价格", 0) or 0)
        high = float(row.get("最高价格", 0) or low or 0)
        if high >= 180 or low >= 100:
            discount = "满200减40"
        elif high >= 80 or low >= 50:
            discount = "满100减20"
        else:
            discount = "满50减10"

        coupon_type = "餐饮团购券" if place_type == "restaurant" else "活动团购券" if place_type in {"activity",
                                                                                                     "sports",
                                                                                                     "leisure"} else "门票/体验团购券"
        theme = infer_coupon_theme(place_name, place_type)
        coupons.append({
            "place_name": place_name,
            "display_name": display_name,
            "resolved_place_name": display_detail.get("resolved_place_name", ""),
            "address": display_detail.get("address", ""),
            "address_source": display_detail.get("address_source", ""),
            "place_type": place_type,
            "theme": theme["theme"],
            "theme_label": theme["label"],
            "icon": theme["icon"],
            "coupon_type": coupon_type,
            "discount": discount,
            "price_range": f"{int(low)}-{int(high)}元" if (low or high) else "价格待核验",
            "note": "mock 数据显示有团购券，出行前需在美团/大众点评核验实时可用性。",
        })

    if coupons:
        summary_lines = ["团购券信息："]
        for idx, coupon in enumerate(coupons, start=1):
            summary_lines.append(
                f"{idx}. {coupon.get('display_name') or coupon['place_name']}｜{coupon['coupon_type']}｜"
                f"{coupon['discount']}｜参考价格 {coupon['price_range']}"
            )
        summary = "\n".join(summary_lines)
    else:
        summary = "团购券信息：当前路线涉及地点在 mock 数据中未发现可用团购券。"

    return {
        "items": coupons,
        "summary": summary,
        "checked_places": checked,
    }


def infer_coupon_theme(place_name: str, place_type: str) -> dict:
    name = str(place_name or "").lower()
    if any(k in name for k in ["火锅", "锅", "串串", "麻辣"]):
        return {"theme": "hotpot", "label": "火锅热辣", "icon": "🍲"}
    if any(k in name for k in ["咖啡", "coffee", "cafe", "拿铁", "seesaw"]):
        return {"theme": "coffee", "label": "咖啡小坐", "icon": "☕"}
    if any(k in name for k in ["江南", "江浙", "浙", "杭帮", "甬", "小笼", "本帮"]):
        return {"theme": "jiangnan", "label": "江浙风味", "icon": "🍜"}
    if any(k in name for k in ["东北", "烧烤", "烤肉", "铁锅"]):
        return {"theme": "northeast", "label": "东北/烧烤", "icon": "🥩"}
    if place_type in {"activity", "sports", "leisure"}:
        return {"theme": "activity", "label": "活动体验", "icon": "🎯"}
    return {"theme": "default", "label": "精选优惠", "icon": "🎟️"}


def filter_coupon_info_for_schedule(coupon_info: dict, structured_plan: dict) -> dict:
    """只保留 structured_plan.schedule 中实际出现的地点团购券。"""
    coupon_info = coupon_info or {}
    items = coupon_info.get("items") or []
    schedule_keys = {
        normalize_place_text(item.get("place", ""))
        for item in (structured_plan or {}).get("schedule", []) or []
        if isinstance(item, dict) and item.get("place")
    }
    filtered = [
        item for item in items
        if normalize_place_text(item.get("place_name", "")) in schedule_keys
    ]
    if filtered:
        summary_lines = ["团购券信息（仅保留最终路线中出现的地点）："]
        for idx, coupon in enumerate(filtered, start=1):
            summary_lines.append(
                f"{idx}. {coupon.get('display_name') or coupon['place_name']}｜{coupon['coupon_type']}｜"
                f"{coupon['discount']}｜参考价格 {coupon['price_range']}"
            )
        summary = "\n".join(summary_lines)
    else:
        summary = "团购券信息：最终方案涉及地点在 mock 数据中未发现可用团购券。"
    return {
        **coupon_info,
        "items": filtered,
        "summary": summary,
    }


def place_role(place_name: str) -> str:
    """把地点分成 meal / light_food / non_food。

    注意：最终路线约束依赖这个函数。高德返回的展示名可能是
    “原地点（解析到的分店名）”这种复合名称，未必能被 _find_place 精确命中，
    所以这里必须有关键词兜底，避免正餐/咖啡被误判为 unknown 后绕过规则。
    """
    raw_name = str(place_name or "").strip()
    lowered_raw = raw_name.lower()

    light_keywords = [
        "咖啡", "coffee", "cafe", "星巴克", "manner", "seesaw", "arabica",
        "下午茶", "甜品", "蛋糕", "面包", "烘焙", "茶饮", "奶茶", "贝果",
    ]
    meal_keywords = [
        "餐厅", "饭店", "酒家", "食堂", "小吃", "火锅", "烤肉", "烧肉",
        "泥炉", "羊肉", "白切羊肉", "本帮", "江浙菜", "上海菜", "韩料",
        "韩国料理", "日料", "寿司", "拉面", "面馆", "小笼", "小笼包",
        "生煎", "汤包", "烧烤", "牛排", "披萨", "海底捞", "西塔老太太",
    ]

    row = _find_place(raw_name)
    if row is not None:
        place_type = str(row.get("地点类型", "")).strip()
        sub_type = str(row.get("sub_type", "") or "").strip()
        name = str(row.get("placeName", raw_name) or raw_name).lower()
        combined = f"{raw_name} {name}".lower()
        if place_type == "restaurant":
            if sub_type in {"cafe", "bakery"} or any(term.lower() in combined for term in light_keywords):
                return "light_food"
            return "meal"
        # 地点表类型不是 restaurant，但复合展示名里明显含餐饮关键词时，仍按餐饮处理。
        if any(term.lower() in combined for term in light_keywords):
            return "light_food"
        if any(term.lower() in combined for term in meal_keywords):
            return "meal"
        return "non_food"

    if any(term.lower() in lowered_raw for term in light_keywords):
        return "light_food"
    if any(term.lower() in lowered_raw for term in meal_keywords):
        return "meal"
    return "unknown"


def same_food_brand(a: str, b: str) -> bool:
    """粗略识别同一餐饮品牌，避免一条路线连续出现两家海底捞/火锅店。"""
    ak = normalize_place_text(re.split(r"[（(]", str(a or ""))[0])
    bk = normalize_place_text(re.split(r"[（(]", str(b or ""))[0])
    if not ak or not bk:
        return False
    if ak in bk or bk in ak:
        return True
    brand_terms = ["海底捞", "火锅", "小笼", "生煎", "面馆", "韩料", "韩国料理", "江浙菜", "烤肉", "泥炉", "羊肉", "白切羊肉", "西塔老太太"]
    return any(term in str(a) and term in str(b) for term in brand_terms)


def route_has_meal(places: list[str]) -> bool:
    return any(place_role(place) == "meal" for place in (places or []))


def candidate_food_conflict(candidate: str, existing_places: list[str]) -> Optional[str]:
    """Return a human-readable reason when candidate would break food rhythm.

    规则：
    - 半日路线最多保留一顿正餐；
    - 不允许连续两家咖啡/甜品/轻食；
    - 不允许同品牌/同类餐饮连续或重复补入。
    """
    role = place_role(candidate)
    if role not in {"meal", "light_food"}:
        return None

    existing = [p for p in (existing_places or []) if p]
    if role == "meal" and route_has_meal(existing):
        return "一条 4-6 小时路线里不安排第二顿正餐"

    last_role = place_role(existing[-1]) if existing else None
    if role == "meal" and last_role == "meal":
        return "避免连续两顿正餐"
    if role == "light_food" and last_role == "light_food":
        return "避免连续两家咖啡/甜品/轻食"

    if role in {"meal", "light_food"} and any(
            place_role(old) in {"meal", "light_food"} and same_food_brand(candidate, old)
            for old in existing
    ):
        return "避免重复安排同品牌/同类型餐饮"
    return None


def can_append_food_safe(candidate: str, existing_places: list[str], state: Optional[AgentState] = None,
                         intent: Optional[dict] = None) -> bool:
    """Whether a generated/complement candidate can be appended without food conflict.

    用户明确锁定的锚点不在这里强删，但普通补点/替换点必须遵守餐饮节奏。
    """
    if state is not None and is_locked_route_place(candidate, state, intent):
        return True
    return candidate_food_conflict(candidate, existing_places) is None


def append_food_safe_route_places(base: list[str], additions: list[str], state: Optional[AgentState] = None,
                                  intent: Optional[dict] = None, limit: Optional[int] = None,
                                  notes: Optional[list[str]] = None) -> list[str]:
    """Append route places while enforcing food sequence constraints."""
    result = unique_preserve_order([p for p in (base or []) if p])
    for place in additions or []:
        place = str(place or "").strip()
        if not place:
            continue
        if any(same_route_place(place, old) or same_route_place(old, place) for old in result):
            continue
        reason = candidate_food_conflict(place, result)
        if reason and not (state is not None and is_locked_route_place(place, state, intent)):
            if notes is not None:
                notes.append(f"已跳过“{place}”：{reason}。")
            continue
        result.append(place)
        if limit and len(result) >= limit:
            break
    return result


def final_sanitize_route_places(places: list[str], state: AgentState, intent: dict) -> tuple[list[str], list[str]]:
    """Final hard guard before schedule rendering.

    前面的 RAG、补点、快捷调整、软排序都可能修改 places；因此必须在生成
    structured_plan.schedule 前再次强制清理，防止“烤肉→羊肉”或“咖啡→甜品”
    这类连续餐饮结构进入前端。
    """
    notes = []
    result = []
    for place in unique_preserve_order([p for p in (places or []) if p and p != "待确认地点"]):
        locked = is_locked_route_place(place, state, intent)
        reason = candidate_food_conflict(place, result)
        if reason and not locked:
            notes.append(f"最终餐饮节奏校验：已移除“{place}”，{reason}。")
            continue
        if reason and locked:
            notes.append(f"餐饮节奏提示：用户锁定地点“{place}”与前一餐饮点存在冲突，系统保留该锚点并优先调整其他地点。")
        result.append(place)

    result = preserve_route_anchors(result, state, intent)
    return result, unique_preserve_order(notes)

def persist_amap_poi_if_needed(poi: dict, spec: dict) -> str:
    """把高德 POI 写入当前地点表，并同步本模块持有的 _df 引用。"""
    try:
        persist_amap_poi_with_mock(poi, spec)
    except Exception as e:
        print(f"⚠️ 高德 POI 入库失败，仅作为本次结构化候选使用: {poi['name']} / {e}")
    return poi["name"]


def structured_plan_fast_mode() -> bool:
    """Keep structured planning inside the 30s target by avoiding sync map fan-out."""
    return os.getenv("STRUCTURED_PLAN_FAST_MODE", "1") == "1"


def sync_amap_complement_enabled() -> bool:
    """Allow expensive POI complement only when explicitly enabled."""
    return os.getenv("ENABLE_SYNC_AMAP_COMPLEMENT", "0") == "1" or not structured_plan_fast_mode()


def route_stop_bounds(state: AgentState, intent: dict, collected: dict) -> tuple[int, int]:
    """Return the minimum/maximum number of real route stops to render.

    默认至少 3 站，避免只生成一个地点；“少走路”快捷调整允许降到 2 站。
    默认最多 4 站，4-6 小时路线不会过度堆点。
    """
    modes = state.get("adjustment_modes") or ([state.get("adjustment_mode")] if state.get("adjustment_mode") else [])
    min_stops = _safe_int(os.getenv("MIN_ROUTE_STOPS", "3"), 3)
    max_stops = _safe_int(os.getenv("MAX_ROUTE_STOPS", "4"), 4)

    if "less_walk" in modes:
        min_stops = min(min_stops, 2)
        max_stops = min(max_stops, 3)

    duration = _safe_int(
        (intent or {}).get("duration_hours") or (collected or {}).get("duration_hours") or 5,
        5,
    )
    if duration <= 4:
        max_stops = min(max_stops, 3)

    min_stops = max(2, min(min_stops, max_stops))
    max_stops = max(min_stops, max_stops)
    return min_stops, max_stops


def row_seat_count(row) -> int:
    """统一余位判断：只要余位信息大于0，就视为有余位，避免“300余位却显示无”的矛盾。"""
    if row is None:
        return 0
    try:
        return _safe_int(row.get("余位信息", 0), 0)
    except Exception:
        return 0


def row_has_seat(row) -> bool:
    """兼容 Excel 中“是否有余位”列填错但“余位信息”大于0的情况。"""
    if row is None:
        return False
    try:
        return bool(row.get("是否有余位", False)) or row_seat_count(row) > 0
    except Exception:
        return row_seat_count(row) > 0


def place_price_detail(place_name: str) -> dict:
    """给 structured_plan 和前端逐站展示用的价格信息。

    注意：find_place_exact_for_route/_find_place 返回的是 pandas Series，
    不能写成 `a or b`，否则会触发 Series 布尔值歧义并导致 /plan 500。
    """
    row = find_place_exact_for_route(place_name)
    if row is None:
        row = _find_place(place_name)
    if row is None:
        return {"price_min": 0, "price_max": 0, "price_text": "价格待核验"}
    low = float(row.get("最低价格", 0) or 0)
    high = float(row.get("最高价格", 0) or low or 0)
    if low <= 0 and high <= 0:
        text = "免费/低消费，出行前以商家实时信息为准"
    elif high and high != low:
        text = f"约{int(low)}-{int(high)}元/人"
    else:
        text = f"约{int(low)}元/人"
    return {"price_min": low, "price_max": high, "price_text": text}


def replaced_destination_key_set(state: AgentState) -> set:
    """Keys of old destination anchors that must not be carried into a new plan.

    多轮修改目的地时，旧目的地可能已经进入 locked_places、route_plan 草稿
    或历史 user_input。这里统一记录要排除的旧目的地，避免“陆家嘴→迪士尼→
    上海动物园”时旧目的地继续混入方案。
    """
    collected = (state or {}).get("collected_info") or {}
    raw_keys = []
    raw_keys.extend((state or {}).get("replaced_destination_keys") or [])
    raw_keys.extend(collected.get("replaced_destination_keys") or [])
    raw_names = []
    raw_names.extend((state or {}).get("replaced_destinations") or [])
    raw_names.extend(collected.get("replaced_destinations") or [])
    for name in raw_names:
        key = normalize_place_text(name)
        if key:
            raw_keys.append(key)
    return {str(k).strip() for k in raw_keys if str(k).strip()}


def is_replaced_destination_place(place: str, state: AgentState, current_anchor: str = "") -> bool:
    """Return True when a place is an old destination anchor that must be removed.

    不能只做 normalize 后的完全相等，因为旧目的地可能以“迪士尼”、
    “上海迪士尼度假区”等不同名字进入 locked_places / RAG 草稿。
    """
    place = str(place or "").strip()
    if not place:
        return False
    if current_anchor and same_route_place(place, current_anchor):
        return False
    key = normalize_place_text(place)
    replaced_keys = replaced_destination_key_set(state)
    if key and key in replaced_keys:
        return True
    collected = (state or {}).get("collected_info") or {}
    replaced_names = []
    replaced_names.extend((state or {}).get("replaced_destinations") or [])
    replaced_names.extend(collected.get("replaced_destinations") or [])
    for old in replaced_names:
        old = str(old or "").strip()
        if old and not (current_anchor and same_route_place(old, current_anchor)) and same_route_place(place, old):
            return True
    return False


def filter_replaced_destinations_from_places(places: list[str], state: AgentState, current_anchor: str = "") -> list[str]:
    """Remove old destination anchors while preserving the latest requested destination."""
    filtered = []
    for place in places or []:
        if is_replaced_destination_place(str(place or ""), state, current_anchor):
            continue
        filtered.append(place)
    return unique_preserve_order([p for p in filtered if p])


def locked_route_place_keys(state: AgentState, intent: Optional[dict] = None) -> set:
    """需要跨轮次保持不变的路线锚点/明确地点。

    只说出发地或只说目的地时，这个单点就是中心锚点；后续快捷调整不能改它。
    同时说出发地和目的地时，两端都固定；快捷调整只能改中间站。
    """
    intent = intent or {}
    collected = state.get("collected_info") or {}
    anchors = []
    anchors.extend(state.get("locked_places") or [])
    for key in ["fixed_departure", "fixed_destination", "center_anchor"]:
        value = state.get(key) or collected.get(key)
        if value:
            anchors.append(str(value).strip())
    if collected.get("_departure_explicit") and (collected.get("departure") or intent.get("departure")):
        anchors.append(str(collected.get("departure") or intent.get("departure")).strip())
    if collected.get("_location_explicit") and (collected.get("location") or intent.get("location")):
        anchors.append(str(collected.get("location") or intent.get("location")).strip())
    anchor_name, anchor_mode = planning_anchor_for_intent(intent, collected)
    if anchor_name and anchor_mode in {"destination", "departure"}:
        anchors.append(anchor_name)
    replaced_keys = replaced_destination_key_set(state)
    current_destination_key = normalize_place_text(
        (state or {}).get("fixed_destination")
        or collected.get("fixed_destination")
        or collected.get("location")
        or intent.get("location")
        or ""
    )
    result = set()
    for p in anchors:
        key = normalize_place_text(p)
        if not key:
            continue
        if key in replaced_keys and key != current_destination_key:
            continue
        result.add(key)
    return result


def is_locked_route_place(place: str, state: AgentState, intent: Optional[dict] = None) -> bool:
    key = normalize_place_text(place)
    if not key:
        return False
    for locked_key in locked_route_place_keys(state, intent):
        if key == locked_key or key in locked_key or locked_key in key:
            return True
    return False


def preserve_route_anchors(places: list[str], state: AgentState, intent: Optional[dict] = None) -> list[str]:
    """把明确目的地放回正确位置，避免快捷调整/补点把中心锚点挤掉。"""
    intent = intent or {}
    collected = state.get("collected_info") or {}
    anchor_name, anchor_mode = planning_anchor_for_intent(intent, collected)
    result = filter_replaced_destinations_from_places(places, state, anchor_name)
    if anchor_mode == "destination" and anchor_name:
        if bool(collected.get("_departure_explicit")):
            result = move_destination_anchor_to_end(result, anchor_name)
        else:
            result = move_destination_anchor_to_start(result, anchor_name)
    return result


def soft_optimize_route_order(places: list[str], state: AgentState, intent: dict) -> tuple[list[str], list[str]]:
    """不设固定10km硬限制，但用高德距离做软排序，避免明显跨区乱跳。"""
    if not places or not get_amap_key() or os.getenv("ENABLE_SOFT_NEARBY_ORDER", "1") != "1":
        return places, []
    collected = state.get("collected_info") or {}
    anchor_name, anchor_mode = planning_anchor_for_intent(intent, collected)
    departure = str(collected.get("departure") or intent.get("departure") or "").strip()
    original = preserve_route_anchors(places, state, intent)
    if len(original) <= 2:
        return original, []

    destination_locked = anchor_name if anchor_mode == "destination" else ""
    keep_destination_start = bool(destination_locked and not collected.get("_departure_explicit"))
    keep_destination_end = bool(destination_locked and collected.get("_departure_explicit"))

    fixed_start = destination_locked if keep_destination_start else ""
    fixed_end = destination_locked if keep_destination_end else ""
    mutable = [p for p in original if not (fixed_start and same_anchor_identity(p, fixed_start)) and not (fixed_end and same_anchor_identity(p, fixed_end))]
    start_anchor = departure or fixed_start or anchor_name or (mutable[0] if mutable else "")

    ordered = []
    current = start_anchor
    remaining = list(mutable)
    while remaining:
        best_i = 0
        best_d = None
        for i, candidate in enumerate(remaining):
            d = route_distance_between_places(current, candidate) if current else None
            if d is not None and (best_d is None or d < best_d):
                best_i = i
                best_d = d
        chosen = remaining.pop(best_i)
        ordered.append(chosen)
        current = chosen

    candidate_order = ordered
    if fixed_start:
        candidate_order = [fixed_start] + candidate_order
    if fixed_end:
        candidate_order = candidate_order + [fixed_end]
    candidate_order = unique_preserve_order(candidate_order)

    def total_known_distance(seq: list[str]) -> int:
        route_start = departure if departure else (seq[0] if seq else "")
        stops = [route_start] + [p for p in seq if p]
        total = 0
        for a, b in zip(stops, stops[1:]):
            d = route_distance_between_places(a, b)
            if d is not None:
                total += d
        return total

    old_total = total_known_distance(original)
    new_total = total_known_distance(candidate_order)
    if new_total and (not old_total or new_total < old_total):
        return candidate_order, [f"已按高德距离做软排序：不设固定公里数硬限制，但把路线从约{format_distance(old_total)}优化到约{format_distance(new_total)}，尽量减少跨区折返。"]
    return original, []


def long_route_segment_warnings(route_segments: list[dict]) -> list[str]:
    warn_m = _safe_int(os.getenv("SOFT_LONG_SEGMENT_WARN_METERS", "22000"), 22000)
    warnings = []
    for seg in route_segments or []:
        d = _safe_int(seg.get("distance_m"), 0)
        if d >= warn_m:
            warnings.append(f"距离提醒：{seg.get('from')} → {seg.get('to')} 约{seg.get('distance_text')}，这段偏远；已不做硬拦截，但建议打车/地铁并预留转场时间。")
    return warnings


def place_effort_level(place_name: str) -> int:
    """Estimate physical intensity of a place on a 1-5 scale for feasibility scoring.

    This function only evaluates the final route. It does not change route generation,
    replace places, or add frontend Tips.
    """
    name = str(place_name or "")
    lowered = name.lower()
    row = _find_place(place_name)
    place_type = ""
    sub_type = ""
    if row is not None:
        place_type = str(row.get("地点类型", "") or "").strip()
        sub_type = str(row.get("sub_type", "") or "").strip()

    role = place_role(place_name)
    if role in {"meal", "light_food"}:
        return 1

    if sub_type in {"theme_park", "sports"} or any(
            term in name for term in ["探险", "森林探险", "乐园", "游乐", "运动", "攀岩", "徒步", "骑行"]
    ):
        return 5
    if sub_type in {"street_walk", "park"} or any(
            term in name for term in ["步行街", "步道", "滨江", "绿道", "森林", "公园", "古镇", "街区", "citywalk"]
    ):
        return 3
    if sub_type in {"museum", "art_exhibition", "shopping", "cinema", "spa_relax"}:
        return 2
    if place_type == "activity":
        return 3
    if place_type in {"attraction", "leisure", "sports"}:
        return 3
    if any(term in lowered for term in ["mall", "plaza", "museum", "gallery"]):
        return 2
    return 2


def place_is_rest_break(place_name: str) -> bool:
    """Return True for places that can reasonably act as a rest or recovery stop."""
    role = place_role(place_name)
    if role in {"meal", "light_food"}:
        return True
    name = str(place_name or "")
    row = _find_place(place_name)
    sub_type = str(row.get("sub_type", "") or "").strip() if row is not None else ""
    return sub_type in {"shopping", "spa_relax", "cinema"} or any(
        term in name for term in ["咖啡", "甜品", "茶", "餐厅", "饭店", "商场", "广场", "汤泉", "温泉", "影院"]
    )


def evaluate_route_feasibility(structured_plan: dict, intent: Optional[dict] = None,
                               route_segments: Optional[list[dict]] = None) -> dict:
    """Evaluate human feasibility of the final route without changing the route.

    Returns only:
    - intensity_score: 0-100, higher means the route is easier to execute.
    - warnings: feasibility concerns for backend/API inspection.
    """
    schedule = [
        item for item in (structured_plan or {}).get("schedule", []) or []
        if isinstance(item, dict) and item.get("place")
    ]
    segments = route_segments if route_segments is not None else ((structured_plan or {}).get("route_segments") or [])
    intent = intent or {}
    modes = set(intent.get("adjustment_modes") or [])
    if intent.get("adjustment_mode"):
        modes.add(intent.get("adjustment_mode"))

    warnings = []
    penalty = 0

    if not schedule:
        return {"intensity_score": 0, "warnings": ["当前结构化方案没有可评估的行程地点。"]}
    if len(schedule) == 1:
        return {"intensity_score": 45, "warnings": ["当前方案只有1个地点，可行性压力不大，但路线完整度不足。"]}

    places = [str(item.get("place") or "") for item in schedule]
    efforts = [place_effort_level(place) for place in places]
    rest_flags = [place_is_rest_break(place) for place in places]

    high_effort_count = sum(1 for value in efforts if value >= 4)
    walking_like_count = sum(1 for value in efforts if value >= 3)
    rest_count = sum(1 for flag in rest_flags if flag)

    if len(schedule) >= 3 and rest_count == 0 and walking_like_count >= 2:
        warnings.append("路线里连续游玩/步行型地点较多，但没有明显吃饭、咖啡或商场休息点，实际走起来可能偏累。")
        penalty += 14
    if high_effort_count >= 2:
        warnings.append("方案包含多个高体力活动点，建议后续在中间插入餐饮/咖啡休息，或减少一个高强度项目。")
        penalty += 18

    for idx in range(len(places) - 1):
        current_place, next_place = places[idx], places[idx + 1]
        current_effort, next_effort = efforts[idx], efforts[idx + 1]
        if current_effort >= 4 and next_effort >= 3 and not rest_flags[idx + 1]:
            warnings.append(f"{current_place} 后面紧接 {next_place}，连续强度偏高，建议中间安排简餐/咖啡/室内休息。")
            penalty += 12
        elif current_effort >= 3 and next_effort >= 3 and not rest_flags[idx] and not rest_flags[idx + 1]:
            warnings.append(f"{current_place} 到 {next_place} 都偏步行/游玩属性，连续逛可能比较累。")
            penalty += 8

    total_distance_m = 0
    long_segment_count = 0
    very_long_segment_count = 0
    for seg in segments or []:
        dist = _safe_int(seg.get("distance_m"), 0)
        if dist <= 0:
            continue
        total_distance_m += dist
        from_name = seg.get("from") or "上一站"
        to_name = seg.get("to") or "下一站"
        distance_text = seg.get("distance_text") or format_distance(dist)
        if dist >= 30000:
            warnings.append(f"{from_name} 到 {to_name} 转场约{distance_text}，这一段明显偏远，实际执行会比较折腾。")
            very_long_segment_count += 1
            penalty += 16
        elif dist >= 18000:
            warnings.append(f"{from_name} 到 {to_name} 转场约{distance_text}，距离偏长，建议后续优先换成同区域备选。")
            long_segment_count += 1
            penalty += 9

    if total_distance_m >= 65000:
        warnings.append(f"整条路线核心转场合计约{format_distance(total_distance_m)}，半日行程可能太赶。")
        penalty += 16
    elif total_distance_m >= 45000:
        warnings.append(f"整条路线核心转场合计约{format_distance(total_distance_m)}，建议确认是否接受较多通勤。")
        penalty += 9

    if "less_walk" in modes and walking_like_count >= 2:
        warnings.append("用户选择了少走路，但路线中仍有多个步行/户外游玩型地点，后续可进一步降低步行强度。")
        penalty += 12
    if "nearer" in modes and (long_segment_count or very_long_segment_count):
        warnings.append("用户选择了换近一点，但当前路线仍存在偏长转场，后续可继续围绕最新锚点收缩范围。")
        penalty += 10
    if len(schedule) >= 4 and walking_like_count >= 3:
        warnings.append("路线站点数和步行型地点都偏多，4-6小时内执行可能会比较赶。")
        penalty += 10

    if rest_count >= 1 and walking_like_count >= 1:
        penalty -= 6
    if len(schedule) <= 3 and high_effort_count <= 1:
        penalty -= 4

    score = max(0, min(100, 100 - penalty))
    return {
        "intensity_score": int(score),
        "warnings": unique_preserve_order(warnings),
    }


def fallback_complement_places_from_local_table(anchor_name: str, existing_places: list[str], intent: dict, limit: int) -> list[str]:
    """Cheap local-table fallback when Amap cannot add enough nearby POIs.

    This is deliberately lower priority than 高德/本地锚点候选. It only prevents a one-stop
    plan from being rendered; distance validation can still reject it when hard limit is enabled.
    """
    if limit <= 0:
        return []
    existing_keys = {normalize_place_text(p) for p in existing_places if p}
    anchor_key = normalize_place_text(anchor_name)
    wanted_place_type = str((intent or {}).get("place_type") or "").strip()
    result = []

    def score_row(row) -> tuple:
        name = str(row.get("placeName", "") or "").strip()
        tags = str(row.get("search_tags", "") or "")
        place_type = str(row.get("地点类型", "") or "")
        role = place_role(name)
        score = 0
        if wanted_place_type and place_type == wanted_place_type:
            score += 30
        if row_has_seat(row):
            score += 20
        if bool(row.get("是否有团购", False)):
            score += 8
        if anchor_key and anchor_key in normalize_place_text(f"{name} {tags}"):
            score += 60
        # 餐饮冲突在候选入池前硬过滤；这里不再用扣分兜底，避免候选不足时仍补入第二顿正餐/连续咖啡。
        try:
            score -= float(row.get("最低价格", 0) or 0) / 50
        except (TypeError, ValueError):
            pass
        return (-score, name)

    rows = []
    for _, row in _df.iterrows():
        name = str(row.get("placeName", "") or "").strip()
        key = normalize_place_text(name)
        if not name or not key or key in existing_keys:
            continue
        if any(same_route_place(name, old) for old in existing_places):
            continue
        if not can_append_food_safe(name, existing_places + result):
            continue
        rows.append((score_row(row), name))

    rows.sort(key=lambda item: item[0])
    for _, name in rows:
        if not can_append_food_safe(name, existing_places + result):
            continue
        result.append(name)
        existing_keys.add(normalize_place_text(name))
        if len(result) >= limit:
            break
    return result


def ensure_minimum_route_places(
        places: list[str],
        state: AgentState,
        intent: dict,
        anchor_name: str,
        anchor_mode: str,
        min_stops: int,
        max_stops: int,
) -> tuple[list[str], list[str]]:
    """Make sure the final route has at least min_stops real places.

    Fill order:
    1. local mock-table candidates from all_place_mock.xlsx / PLACE_DATA_FILE;
    2. RAG/tool-generated route_plan candidates;
    3. a small number of Amap nearby POIs only when local/RAG still cannot reach min_stops;
    4. final local-table fallback to avoid rendering a one-stop plan.
    """
    notes = []
    collected = state.get("collected_info") or {}
    result = unique_preserve_order([p for p in places if p and p != "待确认地点"])
    if len(result) >= min_stops:
        return result[:max_stops], notes

    anchor_candidates = unique_preserve_order([
        anchor_name,
        result[-1] if result else "",
        collected.get("departure") or (intent or {}).get("departure") or "",
        (intent or {}).get("location") or "",
    ])
    anchor_candidates = [a for a in anchor_candidates if a and a != "上海周末休闲活动"]
    if not anchor_candidates:
        anchor_candidates = [result[-1]] if result else []

    for anchor in anchor_candidates:
        if len(result) >= min_stops:
            break
        needed = min_stops - len(result)
        local_places = nearby_existing_places_from_local_pool(anchor, result + [anchor_name], intent, limit=needed)
        if local_places:
            before = len(result)
            result = append_food_safe_route_places(result, local_places, state, intent, max_stops, notes)
            if len(result) > before:
                notes.append(f"路线不足 {min_stops} 站，已优先从本地地点表围绕“{anchor}”补充：{'、'.join(result[before:])}。")

    route_plan_candidates = vary_candidates(
        extract_meal_candidates(state.get("route_plan", "")),
        {**collected, **(intent or {}), "route_variant_seed": state.get("route_variant_seed")},
        "min_route_route_plan_candidates",
    )
    for anchor in anchor_candidates:
        if len(result) >= min_stops:
            break
        needed = min_stops - len(result)
        from_route_plan = nearby_route_plan_places(anchor, route_plan_candidates, result + [anchor_name], limit=needed)
        if from_route_plan:
            before = len(result)
            result = append_food_safe_route_places(result, from_route_plan, state, intent, max_stops, notes)
            if len(result) > before:
                notes.append(f"路线不足 {min_stops} 站，已从路线草稿中补充：{'、'.join(result[before:])}。")

    amap_fill_enabled = (
            os.getenv("ENABLE_MIN_ROUTE_AMAP_FILL", "1") == "1"
            and os.getenv("ENABLE_AMAP_POI_SEARCH", "1") == "1"
            and bool(get_amap_key())
    )
    if len(result) < min_stops and amap_fill_enabled:
        for anchor in anchor_candidates + result:
            if len(result) >= min_stops:
                break
            if not anchor:
                continue
            needed = min_stops - len(result)
            amap_places = find_nearby_complement_places(anchor, result + [anchor_name], intent, limit=needed)
            if amap_places:
                before = len(result)
                result = append_food_safe_route_places(result, amap_places[:needed], state, intent, max_stops, notes)
                if len(result) > before:
                    notes.append(f"本地表和路线草稿仍不足 {min_stops} 站，已少量调用高德围绕“{anchor}”补充：{'、'.join(result[before:])}。")

    if len(result) < min_stops:
        fallback_anchor = anchor_candidates[0] if anchor_candidates else (anchor_name or "当前锚点")
        needed = min_stops - len(result)
        fallback_places = fallback_complement_places_from_local_table(fallback_anchor, result + [anchor_name], intent, needed)
        if fallback_places:
            before = len(result)
            result = append_food_safe_route_places(result, fallback_places, state, intent, max_stops, notes)
            if len(result) > before:
                notes.append(f"路线仍不足 {min_stops} 站，已用本地地点表兜底补充：{'、'.join(result[before:])}。")

    if anchor_mode == "destination" and anchor_name:
        if bool(collected.get("_departure_explicit")):
            result = move_destination_anchor_to_end(result, anchor_name)
        else:
            result = move_destination_anchor_to_start(result, anchor_name)

    result, final_food_notes = final_sanitize_route_places(result, state, intent)
    notes.extend(final_food_notes)
    if len(result) < min_stops:
        notes.append(f"警告：已尝试补点，但最终仍只有 {len(result)} 站；请检查高德 Key、地点表内容或放宽筛选条件。")
    return result[:max_stops], unique_preserve_order(notes)

def find_nearby_complement_places(anchor_name: str, existing_places: list[str], intent: dict, limit: int = 2) -> list[str]:
    """少量调用高德，补充真实地图附近的非重复地点；只在本地/RAG不足时使用。"""
    if os.getenv("ENABLE_AMAP_POI_SEARCH", "1") != "1" or not get_amap_key() or not anchor_name:
        return []

    main_role = place_role(anchor_name)
    time_period = str(intent.get("time_period") or "")
    if main_role == "meal":
        search_specs = [
            {"keyword": "公园", "place_type": "leisure", "sub_type": "park"},
            {"keyword": "咖啡", "place_type": "restaurant", "sub_type": "cafe"},
        ]
        if "晚上" in time_period:
            search_specs.reverse()
    elif main_role == "light_food":
        search_specs = [
            {"keyword": "公园", "place_type": "leisure", "sub_type": "park"},
            {"keyword": "餐厅", "place_type": "restaurant", "sub_type": "restaurant"},
        ]
    else:
        search_specs = [
            {"keyword": "餐厅", "place_type": "restaurant", "sub_type": "restaurant"},
            {"keyword": "公园", "place_type": "leisure", "sub_type": "park"},
        ]

    result = []
    existing_keys = {normalize_place_text(p) for p in existing_places}
    limit = max(1, min(_safe_int(limit, 2), 2))
    for spec in search_specs:
        if len(result) >= limit:
            break
        pois = amap_search_pois_near(
            departure=anchor_name,
            keyword=spec["keyword"],
            radius=_safe_int(os.getenv("AMAP_COMPLEMENT_RADIUS_METERS", "2500"), 2500),
            limit=limit,
        )
        pois = vary_candidates(pois, intent, f"amap_complement:{anchor_name}:{spec['keyword']}")
        for poi in pois:
            name = poi.get("name", "")
            key = normalize_place_text(name)
            if not key or key in existing_keys:
                continue
            if not can_append_food_safe(name, existing_places + result):
                continue
            result.append(persist_amap_poi_if_needed(poi, spec))
            existing_keys.add(key)
            break
        if len(result) >= limit:
            break
    return result[:limit]


def nearby_existing_places_from_local_pool(anchor_name: str, existing_places: list[str], intent: dict,
                                           limit: int = 2) -> list[str]:
    """优先从本地旧案例/结构化地点表里找补充点。

    快速模式下不逐个请求高德距离，避免结构化阶段被多次网络 I/O 拖慢；
    最终交通距离仍由 route_distance_planner 统一补充。
    """
    if not anchor_name:
        return []

    existing_keys = {normalize_place_text(p) for p in existing_places if p}
    main_role = place_role(anchor_name)
    if main_role == "meal":
        wanted_roles = ["non_food", "light_food"]
    else:
        wanted_roles = ["light_food", "meal", "non_food"]

    def local_candidate_matches_anchor(row, name: str) -> bool:
        if not structured_plan_fast_mode():
            return True
        anchor_text = str(anchor_name or "")
        location_text = str((intent or {}).get("location") or "")
        anchor_area = normalize_area_anchor(anchor_text).replace("上海市", "").replace("区", "")
        terms = [
            anchor_text,
            location_text,
            anchor_area,
            str((intent or {}).get("area_anchor") or ""),
        ]
        terms.extend((intent or {}).get("place_keywords", []) or [])
        fields = " ".join(
            str(row.get(col, "") or "")
            for col in ["placeName", "search_tags", "amap_address", "amap_district", "地点类型", "sub_type"]
            if col in row.index
        )
        fields += f" {name}"
        normalized_fields = normalize_place_text(fields)
        for term in terms:
            term = str(term or "").strip()
            if not term or term in GENERIC_LOCATION_TERMS:
                continue
            compact = normalize_place_text(term)
            area = normalize_area_anchor(term).replace("上海市", "").replace("区", "")
            if compact and compact in normalized_fields:
                return True
            if area and area in fields:
                return True
        return False

    candidates = []
    area_hint = normalize_area_anchor(anchor_name).replace("上海市", "").replace("区", "")
    for _, row in _df.iterrows():
        name = str(row.get("placeName", "") or "").strip()
        if not name:
            continue
        if not local_candidate_matches_anchor(row, name):
            continue
        key = normalize_place_text(name)
        if not key or key in existing_keys:
            continue
        role = place_role(name)
        if role not in wanted_roles:
            continue
        if not can_append_food_safe(name, existing_places):
            continue
        try:
            available_bonus = 10 if row_has_seat(row) else 0
            coupon_bonus = 5 if bool(row.get("是否有团购", False)) else 0
            area_bonus = 40 if area_hint and area_hint in name else 0
            price_penalty = float(row.get("最低价格", 0) or 0) / 40
        except (TypeError, ValueError):
            available_bonus = coupon_bonus = area_bonus = 0
            price_penalty = 0
        candidates.append(
            (wanted_roles.index(role), -(available_bonus + coupon_bonus + area_bonus - price_penalty), name))

    candidates.sort()
    candidates = vary_candidates(candidates, intent, f"local_pool:{anchor_name}")
    result = []
    scan_limit = _safe_int(os.getenv("LOCAL_POOL_DISTANCE_SCAN_LIMIT", "18"), 18)
    for _, _, name in candidates[:scan_limit]:
        key = normalize_place_text(name)
        if not key or key in existing_keys:
            continue
        if not can_append_food_safe(name, existing_places + result):
            continue
        result.append(name)
        existing_keys.add(key)
        if len(result) >= limit:
            break
    return result


def nearby_route_plan_places(anchor_name: str, candidate_names: list[str], existing_places: list[str],
                             limit: int = 2) -> list[str]:
    """从 RAG/工具生成的路线草稿里补充非重复地点；距离只在最终高德说明中展示，不做硬拦截。"""
    existing_keys = {normalize_place_text(p) for p in existing_places if p}
    result = []
    anchor = anchor_name
    for name in unique_preserve_order(candidate_names):
        key = normalize_place_text(name)
        if not key or key in existing_keys:
            continue
        if not can_append_food_safe(name, existing_places + result):
            continue
        result.append(name)
        existing_keys.add(key)
        anchor = name
        if len(result) >= limit:
            break
    return result


def sanitize_structured_places(raw_places: list[str], intent: dict) -> tuple[list[str], list[str]]:
    """去重、删除连续正餐，并保留可解释的结构化规则说明。"""
    places = []
    notes = []
    meal_count = 0
    last_role = None
    for place in unique_preserve_order(raw_places):
        role = place_role(place)
        if role == "meal":
            if meal_count >= 1:
                notes.append(f"已移除“{place}”：一条 4-6 小时路线里不安排第二顿正餐。")
                continue
            if last_role == "meal":
                notes.append(f"已移除“{place}”：避免连续正餐。")
                continue
            if places and same_food_brand(place, places[-1]):
                notes.append(f"已移除“{place}”：避免连续安排同品牌/同类型餐饮。")
                continue
            meal_count += 1
        elif role == "light_food" and last_role == "light_food":
            notes.append(f"已移除“{place}”：避免连续安排两家咖啡/甜品/轻食。")
            continue
        places.append(place)
        last_role = role
    return places, notes


def place_is_indoor(place_name: str) -> bool:
    row = find_place_exact_for_route(place_name)
    if row is None:
        row = _find_place(place_name)
    if row is None:
        name = str(place_name or "")
        return any(term in name.lower() for term in ["mall", "plaza", "coffee", "cafe"]) or any(
            term in name for term in ["商场", "广场", "馆", "餐厅", "咖啡", "影院", "影城", "店"])
    place_type = str(row.get("地点类型", "") or "")
    sub_type = str(row.get("sub_type", "") or "")
    name = str(row.get("placeName", place_name) or place_name)
    outdoor_subtypes = {"park", "street_walk", "temple", "theme_park"}
    if sub_type in outdoor_subtypes:
        return False
    if place_type == "restaurant":
        return True
    return any(
        term in name for term in ["馆", "商场", "广场", "中心", "影院", "影城", "店", "室内", "剧院"]) or sub_type in {
        "museum", "art_exhibition", "cinema", "anime", "spa_relax", "shopping"}


def place_has_coupon(place_name: str) -> bool:
    row = find_place_exact_for_route(place_name)
    if row is None:
        row = _find_place(place_name)
    return bool(row is not None and row.get("是否有团购", False))


def place_min_price(place_name: str) -> float:
    row = find_place_exact_for_route(place_name)
    if row is None:
        row = _find_place(place_name)
    if row is None:
        return 9999.0
    try:
        return float(row.get("最低价格", 9999) or 9999)
    except (TypeError, ValueError):
        return 9999.0


def candidate_rows_for_role(role: str, require_indoor: bool = False, require_coupon: bool = False):
    candidates = _df.copy()
    if require_coupon:
        candidates = candidates[candidates["是否有团购"] == True]
    if role == "meal":
        candidates = candidates[candidates["地点类型"] == "restaurant"].copy()
        candidates = candidates[~candidates.get("sub_type", "").isin(
            ["cafe", "bakery"])] if "sub_type" in candidates.columns else candidates
    elif role == "light_food":
        candidates = candidates[
            (candidates["地点类型"] == "restaurant") & (candidates.get("sub_type", "").isin(["cafe", "bakery"]))].copy()
    elif role == "non_food":
        candidates = candidates[candidates["地点类型"].isin(["activity", "attraction", "leisure", "sports"])].copy()
    if require_indoor and not candidates.empty:
        candidates = candidates[candidates["placeName"].apply(place_is_indoor)].copy()
    return candidates


def choose_best_replacement(
        original_place: str,
        mode: str,
        anchor: str,
        intent: dict,
        force_indoor: bool = False,
        force_coupon: bool = False,
        exclude_places: Optional[set] = None,
) -> Optional[str]:
    role = place_role(original_place)
    require_indoor = force_indoor or mode == "indoor"
    require_coupon = force_coupon or (mode == "coupon" and role in {"meal", "light_food", "non_food"})
    candidates = candidate_rows_for_role(role, require_indoor=require_indoor, require_coupon=require_coupon)
    if candidates is None or candidates.empty:
        return None

    original_key = normalize_place_text(original_place)
    exclude_keys = {normalize_place_text(p) for p in (exclude_places or set())}
    exclude_keys.add(original_key)
    candidates = candidates[candidates["placeName"].apply(lambda n: normalize_place_text(n) not in exclude_keys)].copy()
    if candidates.empty:
        return None

    if mode == "cheaper":
        original_price = place_min_price(original_place)
        cheaper = candidates[candidates["最低价格"].fillna(9999).astype(float) < original_price].copy()
        if not cheaper.empty:
            candidates = cheaper
        candidates = candidates.sort_values(by=["最低价格", "余位信息"], ascending=[True, False])
    elif mode == "coupon":
        candidates = candidates.sort_values(by=["最低价格", "余位信息"], ascending=[True, False])
    else:
        candidates["_candidate_score"] = 100
        nearest = choose_nearest_candidate_row(anchor or intent.get("departure", ""), candidates)
        if nearest:
            return str(nearest["row"].get("placeName", "")).strip()
        candidates = candidates.sort_values(by=["最低价格", "余位信息"], ascending=[True, False])

    return str(candidates.iloc[0].get("placeName", "")).strip()


def persist_amap_coupon_candidate(anchor: str, keyword: str, spec: dict) -> Optional[str]:
    """高德补充团购优先候选：地图给真实 POI，团购/余位仍按 demo mock。"""
    if not get_amap_key() or not anchor:
        return None
    pois = amap_search_pois_near(
        departure=anchor,
        keyword=keyword,
        radius=_safe_int(os.getenv("AMAP_POI_RADIUS_METERS", "12000"), 12000),
        limit=5,
    )
    for poi in pois:
        try:
            return persist_amap_poi_with_mock(poi, spec, force_coupon=True)
        except Exception as e:
            print(f"⚠️ 高德团购候选入库失败: {poi.get('name')} / {e}")
    return None


def amap_search_spec_for_replacement(original_place: str, intent: dict, force_indoor: bool = False) -> dict:
    """根据原地点/用户意图生成高德附近搜索关键词。"""
    text = f"{original_place} {intent.get('location', '')} {intent.get('meal_pref', '')} {' '.join(intent.get('place_keywords', []) or [])}"
    spec = infer_amap_search_spec(intent, text)
    if spec:
        if force_indoor and spec["place_type"] in {"leisure", "attraction"}:
            return {"keyword": "商场", "place_type": "leisure", "sub_type": "shopping"}
        return spec

    name = str(original_place or "")
    role = place_role(name)
    if force_indoor and role == "non_food":
        return {"keyword": "商场", "place_type": "leisure", "sub_type": "shopping"}
    if role == "meal":
        if "海底捞" in name:
            return {"keyword": "海底捞", "place_type": "restaurant", "sub_type": "hotpot"}
        if "火锅" in name:
            return {"keyword": "火锅", "place_type": "restaurant", "sub_type": "hotpot"}
        return {"keyword": "餐厅", "place_type": "restaurant", "sub_type": "general_restaurant"}
    if role == "light_food":
        return {"keyword": "咖啡", "place_type": "restaurant", "sub_type": "cafe"}
    if "博物馆" in name:
        return {"keyword": "博物馆", "place_type": "attraction", "sub_type": "museum"}
    if "美术馆" in name or "展" in name:
        return {"keyword": "美术馆", "place_type": "attraction", "sub_type": "art_exhibition"}
    if "公园" in name or "踏青" in text:
        return {"keyword": "公园", "place_type": "leisure", "sub_type": "park"}
    if "影院" in name or "影城" in name:
        return {"keyword": "电影院", "place_type": "activity", "sub_type": "cinema"}
    return {"keyword": "景点", "place_type": "attraction", "sub_type": "nearby_attraction"}


def find_and_persist_nearby_replacement(
        anchor: str,
        original_place: str,
        intent: dict,
        max_m: int,
        force_indoor: bool = False,
        force_coupon: bool = False,
        exclude_places: Optional[set] = None,
) -> Optional[str]:
    """当库内候选无法满足固定距离硬限制时，从高德搜索近距离 POI 并入库。"""
    if os.getenv("ENABLE_AMAP_POI_SEARCH", "1") != "1" or not get_amap_key() or not anchor:
        return None
    exclude_keys = {normalize_place_text(p) for p in (exclude_places or set())}
    spec = amap_search_spec_for_replacement(original_place, intent, force_indoor=force_indoor)
    pois = amap_search_pois_near(
        departure=anchor,
        keyword=spec["keyword"],
        radius=max(1000, max_m),
        limit=_safe_int(os.getenv("AMAP_POI_LIMIT", "5"), 5),
    )
    pois = vary_candidates(pois, intent, f"amap_replacement:{anchor}:{original_place}")
    for poi in pois:
        name = str(poi.get("name") or "").strip()
        distance_m = _safe_int(poi.get("distance_m"), 0)
        if not name or normalize_place_text(name) in exclude_keys:
            continue
        if distance_m <= 0 or distance_m > max_m:
            continue
        if force_indoor and spec["place_type"] in {"leisure", "attraction"} and spec.get("sub_type") == "park":
            continue
        try:
            return persist_amap_poi_with_mock(poi, spec, force_coupon=force_coupon)
        except Exception as e:
            print(f"⚠️ 高德近距离候选入库失败: {name} / {e}")
    return None


def apply_quick_adjustment_to_places(places: list[str], state: AgentState, intent: dict) -> tuple[list[str], list[str]]:
    modes = state.get("adjustment_modes") or ([state.get("adjustment_mode")] if state.get("adjustment_mode") else [])
    modes = [mode for mode in unique_preserve_order(modes) if mode]
    locked_keys = locked_route_place_keys(state, intent)
    avoid_places = {
        place for place in (state.get("avoid_places") or [])
        if normalize_place_text(place) not in locked_keys
    }
    if (not modes and not avoid_places) or not places:
        return preserve_route_anchors(places, state, intent), []

    notes = [f"已执行快捷调整：{', '.join(modes)}"] if modes else ["已根据用户反馈避免复用上一版方案地点。"]
    adjusted = preserve_route_anchors(list(places), state, intent)
    collected = state.get("collected_info") or {}
    departure = collected.get("departure") or intent.get("departure") or ""
    anchor_name, anchor_mode = planning_anchor_for_intent(intent, collected)
    combined_indoor = "indoor" in modes
    combined_coupon = "coupon" in modes
    nearby_radius_m = _safe_int(os.getenv("NEARBY_SEARCH_RADIUS_METERS", "5000"), 5000)

    def keep_or_replace(place: str, mode: str, anchor: str) -> tuple[str, Optional[str]]:
        if is_locked_route_place(place, state, intent):
            return place, None
        replacement = None
        if mode == "cheaper":
            replacement = choose_best_replacement(
                place, "cheaper", anchor, intent,
                force_indoor=combined_indoor,
                force_coupon=combined_coupon,
                exclude_places=avoid_places,
            )
        elif mode == "indoor" and not place_is_indoor(place):
            replacement = choose_best_replacement(
                place, "indoor", anchor, intent,
                force_coupon=combined_coupon,
                exclude_places=avoid_places,
            )
        elif mode == "coupon" and not place_has_coupon(place):
            replacement = choose_best_replacement(
                place, "coupon", anchor, intent,
                force_indoor=combined_indoor,
                exclude_places=avoid_places,
            )
            if not replacement:
                role = place_role(place)
                keyword = "火锅" if role == "meal" else "咖啡" if role == "light_food" else "室内活动"
                replacement = persist_amap_coupon_candidate(
                    anchor or departure or anchor_name or place,
                    keyword,
                    {"keyword": keyword,
                     "place_type": "restaurant" if role in {"meal", "light_food"} else "activity",
                     "sub_type": "hotpot" if role == "meal" else "cafe" if role == "light_food" else "general_activity"},
                )
        elif normalize_place_text(place) in {normalize_place_text(p) for p in avoid_places}:
            replacement = choose_best_replacement(
                place, "nearer", anchor, intent,
                force_indoor=combined_indoor,
                force_coupon=combined_coupon,
                exclude_places=avoid_places,
            ) or find_and_persist_nearby_replacement(
                anchor or departure or anchor_name or place,
                place,
                intent,
                nearby_radius_m,
                force_indoor=combined_indoor,
                force_coupon=combined_coupon,
                exclude_places=avoid_places,
            )

        # 如果本地表没有满足快捷约束的候选，再少量调用高德补点。
        # 这一步只替换非用户锁定锚点，避免“换室内/便宜/团购”完全停留在提示文案。
        if not replacement and mode in {"cheaper", "indoor", "coupon"}:
            replacement = find_and_persist_nearby_replacement(
                anchor or departure or anchor_name or place,
                place,
                intent,
                nearby_radius_m,
                force_indoor=combined_indoor or mode == "indoor",
                force_coupon=combined_coupon or mode == "coupon",
                exclude_places=avoid_places,
            )
        return replacement or place, replacement

    for mode in modes:
        if mode in {"nearer", "less_walk"}:
            # 只改非锚点。单锚点/固定目的地必须保留，围绕它重新找更近的补充点。
            preserved = [p for p in adjusted if is_locked_route_place(p, state, intent)]
            base_anchor = anchor_name or departure or (preserved[0] if preserved else (adjusted[0] if adjusted else ""))
            target_count = 2 if mode == "less_walk" else max(3, len(adjusted))
            if preserved:
                base = preserve_route_anchors(preserved, state, intent)
            elif anchor_mode == "departure" and departure:
                base = []
            else:
                base = adjusted[:1]
            needed = max(0, min(target_count, _safe_int(os.getenv("MAX_ROUTE_STOPS", "4"), 4)) - len(base))
            local_nearby = nearby_existing_places_from_local_pool(base_anchor, base + [base_anchor], intent, limit=needed)
            new_places = append_food_safe_route_places(base, local_nearby, state, intent, None, notes)
            if len(new_places) < target_count:
                amap_needed = max(0, target_count - len(new_places))
                amap_nearby = find_nearby_complement_places(base_anchor, new_places + [base_anchor], intent, limit=amap_needed)
                new_places = append_food_safe_route_places(new_places, amap_nearby, state, intent, None, notes)
            if not new_places:
                new_places = adjusted
            adjusted = preserve_route_anchors(new_places, state, intent)
            if mode == "less_walk":
                adjusted = preserve_route_anchors(adjusted[:max(2, len(preserved))], state, intent)
                notes.append("少走路模式：已保留用户锚点，只压缩中间站数量，交通建议优先地铁/骑行/打车。")
            else:
                notes.append("换近一点模式：已保留用户锚点，只围绕锚点替换/重排其他地点，尽量减少跨区折返。")

        elif mode in {"cheaper", "indoor", "coupon"}:
            new_places = []
            anchor = departure or anchor_name
            for place in adjusted:
                chosen, replacement = keep_or_replace(place, mode, anchor)
                if replacement:
                    notes.append(f"已将“{place}”替换为“{replacement}”。")
                elif is_locked_route_place(place, state, intent):
                    notes.append(f"已保留用户锚点“{place}”，快捷调整只改其他地点。")
                new_places.append(chosen)
                anchor = chosen
            adjusted = preserve_route_anchors(unique_preserve_order(new_places), state, intent)

    # 多条件同时出现时做最终收口，但仍不允许替换用户锚点。
    if combined_indoor or combined_coupon:
        final_places = []
        anchor = departure or anchor_name
        for place in adjusted:
            if is_locked_route_place(place, state, intent):
                final_places.append(place)
                anchor = place
                continue
            needs_replace = (combined_indoor and not place_is_indoor(place)) or (combined_coupon and not place_has_coupon(place))
            replacement = None
            if needs_replace:
                replacement = choose_best_replacement(
                    place,
                    "coupon" if combined_coupon else "indoor",
                    anchor,
                    intent,
                    force_indoor=combined_indoor,
                    force_coupon=combined_coupon,
                    exclude_places=avoid_places,
                ) or find_and_persist_nearby_replacement(
                    anchor or departure or anchor_name or place,
                    place,
                    intent,
                    nearby_radius_m,
                    force_indoor=combined_indoor,
                    force_coupon=combined_coupon,
                    exclude_places=avoid_places,
                )
            chosen = replacement or place
            if replacement:
                notes.append(f"多条件收口：已将“{place}”替换为“{replacement}”。")
            final_places.append(chosen)
            anchor = chosen
        adjusted = preserve_route_anchors(unique_preserve_order(final_places), state, intent)

    if avoid_places:
        history_adjusted = []
        anchor = departure or anchor_name
        avoid_keys = {normalize_place_text(p) for p in avoid_places}
        for place in adjusted:
            replacement = None
            if normalize_place_text(place) in avoid_keys and not is_locked_route_place(place, state, intent):
                replacement = choose_best_replacement(
                    place,
                    "nearer",
                    anchor,
                    intent,
                    force_indoor=combined_indoor,
                    force_coupon=combined_coupon,
                    exclude_places=avoid_places | set(history_adjusted),
                ) or find_and_persist_nearby_replacement(
                    anchor or departure or anchor_name or place,
                    place,
                    intent,
                    nearby_radius_m,
                    force_indoor=combined_indoor,
                    force_coupon=combined_coupon,
                    exclude_places=avoid_places | set(history_adjusted),
                )
            chosen = replacement or place
            if replacement:
                notes.append(f"避免复用上一版：已将“{place}”替换为“{replacement}”。")
            history_adjusted.append(chosen)
            anchor = chosen
        adjusted = preserve_route_anchors(unique_preserve_order(history_adjusted), state, intent)

    adjusted, sanitize_notes = sanitize_structured_places(adjusted, intent)
    adjusted = preserve_route_anchors(adjusted, state, intent)
    final_food_places, final_food_notes = final_sanitize_route_places(adjusted, state, intent)
    adjusted = preserve_route_anchors(final_food_places, state, intent)
    notes.extend(sanitize_notes)
    notes.extend(final_food_notes)
    return adjusted, unique_preserve_order(notes)

def detect_adjustment_conflicts(places: list[str], state: AgentState, intent: dict) -> list[str]:
    """Return user-visible conflicts for quick adjustment combinations."""
    modes = state.get("adjustment_modes") or ([state.get("adjustment_mode")] if state.get("adjustment_mode") else [])
    modes = [mode for mode in unique_preserve_order(modes) if mode]
    conflicts = []
    if not modes:
        return conflicts
    places = [p for p in places if p and p != "待确认地点"]
    if "indoor" in modes:
        outdoor_left = [p for p in places if not place_is_indoor(p)]
        if outdoor_left:
            conflicts.append("换室内：当前地点表/高德候选不足，仍有这些地点不是明确室内：" + "、".join(outdoor_left[:3]))
    if "coupon" in modes:
        coupon_places = [p for p in places if place_has_coupon(p)]
        if not coupon_places:
            conflicts.append("优先有团购：最终路线里没有查到可用团购券，因此没有硬编券；可放宽区域或改成普通推荐。")
    if "cheaper" in modes:
        budget_text = str((state.get("collected_info") or {}).get("budget") or (intent or {}).get("budget") or "")
        max_budget = None
        m = re.search(r"(\d{2,5})", budget_text)
        if m:
            max_budget = _safe_int(m.group(1), 0)
        if max_budget:
            expensive = []
            for place in places:
                row = _find_place(place)
                if row is None:
                    continue
                low = _safe_int(row.get("最低价格"), 0)
                if low and low > max_budget:
                    expensive.append(place)
            if expensive:
                conflicts.append(f"换便宜一点：预算约{max_budget}元，但这些地点最低价仍偏高：" + "、".join(expensive[:3]))
    if {"less_walk", "nearer"}.intersection(modes) and len(places) >= 3:
        conflicts.append("少走路/换近一点：已优先减少跨区和步行，但为了保证至少多站路线，没有压缩成单点。")
    collected = state.get("collected_info") or {}
    if modes and (collected.get("fixed_departure") or collected.get("fixed_destination") or collected.get("center_anchor")):
        anchor_label = collected.get("fixed_destination") or collected.get("fixed_departure") or collected.get("center_anchor")
        conflicts.append(f"锚点锁定：已保留用户明确的中心锚点/起终点“{anchor_label}”；快捷调整只会改其他地点。")
    return unique_preserve_order(conflicts)


def route_distance_between_places(start: str, end: str) -> Optional[int]:
    if not get_amap_key() or not start or not end:
        return None
    start_coord = amap_geocode(start)
    end_coord = amap_geocode(end)
    if not start_coord or not end_coord:
        return None
    route = amap_driving_distance(start_coord, end_coord)
    return route["distance_m"] if route else None


def estimate_queue_minutes(row) -> int:
    seats = _safe_int(row.get("余位信息", 0), 0)
    if seats <= 0:
        return 45
    if seats < 20:
        return 25
    if seats < 60:
        return 15
    if seats < 120:
        return 8
    return 0


def build_reservation_info(place_name: str) -> dict:
    """生成某个团购券地点的详细预约信息。"""
    row = find_place_exact_for_route(place_name)
    if row is None:
        return {
            "place_name": place_name,
            "ok": False,
            "display_name": place_name,
            "resolved_place_name": "",
            "address": "",
            "address_source": "schedule_only",
            "place_type": "unknown",
            "theme": "default",
            "theme_label": "地点状态",
            "icon": "📍",
            "has_seat": False,
            "need_booking": False,
            "seat_count": 0,
            "need_queue": False,
            "queue_minutes": 0,
            "has_coupon": False,
            "discount": "暂无团购券",
            "message": f"{place_name} 来自最终方案，但未在 mock 表中严格匹配到同名地点；不使用模糊匹配，避免串到其他地点。",
        }

    real_name = str(row.get("placeName", place_name)).strip()
    display_detail = build_place_display_detail(real_name)
    display_name = display_detail.get("display_name") or real_name
    place_type = str(row.get("地点类型", "unknown")).strip()
    seat_count = row_seat_count(row)
    has_seat = row_has_seat(row)
    need_booking = bool(row.get("是否需要预约", False))
    queue_minutes = estimate_queue_minutes(row)
    has_coupon = bool(row.get("是否有团购", False))
    theme = infer_coupon_theme(real_name, place_type)

    low = float(row.get("最低价格", 0) or 0)
    high = float(row.get("最高价格", 0) or low or 0)
    if has_coupon:
        if high >= 180 or low >= 100:
            discount = "满200减40"
        elif high >= 80 or low >= 50:
            discount = "满100减20"
        else:
            discount = "满50减10"
    else:
        discount = "暂无团购券"

    order_id = f"CPN-{abs(hash(real_name)) % 100000:05d}"
    message = (
        f"预约信息如下：\n"
        f"- 店铺/地点：{display_name}\n"
        f"- 类型：{theme['label']}（{place_type}）\n"
        f"- 是否需要预约：{'需要预约' if need_booking else '无需预约'}\n"
        f"- 是否有余位：{'有余位' if has_seat else '暂无余位'}（剩余 {seat_count}）\n"
        f"- 是否需要排队：{'需要排队' if queue_minutes > 0 else '基本无需排队'}\n"
        f"- 预计排队：{queue_minutes} 分钟\n"
        f"- 团购券：{discount}\n"
        f"- 预约单号：{order_id}\n"
        f"如果你们确认这个预约信息，可以回复“那就出发”或“就这样”；如果不满意，可以继续说想调整哪里。"
    )
    return {
        "ok": True,
        "place_name": real_name,
        "display_name": display_name,
        "resolved_place_name": display_detail.get("resolved_place_name", ""),
        "address": display_detail.get("address", ""),
        "address_source": display_detail.get("address_source", ""),
        "place_type": place_type,
        "theme": theme["theme"],
        "theme_label": theme["label"],
        "icon": theme["icon"],
        "has_seat": has_seat,
        "need_booking": need_booking,
        "seat_count": seat_count,
        "need_queue": queue_minutes > 0,
        "queue_minutes": queue_minutes,
        "has_coupon": has_coupon,
        "discount": discount,
        "order_id": order_id,
        "message": message,
    }


def build_reservation_options(structured_plan: dict) -> list[dict]:
    """根据最终 structured_plan.schedule 逐站生成地点状态，保证地点/时间和方案一致。"""
    options = []
    seen = set()
    for index, item in enumerate((structured_plan or {}).get("schedule", []) or [], start=1):
        if not isinstance(item, dict):
            continue
        place = str(item.get("place") or "").strip()
        if not place or normalize_place_text(place) in seen:
            continue
        seen.add(normalize_place_text(place))
        row = find_place_exact_for_route(place)
        place_type = str(row.get("地点类型", "") or "") if row is not None else str(item.get("place_role") or "unknown")
        role = place_role(place)
        info = build_reservation_info(place)
        options.append({
            "schedule_index": index,
            "place_name": place,
            "display_name": info.get("display_name") or place,
            "resolved_place_name": info.get("resolved_place_name", ""),
            "address": info.get("address", ""),
            "address_source": info.get("address_source", ""),
            "place_type": place_type,
            "place_role": role,
            "time": item.get("time", ""),
            "purpose": item.get("purpose", ""),
            "transport_from_previous": item.get("transport_from_previous", {}),
            "matches_schedule": True,
            "need_booking": info.get("need_booking", False),
            "has_seat": info.get("has_seat", False),
            "seat_count": info.get("seat_count", 0),
            "queue_minutes": info.get("queue_minutes", 0),
            "has_coupon": info.get("has_coupon", False),
            "discount": info.get("discount", "暂无团购券"),
            "theme": info.get("theme", "default"),
            "theme_label": info.get("theme_label", "预约信息"),
            "icon": info.get("icon", "📍"),
            "price_text": place_price_detail(place).get("price_text", "价格待核验"),
            "price_min": place_price_detail(place).get("price_min", 0),
            "price_max": place_price_detail(place).get("price_max", 0),
        })
    return options


def choose_nearest_candidate_row(anchor_name: str, candidates):
    """在候选 DataFrame 中优先选择离锚点更近且原始得分较高的地点；不做固定距离硬过滤。"""
    if not get_amap_key() or not anchor_name or candidates is None or candidates.empty:
        return None

    anchor_coord = amap_geocode(anchor_name)
    if not anchor_coord:
        return None

    best = None
    max_candidates = _safe_int(os.getenv("AMAP_MAX_PLACE_CANDIDATES", "12"), 12)
    for _, row in candidates.head(max_candidates).iterrows():
        name = str(row.get("placeName", "")).strip()
        if not name:
            continue
        coord = amap_geocode(name)
        route = amap_driving_distance(anchor_coord, coord) if coord else None
        if not route:
            continue
        base_score = float(row.get("_candidate_score", 0) or 0)
        distance_km = route["distance_m"] / 1000 if route["distance_m"] else 999
        combined_score = base_score - distance_km * 3
        item = {
            "row": row,
            "distance_m": route["distance_m"],
            "duration_s": route["duration_s"],
            "combined_score": combined_score,
        }
        if best is None or item["combined_score"] > best["combined_score"]:
            best = item
    return best


def normalize_intent_place_type(intent: dict, user_input: str) -> dict:
    """把用户口语里的地点偏好归一到 mock 表支持的 5 类，避免输出 suburban 等未知类型。"""
    normalized = dict(intent or {})
    text = f"{user_input} {normalized.get('location', '')} {normalized.get('place_type', '')}".lower()
    raw_place_type = str(normalized.get("place_type") or "").strip().lower()

    if raw_place_type in CANONICAL_PLACE_TYPES:
        chosen_type = raw_place_type
    else:
        chosen_type = None
        for place_type, keywords in PLACE_TYPE_KEYWORDS.items():
            if any(keyword.lower() in text for keyword in keywords):
                chosen_type = place_type
                break
        if not chosen_type:
            chosen_type = "attraction"

    expanded_keywords = []
    for keywords in PLACE_TYPE_KEYWORDS.values():
        expanded_keywords.extend([kw for kw in keywords if kw.lower() in text])

    normalized["place_type"] = chosen_type
    normalized["place_keywords"] = sorted(set(expanded_keywords))
    normalized["place_type_reason"] = (
        f"用户需求中的泛化地点词已归一为 {chosen_type}，用于匹配 mock 表中的同类候选地点。"
    )
    return normalized


def parse_people_count(value, default: Optional[int] = None) -> Optional[int]:
    """Parse Arabic / Chinese people counts and common companion phrases.

    规则优先级：
    1. 明确数字/中文数字优先；
    2. “一家三口/四口”等家庭表达；
    3. “我和父母/和爸妈”= 3 人；“和妈妈/和爸爸/和朋友”= 2 人。
    """
    if value is None:
        return default
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    text = str(value).strip()
    if not text:
        return default

    digit_patterns = [
        r"(\d+)\s*(?:个)?人",
        r"(\d+)\s*(?:位|名)",
        r"人数[：: ]*(\d+)",
        r"一家\s*(\d+)\s*口",
    ]
    for pattern in digit_patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))

    chinese_patterns = [
        r"([一二两俩三四五六七八九十])\s*(?:个)?人",
        r"([一二两俩三四五六七八九十])\s*(?:位|名)",
        r"一家\s*([一二两俩三四五六七八九十])\s*口",
    ]
    for pattern in chinese_patterns:
        match = re.search(pattern, text)
        if match:
            return CHINESE_NUMERAL_MAP.get(match.group(1), default)

    normalized = re.sub(r"\s+", "", text)
    parents_words = ["父母", "爸妈", "爸爸妈妈", "爹妈", "双亲"]
    single_parent_words = ["妈妈", "母亲", "爸爸", "父亲", "老妈", "老爸"]
    friend_words = ["朋友", "同学", "同事", "对象", "男朋友", "女朋友", "伴侣", "闺蜜"]

    if any(word in normalized for word in parents_words) and re.search(r"(?:我|我们)?(?:和|跟|带|陪)", normalized):
        return 3
    if any(word in normalized for word in single_parent_words) and re.search(r"(?:我|我们)?(?:和|跟|带|陪)", normalized):
        return 2
    if any(word in normalized for word in friend_words) and re.search(r"(?:我|我们)?(?:和|跟|带|陪)", normalized):
        return 2
    if any(word in normalized for word in ["我俩", "我们俩", "两个人", "两人"]):
        return 2
    return default


def repair_mojibake_text(value) -> str:
    """Best-effort repair for occasional UTF-8 text decoded as latin-1 in UI/LLM hops."""
    text = str(value or "").strip()
    if not text:
        return ""
    if any(marker in text for marker in ["Ã", "Â", "ä", "å", "æ", " "]):
        try:
            repaired = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore").strip()
            # 只有修复后确实更像中文/正常文本时才采用，避免误伤英文。
            if repaired and (re.search(r"[一-鿿]", repaired) or len(repaired) >= len(text) - 2):
                text = repaired
        except Exception:
            pass
    return text.replace(" ", "").strip()


def normalize_budget_text(value, default: str = "") -> str:
    """Normalize budget into a stable human-readable Chinese string for backend and frontend.

    解决预算字段在 LLM/前端之间被写成数字、None、NaN、错误编码或重复单位的问题。
    """
    if value is None:
        return default
    if isinstance(value, (int, float)):
        try:
            amount = int(value)
            return f"人均{amount}元以内" if amount > 0 else (default or "预算待定")
        except (TypeError, ValueError):
            return default

    text = repair_mojibake_text(value)
    if not text or text.lower() in {"none", "null", "nan", "undefined"}:
        return default
    text = text.replace("￥", "").replace("¥", "").replace("RMB", "").replace("rmb", "")
    text = re.sub(r"\s+", "", text)
    text = text.replace("每人", "人均").replace("一个人", "人均")
    text = re.sub(r"(以内|以下|左右|上下){2,}", r"", text)

    total_like = any(token in text for token in ["总预算", "总共", "合计", "一共"])
    per_like = any(token in text for token in ["人均", "每人", "单人"])
    match = re.search(r"(\d{1,5})(?:\.0+)?", text)
    if match:
        amount = int(match.group(1))
        suffix = "左右" if "左右" in text or "上下" in text else "以内"
        if total_like and not per_like:
            return f"总预算{amount}元{suffix}"
        return f"人均{amount}元{suffix}"
    return text or default


def extract_start_time_hint_from_user_text(user_input: str) -> Optional[str]:
    """规则级抽取出发时间，避免 LLM 漏掉“晚上七点出发/19:00出发”."""
    text = str(user_input or "")
    if not text:
        return None

    time_patterns = [
        r"((?:上午|早上|下午|晚上|夜里|傍晚|中午|凌晨)?\s*\d{1,2}\s*[:：点]\s*\d{0,2}\s*(?:分)?)\s*(?:出发|开始|到|去|走)?",
        r"((?:上午|早上|下午|晚上|夜里|傍晚|中午|凌晨)?\s*[一二两俩三四五六七八九十十一十二]+\s*点\s*(?:半|[一二三四五六七八九十]刻)?)\s*(?:出发|开始|到|去|走)?",
    ]
    for pattern in time_patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        candidate = re.sub(r"\s+", "", match.group(1))
        if candidate:
            return candidate
    return None


def _safe_int(value, default: int) -> int:
    parsed = parse_people_count(value, None)
    if parsed is not None:
        return parsed
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_place_text(text: str) -> str:
    value = str(text or "").lower()
    for ch in " \t\r\n·•-_ /（）()【】[]《》<>，,。；;：:、|":
        value = value.replace(ch, "")
    return value


def place_aliases(place_name: str) -> list[str]:
    raw = str(place_name or "").strip()
    compact = normalize_place_text(raw)
    aliases = [raw, compact]

    base = re.split(r"[（(]", raw)[0].strip()
    if base:
        aliases.extend([base, normalize_place_text(base)])

    no_city_base = re.sub(r"^上海市?", "", base).strip()
    if no_city_base:
        aliases.extend([no_city_base, normalize_place_text(no_city_base)])

    simplified = re.sub(r"(上海市?|火锅|餐厅|咖啡馆|咖啡|园区|公园|美术馆|博物馆|店)$", "", base).strip()
    if simplified and len(normalize_place_text(simplified)) >= 2:
        aliases.extend([simplified, normalize_place_text(simplified)])

    if "EKA" in raw.upper():
        aliases.extend(["EKA", "eka", "EKA园区", "eka园区", "EKA天物空间", "eka天物空间"])

    seen = set()
    result = []
    for alias in aliases:
        key = normalize_place_text(alias)
        if key and key not in seen:
            seen.add(key)
            result.append(alias)
    return result


def find_explicit_place_in_user_input(user_input: str):
    """Prefer a concrete place/brand the user explicitly named over generic type matching."""
    user_compact = normalize_place_text(user_input)
    if not user_compact:
        return None
    user_key_is_broad_area = user_compact in {normalize_place_text(term) for term in SHANGHAI_DISTRICT_TERMS}

    best = None
    for _, row in _df.iterrows():
        place_name = str(row.get("placeName", "")).strip()
        if not place_name:
            continue
        for alias in place_aliases(place_name):
            alias_key = normalize_place_text(alias)
            if len(alias_key) < 2:
                continue
            if alias_key in GENERIC_LOCATION_TERMS:
                continue
            reverse_contains_ok = (
                    user_compact in alias_key
                    and len(user_compact) >= 4
                    and not user_key_is_broad_area
            )
            if alias_key in user_compact or reverse_contains_ok:
                score = len(alias_key)
                if alias_key in user_compact:
                    score += 20
                if best is None or score > best["score"]:
                    best = {"row": row, "place_name": place_name, "alias": alias, "score": score}
    return best


def detect_unmatched_specific_place(user_input: str) -> Optional[str]:
    """Return a likely named place/brand that is not in mock data, so it is not treated as a generic type."""
    patterns = [
        r"(?:想吃|要吃|去吃|吃|想去|要去|去)([^，。,.！!？?、\s]{2,20}(?:火锅|餐厅|饭店|公园|园区|植物园|美术馆|博物馆|咖啡|咖啡馆))",
        r"(海底捞|七七火锅|EKA|鲁迅公园)",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_input or "", re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if find_explicit_place_in_user_input(candidate):
                continue
            if candidate and candidate not in GENERIC_LOCATION_TERMS:
                return candidate
    return None


def extract_location_hint_from_user_text(user_input: str) -> Optional[str]:
    """在 LLM 漏抽时，用规则兜底识别用户明确说的上海区域或景点名。"""
    text = str(user_input or "")
    if not text:
        return None

    for district in sorted(SHANGHAI_DISTRICT_TERMS, key=len, reverse=True):
        if district and district in text:
            departure_context = re.search(rf"(?:从|出发地|起点)[^，。！？；;]{{0,8}}{re.escape(district)}", text)
            destination_context = re.search(rf"(?:想去|要去|去|逛|玩|安排|目的地)[^，。！？；;]{{0,8}}{re.escape(district)}",
                                            text)
            if destination_context or not departure_context:
                return district

    suffixes = "动物园|植物园|公园|古镇|乐园|大学城|景区|博物馆|美术馆|展览馆|广场|商场|街区|寺庙|寺|园区|步道|滨江|外滩|沙滩|海滩|海湾"
    patterns = [
        rf"(?:想去|要去|去|逛|玩|安排|目的地是)([^，。！？、/／\s]{{2,24}}(?:{suffixes}))",
        rf"([^，。！？、/／\s]{{2,24}}(?:{suffixes}))",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            candidate = clean_location_hint_candidate(match.group(1))
            if candidate and candidate not in GENERIC_LOCATION_TERMS and not is_departure_only_mention(text, candidate):
                return candidate
    return None


def clean_location_hint_candidate(candidate: str) -> str:
    """Clean route verbs accidentally captured with a location, e.g. “松江玩” -> “松江”.

    多轮修改里用户常写“更换目的地为松江大学城/换近一点”。这里必须只保留
    “松江大学城”，不能把“/换近一点”吞进地点名，也不能把“松江大学城”简化成“松江”。
    """
    text = str(candidate or "").strip()
    if not text:
        return ""
    # 先切掉快捷调整或后半句，避免“松江大学城/换近一点”成为一个假地点。
    text = re.split(r"[/／,，。！？；;\n]", text, maxsplit=1)[0].strip()
    text = re.sub(r"(换近一点|换便宜一点|换室内|换成室内|优先有团购|少走路|重新生成|再来一版)$", "", text).strip()
    # “松江大学城/浦东新区/上海野生动物园”这类完整地名应优先保留，不要被行政区短词截断。
    strong_suffixes = ("大学城", "动物园", "植物园", "迪士尼", "乐园", "博物馆", "美术馆", "展览馆", "古镇")
    if any(text.endswith(suffix) for suffix in strong_suffixes):
        return text
    for district in sorted(SHANGHAI_DISTRICT_TERMS, key=len, reverse=True):
        if text == district:
            return district
    text = re.sub(r"(附近|周边|一带|那边|这边)$", "", text).strip()
    text = re.sub(r"(轻松逛吃|逛吃|吃喝玩乐|吃喝|玩乐|玩|逛|走走|散步|吃饭|吃东西|看看|打卡)$", "", text).strip()
    return text


def is_departure_only_mention(text: str, candidate: str) -> bool:
    """判断候选地点是否只是“从 X 出发”的出发地，而不是目的地。"""
    text = str(text or "")
    candidate = str(candidate or "").strip()
    if not text or not candidate:
        return False
    escaped = re.escape(candidate)
    departure_hit = re.search(rf"(?:从|出发地|起点)[^，。！？；;]{{0,10}}{escaped}[^，。！？；;]{{0,8}}(?:出发|走|开始)?", text)
    destination_hit = re.search(rf"(?:想去|要去|去|逛|玩|安排|目的地)[^，。！？；;]{{0,10}}{escaped}", text)
    return bool(departure_hit and not destination_hit)


def extract_departure_hint_from_user_text(user_input: str) -> Optional[str]:
    """规则兜底抽取出发地。

    覆盖：
    - “想从长乐路出发”
    - “把出发地换为长乐路 / 出发地改成长乐路”
    - “起点在长乐路”
    """
    text = str(user_input or "")
    patterns = [
        r"(?:把|将)?(?:出发地|起点)(?:换成|换为|改成|改为|换到|改到|设为|设置为|是|在)([^，。！？；;、/／\s]{2,24}?)(?:出发|开始|走|$|，|。|！|？|；|;)",
        r"(?:想|要|准备)?从([^，。！？；;、/／\s]{2,24}?)(?:出发|开始|走|$|，|。|！|？|；|;)",
        r"([^，。！？；;、/／\s]{2,24}?)(?:出发)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            candidate = re.sub(r"^上海市", "", match.group(1).strip())
            candidate = clean_location_hint_candidate(candidate)
            candidate = re.sub(r"(附近|周边|一带)$", "", candidate).strip()
            if candidate and candidate not in GENERIC_LOCATION_TERMS:
                return candidate
    return None


def extract_destination_hint_from_user_text(user_input: str) -> Optional[str]:
    """规则兜底抽取“到 X / 目的地 X / 想去 X / 目的地换成 X”。

    注意：多轮修改时，latest_user_input 里常见“目的地换成陆家嘴”、
    “不去田子坊了，改成陆家嘴”。这些必须覆盖旧目的地，不能被历史
    user_input 里的旧目的地干扰。
    """
    text = str(user_input or "")
    suffixes = "动物园|植物园|公园|古镇|乐园|大学城|景区|博物馆|美术馆|展览馆|广场|商场|街区|寺庙|寺|园区|步道|滨江|外滩|沙滩|海滩|海湾|新区|区"
    patterns = [
        # 明确修改目的地：更换目的地为X / 目的地换成X / 终点改为X / 不去A了，改成X
        rf"(?:更换|修改|调整|重新设置)?(?:目的地|终点|想去的地方|要去的地方)(?:换成|改成|改为|换为|换到|改到|为|是|到)([^，。！？；;、/／\s]{{2,24}}(?:{suffixes})?)",
        rf"(?:目的地|终点|想去的地方|要去的地方)?(?:换成|改成|改为|换为|换到|改到)([^，。！？；;、/／\s]{{2,24}}(?:{suffixes})?)",
        rf"(?:不去|不要去|换掉)[^，。！？；;]{{0,18}}(?:了|啦)?[，,、\s]*(?:去|换成|改成|改为|换为|换到|改到)([^，。！？；;、/／\s]{{2,24}}(?:{suffixes})?)",
        rf"(?:到|去到|目的地是|目的地为|目的地在|终点是|终点为|想去|要去|去|逛|玩|安排)([^，。！？；;、/／\s]{{2,24}}(?:{suffixes})?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            candidate = clean_location_hint_candidate(match.group(1))
            if candidate and candidate not in GENERIC_LOCATION_TERMS and not is_departure_only_mention(text, candidate):
                return candidate
    return extract_location_hint_from_user_text(text)


def is_destination_anchor_intent(intent: dict, collected: Optional[dict] = None) -> bool:
    """用户明确给了目的地/区域时，后续路线应围绕该目的地，而不是围绕默认旧案例。"""
    location = str((intent or {}).get("location") or "").strip()
    if not is_concrete_location_anchor(location):
        return False
    if bool((collected or {}).get("_location_explicit")):
        return True
    if (intent or {}).get("amap_anchor_type") in {"area", "explicit_poi"}:
        return True
    if (intent or {}).get("explicit_place_match") is True:
        return True
    if (intent or {}).get("area_anchor"):
        return True
    return False


def planning_anchor_for_intent(intent: dict, collected: dict) -> tuple[str, str]:
    """返回规划锚点。destination=围绕目的地；departure=围绕出发地。

    目的地锚点必须以 collected/fixed_destination 为最高优先级，不能被历史
    user_input、旧 intent.location 或上一版 structured_plan 里的旧目的地覆盖。
    """
    fixed_destination = str((collected or {}).get("fixed_destination") or (collected or {}).get("active_destination_anchor") or "").strip()
    if fixed_destination and is_concrete_location_anchor(fixed_destination):
        return fixed_destination, "destination"
    collected_location = str((collected or {}).get("location") or "").strip()
    if bool((collected or {}).get("_location_explicit")) and is_concrete_location_anchor(collected_location):
        return collected_location, "destination"
    if is_destination_anchor_intent(intent, collected):
        return str((intent or {}).get("location") or "").strip(), "destination"
    departure = str((collected or {}).get("fixed_departure") or (collected or {}).get("departure") or (intent or {}).get("departure") or "").strip()
    return departure, "departure"


def same_route_place(a: str, b: str) -> bool:
    """路线级地点等价判断，用于保证目的地锚点不重复又能作为终点校验。"""
    if not a or not b:
        return False
    return place_matches_text(a, b) or place_matches_text(b, a)


def user_requests_departure_change(latest_text: str, current_departure: str = "") -> bool:
    """Only unlock a saved departure when the latest message explicitly gives a new one."""
    text = str(latest_text or "")
    hint = extract_departure_hint_from_user_text(text)
    explicit_change = any(token in text for token in [
        "换出发地", "改出发地", "出发地改", "出发地换", "出发地改为", "出发地换成", "出发地换为",
        "把出发地", "设置出发地", "出发地设为",
        "起点改", "起点换", "起点改为", "起点换成", "起点换为", "把起点", "设置起点", "起点设为",
    ])
    if not hint:
        return explicit_change
    if not current_departure:
        return True
    return explicit_change or not same_route_place(hint, current_departure)


def user_requests_destination_change(latest_text: str, current_destination: str = "") -> bool:
    """Only unlock a saved destination when the latest message explicitly names a new destination."""
    text = str(latest_text or "")
    hint = extract_destination_hint_from_user_text(text)
    explicit_change = any(token in text for token in [
        "换目的地", "改目的地", "目的地改", "目的地换", "目的地换成", "目的地改成",
        "换终点", "改终点", "终点改", "终点换", "终点换成", "终点改成",
        "不去", "不要去", "换掉", "改成", "改为", "换成", "换到", "改到"
    ])
    if not hint:
        return explicit_change
    if not current_destination:
        return True
    return explicit_change or not same_route_place(hint, current_destination)


def apply_fixed_anchor_guards(extracted: dict, collected: dict, state: AgentState) -> tuple[dict, dict, list[str]]:
    """Preserve or overwrite user-fixed start/end anchors across multi-turn revisions.

    规则：
    - 用户只说“换近一点/换便宜一点/换室内/重新生成”时，固定出发地、固定目的地、
      单中心锚点都不能被历史方案或模型抽取结果改掉。
    - 用户明确说“目的地换成 X / 终点改为 X”时，只覆盖目的地锚点；如果没有同时
      修改出发地，原出发地必须保留，不能变成上一版目的地。
    - 用户明确说“出发地换成 X / 起点改为 X”时，只覆盖出发地。
    """
    fixed_departure = str((state.get("fixed_departure") or collected.get("fixed_departure") or "") or "").strip()
    fixed_destination = str((state.get("fixed_destination") or collected.get("fixed_destination") or "") or "").strip()
    latest_text = str(state.get("latest_user_input") or "")
    if not latest_text:
        latest_text = str(state.get("user_input") or "")

    latest_departure_hint = extract_departure_hint_from_user_text(latest_text)
    latest_destination_hint = extract_destination_hint_from_user_text(latest_text)
    departure_change = bool(latest_departure_hint) and user_requests_departure_change(latest_text, fixed_departure)
    destination_change = bool(latest_destination_hint) and user_requests_destination_change(latest_text, fixed_destination)
    notes = []

    # 出发地：没有明确修改时，强制保留旧出发地，防止被“上一版目的地”污染。
    if departure_change:
        new_departure = str(latest_departure_hint).strip()
        extracted["departure"] = new_departure
        collected["departure"] = new_departure
        collected["fixed_departure"] = new_departure
        collected["_departure_explicit"] = True
        notes.append(f"已按最新输入更新固定出发地：{new_departure}。")
    elif fixed_departure:
        extracted["departure"] = fixed_departure
        collected["departure"] = fixed_departure
        collected["fixed_departure"] = fixed_departure
        collected["_departure_explicit"] = True
        notes.append(f"已保留用户固定出发地：{fixed_departure}。")
    elif extracted.get("departure"):
        collected["fixed_departure"] = str(extracted.get("departure")).strip()

    # 目的地：明确修改目的地时用最新目的地覆盖旧目的地；否则保留旧目的地。
    if destination_change:
        new_destination = str(latest_destination_hint).strip()
        previous_candidates = [
            fixed_destination,
            state.get("fixed_destination"),
            (state.get("intent") or {}).get("location"),
            collected.get("fixed_destination"),
            collected.get("location"),
        ]
        replaced_names = [
            str(p).strip() for p in previous_candidates
            if p and str(p).strip() and not same_route_place(str(p).strip(), new_destination)
        ]
        replaced_keys = {normalize_place_text(p) for p in replaced_names if normalize_place_text(p)}
        existing_replaced = set(state.get("replaced_destination_keys") or [])
        state["replaced_destination_keys"] = sorted(existing_replaced | replaced_keys)
        state["replaced_destinations"] = unique_preserve_order(
            list(state.get("replaced_destinations") or []) + replaced_names
        )
        collected["replaced_destination_keys"] = state["replaced_destination_keys"]
        collected["replaced_destinations"] = state["replaced_destinations"]
        extracted["location"] = new_destination
        collected["location"] = new_destination
        collected["fixed_destination"] = new_destination
        collected["active_destination_anchor"] = new_destination
        collected["_location_explicit"] = True
        collected["center_anchor"] = new_destination
        state["fixed_destination"] = new_destination
        state["active_destination_anchor"] = new_destination
        # 清掉本轮之前可能残留在 state.intent 里的旧目的地，后续节点只能看到最新目的地。
        if isinstance(state.get("intent"), dict):
            state["intent"] = {**state.get("intent", {}), "location": new_destination}
        notes.append(f"已按最新输入更新固定目的地：{new_destination}。")
        if replaced_names:
            notes.append(f"已移除旧目的地锚点：{'、'.join(unique_preserve_order(replaced_names))}。")
    elif fixed_destination:
        extracted["location"] = fixed_destination
        collected["location"] = fixed_destination
        collected["fixed_destination"] = fixed_destination
        collected["active_destination_anchor"] = fixed_destination
        collected["_location_explicit"] = True
        state["active_destination_anchor"] = fixed_destination
        notes.append(f"已保留用户固定目的地：{fixed_destination}。")
    elif extracted.get("location") and is_concrete_location_anchor(str(extracted.get("location"))):
        collected["fixed_destination"] = str(extracted.get("location")).strip()

    # 单锚点逻辑：只说出发地或只说目的地时，也要跨轮次锁定这个中心锚点。
    # 如果本轮明确改了目的地，中心锚点必须跟随最新目的地。
    if collected.get("_location_explicit") and collected.get("location"):
        collected["center_anchor"] = str(collected.get("location")).strip()
    elif collected.get("_departure_explicit") and collected.get("departure"):
        collected["center_anchor"] = str(collected.get("departure")).strip()

    return extracted, collected, notes

def same_anchor_identity(place: str, anchor: str) -> bool:
    """Only treat a route place as the same anchor when it is the exact anchor, not just nearby text."""
    place_key = normalize_place_text(place)
    anchor_key = normalize_place_text(anchor)
    if not place_key or not anchor_key:
        return False
    if place_key == anchor_key:
        return True
    stripped_place = re.sub(r"^上海市?", "", str(place or "").strip())
    stripped_anchor = re.sub(r"^上海市?", "", str(anchor or "").strip())
    return normalize_place_text(stripped_place) == normalize_place_text(stripped_anchor)


def move_destination_anchor_to_end(places: list[str], destination_anchor: str) -> list[str]:
    """目的地模式下，把用户目的地放到最后，便于校验 C -> 目的地 <= 固定距离。"""
    destination_anchor = str(destination_anchor or "").strip()
    if not destination_anchor:
        return unique_preserve_order([p for p in places if p])
    without_anchor = [
        p for p in places
        if p and not same_anchor_identity(str(p), destination_anchor)
    ]
    return unique_preserve_order(without_anchor + [destination_anchor])


def move_destination_anchor_to_start(places: list[str], destination_anchor: str) -> list[str]:
    """未明确出发地时，把用户目的地作为路线第一站和后续补点锚点。"""
    destination_anchor = str(destination_anchor or "").strip()
    if not destination_anchor:
        return unique_preserve_order([p for p in places if p])
    without_anchor = [
        p for p in places
        if p and not same_anchor_identity(str(p), destination_anchor)
    ]
    return unique_preserve_order([destination_anchor] + without_anchor)


def reconcile_intent_with_rules(intent: dict, state: AgentState) -> dict:
    """Deterministically fix high-risk fields after LLM extraction.

    这里必须优先看 latest_user_input。多轮修改时 user_input 会包含历史，
    如果从整段历史抽取，旧目的地（如迪士尼）会覆盖用户最新说的
    “上海野生动物园/松江大学城”。
    """
    fixed = dict(intent or {})
    user_input = state.get("user_input", "")
    latest_text = str(state.get("latest_user_input") or "")
    collected = state.get("collected_info", {}) or {}

    fixed_departure = str(collected.get("fixed_departure") or state.get("fixed_departure") or "").strip()
    fixed_destination = str(collected.get("fixed_destination") or state.get("fixed_destination") or "").strip()

    latest_departure_hint = extract_departure_hint_from_user_text(latest_text) if latest_text else None
    latest_destination_hint = extract_destination_hint_from_user_text(latest_text) if latest_text else None
    departure_hint = latest_departure_hint if (latest_departure_hint and user_requests_departure_change(latest_text, fixed_departure)) else None
    destination_hint = latest_destination_hint if (latest_destination_hint and user_requests_destination_change(latest_text, fixed_destination)) else None

    # 没有显式修改时，保留已锁定的锚点；只有首轮/无锁定时才从完整输入兜底抽取。
    if not departure_hint and not fixed_departure:
        departure_hint = extract_departure_hint_from_user_text(user_input)
    if not destination_hint:
        if fixed_destination:
            destination_hint = fixed_destination
        else:
            destination_hint = extract_destination_hint_from_user_text(user_input)

    if departure_hint:
        fixed["departure"] = departure_hint
        collected["departure"] = departure_hint
        collected["fixed_departure"] = departure_hint
        collected["_departure_explicit"] = True
    if destination_hint and (destination_hint == fixed_destination or not fixed.get("location") or not is_concrete_location_anchor(str(
            fixed.get("location"))) or is_departure_only_mention(latest_text or user_input,
                                                                 str(fixed.get("location")))):
        fixed["location"] = destination_hint
        collected["location"] = destination_hint
        collected["fixed_destination"] = destination_hint
        collected["active_destination_anchor"] = destination_hint
        collected["_location_explicit"] = True
    if fixed.get("location") and is_departure_only_mention(user_input, str(fixed.get("location"))):
        fixed["location"] = ""
    if fixed.get("location") and not is_concrete_location_anchor(str(fixed.get("location"))):
        fixed["location"] = ""
    if not is_concrete_location_anchor(str(fixed.get("location") or "")):
        if not is_concrete_location_anchor(str(collected.get("location") or "")):
            collected.pop("location", None)
        collected["_location_explicit"] = False

    explicit_people = parse_people_count(user_input, None)
    collected_people = parse_people_count(collected.get("num_people"), None)
    intent_people = parse_people_count(fixed.get("num_people"), None)
    final_people = explicit_people or collected_people or intent_people
    if final_people:
        fixed["num_people"] = final_people

    explicit_place = None
    if not destination_hint or _find_place(destination_hint) is not None:
        explicit_place = find_explicit_place_in_user_input(user_input)
        if explicit_place and (
                is_departure_only_mention(user_input, explicit_place.get("alias", ""))
                or is_departure_only_mention(user_input, explicit_place.get("place_name", ""))
        ):
            explicit_place = None
    if explicit_place:
        row = explicit_place["row"]
        fixed["location"] = explicit_place["place_name"]
        fixed["place_type"] = str(row.get("地点类型") or fixed.get("place_type") or "attraction")
        fixed["explicit_place_match"] = True
        fixed["explicit_place_note"] = (
            f"用户点名“{explicit_place['alias']}”，已匹配到 mock 库地点：{explicit_place['place_name']}。"
        )
        if fixed["place_type"] == "restaurant":
            fixed["meal_pref"] = explicit_place["place_name"]
    else:
        unmatched = detect_unmatched_specific_place(user_input)
        if unmatched:
            fixed["location"] = unmatched
            fixed["explicit_place_match"] = False
            fixed["explicit_place_note"] = (
                f"用户点名“{unmatched}”，但当前 mock 库未找到该地点；规划时必须提示需接入商家/地图 API 核验，不要改成其他店冒充。"
            )

    fixed = resolve_unmatched_location_with_amap(fixed, state)
    fixed = maybe_add_nearby_amap_candidate(fixed, state)
    return fixed


def clamp_duration_hours(value) -> tuple[int, Optional[str]]:
    """总行程统一限制在 4-6 小时内。"""
    duration = _safe_int(value, 5)
    if duration < 4:
        return 4, f"用户原始时长 {duration}h 低于下限，已按比赛约束调整为 4h。"
    if duration > 6:
        return 6, f"用户原始时长 {duration}h 超出上限，已按比赛约束调整为 6h。"
    return duration, None


def extract_transport_mode_from_user_text(user_input: str) -> Optional[str]:
    text = str(user_input or "")
    if any(token in text for token in ["自驾", "开车", "驾车"]):
        return "自驾"
    if any(token in text for token in ["公交", "地铁", "公共交通", "公交地铁"]):
        return "公交地铁"
    if any(token in text for token in ["步行", "走路", "徒步"]):
        return "步行"
    return None


def expand_place_match_terms(user_keywords: set[str], location: str, meal_pref: str = "") -> tuple[
    list[str], list[str]]:
    """把用户口语需求扩展成地点名/标签可匹配的细粒度关键词。"""
    text = f"{location} {meal_pref} {' '.join(user_keywords)}".lower()
    name_terms = []
    negative_name_terms = []
    rules = [
        (["郊区", "近郊", "远郊", "户外", "踏青", "公园"],
         ["公园", "古镇", "农场", "花园", "滨江", "森林", "郊野", "湿地", "罗店", "顾村", "步道"],
         ["万达", "商场", "广场店", "咖啡", "coffee", "cafe"]),
        (["散步", "遛弯", "citywalk", "城市漫步", "街道", "大学路"],
         ["路", "街", "大学路", "武康路", "安福路", "多伦路", "滨江", "步道"], []),
        (["咖啡", "下午茶"], ["咖啡", "coffee", "cafe", "星巴克", "% arabica", "manner"], []),
        (["火锅", "海底捞"], ["火锅", "海底捞", "湊湊", "凑凑", "小龙坎", "哥老官"], []),
        (["小笼包", "小笼", "汤包"], ["小笼", "小笼包", "汤包", "南翔", "来来", "佳家汤包"], []),
        (["生煎"], ["生煎", "小杨生煎", "大壶春"], []),
        (["面馆", "吃面", "汤面", "拉面"], ["面馆", "拉面", "汤面", "小桃面馆"], []),
        (["韩料", "韩国料理", "韩式"], ["韩式", "韩国", "烤肉", "部队锅"], []),
        (["江浙菜", "本帮菜", "上海菜"], ["江浙", "杭帮", "本帮", "上海菜", "半亩田"], []),
        (["面包", "烘焙", "甜品"], ["面包", "烘焙", "蛋糕", "贝果", "甜品"], []),
        (["博物馆", "美术馆", "展馆", "看展", "艺术展"], ["博物馆", "美术馆", "展馆", "画廊", "艺术"], []),
        (["寺庙", "寺"], ["寺", "宝山寺", "龙华寺", "静安寺"], []),
        (["泡汤", "汤泉", "温泉"], ["汤泉", "泡汤", "温泉", "浅山"], []),
        (["二次元", "动漫"], ["二次元", "百联zx", "animate", "谷子", "动漫"], []),
        (["电影", "影院", "影城"], ["影院", "影城", "电影", "cinema"], []),
    ]
    for triggers, positives, negatives in rules:
        if any(trigger.lower() in text for trigger in triggers):
            name_terms.extend(positives)
            negative_name_terms.extend(negatives)
    return sorted(set(name_terms)), sorted(set(negative_name_terms))


def resolve_generic_location(intent: dict) -> dict:
    """用户只说郊区/踏青/户外等泛词时，从结构化 mock 表中选一个具体可用地点。"""
    resolved = dict(intent or {})
    location = str(resolved.get("location") or "").strip()
    compact_location = location.replace(" ", "")
    if compact_location and compact_location not in GENERIC_LOCATION_TERMS and _find_place(location) is not None:
        return resolved
    if resolved.get("explicit_place_match") is False and compact_location:
        resolved["resolved_location_note"] = resolved.get("explicit_place_note", "")
        return resolved

    place_type = str(resolved.get("place_type") or "attraction").strip().lower()
    if place_type not in CANONICAL_PLACE_TYPES:
        place_type = "attraction"

    preferred_types = [place_type]
    if place_type == "leisure":
        preferred_types = ["leisure", "attraction", "activity"]
    elif place_type == "activity":
        preferred_types = ["activity", "leisure", "attraction"]
    elif place_type == "attraction":
        preferred_types = ["attraction", "leisure", "activity"]

    candidates = _df[_df["地点类型"].isin(preferred_types)].copy()
    if candidates.empty:
        candidates = _df.copy()

    user_keywords = set(resolved.get("place_keywords", []) or [])
    name_terms, negative_name_terms = expand_place_match_terms(
        user_keywords,
        location,
        resolved.get("meal_pref", ""),
    )

    def candidate_score(row) -> float:
        name = str(row.get("placeName", "")).lower()
        tags = str(row.get("search_tags", "") or "").lower()
        sub_type = str(row.get("sub_type", "") or "").lower()
        haystack = f"{name} {tags} {sub_type}"
        score = 0.0
        if row.get("地点类型") == place_type:
            score += 30
        if row_has_seat(row):
            score += 40
        if name_terms and any(term.lower() in haystack for term in name_terms):
            score += 60
        if user_keywords and any(str(kw).lower() in haystack for kw in user_keywords):
            score += 45
        if negative_name_terms and any(term.lower() in haystack for term in negative_name_terms):
            score -= 70
        if location and location not in GENERIC_LOCATION_TERMS and location.lower() in haystack:
            score += 80
        try:
            score -= float(row.get("最低价格", 0) or 0) / 20
        except (TypeError, ValueError):
            pass
        return score

    candidates["_candidate_score"] = candidates.apply(candidate_score, axis=1)
    sort_cols = ["_candidate_score"]
    ascending = [False]
    if "最低价格" in candidates.columns:
        sort_cols.append("最低价格")
        ascending.append(True)
    candidates = candidates.sort_values(by=sort_cols, ascending=ascending)

    if candidates.empty:
        return resolved

    nearest = choose_nearest_candidate_row(resolved.get("departure", ""), candidates)
    if resolved.get("departure") and nearest is None:
        resolved["resolved_location_note"] = (
            f"用户输入的是“{location or '未明确'}”这类泛化地点；"
            f"系统没有在出发地“{resolved.get('departure')}”附近找到合格的本地候选，"
            "将交给 structured_plan 阶段优先用高德附近 POI 补充，避免退回全局远距离地点。"
        )
        return resolved
    chosen = nearest["row"] if nearest else candidates.iloc[0]
    chosen_name = str(chosen.get("placeName", "")).strip()
    if chosen_name:
        resolved["location"] = chosen_name
        distance_note = ""
        if nearest:
            distance_note = (
                f"；并结合高德距离，距出发地约 {format_distance(nearest['distance_m'])}，"
                f"约{format_duration(nearest['duration_s'])}"
            )
        resolved["resolved_location_note"] = (
            f"用户输入的是“{location or '未明确'}”这类泛化地点，"
            f"系统按 {place_type} 类型从 mock 表中匹配为：{chosen_name}{distance_note}。"
        )
    return resolved


# ==========================================
# 3. 节点函数
# ==========================================

# ✅ 新增：供 API 调用的信息收集函数（不阻塞，返回追问话术）
def collect_required_info_for_api(state: AgentState) -> AgentState:
    """
    API 版本的信息收集节点：
    - 不使用 input() 阻塞等待
    - 信息不齐全时，将追问话术写入 pending_question 并返回
    - 由 app_api.py 负责将 question 返回给前端
    """
    extract_prompt = ChatPromptTemplate.from_template("""
请从用户输入中提取以下信息，以 JSON 格式返回（只返回 JSON，不加任何说明和 Markdown 标记）：
- departure: 完整出发地点（尽量包含城市/区/地铁站/小区/地标，如"上海人民广场地铁站"，如未提及填 null）
- location: 用户想去的目的地、商圈、店铺、活动类型或地点偏好（如"海底捞""公园""看展""你看着办"，如未提及填 null）
- num_people: 出行人数（整数，如未提及填 null）
- date: 出行日期（具体日期或描述，如"本周六""5月20日"，如未提及填 null）
- time_period: 出行时间段，只能从 ["上午", "下午", "晚上"] 中选择；如果用户说具体几点，按上午/下午/晚上归类
- start_time: 具体出发时间（如"14:00""下午两点""晚上7点"，如未提及填 null）
- duration_hours: 出行时长小时数（如"4小时""半天"，如未提及填 null）
- weather: 出行天气或天气偏好，如"晴天""下雨""阴天""热""冷"，如未提及填 null
- budget: 大致预算（人均金额整数或描述，如"人均100元""200以内"，如未提及填 null）
- group_type: 人群类型，从 [情侣, 家庭, 朋友, 独行] 中选择；如未提及填 null

用户输入: {input}
""")

    extract_chain = extract_prompt | llm | StrOutputParser()

    current_input = state.get("user_input", "")
    collected = dict(state.get("collected_info") or {})

    # 从最新输入中提取信息
    result = extract_chain.invoke({"input": current_input})
    cleaned = re.sub(r"```json|```", "", result).strip()
    try:
        extracted = json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"⚠️ 信息提取解析失败: {result}")
        extracted = {}

    explicit_people = parse_people_count(current_input, None)
    if explicit_people is not None:
        extracted["num_people"] = explicit_people
    elif extracted.get("num_people") is not None:
        extracted["num_people"] = parse_people_count(extracted.get("num_people"), extracted.get("num_people"))

    skip_words = ["看着办", "随便", "都行", "你决定", "你来定", "随机", "无所谓"]
    if any(word in str(extracted.get("location") or "") for word in skip_words):
        extracted["location"] = None
    if extracted.get("budget") is not None:
        extracted["budget"] = normalize_budget_text(extracted.get("budget"))

    departure_hint = extract_departure_hint_from_user_text(current_input)
    destination_hint = extract_destination_hint_from_user_text(current_input)
    latest_text_for_hints = str(state.get("latest_user_input") or "")
    if latest_text_for_hints:
        latest_departure_hint = extract_departure_hint_from_user_text(latest_text_for_hints)
        latest_destination_hint = extract_destination_hint_from_user_text(latest_text_for_hints)
        fixed_departure_for_hint = str(state.get("fixed_departure") or collected.get("fixed_departure") or "")
        fixed_destination_for_hint = str(state.get("fixed_destination") or collected.get("fixed_destination") or "")
        if latest_departure_hint and user_requests_departure_change(latest_text_for_hints, fixed_departure_for_hint):
            departure_hint = latest_departure_hint
        if latest_destination_hint and user_requests_destination_change(latest_text_for_hints, fixed_destination_for_hint):
            destination_hint = latest_destination_hint
    start_time_hint = extract_start_time_hint_from_user_text(current_input)
    transport_mode_hint = extract_transport_mode_from_user_text(current_input)
    if departure_hint:
        extracted["departure"] = departure_hint
    if destination_hint:
        extracted["location"] = destination_hint
    elif departure_hint and normalize_place_text(extracted.get("location")) == normalize_place_text(departure_hint):
        extracted["location"] = None
    elif extracted.get("location") is not None:
        cleaned_location = clean_location_hint_candidate(str(extracted.get("location") or ""))
        if cleaned_location != str(extracted.get("location") or ""):
            extracted["location"] = cleaned_location or None
        if extracted.get("location") and not is_concrete_location_anchor(str(extracted.get("location"))):
            extracted["location"] = None
    if start_time_hint:
        extracted["start_time"] = start_time_hint
        if any(token in start_time_hint for token in ["晚上", "夜里", "傍晚"]):
            extracted["time_period"] = "晚上"
        elif any(token in start_time_hint for token in ["下午"]):
            extracted["time_period"] = "下午"
        elif any(token in start_time_hint for token in ["上午", "早上"]):
            extracted["time_period"] = "上午"

    # 合并已有信息与新提取信息（非 null 值覆盖旧值）
    if transport_mode_hint:
        extracted["transport_mode"] = transport_mode_hint

    extracted, collected, anchor_guard_notes = apply_fixed_anchor_guards(extracted, collected, state)
    if anchor_guard_notes:
        existing_notes = list(state.get("anchor_guard_notes") or [])
        state = {**state, "anchor_guard_notes": existing_notes + anchor_guard_notes}

    for key in ["departure", "location", "num_people", "date", "time_period", "start_time", "duration_hours", "weather",
                "budget", "group_type", "transport_mode"]:
        new_val = extracted.get(key)
        if new_val is not None:
            if key == "budget":
                new_val = normalize_budget_text(new_val, collected.get("budget") or "人均200元以内")
            collected[key] = new_val
            if key == "departure" and str(new_val).strip():
                collected["_departure_explicit"] = True
            if key == "location" and is_concrete_location_anchor(str(new_val)):
                collected["_location_explicit"] = True

    print(f"📝 当前已收集信息: {collected}")

    has_anchor = bool(collected.get("departure")) or is_concrete_location_anchor(str(collected.get("location") or ""))
    missing_followups = []
    if not has_anchor:
        missing_followups.append("location_or_departure")
    for field in ["budget", "transport_mode"]:
        if not collected.get(field):
            missing_followups.append(field)
    already_asked = bool(state.get("info_followup_asked"))
    user_allows_defaults = any(word in current_input for word in skip_words)

    if missing_followups and not already_asked and not user_allows_defaults:
        question = (
            "我再确认一次就开始规划：目的地或想玩的类型是哪里/什么？从哪里出发？"
            "人均预算大概多少？交通方式偏好是步行、公交地铁还是自驾？如果不想细填，也可以直接回“你看着办”。"
        )
        print(f"🤖 一次性追问: {question}")
        return {
            **state,
            "collected_info": collected,
            "info_complete": False,
            "info_followup_asked": True,
            "pending_question": question
        }

    defaults = {
        "departure": None,
        "location": "上海周末休闲活动",
        "num_people": 2,
        "date": "本周末",
        "time_period": "下午",
        "start_time": None,
        "duration_hours": 5,
        "weather": "天气正常",
        "budget": "人均200元以内",
        "group_type": "朋友",
        "transport_mode": "公交地铁",
    }
    for key, value in defaults.items():
        if not collected.get(key):
            collected[key] = value
    collected["budget"] = normalize_budget_text(collected.get("budget"), "人均200元以内")
    collected.setdefault("_departure_explicit", False)
    collected.setdefault("_location_explicit", False)

    print("✅ 信息收集结束：缺失项已按默认值处理")
    enriched_input = (
        f"{current_input} "
        f"（目的地/偏好：{collected['location']}，"
        f"出发地点：{collected.get('departure') or '未指定，以目的地作为起点'}，"
        f"出行人数：{collected['num_people']}人，"
        f"出行日期：{collected['date']}，"
        f"出行时间段：{collected['time_period']}，"
        f"出发时间：{collected.get('start_time') or '未指定'}，"
        f"天气情况：{collected['weather']}，"
        f"大致预算：{collected['budget']}，"
        f"交通方式：{collected['transport_mode']}，"
        f"出行时长：{collected['duration_hours']}小时）"
    )
    return {
        **state,
        "user_input": enriched_input,
        "collected_info": collected,
        "info_complete": True,
        "pending_question": None
    }


def parse_intent(state: AgentState) -> AgentState:
    """解析用户输入中的人群、时间、地点、偏好等约束"""
    if os.getenv("FAST_INTENT_PARSE", "1") == "1":
        collected = state.get("collected_info", {}) or {}
        intent = {
            "group_type": collected.get("group_type", "朋友"),
            "date": collected.get("date", "本周末"),
            "time_period": collected.get("time_period", "下午"),
            "weather": collected.get("weather", "天气正常"),
            "departure": collected.get("departure") or "",
            "location": collected.get("location", "上海周末休闲活动"),
            "duration_hours": collected.get("duration_hours", 5),
            "meal_pref": collected.get("meal_pref", "中餐"),
            "num_people": collected.get("num_people", 2),
            "budget": normalize_budget_text(collected.get("budget"), "人均200元以内"),
            "place_type": collected.get("place_type", "attraction"),
            "place_keywords": [],
            "start_time": collected.get("start_time"),
            "transport_mode": collected.get("transport_mode", "公交地铁"),
        }
        start_time_hint = extract_start_time_hint_from_user_text(state.get("user_input", ""))
        if start_time_hint:
            intent["start_time"] = start_time_hint
            collected["start_time"] = start_time_hint
            if any(token in start_time_hint for token in ["晚上", "夜里", "傍晚"]):
                intent["time_period"] = collected["time_period"] = "晚上"
            elif "下午" in start_time_hint:
                intent["time_period"] = collected["time_period"] = "下午"
            elif any(token in start_time_hint for token in ["上午", "早上"]):
                intent["time_period"] = collected["time_period"] = "上午"
        duration, duration_note = clamp_duration_hours(intent.get("duration_hours", 5))
        intent["duration_hours"] = duration
        if duration_note:
            intent["duration_note"] = duration_note
        intent = normalize_intent_place_type(intent, state.get("user_input", ""))
        intent = reconcile_intent_with_rules(intent, {**state, "collected_info": collected})
        print(f"📋 意图解析结果: {intent}")
        print(f"📍 地点类型归一结果: {intent.get('place_type')} / {intent.get('place_keywords', [])}")
        return {**state, "collected_info": collected, "intent": intent}

    prompt = ChatPromptTemplate.from_template("""
从用户输入中提取以下信息，以JSON格式返回（只返回JSON，不加任何说明和Markdown标记）：
- group_type: 人群类型，从 [情侣, 家庭, 朋友, 独行] 中选择
- date: 出行日期，如未提及填"本周末"
- time_period: 出行时间段，从 [上午, 下午, 晚上] 中选择；如未提及，优先使用已收集信息
- weather: 天气情况或天气偏好，如"晴天""下雨""阴天""热""冷"；如未提及，优先使用已收集信息
- departure: 出发地点
- location: 目标地点或景区名称
- duration_hours: 计划游玩时长（小时，整数），如未提及填 5
  - 总行程必须限制在 4-6 小时，如果用户说半小时/2小时/一天等，也先抽取原始意图，后续系统会强制校正到 4-6 小时
- meal_pref: 餐饮偏好关键词，如未提及填"中餐"
- num_people: 出行人数（整数），如未提及填 2
- budget: 大致预算描述
- place_type: 偏好的地点类型，只能从 [attraction, restaurant, activity, leisure, sports] 中选择，如未提及填"attraction"
  - 如果用户说“郊区/近郊/远郊/踏青/户外/散步/放松”，不要输出 suburban/outdoor，优先填 "leisure"
  - 如果用户说“露营/团建/手作/体验/亲子活动”，填 "activity"
  - 如果用户说“公园/古镇/博物馆/美术馆/展馆/景区”，填 "attraction"
- place_keywords: 从用户原话中提取地点偏好关键词数组，如 ["郊区", "踏青", "户外"]；没有则填 []

用户输入: {input}
""")
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"input": state["user_input"]})

    cleaned = re.sub(r"```json|```", "", result).strip()
    try:
        intent = json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"⚠️ 意图解析 JSON 失败，使用默认值。原始返回: {result}")
        collected = state.get("collected_info", {}) or {}
        intent = {
            "group_type": "朋友",
            "location": "景区",
            "departure": collected.get("departure", "市区"),
            "date": collected.get("date", "本周末"),
            "time_period": collected.get("time_period", "下午"),
            "weather": collected.get("weather", "晴天"),
            "num_people": collected.get("num_people", 2),
            "duration_hours": 5,
            "meal_pref": "中餐",
            "budget": normalize_budget_text(collected.get("budget"), "适中"),
            "place_type": "attraction"
        }

    print(f"📋 意图解析结果: {intent}")
    collected = state.get("collected_info", {}) or {}
    transport_mode_hint = extract_transport_mode_from_user_text(state.get("user_input", ""))
    if transport_mode_hint:
        collected["transport_mode"] = transport_mode_hint
    for key in ["departure", "location", "date", "time_period", "start_time", "weather", "num_people", "budget",
                "duration_hours", "group_type", "transport_mode"]:
        if collected.get(key):
            intent[key] = collected[key]
    start_time_hint = extract_start_time_hint_from_user_text(state.get("user_input", ""))
    if start_time_hint:
        intent["start_time"] = start_time_hint
        collected["start_time"] = start_time_hint
        if any(token in start_time_hint for token in ["晚上", "夜里", "傍晚"]):
            intent["time_period"] = collected["time_period"] = "晚上"
        elif "下午" in start_time_hint:
            intent["time_period"] = collected["time_period"] = "下午"
        elif any(token in start_time_hint for token in ["上午", "早上"]):
            intent["time_period"] = collected["time_period"] = "上午"
    duration, duration_note = clamp_duration_hours(intent.get("duration_hours", 5))
    intent["duration_hours"] = duration
    if duration_note:
        intent["duration_note"] = duration_note
    intent = normalize_intent_place_type(intent, state.get("user_input", ""))
    intent = reconcile_intent_with_rules(intent, state)
    print(f"📍 地点类型归一结果: {intent.get('place_type')} / {intent.get('place_keywords', [])}")
    return {**state, "intent": intent}


def rag_retrieval(state: AgentState) -> AgentState:
    """从知识库检索相关出行案例。

    改进点：
    - 原来只用「人群 + 地点 + 类型 + 出行规划」检索，容易丢失天气、预算、时长、室内外等关键约束；
    - 现在把用户原始需求和解析出的结构化字段一起放进 query；
    - context 保留来源分隔符，方便后续提示词更稳定地引用。
    """
    intent = state["intent"] or {}
    collected = state.get("collected_info", {}) or {}
    weather_info = state.get("weather_info") or {}

    query_parts = [
        state.get("user_input", ""),
        intent.get("group_type", ""),
        intent.get("date", ""),
        intent.get("time_period", collected.get("time_period", "")),
        intent.get("weather", collected.get("weather", "")),
        weather_info.get("summary", ""),
        intent.get("departure", collected.get("departure", "")),
        intent.get("location", ""),
        intent.get("place_type", ""),
        " ".join(intent.get("place_keywords", []) or []),
        intent.get("place_type_reason", ""),
        intent.get("meal_pref", ""),
        str(intent.get("budget", collected.get("budget", ""))),
        f"{intent.get('duration_hours', '')}小时" if intent.get("duration_hours") else "",
        "上海 周末 本地 活动 出行 规划 路线 预算 门票 预约",
    ]
    query = " ".join([str(x) for x in query_parts if x])

    docs = retriever.invoke(query)

    context_blocks = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", doc.metadata.get("doc_id", ""))
        context_blocks.append(
            f"【参考案例{idx}｜source={source}｜chunk={chunk_id}】\n{doc.page_content}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    print(f"📚 RAG 检索完成，共召回 {len(docs)} 条相关案例")
    return {**state, "rag_context": context}


def tool_dispatch(state: AgentState) -> AgentState:
    """并行调用工具：查景点、查门票、规划路线"""
    intent = resolve_generic_location(state["intent"] or {})
    location = intent.get("location", "景区")
    date = intent.get("date", "本周末")
    num_people = _safe_int(intent.get("num_people", 2), 2)
    meal_pref = intent.get("meal_pref", "中餐")
    duration, _ = clamp_duration_hours(intent.get("duration_hours", 5))

    print(f"🔧 开始工具调用: 地点={location}, 日期={date}, 人数={num_people}")
    if intent.get("resolved_location_note"):
        print(f"  ├─ 地点解析: {intent['resolved_location_note']}")

    if (
            os.getenv("BLOCK_UNMATCHED_PLACE_EARLY", "0") == "1"
            and intent.get("explicit_place_match") is False
            and _find_place(location) is None
    ):
        attraction_info = f"未找到 {location} 的相关信息"
        ticket_info = f"未找到 {location} 的价格信息"
        route_plan = (
            f"用户点名地点 {location} 当前 mock 库未找到；系统不自动替换为其他同类地点。"
            "建议接入商家/地图 API 核验，或让用户换一个已入库地点。"
        )
        coupon_info = {"items": [], "summary": "团购券信息：当前点名地点未入库，未发现可用团购券。",
                       "checked_places": [location]}
    else:
        attraction_info = search_attraction.invoke({"location": location, "date": date})
        ticket_info = check_ticket.invoke({"attraction": location, "date": date, "num_people": num_people})
        route_plan = (
            f"候选路线草稿: {location} → 周边餐饮/轻量休闲 → 返程, "
            f"预计总时长 {duration}h。具体餐饮、团购券和可预订地点以后续 structured_plan.schedule 为准。"
        )
        coupon_info = {
            "items": [],
            "summary": "团购券信息：等待结构化路线生成后，按最终方案地点统一校验。",
            "checked_places": [],
        }

    print(f"  ├─ 景点信息: {attraction_info}")
    print(f"  ├─ 门票信息: {ticket_info}")
    print(f"  └─ 路线规划: {route_plan}")

    return {
        **state,
        "intent": intent,
        "attraction_info": attraction_info,
        "ticket_info": ticket_info,
        "route_plan": route_plan,
        "coupon_info": coupon_info,
    }


def extract_tool_place_name(tool_text: str, fallback: str = "") -> str:
    text = str(tool_text or "").strip()
    if "|" in text:
        return text.split("|", 1)[0].strip() or fallback
    match = re.search(r"(?:未找到|找到)?\s*([^，。|]+?)(?:\s*的|\s*\|)", text)
    return (match.group(1).strip() if match else str(fallback or "").strip())


def find_available_alternative_for_unavailable(original_place: str, intent: dict, state: AgentState) -> Optional[str]:
    """Find an available backup without changing a user-fixed start/destination anchor.

    Priority is local table first; Amap is used only as a small last-mile supplement.
    """
    original_key = normalize_place_text(original_place)
    collected = state.get("collected_info") or {}
    place_type = str((intent or {}).get("place_type") or "").strip()
    original_row = _find_place(original_place)
    if original_row is not None:
        place_type = str(original_row.get("地点类型") or place_type or "attraction")
    anchor = (
        collected.get("fixed_destination")
        or collected.get("location")
        or (intent or {}).get("location")
        or collected.get("departure")
        or (intent or {}).get("departure")
        or ""
    )
    anchor_key = normalize_place_text(anchor)
    candidates = []
    for _, row in _df.iterrows():
        name = str(row.get("placeName", "") or "").strip()
        if not name:
            continue
        key = normalize_place_text(name)
        if not key or key == original_key or same_route_place(name, original_place):
            continue
        if place_type and str(row.get("地点类型") or "") != place_type:
            continue
        if not row_has_seat(row):
            continue
        searchable = normalize_place_text(" ".join([
            name,
            str(row.get("search_tags", "") or ""),
            str(row.get("amap_address", "") or ""),
            str(row.get("source_note", "") or ""),
        ]))
        score = 0
        if anchor_key and (anchor_key in searchable or searchable in anchor_key):
            score += 120
        if place_has_coupon(name):
            score += 10
        if place_is_indoor(name):
            score += 5
        try:
            score -= float(row.get("最低价格", 0) or 0) / 50
        except (TypeError, ValueError):
            pass
        candidates.append((-score, name))
    candidates.sort()
    if candidates:
        return candidates[0][1]

    # Small Amap supplement only if local table cannot provide an available backup.
    if os.getenv("ENABLE_AMAP_POI_SEARCH", "1") == "1" and get_amap_key() and anchor:
        spec = default_amap_search_spec(intent or {}, state.get("user_input", ""))
        pois = amap_search_pois_near(anchor, spec.get("keyword") or "餐厅", limit=2)
        for poi in pois[:2]:
            name = str(poi.get("name") or "").strip()
            if name and not same_route_place(name, original_place):
                try:
                    return persist_amap_poi_with_mock(poi, spec)
                except Exception:
                    return name
    return None


def build_exception_event(kind: str, original_place: str, backup_place: Optional[str], state: AgentState, reason: str) -> dict:
    collected = state.get("collected_info") or {}
    fixed_bits = []
    if collected.get("fixed_departure") or collected.get("departure"):
        fixed_bits.append(f"出发地“{collected.get('fixed_departure') or collected.get('departure')}”")
    if collected.get("fixed_destination") or (collected.get("_location_explicit") and collected.get("location")):
        fixed_bits.append(f"目的地“{collected.get('fixed_destination') or collected.get('location')}”")
    fixed_text = "、".join(fixed_bits) if fixed_bits else "用户已明确的路线锚点"
    if backup_place:
        message = f"{original_place} 当前{reason}，不会改掉{fixed_text}；已切换到附近/同类型备选“{backup_place}”。"
    else:
        message = f"{original_place} 当前{reason}，不会改掉{fixed_text}；暂未找到可用备选，请放宽区域/类型或出行前二次核验。"
    return {
        "type": kind,
        "original_place": original_place,
        "backup_place": backup_place or "",
        "message": message,
        "display_level": "warning",
    }


def exception_handler(state: AgentState) -> AgentState:
    """检测无座/无票/时间冲突；不再全局改目的地，只给出前端可见异常和附近备选。"""
    intent = state["intent"]
    attraction_info = state.get("attraction_info", "")
    ticket_info = state.get("ticket_info", "")
    duration, duration_note = clamp_duration_hours(intent.get("duration_hours", 5))
    place_type = intent.get("place_type", "attraction")

    exception_events = list(state.get("exception_events") or [])
    unavailable_places = list(state.get("unavailable_places") or [])
    suggested_backup_places = list(state.get("suggested_backup_places") or [])

    if "已满" in attraction_info:
        original = extract_tool_place_name(attraction_info, intent.get("location", "原目标地点"))
        backup_name = find_available_alternative_for_unavailable(original, intent, state)
        event = build_exception_event("place_full", original, backup_name, state, "已满/暂不可预约")
        exception_events.append(event)
        unavailable_places.append(original)
        if backup_name:
            suggested_backup_places.append(backup_name)
            backup_info = search_attraction.invoke({"location": backup_name, "date": intent.get("date", "本周末")})
            backup_ticket = check_ticket.invoke({
                "attraction": backup_name,
                "date": intent.get("date", "本周末"),
                "num_people": int(parse_people_count(intent.get("num_people", 2), 2) or 2),
            })
        else:
            backup_info = attraction_info
            backup_ticket = ticket_info
        exception_msg = event["message"]
        print(f"⚠️ 异常处理: {exception_msg}")
        return {
            **state,
            "attraction_info": backup_info,
            "ticket_info": backup_ticket,
            "exception": exception_msg,
            "exception_events": exception_events,
            "unavailable_places": unique_preserve_order(unavailable_places),
            "suggested_backup_places": unique_preserve_order(suggested_backup_places),
            "avoid_places": unique_preserve_order((state.get("avoid_places") or []) + [original]),
        }

    if "售罄" in ticket_info or "库存不足" in ticket_info:
        original = extract_tool_place_name(ticket_info, intent.get("location", "原目标地点"))
        backup_name = find_available_alternative_for_unavailable(original, intent, state)
        event = build_exception_event("ticket_sold_out", original, backup_name, state, "售罄/库存不足")
        exception_events.append(event)
        unavailable_places.append(original)
        if backup_name:
            suggested_backup_places.append(backup_name)
            backup_info = search_attraction.invoke({"location": backup_name, "date": intent.get("date", "本周末")})
            backup_ticket = check_ticket.invoke({
                "attraction": backup_name,
                "date": intent.get("date", "本周末"),
                "num_people": int(parse_people_count(intent.get("num_people", 2), 2) or 2),
            })
        else:
            backup_info = attraction_info
            backup_ticket = ticket_info
        exception_msg = event["message"]
        print(f"⚠️ 异常处理: {exception_msg}")
        return {
            **state,
            "attraction_info": backup_info,
            "ticket_info": backup_ticket,
            "exception": exception_msg,
            "exception_events": exception_events,
            "unavailable_places": unique_preserve_order(unavailable_places),
            "suggested_backup_places": unique_preserve_order(suggested_backup_places),
            "avoid_places": unique_preserve_order((state.get("avoid_places") or []) + [original]),
        }

    if duration_note:
        exception_msg = duration_note
        print(f"⚠️ 异常处理: {exception_msg}")
        return {
            **state,
            "intent": {**intent, "duration_hours": duration, "duration_note": duration_note},
            "exception": exception_msg
        }

    print("✅ 异常检测通过，无需切换备选")
    return {**state, "exception": None}


def route_distance_planner(state: AgentState) -> AgentState:
    """基于高德 API 计算 structured_plan.schedule 的真实转场距离；仅作为参考，不做固定距离硬拦截。"""
    intent = state.get("intent") or {}
    collected = state.get("collected_info", {}) or {}
    location = intent.get("location", "景区")
    structured_plan = state.get("structured_plan") or {}
    hard_constraints = structured_plan.get("hard_constraints", {}) if isinstance(structured_plan, dict) else {}
    departure = hard_constraints.get("departure") or collected.get("departure") or intent.get("departure", "")
    if not bool(collected.get("_departure_explicit")) and hard_constraints.get("planning_anchor_mode") == "destination":
        departure = hard_constraints.get("planning_anchor") or (structured_plan.get("places") or [""])[0]
    meal_pref = intent.get("meal_pref", "中餐")
    duration, _ = clamp_duration_hours(intent.get("duration_hours", 5))
    route_plan_text = state.get("route_plan", "")

    if (
            os.getenv("BLOCK_UNMATCHED_PLACE_EARLY", "0") == "1"
            and intent.get("explicit_place_match") is False
            and _find_place(location) is None
    ):
        route_distance_info = f"高德距离参考未执行：用户点名地点 {location} 当前 mock 库未找到，需要先接入地图/商家 API 核验。"
        coupon_info = state.get("coupon_info") or {
            "items": [],
            "summary": "团购券信息：当前点名地点未入库，未发现可用团购券。",
            "checked_places": [location],
        }
        print(f"📏 高德距离参考完成:\n{route_distance_info}")
        print(f"🎟️ 团购券校验完成:\n{coupon_info.get('summary')}")
        return {
            **state,
            "route_plan": route_plan_text,
            "route_distance_info": route_distance_info,
            "coupon_info": coupon_info,
        }

    if location and location not in str(route_plan_text):
        route_plan_text = plan_route.invoke({
            "attractions": location,
            "meal_pref": meal_pref,
            "total_hours": duration,
        })

    schedule_stops = [
        item.get("place")
        for item in structured_plan.get("schedule", [])
        if isinstance(item, dict) and item.get("place")
    ]
    if schedule_stops:
        route_distance_info, route_segments, failed_segments = compute_route_segments(departure, schedule_stops)
        structured_plan = attach_route_segments_to_structured_plan(structured_plan, route_segments)
        coupon_info = build_coupon_info_for_places(schedule_stops)
        validation = structured_plan.setdefault("route_logic_validation", {})
        validation["distance_reference_only"] = True
        validation["failed_segments"] = failed_segments
        validation.setdefault("notes", []).append("高德距离仅作为转场参考和前端展示，不再按固定公里数硬拦截或移除地点。")
        soft_warnings = long_route_segment_warnings(route_segments)
        if soft_warnings:
            validation.setdefault("soft_distance_warnings", []).extend(soft_warnings)
            validation.setdefault("adjustment_conflicts", []).extend(soft_warnings)
            validation.setdefault("notes", []).extend(soft_warnings)
    else:
        route_segments = []
        failed_segments = []
        validation = structured_plan.setdefault("route_logic_validation", {}) if isinstance(structured_plan, dict) else {}
        validation["ok"] = False
        validation["distance_reference_only"] = True
        validation.setdefault("notes", []).append(
            "structured_plan 中没有剩余可用地点；已停止使用旧 route_plan/location 兜底，避免重新引入不一致地点。"
        )
        route_distance_info = (
            "高德距离参考未生成可执行路线：structured_plan 中没有剩余可用地点。"
            "请放宽地点/预算/室内外要求，或检查高德 API 是否能返回附近 POI。"
        )
        coupon_info = build_coupon_info_for_places([])

    if state.get("adjustment_mode") == "less_walk":
        route_distance_info = (
            f"{route_distance_info}\n少走路模式：除站内/店内必要步行外，优先地铁、骑行或打车转场；不建议安排长距离步行串联。"
        )
    elif "less_walk" in (state.get("adjustment_modes") or []):
        route_distance_info = (
            f"{route_distance_info}\n少走路模式：除站内/店内必要步行外，优先地铁、骑行或打车转场；不建议安排长距离步行串联。"
        )
    feasibility_report = evaluate_route_feasibility(structured_plan, intent, route_segments) if structured_plan else {"intensity_score": 0, "warnings": ["结构化方案为空，无法评估路线强度。"]}
    if structured_plan:
        structured_plan["feasibility_report"] = feasibility_report
        structured_plan.setdefault("tool_facts", {})
        structured_plan["tool_facts"]["route_distance_info"] = route_distance_info
        structured_plan["tool_facts"]["coupon_summary"] = coupon_info.get("summary", "")
    route_map = build_route_map_placeholder(structured_plan) if structured_plan else {"available": False,
                                                                                      "reason": "结构化方案为空，未生成地图。"}
    print(f"📏 高德距离参考完成:\n{route_distance_info}")
    print(f"🎟️ 团购券校验完成:\n{coupon_info.get('summary')}")
    print(f"🗺️ 路线地图状态: {route_map.get('note') or route_map.get('reason')}")
    return {
        **state,
        "route_plan": route_plan_text,
        "route_distance_info": route_distance_info,
        "route_map": route_map,
        "coupon_info": coupon_info,
        "feasibility_report": feasibility_report,
        "structured_plan": structured_plan or state.get("structured_plan"),
    }

def truncate_text(text: str, limit: int = 2600) -> str:
    """控制进入规划模型的上下文长度，避免挤占回答空间。"""
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n……（已截取最相关的前半部分，避免方案生成过长）"


def complete_plan_if_needed(plan: str, plan_llm: ChatDashScope) -> str:
    """如果模型没有输出结束标记，补一段简短结尾，避免前端看到半句话。"""
    marker = "【方案结束】"
    if marker in (plan or "")[-120:]:
        return plan

    continuation_prompt = ChatPromptTemplate.from_template("""
下面是一份出行方案草稿，但它可能在结尾处被截断了。
请只补全最后缺失的结尾，不要重复前文，不要新增不存在的地点。
补全内容控制在 200 字以内，并必须以【方案结束】结尾。

【方案草稿末尾】:
{tail}
""")
    continuation_chain = continuation_prompt | plan_llm | StrOutputParser()
    try:
        tail = plan[-900:] if plan else ""
        continuation = continuation_chain.invoke({"tail": tail}).strip()
        if continuation:
            plan = plan.rstrip() + "\n\n" + continuation
    except Exception as e:
        print(f"⚠️ 方案补全失败，使用兜底结束语: {e}")
        plan = plan.rstrip() + "\n\n以上方案可以先作为当前版本参考，后续可根据你的反馈继续调整。【方案结束】"

    if marker not in plan[-220:]:
        plan = plan.rstrip() + "\n\n【方案结束】"
    return plan


def plan_has_coupon_claim(plan: str) -> bool:
    text = str(plan or "")
    negative_patterns = [
        "无可用团购券", "未发现可用团购券", "没有可用团购券", "暂无团购券",
        "当前路线无可用团购券", "未找到可用团购券",
    ]
    stripped = text
    for pattern in negative_patterns:
        stripped = stripped.replace(pattern, "")

    positive_patterns = [
        r"满\s*\d+\s*减\s*\d+",
        r"\d+(?:\.\d+)?\s*元?\s*代\s*\d+(?:\.\d+)?\s*元?",
        r"券码", r"美团券", r"抖音券", r"抖音生活", r"团购价",
        r"立减", r"折扣券", r"套餐券", r"今日可用券",
        r"已锁定", r"下单即用", r"VIP通道", r"免排队权益",
        r"有团购券", r"可用团购券",
    ]
    return any(re.search(pattern, stripped) for pattern in positive_patterns)


def remove_unsupported_coupon_claims(plan: str, coupon_info: dict, plan_llm: ChatDashScope) -> str:
    """如果 mock 数据没有券但模型编了券，二次改写，清掉无依据团购描述。"""
    items = (coupon_info or {}).get("items") or []
    if items or not plan_has_coupon_claim(plan):
        return plan

    repair_prompt = ChatPromptTemplate.from_template("""
下面这份出行方案里出现了团购券/代金券/美团券/抖音券/满减/VIP免排队等说法，但结构化 mock 数据明确显示：当前路线无可用团购券。

请在不改变核心路线、时间段、地点顺序的前提下，改写这份方案：
- 删除所有无依据的团购券、代金券、券码、抖音/美团优惠、VIP免排队、已锁定券码等说法。
- 费用部分改成普通价格估算，只能使用【景点/场馆信息】【门票/费用信息】【路线规划】中有依据的信息。
- 明确写一句：当前路线涉及地点在 mock 数据中未发现可用团购券。
- 不要新增新地点。
- 输出完整方案，最后必须以【方案结束】结尾。

【原方案】
{plan}

【景点/场馆信息】
{attraction_info}

【门票/费用信息】
{ticket_info}

【路线规划】
{route_plan}
""")
    try:
        repaired = (repair_prompt | plan_llm | StrOutputParser()).invoke({
            "plan": plan,
            "attraction_info": "见当前规划上下文，禁止补充未给出的团购券。",
            "ticket_info": "见当前规划上下文，禁止补充未给出的团购券。",
            "route_plan": "见当前规划上下文，保持原路线，不新增地点。",
        }).strip()
        repaired = complete_plan_if_needed(repaired, plan_llm)
        return repaired.replace("【方案结束】", "").strip()
    except Exception as e:
        print(f"⚠️ 团购券幻觉修正失败，使用规则兜底: {e}")
        return (
                plan.rstrip()
                + "\n\n⚠️ 团购券核验：当前路线涉及地点在 mock 数据中未发现可用团购券；"
                  "上文如出现团购券、代金券或满减说法，请以本条结构化核验结果为准。"
        )


def sanitize_final_plan_text(plan: str, num_people) -> str:
    """Clean display markdown noise and enforce the collected people count in obvious group phrases."""
    text = str(plan or "")
    people = parse_people_count(num_people, None)

    # Convert markdown bullets before removing emphasis markers.
    text = re.sub(r"(?m)^\s*[\-*]\s+", "• ", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = text.replace("**", "").replace("*", "")

    if people:
        people_text = f"{people}人"
        numeral = "一二两俩三四五六七八九十"
        wrong_group_pattern = rf"(\d+|[{numeral}])\s*个?\s*人(?=(?:组队|同行|一起|出行|小队|局|聚会|朋友|好友|去|吃|玩))"
        text = re.sub(wrong_group_pattern, people_text, text)
        title_group_pattern = rf"(\d+|[{numeral}])\s*人(组队|小分队|同行|出行|吃喝|饭局)"
        text = re.sub(title_group_pattern, lambda m: people_text + m.group(2), text)

        if f"【出行人数】" not in text and not re.search(rf"(本次|这次|适合).{{0,8}}{people_text}", text):
            text = f"本次按 {people_text} 出行规划。\n" + text

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def ensure_schedule_places_rendered(plan: str, structured_plan: dict) -> str:
    """如果模型漏写了 structured_plan 中的地点，追加一段结构化核对，避免核心地点消失。"""
    text = str(plan or "").strip()
    missing_items = []
    for item in (structured_plan or {}).get("schedule", []) or []:
        if not isinstance(item, dict):
            continue
        place = str(item.get("place") or "").strip()
        if not place or place == "待确认地点" or place_matches_text(place, text):
            continue
        transport = (item.get("transport_from_previous") or {}).get("summary", "")
        purpose = item.get("purpose", "")
        time_slot = item.get("time", "")
        missing_items.append((time_slot, place, transport, purpose))

    if not missing_items:
        return text

    lines = ["", "路线地点核对："]
    for time_slot, place, transport, purpose in missing_items:
        detail = f"{time_slot} {place}"
        if transport:
            detail += f"｜{transport}"
        if purpose:
            detail += f"｜{purpose}"
        lines.append(f"• {detail}")
    return (text + "\n".join(lines)).strip()


def append_canonical_schedule_table(plan: str, structured_plan: dict) -> str:
    schedule = [
        item for item in (structured_plan or {}).get("schedule", []) or []
        if isinstance(item, dict) and item.get("place")
    ]
    if not schedule:
        return str(plan or "").strip()
    lines = ["", "最终时间表："]
    for item in schedule:
        lines.append(f"• {item.get('time', '时间待定')}｜{item.get('place')}｜{item.get('purpose', '行程地点')}")
    text = str(plan or "").strip()
    if "最终时间表：" in text:
        return text
    return (text + "\n" + "\n".join(lines)).strip()


def parse_start_time_minutes(start_time, time_period: str = "") -> Optional[int]:
    text = str(start_time or "").strip()
    if not text or text.lower() in {"none", "null"}:
        return None

    match = re.search(r"(\d{1,2})\s*[:：点]\s*(\d{1,2})?", text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
    else:
        cn_map = {
            "一": 1, "二": 2, "两": 2, "俩": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12,
        }
        hour = None
        for token, value in sorted(cn_map.items(), key=lambda kv: len(kv[0]), reverse=True):
            if token in text:
                hour = value
                break
        if hour is None:
            return None
        minute = 30 if "半" in text else 0

    period_text = f"{start_time} {time_period}"
    if hour <= 12 and any(token in period_text for token in ["下午", "晚上", "夜里", "傍晚"]):
        if hour < 12:
            hour += 12
    if hour == 12 and any(token in period_text for token in ["凌晨", "早上", "上午"]):
        hour = 0 if "凌晨" in period_text else 12
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour * 60 + minute


def format_time_minutes(minutes: int) -> str:
    minutes = max(0, min(minutes, 23 * 60 + 59))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def build_schedule_slots(collected: dict, intent: dict, count: int) -> list[str]:
    time_period = collected.get("time_period") or intent.get("time_period") or "下午"
    start_minutes = parse_start_time_minutes(collected.get("start_time") or intent.get("start_time"), time_period)
    if start_minutes is None:
        default_start = {"上午": 9 * 60, "下午": 13 * 60 + 30, "晚上": 18 * 60}
        start_minutes = default_start.get(str(time_period), 13 * 60 + 30)

    total_hours, _ = clamp_duration_hours(
        collected.get("duration_hours") or intent.get("duration_hours") or 5
    )
    total_minutes = total_hours * 60
    count = max(1, int(count or 1))
    base = total_minutes // count
    remainder = total_minutes % count
    durations = [base + (1 if index < remainder else 0) for index in range(count)]
    slots = []
    cursor = start_minutes
    for index, duration in enumerate(durations):
        end = min(cursor + duration, 23 * 60 + 59)
        slots.append(f"{format_time_minutes(cursor)}-{format_time_minutes(end)}")
        cursor = end
    return slots


def build_structured_plan(state: AgentState) -> AgentState:
    """Build the strict structured plan first; result_formatter only renders it into copy."""
    intent = dict(state.get("intent", {}) or {})
    if state.get("route_variant_seed"):
        intent["route_variant_seed"] = state.get("route_variant_seed")
    collected = state.get("collected_info", {}) or {}
    weather_info = state.get("weather_info") or {}
    route_plan = state.get("route_plan", "") or ""
    coupon_info = state.get("coupon_info") or {}
    locked_places = state.get("locked_places") or []
    anchor_name, anchor_mode = planning_anchor_for_intent(intent, collected)
    locked_places = filter_replaced_destinations_from_places(locked_places, state, anchor_name)
    planning_context = {**collected, **intent, "route_variant_seed": state.get("route_variant_seed")}
    min_route_stops, max_route_stops = route_stop_bounds(state, intent, collected)
    departure_explicit = bool(collected.get("_departure_explicit"))
    location_explicit = bool(collected.get("_location_explicit"))
    if anchor_mode == "destination" and departure_explicit:
        route_logic_mode = "fixed_start_destination"
    elif anchor_mode == "destination":
        route_logic_mode = "single_anchor_destination"
    elif departure_explicit:
        route_logic_mode = "single_anchor_departure"
    else:
        route_logic_mode = "open_default"
    departure_only_mode = bool(collected.get("_departure_explicit")) and not is_destination_anchor_intent(intent,
                                                                                                          collected)
    if departure_only_mode:
        raw_places = unique_preserve_order(list(locked_places))
        route_plan_candidates = vary_candidates(
            extract_meal_candidates(route_plan),
            planning_context,
            "route_plan_departure",
        )
        local_pool_places = nearby_existing_places_from_local_pool(
            anchor_name,
            raw_places,
            planning_context,
            limit=max(0, min_route_stops - len(raw_places)),
        )
        raw_places.extend(local_pool_places)
        route_plan_places = []
        if len(raw_places) < min_route_stops:
            route_plan_places = nearby_route_plan_places(
                anchor_name,
                route_plan_candidates,
                raw_places,
                limit=max(0, min_route_stops - len(raw_places)),
            )
            raw_places.extend(route_plan_places)
        amap_places = []
        if len(raw_places) < min_route_stops and sync_amap_complement_enabled():
            search_target = intent.get("location") if not is_category_like_location(
                intent.get("location")) else intent.get("meal_pref")
            needed = max(0, min_route_stops - len(raw_places))
            amap_places = find_nearby_complement_places(
                anchor_name,
                raw_places + [anchor_name],
                {**planning_context, "location": search_target or intent.get("location")},
                limit=needed,
            )
            raw_places.extend(amap_places[:needed])
        if route_plan_places or local_pool_places:
            structure_anchor_note = (
                f"用户明确从“{anchor_name}”出发，已优先使用本地地点表/路线草稿中的候选："
                f"{'、'.join(local_pool_places + route_plan_places)}。"
            )
            if amap_places:
                structure_anchor_note += f" 本地近距离候选不足，已用高德补充：{'、'.join(amap_places)}。"
        elif amap_places:
            structure_anchor_note = f"用户明确从“{anchor_name}”出发，旧案例附近候选不足，已调用高德补充：{'、'.join(amap_places)}。"
        else:
            structure_anchor_note = f"用户明确从“{anchor_name}”出发，但 RAG/本地旧案例和高德周边均未补到足够候选。"
    elif anchor_mode == "destination":
        raw_places = unique_preserve_order(list(locked_places))
        route_plan_candidates = vary_candidates(
            extract_meal_candidates(route_plan),
            planning_context,
            "route_plan_destination",
        )
        local_pool_places = nearby_existing_places_from_local_pool(
            anchor_name,
            raw_places + [anchor_name],
            planning_context,
            limit=max(0, min_route_stops - 1 - len(raw_places)),
        )
        raw_places.extend(local_pool_places)
        route_plan_places = []
        if len(raw_places) < max(1, min_route_stops - 1):
            route_plan_places = nearby_route_plan_places(
                anchor_name,
                route_plan_candidates,
                raw_places + [anchor_name],
                limit=max(0, min_route_stops - 1 - len(raw_places)),
            )
            raw_places.extend(route_plan_places)
        amap_places = []
        if len(raw_places) < max(1, min_route_stops - 1) and sync_amap_complement_enabled():
            needed = max(0, min_route_stops - 1 - len(raw_places))
            amap_places = find_nearby_complement_places(anchor_name, raw_places + [anchor_name], planning_context, limit=needed)
            raw_places.extend(amap_places[:needed])
        if bool(collected.get("_departure_explicit")):
            raw_places = move_destination_anchor_to_end(raw_places, anchor_name)
        else:
            raw_places = move_destination_anchor_to_start(raw_places, anchor_name)
        if route_plan_places or local_pool_places:
            structure_anchor_note = (
                f"已按用户目的地“{anchor_name}”优先使用本地地点表/路线草稿中的候选："
                f"{'、'.join(local_pool_places + route_plan_places)}。"
            )
            if amap_places:
                structure_anchor_note += f" 本地近距离候选不足，已用高德补充：{'、'.join(amap_places)}。"
        elif amap_places:
            structure_anchor_note = f"用户目的地“{anchor_name}”周边旧案例不足，已调用高德补充路线地点：{'、'.join(amap_places)}。"
        else:
            structure_anchor_note = f"用户目的地“{anchor_name}”已作为路线锚点；RAG/本地旧案例和高德周边均未补到足够候选。"
    else:
        raw_places = unique_preserve_order(
            list(locked_places) + [intent.get("location", "")] + extract_meal_candidates(route_plan))
        structure_anchor_note = f"未检测到明确目的地，已按出发地“{anchor_name or '默认出发地'}”附近优先规划。"
    raw_places = filter_replaced_destinations_from_places([p for p in raw_places if p], state, anchor_name)

    # Reservation/availability exceptions are visible to the frontend and should only
    # affect the problematic stop, not user-fixed start/end anchors.
    exception_events = list(state.get("exception_events") or [])
    unavailable_keys = {normalize_place_text(p) for p in (state.get("unavailable_places") or []) if p}
    if unavailable_keys:
        before_unavailable = list(raw_places)
        raw_places = [p for p in raw_places if normalize_place_text(p) not in unavailable_keys]
        removed = [p for p in before_unavailable if normalize_place_text(p) in unavailable_keys]
        if removed:
            structure_anchor_note += f" 已避开当前不可预约/已满地点：{'、'.join(removed)}。"
    backup_places = [p for p in (state.get("suggested_backup_places") or []) if p]
    if backup_places:
        raw_places = unique_preserve_order(list(locked_places) + raw_places + backup_places)
        structure_anchor_note += f" 已把可用备选加入候选池：{'、'.join(backup_places)}。"
    anchor_guard_notes = list(state.get("anchor_guard_notes") or [])
    if anchor_guard_notes:
        structure_anchor_note += " " + " ".join(anchor_guard_notes[-2:])

    if structure_anchor_note:
        if route_logic_mode == "fixed_start_destination":
            structure_anchor_note = (
                f"用户同时明确出发地“{collected.get('departure')}”和目的地“{anchor_name}”；"
                "本次把两端固定，只在中间补充尽量顺路、少绕行的候选。"
                f"{structure_anchor_note}"
            )
        elif route_logic_mode == "single_anchor_destination":
            structure_anchor_note = (
                f"用户只明确目的地“{anchor_name}”；本次把它作为单点锚点，围绕它生成附近游玩路线。"
                f"{structure_anchor_note}"
            )
        elif route_logic_mode == "single_anchor_departure":
            structure_anchor_note = (
                f"用户只明确出发地“{anchor_name}”；本次把它作为单点锚点，围绕它生成附近游玩路线。"
                f"{structure_anchor_note}"
            )
        else:
            structure_anchor_note = (
                f"已按默认锚点“{anchor_name or '上海人民广场'}”生成附近候选；"
                "餐饮/咖啡候选不再使用固定名单，并会清理连续咖啡或连续正餐。"
            )
    if anchor_mode != "destination" and sync_amap_complement_enabled() and (
            len(raw_places) < 2 or os.getenv("ENABLE_STRUCTURED_COMPLEMENT_SEARCH", "0") == "1"):
        complement_anchor = anchor_name or intent.get("location", "")
        raw_places.extend(find_nearby_complement_places(complement_anchor, raw_places, planning_context, limit=max(1, min_route_stops - len(raw_places))))
    places, structure_notes = sanitize_structured_places(raw_places, intent)
    places = filter_replaced_destinations_from_places(places, state, anchor_name)
    structure_notes.insert(0, structure_anchor_note)
    if len(places) < 2 and sync_amap_complement_enabled():
        refill_anchor = places[-1] if places else (anchor_name or intent.get("location", ""))
        refill_places = find_nearby_complement_places(refill_anchor, places + [anchor_name], planning_context, limit=max(1, min_route_stops - len(places)))
        if refill_places:
            places, refill_notes = sanitize_structured_places(places + refill_places, intent)
            structure_notes.extend(refill_notes)
            structure_notes.append(
                f"清理连续正餐/重复地点后路线不足，已围绕“{refill_anchor}”补充近距离候选：{'、'.join(refill_places)}。")
    places, adjustment_notes = apply_quick_adjustment_to_places(places, state, intent)
    places = filter_replaced_destinations_from_places(places, state, anchor_name)
    structure_notes.extend(adjustment_notes)

    places, min_route_notes = ensure_minimum_route_places(
        places,
        state,
        intent,
        anchor_name,
        anchor_mode,
        min_route_stops,
        max_route_stops,
    )
    places = filter_replaced_destinations_from_places(places, state, anchor_name)
    structure_notes.extend(min_route_notes)

    places, soft_order_notes = soft_optimize_route_order(places, state, intent)
    structure_notes.extend(soft_order_notes)

    places, final_food_notes = final_sanitize_route_places(places, state, intent)
    structure_notes.extend(final_food_notes)
    if len(places) < min_route_stops:
        places, refill_food_safe_notes = ensure_minimum_route_places(
            places, state, intent, anchor_name, anchor_mode, min_route_stops, max_route_stops
        )
        structure_notes.extend(refill_food_safe_notes)
        places, final_food_notes = final_sanitize_route_places(places, state, intent)
        structure_notes.extend(final_food_notes)

    # 最终保险：无论前面 RAG、历史 user_input、快捷调整怎样改动，
    # 旧目的地都不能留在方案里；最新目的地必须作为当前锚点出现。
    anchor_name, anchor_mode = planning_anchor_for_intent(intent, collected)
    places = filter_replaced_destinations_from_places(places, state, anchor_name)
    if anchor_mode == "destination" and anchor_name:
        places = move_destination_anchor_to_end(places, anchor_name) if bool(collected.get("_departure_explicit")) else move_destination_anchor_to_start(places, anchor_name)
    places, final_food_notes = final_sanitize_route_places(places, state, intent)
    structure_notes.extend(final_food_notes)

    if not places and not sync_amap_complement_enabled():
        static_fallback = anchor_name or intent.get("location") or collected.get("departure")
        if static_fallback:
            places = [static_fallback]
            places, fallback_min_notes = ensure_minimum_route_places(
                places,
                state,
                intent,
                anchor_name,
                anchor_mode,
                min_route_stops,
                max_route_stops,
            )
            structure_notes.extend(fallback_min_notes)
            structure_notes.append(
                f"快速规划模式下已保留锚点“{static_fallback}”，并尝试补齐至少 {min_route_stops} 个路线地点。"
            )
    if not places and sync_amap_complement_enabled():
        departure = collected.get("departure") or intent.get("departure") or ""
        fallback_anchor = anchor_name if anchor_mode == "destination" else departure
        modes = state.get("adjustment_modes") or (
            [state.get("adjustment_mode")] if state.get("adjustment_mode") else [])
        fallback = find_and_persist_nearby_replacement(
            fallback_anchor,
            intent.get("location") or intent.get("meal_pref") or "附近可去地点",
            intent,
            _safe_int(os.getenv("NEARBY_SEARCH_RADIUS_METERS", "5000"), 5000),
            force_indoor="indoor" in modes,
            force_coupon="coupon" in modes,
            exclude_places=set(state.get("avoid_places") or []),
        )
        if fallback:
            places = [fallback]
            places, fallback_min_notes = ensure_minimum_route_places(
                places,
                state,
                intent,
                fallback_anchor,
                anchor_mode,
                min_route_stops,
                max_route_stops,
            )
            structure_notes.extend(fallback_min_notes)
            structure_notes.append(f"本地/RAG候选为空，已改用锚点“{fallback_anchor}”附近高德候选“{fallback}”，并继续补齐多站路线。")

    places, final_food_notes = final_sanitize_route_places(places, state, intent)
    structure_notes.extend(final_food_notes)
    route_places_for_schedule = places[:max_route_stops]
    slots = build_schedule_slots(collected, intent, max(1, len(route_places_for_schedule)))
    schedule = []
    for index, place in enumerate(route_places_for_schedule):
        role = place_role(place)
        if role == "meal":
            purpose = "正餐/核心用餐"
        elif role == "light_food":
            purpose = "轻量补充/咖啡休息"
        else:
            purpose = "顺路游玩/散步体验" if index else "核心游玩"
        price_detail = place_price_detail(place)
        schedule.append({
            "time": slots[min(index, len(slots) - 1)],
            "place": place,
            "place_role": role,
            "purpose": purpose,
            "price_text": price_detail.get("price_text", "价格待核验"),
            "price_min": price_detail.get("price_min", 0),
            "price_max": price_detail.get("price_max", 0),
        })
    if not schedule:
        schedule.append({
            "time": slots[0],
            "place": "待确认地点",
            "place_role": "unknown",
            "purpose": "当前没有找到可核验地点，需要用户放宽条件或检查高德 API",
        })
    schedule_places = [item["place"] for item in schedule]
    places = unique_preserve_order(schedule_places)
    effective_departure = collected.get("departure") or intent.get("departure")
    if anchor_mode == "destination" and not bool(collected.get("_departure_explicit")):
        effective_departure = anchor_name or (places[0] if places else "")
        structure_notes.append(
            f"用户只明确目的地，未明确出发地；系统以目的地“{effective_departure}”作为路线起点，不再使用默认市中心出发地。"
        )
    adjustment_conflicts = detect_adjustment_conflicts(places, state, intent)
    if adjustment_conflicts:
        structure_notes.extend(adjustment_conflicts)

    route_logic_validation = {
        "ok": not adjustment_conflicts,
        "notes": structure_notes,
        "adjustment_conflicts": adjustment_conflicts,
        "exception_events": exception_events,
        "rules": [
            "structured_plan.schedule 是最终文案唯一主路线",
            "同一条 4-6 小时路线最多安排一顿正餐",
            "最终 schedule 前会再次清理连续正餐和连续咖啡/甜品/轻食",
            "允许正餐后接散步/咖啡/轻量体验",
            "高德 POI 真实检索到的附近地点可以进入 structured_plan",
            "快捷按钮必须改动 structured_plan，而不是只改文案",
            "最终路线默认至少包含 2-3 个真实地点；少走路模式也不得退化为单点方案",
            "高德距离用于转场参考和尽量靠近的排序，不再按固定公里数做硬拦截",
        ],
    }

    structured = {
        "schema": "localmate_structured_plan_v1",
        "places": places,
        "schedule": schedule,
        "route_logic_validation": route_logic_validation,
        "hard_constraints": {
            "departure": effective_departure,
            "num_people": parse_people_count(collected.get("num_people") or intent.get("num_people"), None),
            "date": collected.get("date") or intent.get("date"),
            "time_period": collected.get("time_period") or intent.get("time_period"),
            "start_time": collected.get("start_time") or intent.get("start_time"),
            "weather": weather_info.get("weather") or collected.get("weather") or intent.get("weather"),
            "weather_reference": weather_info.get("summary", ""),
            "budget": normalize_budget_text(collected.get("budget") or intent.get("budget"), "人均200元以内"),
            "duration_hours": intent.get("duration_hours"),
            "transport_mode": collected.get("transport_mode") or intent.get("transport_mode") or "公交地铁",
            "planning_anchor": anchor_name,
            "planning_anchor_mode": anchor_mode,
            "route_logic_mode": route_logic_mode,
            "min_route_stops": min_route_stops,
            "max_route_stops": max_route_stops,
            "destination": anchor_name if anchor_mode == "destination" else None,
            "departure_explicit": departure_explicit,
            "location_explicit": location_explicit,
            "adjustment_mode": state.get("adjustment_mode"),
            "adjustment_modes": state.get("adjustment_modes") or (
                [state.get("adjustment_mode")] if state.get("adjustment_mode") else []),
            "locked_places": state.get("locked_places") or [],
        },
        "tool_facts": {
            "attraction_info": state.get("attraction_info", ""),
            "ticket_info": state.get("ticket_info", ""),
            "route_plan": "最终路线以 structured_plan.schedule 为准：" + " -> ".join(schedule_places),
            "route_distance_info": state.get("route_distance_info", ""),
            "coupon_summary": (coupon_info or {}).get("summary", ""),
            "weather_info": weather_info.get("summary", ""),
        },
        "notes": {
            "exception": state.get("exception"),
            "exception_events": exception_events,
            "adjustment_conflicts": adjustment_conflicts,
            "explicit_place_note": intent.get("explicit_place_note"),
            "resolved_location_note": intent.get("resolved_location_note"),
        },
        "rendering_rules": [
            "只渲染 structured_plan 中的地点，不新增不存在的景点或餐厅",
            "人数、预算、时间段、天气以 hard_constraints 为准",
            "团购券只写 coupon_summary 明确列出的内容",
            "距离和交通只写 route_distance_info 支持的信息",
            "不得写 structured_plan.schedule 以外的地点作为主路线",
            "不得安排连续两顿正餐或连续两家同类正餐店",
        ],
    }
    print("🧩 结构化方案已完整生成:")
    print(json.dumps(structured, ensure_ascii=False, indent=2))
    return {**state, "structured_plan": structured}


def validate_generated_plan(plan: str, state: AgentState, coupon_info: dict) -> dict:
    """Product-level rule checks: harder constraints than RAGAS retrieval metrics."""
    issues = []
    intent = state.get("intent", {}) or {}
    collected = state.get("collected_info", {}) or {}
    people = parse_people_count(collected.get("num_people") or intent.get("num_people"), None)
    text = str(plan or "")
    structured_plan = state.get("structured_plan") or {}
    schedule = structured_plan.get("schedule") or []
    schedule_places = [
        item.get("place")
        for item in schedule
        if isinstance(item, dict) and item.get("place")
    ]

    if people:
        wrong_counts = []
        for match in re.finditer(r"(\d+|[一二两俩三四五六七八九十])\s*个?\s*人(?![均民])", text):
            parsed = parse_people_count(match.group(0), None)
            if parsed and parsed != people and parsed >= 2:
                wrong_counts.append(match.group(0))
        if wrong_counts:
            issues.append(
                f"文案出现与用户人数不一致的人数表达：{', '.join(sorted(set(wrong_counts)))}；用户人数应为 {people}人")

    target_location = str(intent.get("location") or "").strip()
    target_is_scheduled = any(
        place_matches_text(target_location, place) or place_matches_text(place, target_location)
        for place in schedule_places
    )
    if target_location and _find_place(target_location) is not None and target_is_scheduled:
        if not place_matches_text(target_location, text):
            issues.append(f"用户/系统确定的核心地点“{target_location}”未出现在最终方案正文中")

    missing_schedule_places = [
        place for place in schedule_places
        if not place_matches_text(place, text)
    ]
    if missing_schedule_places:
        issues.append(f"structured_plan.schedule 中的地点没有完整渲染到正文：{', '.join(missing_schedule_places)}")

    meal_positions = [
        idx for idx, place in enumerate(schedule_places)
        if place_role(place) == "meal"
    ]
    if len(meal_positions) > 1:
        issues.append("structured_plan 中仍存在多顿正餐，需删除第二个及之后的正餐地点")
    for idx in range(len(schedule_places) - 1):
        if place_role(schedule_places[idx]) == "meal" and place_role(schedule_places[idx + 1]) == "meal":
            issues.append(f"structured_plan 中存在连续正餐：{schedule_places[idx]} -> {schedule_places[idx + 1]}")
        if place_role(schedule_places[idx]) == "light_food" and place_role(schedule_places[idx + 1]) == "light_food":
            issues.append(
                f"structured_plan 中存在连续咖啡/甜品/轻食：{schedule_places[idx]} -> {schedule_places[idx + 1]}")

    coupon_items = (coupon_info or {}).get("items") or []
    invalid_coupon_places = [
        item.get("place_name", "")
        for item in coupon_items
        if item.get("place_name") and not any(
            place_matches_text(item.get("place_name", ""), place) or place_matches_text(place,
                                                                                        item.get("place_name", "")) for
            place in schedule_places)
    ]
    if invalid_coupon_places:
        issues.append(f"团购券地点不在 structured_plan.schedule 中：{', '.join(invalid_coupon_places)}")
    if not coupon_items and plan_has_coupon_claim(text):
        issues.append("文案出现团购/券/满减说法，但结构化 coupon_info 没有可用券")

    if "**" in text or re.search(r"(?m)^\s*\*\s+", text):
        issues.append("文案仍包含 Markdown 星号，前端展示不够干净")

    if "未找到" in str(state.get("attraction_info", "")) and target_location and not place_matches_text(target_location,
                                                                                                        text):
        issues.append("工具提示地点未入库，但最终方案没有明确说明点名地点未入库")

    return {
        "ok": not issues,
        "issues": issues,
        "checked": [
            "人数一致性",
            "核心地点一致性",
            "structured_plan地点完整渲染",
            "不连续正餐/咖啡",
            "相邻距离软提醒",
            "团购券结构化依据",
            "团购券地点必须属于structured_plan",
            "Markdown星号清理",
            "未入库地点提示",
        ],
    }


def add_reservation_consistency_checks(validation_report: dict, structured_plan: dict,
                                       reservation_options: list[dict]) -> dict:
    report = dict(validation_report or {})
    issues = list(report.get("issues") or [])
    schedule = [
        item for item in (structured_plan or {}).get("schedule", []) or []
        if isinstance(item, dict) and item.get("place")
    ]
    options = reservation_options or []
    pair_mismatch = len(options) != len(schedule)
    if not pair_mismatch:
        for schedule_item, option_item in zip(schedule, options):
            schedule_place = str(schedule_item.get("place", ""))
            option_place = str(option_item.get("place_name", ""))
            same_place = place_matches_text(schedule_place, option_place) or place_matches_text(option_place,
                                                                                                schedule_place)
            same_time = str(schedule_item.get("time", "")) == str(option_item.get("time", ""))
            if not same_place or not same_time:
                pair_mismatch = True
                break
    if pair_mismatch:
        issues.append("可预订地点与 structured_plan.schedule 的地点或时间不一致")
    option_pairs = [(normalize_place_text(item.get("place_name", "")), str(item.get("time", ""))) for item in options]
    if len(option_pairs) != len(set(option_pairs)):
        issues.append("可预订地点中存在重复地点/时间项")
    report["issues"] = issues
    report["ok"] = not issues
    checked = list(report.get("checked") or [])
    checked.extend(["可预订地点与schedule一致", "可预订地点不重复"])
    report["checked"] = list(dict.fromkeys(checked))
    return report


def quick_mode_label(mode: str) -> str:
    return {
        "nearer": "换近一点",
        "cheaper": "换便宜一点",
        "indoor": "换室内",
        "coupon": "优先有团购",
        "less_walk": "少走路",
    }.get(str(mode or ""), str(mode or "调整需求"))


def quick_mode_default_explanation(mode: str, coupon_info: dict) -> str:
    coupon_items = (coupon_info or {}).get("items") or []
    if mode == "nearer":
        return "已优先围绕核心地点和同一区域补点，减少跨区折返。"
    if mode == "cheaper":
        return "已优先选择免费、低消费或人均更可控的地点。"
    if mode == "indoor":
        return "已优先替换为商场、室内展馆、咖啡店等更稳定的空间。"
    if mode == "coupon":
        if coupon_items:
            names = "、".join(str(item.get("place_name") or item.get("name") or "") for item in coupon_items[:3] if item)
            return f"已只按最终路线里的店铺校验团购券，可用券集中在：{names}。"
        return "已按最终路线校验团购券，当前没有查到可用券，所以没有硬塞不在路线里的店。"
    if mode == "less_walk":
        return "已压缩不必要步行，转场优先写地铁、骑行或打车。"
    return "已根据这条反馈重新调整路线。"


def build_adjustment_summary_lines(state: AgentState, structured_plan: dict, coupon_info: dict) -> list[str]:
    modes = state.get("adjustment_modes") or ([state.get("adjustment_mode")] if state.get("adjustment_mode") else [])
    modes = [mode for mode in unique_preserve_order(modes) if mode]
    if not modes:
        return []

    validation = (structured_plan or {}).get("route_logic_validation") or {}
    raw_notes = [str(note) for note in (validation.get("notes") or []) if note]
    useful_notes = [
        note for note in raw_notes
        if
        any(key in note for key in ["已将", "已移除", "已用", "替换", "团购", "室内", "少走路", "换近", "便宜", "连续"])
    ]

    lines = ["这版我已经按你的反馈重新改过："]
    for mode in modes:
        lines.append(f"• {quick_mode_label(mode)}：{quick_mode_default_explanation(mode, coupon_info)}")
    if useful_notes:
        compact_notes = "；".join(useful_notes[-3:])
        lines.append(f"具体改动：{compact_notes}")
    lines.append("")
    return lines


def stop_vibe_text(index: int, item: dict, total: int) -> str:
    role = item.get("place_role")
    purpose = item.get("purpose") or "顺路体验"
    if role == "meal":
        return "这一段安排成坐下来好好吃点东西的节奏，前后都留了转场缓冲，不会变成赶路吃饭。"
    if role == "light_food":
        return "这里适合轻轻补能：点杯喝的、整理照片、聊一下下一段怎么走，让半日路线有喘口气的地方。"
    if index == total:
        return "收尾放在这里比较稳，逛完可以直接返程，也可以看体力临时多停一会儿。"
    if index == 1:
        return "开场先选一个好进入状态的点，不急着赶路，先把今天的节奏和氛围拉起来。"
    return f"这一段主要负责换场景，{purpose} 的属性比较轻，不会让路线突然断掉。"


def render_fast_plan_text(state: AgentState, structured_plan: dict, coupon_info: dict) -> str:
    """Fast deterministic renderer used to keep first response under 30s."""
    collected = state.get("collected_info", {}) or {}
    hard = (structured_plan or {}).get("hard_constraints", {}) or {}
    schedule = (structured_plan or {}).get("schedule", []) or []
    weather_ref = hard.get("weather_reference") or ((state.get("weather_info") or {}).get("summary")) or "天气待出行前核验"
    num_people = hard.get("num_people") or collected.get("num_people") or "未知"
    budget = normalize_budget_text(hard.get("budget") or collected.get("budget"), "预算待定")
    departure = hard.get("departure") or collected.get("departure") or "出发地未知"
    date_text = hard.get("date") or collected.get("date") or "本周末"
    transport_mode = hard.get("transport_mode") or collected.get("transport_mode") or "公交地铁"
    route_logic_mode = hard.get("route_logic_mode")
    destination = hard.get("destination")
    coupon_summary = (coupon_info or {}).get("summary") or "团购券信息：当前路线暂未计算。"

    if route_logic_mode == "fixed_start_destination" and destination:
        route_intro = f"{num_people}人，从{departure}出发，目的地锁定在{destination}。中间只塞顺路点，不改你的起点和终点。"
    elif route_logic_mode == "single_anchor_destination" and destination:
        route_intro = f"这次就围绕{destination}展开，不默认拉去市中心；后续你点换近/便宜/室内，也只换旁边的补充点。"
    elif route_logic_mode == "single_anchor_departure":
        route_intro = f"这次以{departure}为中心锚点往外找，后续调整也不会把这个锚点换掉。"
    else:
        route_intro = f"{num_people}人从{departure}出发，预算按{budget}来控，交通优先按{transport_mode}走。"

    lines = [
        "这条上海半日路线，可以直接照着走 ✨",
        route_intro,
        f"天气我先帮你看过：{weather_ref}。路线不做硬凑景点，重点是少绕、好落地、每段都有休息点。",
        "",
    ]
    exception_events = (structured_plan.get("route_logic_validation") or {}).get("exception_events") or state.get("exception_events") or []
    if exception_events:
        lines.append("⚠️ 先说一个现场小情况：")
        for event in exception_events[:3]:
            lines.append(f"• {event.get('message') or event}")
        lines.append("")
    conflicts = (structured_plan.get("route_logic_validation") or {}).get("adjustment_conflicts") or []
    if conflicts:
        lines.append("⚠️ 本次调整说明：")
        for conflict in conflicts[:4]:
            lines.append(f"• {conflict}")
        lines.append("")
    lines.extend(build_adjustment_summary_lines(state, structured_plan, coupon_info))

    if schedule:
        route_names = " → ".join((item.get("display_name") or item.get("place") or "待确认地点") for item in schedule)
        lines.extend([
            f"路线顺序：{route_names}",
            "",
            "我会这样走：",
        ])

    total = len(schedule)
    for index, item in enumerate(schedule, start=1):
        place = item.get("place", "待确认地点")
        display_name = item.get("display_name") or place
        address = item.get("address") or "地址待高德/商家二次核验"
        time_slot = item.get("time", "时间待定")
        purpose = item.get("purpose", "行程地点")
        price_text = item.get("price_text") or place_price_detail(place).get("price_text", "价格待核验")
        transport = (item.get("transport_from_previous") or {}).get("summary")
        vibe = stop_vibe_text(index, item, total)
        lines.extend([
            f"📍{time_slot}｜{display_name}",
            f"💰参考花费：{price_text}",
            f"🎯适合做什么：{purpose}。{vibe}",
            f"📌地址：{address}",
        ])
        if transport:
            lines.append(f"🚇转场参考：{transport}")
        lines.append("")

    lines.extend([
        f"费用与团购：{coupon_summary}",
        "整体体感：不是那种硬排满的打卡表，而是能走、能停、能吃，也能根据体力临时微调的路线。",
        "出发前建议再核验营业时间、余位和实时交通；想换近一点、换便宜一点、换室内、优先有团购或少走路，直接继续说就行。",
    ])
    return "\n".join(lines).strip()


def result_formatter(state: AgentState) -> AgentState:
    """把严格 structured_plan 渲染成小红书风格文案。"""
    structured_plan = state.get("structured_plan") or build_structured_plan(state).get("structured_plan", {})

    exception_note = f"【注意事项】: {state['exception']}" if state.get("exception") else ""
    collected = state.get("collected_info", {}) or {}
    intent = state.get("intent", {}) or {}
    location_note = "；".join(
        note for note in [intent.get("explicit_place_note", ""), intent.get("resolved_location_note", "")]
        if note
    )
    if structured_plan.get("schedule"):
        structured_plan["schedule"] = enrich_schedule_addresses(structured_plan.get("schedule") or [])
    structured_plan_json = json.dumps(structured_plan, ensure_ascii=False, indent=2)
    weather_info = state.get("weather_info") or {}
    weather_reference = weather_info.get("summary") or "未查询到实时/预报天气，按用户输入天气偏好或默认天气处理。"
    coupon_info = state.get("coupon_info") or {}

    if os.getenv("FAST_PLAN_RENDER", "1") == "1":
        coupon_info = filter_coupon_info_for_schedule(coupon_info, structured_plan)
        plan = render_fast_plan_text(state, structured_plan, coupon_info)
        structured_plan["plan_preview"] = str(plan or "")[:240]
        validation_report = validate_generated_plan(plan, state, coupon_info)
        reservation_options = build_reservation_options(structured_plan)
        validation_report = add_reservation_consistency_checks(validation_report, structured_plan, reservation_options)

        print("\n" + "=" * 60)
        print("📋 为您生成的出行方案：")
        print("=" * 60)
        print(plan)
        print("=" * 60)

        return {
            **state,
            "final_plan": plan,
            "coupon_info": coupon_info,
            "structured_plan": structured_plan,
            "validation_report": validation_report,
            "feasibility_report": state.get("feasibility_report") or structured_plan.get("feasibility_report") or evaluate_route_feasibility(structured_plan, intent),
            "reservation_options": reservation_options,
        }

    plan_llm = ChatDashScope(
        model=os.getenv("PLAN_MODEL", "qwen-plus"),
        temperature=0.25,
        max_tokens=int(os.getenv("PLAN_MAX_TOKENS", "2600")),
    )
    plan_llm.client = dashscope.Generation

    prompt = ChatPromptTemplate.from_template("""
你是一个活跃在上海的生活方式博主，同时也是专业的周末出行规划师。
请先把【结构化方案JSON】当作唯一方案骨架，再渲染成一份充满感染力的出行攻略，风格参考小红书种草笔记。

【结构化方案JSON】:
{structured_plan_json}

【用户需求】: {user_input}
【出发地点】: {departure}
【出行人数】: {num_people}人
【出行日期】: {date}
【出行时间段】: {time_period}
【用户指定出发时间】: {start_time}
【天气情况】: {weather}
【天气查询参考】: {weather_reference}
【大致预算】: {budget}
【参考案例，仅作需求理解和表达参考，禁止引入其中地点】: {rag_context}
【景点/场馆信息】: {attraction_info}
【门票/费用信息】: {ticket_info}
【高德距离参考】: {route_distance_info}
【团购券信息】: {coupon_summary}
【地点解析说明】: {location_note}
{exception_note}

【输出长度与完整性要求】：
- 控制在 700-1000 字以内，必须完整收尾，不要写到半句停住。
- 必须包含：路线概览、分时段安排、交通、费用、注意事项。
- 总行程必须控制在 4-6 小时之间，不要生成少于 4 小时或超过 6 小时的完整行程。
- 最后一行必须输出【方案结束】。
- 如果信息不足，明确写“需要出行前核验”，不要编造营业时间、精确耗时或不存在的地点。
- 交通距离、转场时间、交通建议必须优先使用【结构化方案JSON】中每个 schedule 项的 transport_from_previous；如果没有该字段，再参考【高德距离参考】。如果里面写“未配置”或“失败”，只能说明“暂未计算成功”，严禁自行估算或编造任何公里数、分钟数、打车费。
- 每个 schedule 项如果有 address 字段，必须把该具体地址融合进该地点的 📍 描述里，例如“地址：上海市xx区xx路xx号”。不要把所有地址集中放到文末。
- 团购券是结构化数据字段，不是写作素材。只有【团购券信息】明确列出的店铺才允许写团购券。
- 如果【团购券信息】写“未发现可用团购券”或“无可用团购券”，标题、正文、费用和 Tips 中都禁止出现“团购、代金券、满减、美团券、抖音券、券码、VIP免排队、已锁定券码”等说法。
- 【出行人数】是关键约束，必须始终写成 {num_people}人。参考案例里的其他人数（如8人、9人、12人）只能当案例背景，禁止带入本次方案。
- 如果【地点解析说明】显示用户点名的地点已匹配到 mock 库，必须围绕该地点规划；如果显示未找到，必须说明当前 mock 库未找到，不要换成其他同类店铺冒充。
- 输出不要使用 Markdown 加粗星号，不要出现 **文本** 或单独的 * 列表符号。
- 必须只使用【结构化方案JSON】里的 places 和 schedule 作为主路线，不要自己新增路线地点。
- 如果【用户指定出发时间】不是“未指定”，正文第一站必须从该时间开始，并严格使用 structured_plan.schedule 中的时间段。
- 如果其他上下文、参考案例、工具草稿中出现了 structured_plan.places 以外的地点，一律忽略，不得写入正文路线。
- 如果 structured_plan.route_logic_validation 说明移除了某些餐饮地点，正文不得再写这些被移除地点。
- 如果 hard_constraints.adjustment_modes 不为空，正文必须明确体现所有快捷调整都已经执行，比如更近、更便宜、室内、有团购或少走路，不要只写其中一个。
- hard_constraints.locked_places 是用户前文明确说过且未推翻的核心地点，正文必须尽量保留这些地点；除非 route_logic_validation 明确说明因距离或不可核验被替换/移除。
- 如果 route_logic_validation.notes 里出现“本地表补充”“高德补充”“替换”，正文需要自然说明这是为了让路线更顺、更可落地。

【写作风格要求】：
1. 标题：用一句戳心的话作为开头，搭配 emoji，制造画面感或情绪共鸣
   例如："周末不知道去哪？这条路线直接拿捏！🎯"
2. 开场白：1-2句话交代出行背景或氛围，像朋友聊天一样自然
3. 路线正文：
   - 每个地点用 📍 标注
   - 每个地点必须写具体地址；优先使用 schedule.address，如果 address 为空，就写“地址待高德/商家二次核验”，不要编造门牌号。
   - 每个地点必须先写“怎么到这里”：第一站写从【出发地点】到该地点；后续站写从上一站到该地点。距离、耗时、交通方式直接使用 schedule.transport_from_previous.summary。
   - 每个地点包含：氛围描述 + 亮点推荐 + 拍照/游玩技巧
   - 最多展开 3 个核心地点，避免无限扩写导致结尾被截断
   - 多用短句，节奏轻快，避免大段堆砌
   - 穿插"真的""超级""直接""无脑冲""氛围感拉满"等口语词
4. 实用信息：
   - 不要单独写一大段“高德距离参考”或“交通出行贴士”；交通信息必须融合到每一站的 📍 路线描述中。
   - ⏰ 行程安排必须匹配【出行时间段】，上午/下午/晚上不能混用
   - 🎟️ 如果【团购券信息】显示有券，必须在费用部分列出店名、券类型和优惠；如果没有券，说明当前路线无可用团购券
   - 🌦️ 路线和室内外安排必须同时考虑【天气情况】和【天气查询参考】；如果查询结果显示下雨、高温、低温或大风，必须相应调整为室内、遮阳、防雨或少步行方案。
   - 💰 费用预估结合【大致预算】和【出行人数】列出分项明细
   - 💡 tip：最佳时间、排队建议、注意事项等
   - ⚠️ 如有备选方案切换，自然融入说明
5. 结尾：用一句温暖或俏皮的话收尾，鼓励读者行动

请严格基于以上信息生成方案，不要编造不存在的地点、门牌号、营业时间、价格、交通耗时、公里数、打车费或活动。
尤其不要编造任何未在【团购券信息】中出现的优惠券、套餐券、代金券、团购价、平台券或满减规则。
如果【景点/场馆信息】或【门票/费用信息】显示“未找到”，必须明确提示“当前 mock 库未找到该地点，建议换一个已入库地点或接入地图/商家 API 核验”，不要继续围绕该地点编造攻略。
如果参考案例、RAG 文本或用户历史里出现与【出行人数】矛盾的人数，以【出行人数】为唯一准。
""")

    chain = prompt | plan_llm | StrOutputParser()
    coupon_info = state.get("coupon_info") or {}
    plan = chain.invoke({
        "user_input": state["user_input"],
        "structured_plan_json": structured_plan_json,
        "departure": collected.get("departure", "出发地未知"),
        "num_people": collected.get("num_people", "未知"),
        "date": collected.get("date", "本周末"),
        "time_period": collected.get("time_period", "下午"),
        "start_time": collected.get("start_time") or "未指定",
        "weather": weather_info.get("weather") or collected.get("weather", "晴天"),
        "weather_reference": truncate_text(weather_reference, 500),
        "budget": normalize_budget_text(collected.get("budget"), "适中"),
        "rag_context": truncate_text(state.get("rag_context", "暂无参考案例"), 1200),
        "attraction_info": truncate_text(state.get("attraction_info", ""), 450),
        "ticket_info": truncate_text(state.get("ticket_info", ""), 450),
        "route_distance_info": truncate_text(state.get("route_distance_info", ""), 800),
        "coupon_summary": truncate_text(coupon_info.get("summary", "团购券信息：当前路线暂未计算。"), 600),
        "location_note": location_note,
        "exception_note": exception_note
    })
    plan = complete_plan_if_needed(plan, plan_llm)
    plan = plan.replace("【方案结束】", "").strip()
    plan = remove_unsupported_coupon_claims(plan, coupon_info, plan_llm)
    plan = sanitize_final_plan_text(plan, collected.get("num_people") or intent.get("num_people"))
    plan = ensure_schedule_places_rendered(plan, structured_plan)
    plan = append_canonical_schedule_table(plan, structured_plan)
    coupon_info = filter_coupon_info_for_schedule(coupon_info, structured_plan)
    structured_plan["plan_preview"] = str(plan or "")[:240]
    validation_report = validate_generated_plan(plan, state, coupon_info)
    reservation_options = build_reservation_options(structured_plan)
    validation_report = add_reservation_consistency_checks(validation_report, structured_plan, reservation_options)

    print("\n" + "=" * 60)
    print("📋 为您生成的出行方案：")
    print("=" * 60)
    print(plan)
    print("=" * 60)

    return {
        **state,
        "final_plan": plan,
        "coupon_info": coupon_info,
        "structured_plan": structured_plan,
        "validation_report": validation_report,
        "feasibility_report": state.get("feasibility_report") or structured_plan.get("feasibility_report") or evaluate_route_feasibility(structured_plan, intent),
        "reservation_options": reservation_options,
    }


def enrich_new_places(state: AgentState) -> AgentState:
    """从方案中提取地点，对 mock 库中不存在的地点生成数据并入库。

    默认关闭：自动把模型输出的新地点写入 mock 库，会把一次幻觉固化为后续数据源。
    如确实需要演示自增长能力，可设置环境变量 ENABLE_AUTO_ENRICH_PLACES=1。
    """
    if os.getenv("ENABLE_AUTO_ENRICH_PLACES", "0") != "1":
        print("ℹ️ 已跳过自动入库新地点；如需开启，设置 ENABLE_AUTO_ENRICH_PLACES=1")
        return state

    plan = state.get("final_plan", "")
    if not plan:
        return state

    extract_prompt = ChatPromptTemplate.from_template("""
从以下出行方案文本中，提取所有出现的景点、餐厅、场馆名称，以JSON数组返回，只返回名称列表，不加说明。
例如: ["鲁迅公园", "万寿斋", "多伦路文化名人街"]

方案文本:
{plan}
""")
    names_str = (extract_prompt | llm | StrOutputParser()).invoke({"plan": plan})
    cleaned = re.sub(r"```json|```", "", names_str).strip()
    try:
        place_names = json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"⚠️ 地点提取解析失败: {names_str}")
        return state

    new_places = [n for n in place_names if _find_place(n) is None]
    if not new_places:
        print("✅ 方案中所有地点均已在 mock 库中")
        return state

    print(f"🆕 发现 {len(new_places)} 个新地点，正在生成 mock 数据: {new_places}")

    gen_prompt = ChatPromptTemplate.from_template("""
请为以下上海地点生成模拟数据，以JSON数组返回，每个对象包含以下字段（只返回JSON，不加说明）：
- placeName: string
- 是否需要预约: boolean
- 是否有余位: boolean（随机，大概率true）
- 余位信息: number（50-200之间的整数）
- 是否有团购: boolean
- 最低价格: number
- 最高价格: number
- 地点类型: string（从 attraction/restaurant/activity/leisure/sports 中选）

地点列表: {places}
""")
    gen_result = (gen_prompt | llm | StrOutputParser()).invoke({"places": str(new_places)})
    gen_cleaned = re.sub(r"```json|```", "", gen_result).strip()
    try:
        new_records = json.loads(gen_cleaned)
        for record in new_records:
            add_new_place(record)
    except (json.JSONDecodeError, Exception) as e:
        print(f"⚠️ 新地点数据生成失败: {e}")

    return state


def book_order_node(state: AgentState) -> AgentState:
    """执行预订下单"""
    intent = state["intent"]
    location = intent.get("location", "景区")
    result = book_order.invoke({
        "attraction": location,
        "date": intent.get("date", "本周末"),
        "num_people": int(intent.get("num_people", 2)),
        "meal": intent.get("meal_pref", "中餐")
    })
    print("\n" + result)

    mask = _df["placeName"] == location
    if mask.any():
        _df.loc[mask, "余位信息"] = (_df.loc[mask, "余位信息"] - 1).clip(lower=0)
        new_count = _df.loc[mask, "余位信息"].values[0]
        if new_count == 0:
            _df.loc[mask, "是否有余位"] = False
        print(f"📉 {location} 剩余余位已更新为: {new_count}")

    return {**state, "order_result": result}


def skip_booking(state: AgentState) -> AgentState:
    """用户取消预订"""
    print("💔 已取消预订。如需重新规划，请重新输入需求～")
    return {**state, "order_result": "用户取消预订"}


# ==========================================
# 4. 构建 LangGraph 工作流
# ==========================================
TOOL_TIMING_NODES = {
    "weather_lookup",
    "rag_retrieval",
    "tool_dispatch",
    "route_distance_planner",
}


def timed_workflow_node(node_name: str, func):
    def wrapper(state: AgentState) -> AgentState:
        started = time.perf_counter()
        result = func(state)
        elapsed = time.perf_counter() - started
        timings = dict((result or {}).get("node_timings") or (state or {}).get("node_timings") or {})
        timings[node_name] = round(elapsed, 2)
        label = "工具调用耗时" if node_name in TOOL_TIMING_NODES else "节点耗时"
        print(f"⏱️ {label} [{node_name}]: {elapsed:.2f}s")
        return {**result, "node_timings": timings}

    return wrapper


def build_plan_workflow():
    """规划工作流（信息收集在 API 层处理，从 parse_intent 开始）"""
    workflow = StateGraph(AgentState)

    workflow.add_node("parse_intent", timed_workflow_node("parse_intent", parse_intent))
    workflow.add_node("weather_lookup", timed_workflow_node("weather_lookup", weather_lookup))
    workflow.add_node("rag_retrieval", timed_workflow_node("rag_retrieval", rag_retrieval))
    workflow.add_node("tool_dispatch", timed_workflow_node("tool_dispatch", tool_dispatch))
    workflow.add_node("exception_handler", timed_workflow_node("exception_handler", exception_handler))
    workflow.add_node("route_distance_planner", timed_workflow_node("route_distance_planner", route_distance_planner))
    workflow.add_node("build_structured_plan", timed_workflow_node("build_structured_plan", build_structured_plan))
    workflow.add_node("result_formatter", timed_workflow_node("result_formatter", result_formatter))
    workflow.add_node("enrich_new_places", timed_workflow_node("enrich_new_places", enrich_new_places))

    workflow.set_entry_point("parse_intent")
    workflow.add_edge("parse_intent", "weather_lookup")
    workflow.add_edge("weather_lookup", "rag_retrieval")
    workflow.add_edge("rag_retrieval", "tool_dispatch")
    workflow.add_edge("tool_dispatch", "exception_handler")
    workflow.add_edge("exception_handler", "build_structured_plan")
    workflow.add_edge("build_structured_plan", "route_distance_planner")
    workflow.add_edge("route_distance_planner", "result_formatter")
    workflow.add_edge("result_formatter", "enrich_new_places")
    workflow.add_edge("enrich_new_places", END)

    return workflow.compile()


def build_book_workflow():
    """预订工作流"""
    workflow = StateGraph(AgentState)
    workflow.add_node("book_order", book_order_node)
    workflow.set_entry_point("book_order")
    workflow.add_edge("book_order", END)
    return workflow.compile()


# ==========================================
# 5. 全局 workflow 实例（延迟初始化）
# ==========================================
plan_workflow = None
book_workflow = None


def init_workflows():
    global plan_workflow, book_workflow
    plan_workflow = build_plan_workflow()
    book_workflow = build_book_workflow()
    print("✅ Workflow 构建完成")


# ==========================================
# 6. 命令行主入口
# ==========================================
if __name__ == "__main__":
    init_models()
    init_workflows()

    print("🎯 上海周末出行规划 Agent 已启动！")
    print("💡 请告诉我您的出行需求，我会根据出发地点、人数、时间和预算为您规划～")
    print("-" * 60)

    user_input = input("👤 请输入您的出行需求: ").strip()
    if not user_input:
        print("⚠️ 输入为空，程序退出")
        exit()

    initial_state = AgentState(
        user_input=user_input,
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
        latest_user_input=user_input,
        node_timings={}
    )

    result = plan_workflow.invoke(initial_state)

    print("🏁 流程结束")
    if result.get("order_result"):
        print(f"📌 最终结果: {result['order_result']}")
