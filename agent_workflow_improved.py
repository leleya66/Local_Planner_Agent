import sys

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

# agent_workflow.py
# 核心 Agent 工作流文件：
# - 负责把用户自然语言需求识别成结构化约束。
# - 调用 RAG、地点库、高德/模拟工具，生成可渲染的 structured_plan。
# - 最后把 structured_plan 渲染成前端时间轴和最终文案。
# - app_api.py 只负责 HTTP/session/协作房间；真正的规划逻辑主要在这里。
import os
import re
import json
import urllib.parse
import urllib.request
import urllib.error
import time
import random
import hashlib
import math
from datetime import date, datetime, timedelta
from typing import TypedDict, Optional
from dotenv import load_dotenv
import dashscope

from langchain_chroma import Chroma
from langchain_dashscope import DashScopeEmbeddings, ChatDashScope
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from mock_api_improved import search_attraction, check_ticket, plan_route, book_order, _df, add_new_place, _find_place, stable_mock_seat_count

_mock_find_place = _find_place
_runtime_amap_place_records: dict[str, dict] = {}
_runtime_amap_poi_records: dict[str, dict] = {}


def _find_place(place_name: str):
    """Read mock places plus non-persistent Amap POIs created during this process."""
    row = _mock_find_place(place_name)
    if row is not None:
        return row
    key = normalize_place_text(place_name)
    if key in _runtime_amap_place_records:
        return _runtime_amap_place_records[key]
    return None


def _find_place_runtime_aware(place_name: str):
    row = _mock_find_place(place_name)
    if row is not None:
        return row
    key = normalize_place_text(place_name)
    if key in _runtime_amap_place_records:
        return _runtime_amap_place_records[key]
    return None


def refresh_place_data_if_changed(force: bool = False) -> bool:
    """Sync this module's cached _df reference with mock_api_improved.

    The mock Excel file may be edited while uvicorn is running. Without syncing,
    agent_workflow keeps the old imported _df and availability tests read stale seats.
    """
    try:
        import mock_api_improved as mock_api_module
        changed = False
        if hasattr(mock_api_module, "reload_place_data_if_changed"):
            changed = bool(mock_api_module.reload_place_data_if_changed(force))
        globals()["_df"] = mock_api_module._df
        globals()["_mock_find_place"] = mock_api_module._find_place
        globals()["_find_place"] = _find_place_runtime_aware
        return changed
    except Exception as exc:
        print(f"⚠️ mock地点表刷新失败: {exc}")
        return False


def place_data_signature() -> str:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    try:
        import mock_api_improved as mock_api_module
        if hasattr(mock_api_module, "place_data_signature"):
            sig = mock_api_module.place_data_signature()
            globals()["_df"] = mock_api_module._df
            return sig
    except Exception:
        pass
    return ""

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
    """中文说明：初始化模型、数据或运行时依赖。"""
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
            # 默认检索量下调：RAG 只做风格/案例参考，最终事实以结构化地点表和高德为准。
            # 减少 chunk 数可以显著降低嵌入检索和后续字符串处理耗时；如需更强 RAG 可用环境变量调大。
            "k": int(os.getenv("RETRIEVER_K", "4")),
            "fetch_k": int(os.getenv("RETRIEVER_FETCH_K", "12")),
            "lambda_mult": 0.45,
        },
    )
    # 规划类输出不需要太高随机性，否则容易编造细节。
    llm = ChatDashScope(model="qwen-flash", temperature=0.35)
    llm.client = dashscope.Generation
    print("✅ 向量数据库连接成功")


# ==========================================
# 2. Agent 状态定义
# ==========================================
class AgentState(TypedDict):
    # LangGraph 节点之间传递的统一状态。
    # 所有节点都读取/写入这个 dict，所以字段越清晰，越容易排查多轮修改和多人协作问题。
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
    requirement_revision_mode: Optional[bool]
    requirement_change_source_text: Optional[str]
    latest_requirement_explicit_fields: Optional[list]
    latest_requirement_changes: Optional[list]
    requirement_change_log: Optional[list]


# LLM 结构化输出只能落到这些本地支持的枚举里；超出枚举的类型会被 sanitize 丢弃。
CANONICAL_PLACE_TYPES = {"attraction", "restaurant", "cafe", "activity", "leisure", "shopping", "sports"}

# 细分类用于更细的地点筛选和价格/时长估算，例如 park/cafe/hotpot/street_walk。
CANONICAL_SUB_TYPES = {
    "milk_tea", "cafe", "dessert", "bakery",
    "hotpot", "xiaolongbao", "shengjian", "noodle",
    "jiangzhe_cuisine", "korean_cuisine", "japanese_cuisine", "western_cuisine",
    "temple", "street_walk", "art_exhibition", "museum", "cinema", "anime", "escape_room",
    "spa_relax", "theme_park", "park", "shopping_mall", "business_district",
}


# ==========================================
# Amap authoritative tag mode
# ==========================================
# 用户需求阶段只做二分类：吃喝 => restaurant；其它 => attraction。
# 具体小类型（KTV/火锅/电影院/公园等）以后端实际检索到的高德 POI type 为准。
SIMPLIFY_USER_DEMAND_TYPES = os.getenv("SIMPLIFY_USER_DEMAND_TYPES", "1") == "1"
AMAP_AUTHORITATIVE_TAGS = os.getenv("AMAP_AUTHORITATIVE_TAGS", "1") == "1"

FOOD_DRINK_REQUIREMENT_WORDS = {
    "吃", "饭", "餐", "餐厅", "饭店", "美食", "小吃", "正餐", "午餐", "晚餐", "早饭", "午饭", "晚饭",
    "火锅", "烧烤", "烤肉", "烧肉", "海底捞", "小笼", "小笼包", "汤包", "生煎", "面馆", "拉面", "面条",
    "韩料", "韩国料理", "日料", "寿司", "江浙菜", "本帮菜", "西餐", "牛排", "披萨", "自助", "甜品",
    "咖啡", "奶茶", "茶饮", "下午茶", "面包", "蛋糕", "酒吧", "清吧", "喝", "饮品", "饮料", "烘焙",
}

AMAP_GENERIC_TYPE_PARTS = {
    "", "餐饮服务", "购物服务", "生活服务", "体育休闲服务", "风景名胜", "科教文化服务", "商务住宅",
    "公司企业", "道路附属设施", "地名地址信息", "公共设施", "政府机构及社会团体", "住宿服务",
    "医疗保健服务", "汽车服务", "汽车销售", "汽车维修", "摩托车服务", "金融保险服务", "交通设施服务",
    "室内设施", "通行设施", "事件活动", "娱乐场所", "休闲场所", "综合市场", "其它", "其他",
}


def _contains_any(text: str, words) -> bool:
    raw = str(text or "")
    lower = raw.lower()
    return any(str(word).lower() in lower for word in words if str(word or "").strip())


def is_food_drink_requirement_text(*parts) -> bool:
    """二分类需求识别：只判断是否涉及吃喝。"""
    text = " ".join(str(part or "") for part in parts)
    return _contains_any(text, FOOD_DRINK_REQUIREMENT_WORDS)


def simplified_requirement_place_type(keyword: str = "", llm_sub_type: str = "", category: str = "") -> str:
    return "restaurant" if is_food_drink_requirement_text(keyword, llm_sub_type, category) else "attraction"


def simplified_requirement_category(keyword: str = "", llm_sub_type: str = "", category: str = "") -> str:
    return "餐饮" if simplified_requirement_place_type(keyword, llm_sub_type, category) == "restaurant" else "景点"


def split_amap_type(raw_type: str) -> list[str]:
    text = str(raw_type or "").strip()
    if not text:
        return []
    # 高德 type 常见格式："体育休闲服务;娱乐场所;KTV"。
    parts = re.split(r"[;；|/／>＞]+", text)
    result = []
    for part in parts:
        clean = str(part or "").strip()
        if clean and clean not in result:
            result.append(clean)
    return result


def amap_display_tag_from_type(raw_type: str, name: str = "") -> str:
    parts = split_amap_type(raw_type)
    for part in reversed(parts):
        if part and part not in AMAP_GENERIC_TYPE_PARTS:
            return part
    # 高德少数 POI 不返回细类型时，用名称关键词兜底只做展示，不覆盖高德已有 tag。
    name = str(name or "")
    fallback_rules = [
        (["KTV", "ktv", "唱歌", "卡拉OK", "纯K", "好乐迪", "麦乐迪"], "KTV"),
        (["影院", "影城", "电影"], "电影院"),
        (["火锅"], "火锅"),
        (["烤肉", "烧肉", "烧烤", "泥炉"], "烤肉"),
        (["咖啡", "Coffee", "COFFEE", "Manner", "星巴克"], "咖啡厅"),
        (["公园", "森林", "湿地"], "公园"),
        (["博物馆"], "博物馆"),
        (["美术馆", "艺术馆", "展览"], "美术馆"),
        (["商场", "购物中心", "百联", "万达", "合生汇"], "购物中心"),
    ]
    for words, tag in fallback_rules:
        if _contains_any(name, words):
            return tag
    return "餐饮" if is_food_drink_requirement_text(name) else "景点"


def amap_text_is_food_drink(name: str = "", raw_type: str = "", display_tag: str = "") -> bool:
    text = f"{name} {raw_type} {display_tag}"
    if any(token in str(raw_type or "") for token in ["餐饮服务", "中餐厅", "外国餐厅", "快餐厅", "咖啡厅", "茶艺馆", "冷饮店", "甜品店"]):
        return True
    return is_food_drink_requirement_text(text)


def amap_route_category_from_text(name: str = "", raw_type: str = "", display_tag: str = "") -> str:
    text = f"{name} {raw_type} {display_tag}"
    rules = [
        (["KTV", "ktv", "唱歌", "卡拉OK", "纯K", "好乐迪", "麦乐迪"], "ktv"),
        (["电影院", "影院", "影城", "电影"], "cinema"),
        (["密室", "剧本杀", "推理馆", "沉浸"], "escape_room"),
        (["火锅"], "hotpot"),
        (["烤肉", "烧肉", "烧烤", "泥炉"], "bbq"),
        (["咖啡厅", "咖啡", "Coffee", "COFFEE"], "cafe"),
        (["茶饮", "奶茶", "冷饮", "甜品", "面包", "蛋糕", "烘焙"], "light_food"),
        (["公园", "森林", "湿地", "郊野"], "park"),
        (["博物馆"], "museum"),
        (["美术馆", "艺术馆", "展览", "画廊"], "art_exhibition"),
        (["购物中心", "商场", "百货", "百联", "万达", "合生汇"], "shopping"),
        (["酒吧", "清吧", "Live", "livehouse"], "bar"),
        (["温泉", "汤泉", "洗浴", "SPA", "spa"], "spa_relax"),
    ]
    for words, category in rules:
        if _contains_any(text, words):
            return category
    return "meal" if amap_text_is_food_drink(name, raw_type, display_tag) else "scenic"


def derive_amap_authoritative_spec(poi: dict, fallback_keyword: str = "", fallback_place_type: str = "") -> dict:
    """由高德 POI type/name 生成最终权威标签。不会被 LLM 类型覆盖。"""
    poi = poi or {}
    name = str(poi.get("name") or "").strip()
    raw_type = str(poi.get("type") or poi.get("poi_type") or "").strip()
    type_path = split_amap_type(raw_type)
    display_tag = amap_display_tag_from_type(raw_type, name)
    primary_type = "restaurant" if amap_text_is_food_drink(name, raw_type, display_tag) else "attraction"
    route_category = amap_route_category_from_text(name, raw_type, display_tag)
    tags = []
    for item in [display_tag, route_category, primary_type] + type_path:
        item = str(item or "").strip()
        if item and item not in tags:
            tags.append(item)
    keyword = str(fallback_keyword or name or display_tag or "景点").strip()
    return {
        "keyword": keyword,
        "place_type": primary_type,             # 二分类：吃喝 restaurant，其它 attraction
        "sub_type": route_category,             # 内部去重用；展示仍用 amap_display_tag
        "amap_raw_type": raw_type,
        "amap_type_path": type_path,
        "amap_tags": tags,
        "amap_display_tag": display_tag,
        "amap_route_category": route_category,
        "amap_primary_type": primary_type,
        "tag_source": "amap" if raw_type else "amap_name_fallback",
    }


def schedule_place_type_metadata(place_name: str) -> dict:
    """从运行时 POI/地点表读取最终展示类型，确保前端和高德标签一致。"""
    row = _find_place(place_name)
    if row is None:
        cat = route_place_category(place_name)
        return {
            "route_category": cat,
            "display_type": "餐饮" if place_role(place_name) in {"meal", "light_food"} else ("景点" if cat in {"unknown", "scenic", ""} else cat),
            "tag_source": "local_rule_fallback",
        }
    display_tag = str(row.get("amap_display_tag") or "").strip()
    raw_type = str(row.get("amap_raw_type") or "").strip()
    route_category = str(row.get("amap_route_category") or "").strip() or route_place_category(place_name)
    primary_type = str(row.get("amap_primary_type") or row.get("地点类型") or "").strip()
    display_type = display_tag or ("餐饮" if primary_type == "restaurant" else "景点")
    return {
        "route_category": route_category,
        "display_type": display_type,
        "amap_display_tag": display_tag,
        "amap_raw_type": raw_type,
        "amap_tags": row.get("amap_tags", ""),
        "tag_source": row.get("tag_source", "place_table"),
    }


# 本地关键词兜底：LLM 失败或超时时，仍能把“咖啡/公园/看展”等映射到地点类型。
PLACE_TYPE_KEYWORDS = {
    "restaurant": ["餐厅", "美食", "饭店", "小吃", "吃饭", "火锅", "海底捞", "小笼包",
                   "小笼", "生煎", "面馆", "吃面", "韩料", "韩国料理", "江浙菜", "本帮菜", "正餐"],
    "cafe": ["咖啡", "咖啡馆", "下午茶", "甜品", "面包", "奶茶", "茶饮", "蜜雪冰城", "cafe", "coffee", "轻食"],
    "activity": ["活动", "体验", "露营", "团建", "亲子", "电影", "影院", "影城", "手作", "展览", "看展", "艺术展",
                 "二次元", "动漫", "泡汤", "汤泉", "温泉", "camping", "唱歌", "ktv"],
    "shopping": ["购物", "商场", "商圈", "购物中心", "百联", "万达", "合生汇", "环球港", "印象城", "逛街"],
    "sports": ["运动", "徒步", "骑行", "羽毛球", "保龄球", "射箭", "健身"],
    "leisure": ["郊区", "近郊", "远郊", "户外", "踏青", "散步", "遛弯", "放松", "休闲", "江边", "滨江", "步道", "街道",
                "大学路", "citywalk", "城市漫步", "田园", "outdoor", "suburban", "suburb", "nature"],
    "attraction": ["景点", "景区", "公园", "博物馆", "美术馆", "展馆", "古镇", "寺庙", "寺", "迪士尼", "乐园", "park"],
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



# ==========================================
# 0.5 意图/空间锚点辅助函数
# ==========================================
NEARBY_MARKERS = ("附近", "周边", "一带", "周围", "旁边", "边上", "周遭")
AREA_ANCHOR_TERMS = {
    "陆家嘴", "徐家汇", "五角场",  "南京西路", "南京东路",
    "淮海路", "虹桥", "古北", "大学路",
    "七宝", "莘庄", "张江", "花木路", 
     "打浦桥",  "武康路", "安福路", "愚园路", "巨鹿路",
}

# These names can describe an area, but they are also valid visitable landmarks.
# "人民广场附近" remains a nearby search anchor; "想去人民广场" must keep the
# landmark itself in the final schedule.
DIRECT_SCHEDULE_LANDMARK_TERMS = {
    "人民广场", "外滩", "世纪公园", "中山公园", "长风公园", "豫园", "田子坊", "静安寺",
}


INVALID_ANCHOR_MARKERS = {
    "保留用户明确", "保留已锁定", "用户明确的", "当前偏好", "上一版", "不同路线",
    "重新生成", "不要重复", "非锁定地点", "目的地区域", "出发地目的地",
    "作为起点", "以目的地作为起点", "未指定以目的地作为起点", "未指定作为起点",
}


def is_invalid_anchor_text(value: str) -> bool:
    """Return True when a captured anchor is actually an instruction/placeholder."""
    text = str(value or "").strip()
    if not text:
        return True
    compact = re.sub(r"\s+", "", text)
    if any(marker in compact for marker in INVALID_ANCHOR_MARKERS):
        return True
    if re.search(r"(?:出发地|目的地|终点|起点).*(?:预算|人数|交通|偏好|区域)", compact):
        return True
    if len(compact) > 30 and not re.search(r"(?:路|街|站|店|园|馆|城|区|镇|广场|中心|天地|小镇|公园)$", compact):
        return True
    return False


def clean_anchor_for_display(value: str) -> str:
    """中文说明：清洗并归一化数据，避免脏值进入后续规划。"""
    text = clean_location_hint_candidate(value) if value else ""
    if is_invalid_anchor_text(text):
        return ""
    return text.strip()

def strip_anchor_quotes(value: str) -> str:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    text = str(value or "").strip()
    text = text.strip("\"'“”‘’《》<>（）()[]【】")
    return text.strip()


def strip_anchor_edit_prefix(value: str) -> str:
    """把“出发地换为X / 目的地设为X”清洗成 X，再交给地点合法性判断。"""
    text = strip_anchor_quotes(value)
    text = re.sub(
        r"^(?:把|将|请把|帮我把)?(?:出发地|起点|始发地|目的地|终点|想去的地方|要去的地方)"
        r"(?:换成|换为|改成|改为|换到|改到|设为|设置为|为|是|在|到)",
        "",
        text,
    ).strip()
    text = re.sub(r"^(?:换成|换为|改成|改为|换到|改到|调整为|调整到|设为|设置为)", "", text).strip()
    return strip_anchor_quotes(text)


def text_has_departure_edit_prefix(value: str) -> bool:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    compact = re.sub(r"\s+", "", str(value or ""))
    return bool(re.search(r"^(?:把|将|请把)?(?:出发地|起点|始发地)(?:换成|换为|改成|改为|换到|改到|设为|设置为|为|是|在|到)", compact))


def text_has_destination_edit_prefix(value: str) -> bool:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    compact = re.sub(r"\s+", "", str(value or ""))
    return bool(re.search(r"^(?:把|将|请把)?(?:目的地|终点|想去的地方|要去的地方)(?:换成|换为|改成|改为|换到|改到|设为|设置为|为|是|在|到)", compact))


def is_departure_update_only_text(text: str) -> bool:
    """本轮只是在修改出发地时，不允许同一个地点进入 location/fixed_destination。"""
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    departure_hit = bool(re.search(r"(?:从.+出发|(?:把|将|请把)?(?:出发地|起点|始发地)(?:换成|换为|改成|改为|设为|设置为|为|是|在|到))", compact))
    destination_hit = bool(re.search(r"(?:目的地|终点|想去|要去|去玩|去逛|安排到|到[^，。！？；;、|｜]{2,30}|逛一下|玩一下)", compact))
    return departure_hit and not destination_hit


def is_departure_edit_value(value: str, latest_text: str = "", departure_hint: str = "") -> bool:
    """判断某个 location/fixed_destination 值是否其实是出发地修改误写入。"""
    raw = str(value or "").strip()
    if not raw:
        return False
    if text_has_departure_edit_prefix(raw):
        return True
    cleaned = strip_anchor_edit_prefix(raw)
    hint = str(departure_hint or "").strip()
    if hint and is_departure_update_only_text(latest_text) and normalize_place_text(cleaned) == normalize_place_text(hint):
        return True
    return False


def normalize_area_like_anchor(value: str) -> str:
    """中文说明：清洗并归一化数据，避免脏值进入后续规划。"""
    text = strip_anchor_edit_prefix(value)
    text = re.sub(r"(附近|周边|一带|周围|旁边|边上|周遭)$", "", text).strip()
    text = re.sub(r"^(上海市?)", "", text).strip()
    return text


def clean_spatial_anchor_candidate(candidate: str) -> str:
    """把“从徐汇出发想去中山公园附近/我们想在陆家嘴周边”清洗成真正锚点。

    这个函数专门处理“附近/周边”语义，不能只按后缀判断；否则会把
    “从徐汇出发想去中山公园”整段误当成锚点。
    """
    text = strip_anchor_edit_prefix(candidate)
    text = re.split(r"[，。！？；;、/／|｜\n]", text)[-1].strip()
    text = re.sub(r"^(?:我|我们|咱们|大家|想|要|希望|打算|计划)+", "", text).strip()
    # 若同一句里有出发地和目的地，出发地之前的内容全部丢掉，只保留“想去/在/到”之后的锚点。
    if "出发" in text:
        text = text.split("出发")[-1].strip()
    markers = ["想去", "要去", "想在", "要在", "计划去", "打算去", "安排在", "安排到", "围绕", "以", "去", "在", "到", "逛", "玩"]
    best_pos = -1
    best_marker = ""
    for marker in markers:
        pos = text.rfind(marker)
        if pos > best_pos:
            best_pos = pos
            best_marker = marker
    if best_pos >= 0:
        text = text[best_pos + len(best_marker):].strip()
    text = re.sub(r"^(?:为中心|为锚点|附近|周边|一带|周围|旁边|边上|周遭)", "", text).strip()
    text = re.sub(r"(?:附近|周边|一带|周围|旁边|边上|周遭)$", "", text).strip()
    text = re.sub(r"^(上海市?)", "", text).strip()
    text = re.sub(r"^(?:的|这边|那边)", "", text).strip()
    return strip_anchor_quotes(text)


def infer_anchor_query_from_text(text: str) -> str:
    """区域/附近锚点不直接入路线，需提取“周边找什么”的查询意图。

    Only use positive requirement text here.  Otherwise "想去陆家嘴；不要咖啡" would
    incorrectly turn the area query into "咖啡".
    """
    raw = positive_requirement_text_for_matching(str(text or ""))
    if any(w in raw for w in ["吃饭", "餐厅", "美食", "火锅", "韩料", "小笼", "生煎", "面馆"]):
        return "餐厅"
    if any(w in raw for w in ["咖啡", "下午茶", "甜品", "奶茶"]):
        return "咖啡"
    if any(w in raw for w in ["看展", "展览", "美术馆", "博物馆"]):
        return "展览"
    if any(w in raw for w in ["室内", "下雨", "别晒", "避雨"]):
        return "室内活动"
    if any(w in raw for w in ["散步", "逛", "走走", "citywalk", "休闲", "放松"]):
        return "休闲景点"
    return "周边休闲活动"


def extract_spatial_anchor_from_user_text(user_input: str) -> Optional[dict]:
    """识别“中山公园附近/陆家嘴/浦东新区/松江区”这类空间锚点。

    返回的 anchor 只用于搜索附近 POI，不直接进入最终 schedule。
    """
    text = positive_requirement_text_for_matching(str(user_input or ""))
    if not text:
        return None

    # 明确“X 附近/周边/一带”：X 必须作为空间锚点，不作为最终站点。
    nearby_patterns = [
        r"(?:想去|要去|去|逛|玩|安排|在|到)?\s*([^，。！？；;、/／\s]{2,30}?)(?:附近|周边|一带|周围|旁边|边上|周遭)",
        r"(?:围绕|以)([^，。！？；;、/／\s]{2,30}?)(?:为中心|为锚点|附近|周边)",
    ]
    for pattern in nearby_patterns:
        match = re.search(pattern, text)
        if match:
            anchor = clean_spatial_anchor_candidate(match.group(1))
            if anchor and anchor not in GENERIC_LOCATION_TERMS and not text_has_departure_edit_prefix(anchor):
                return {
                    "anchor": anchor,
                    "mode": "nearby",
                    "query": infer_anchor_query_from_text(text),
                    "exclude_anchor_from_schedule": True,
                    "note": f"用户表达为“{anchor}附近/周边”，系统将它作为搜索锚点，只推荐周边具体地点，不把锚点本身硬塞进路线。",
                }

    # 行政区和典型商圈/片区：同样作为区域锚点，不直接作为最终站点。
    candidate_terms = sorted(set(SHANGHAI_DISTRICT_TERMS) | AREA_ANCHOR_TERMS, key=len, reverse=True)
    for term in candidate_terms:
        if not term or term not in text:
            continue
        departure_context = re.search(rf"(?:从|出发地|起点|始发地)[^，。！？；;]{{0,10}}{re.escape(term)}", text)
        destination_context = re.search(rf"(?:想去|要去|去|逛|玩|安排|目的地|终点|附近|周边|一带)[^，。！？；;]{{0,12}}{re.escape(term)}", text)
        if departure_context and not destination_context:
            continue
        direct_landmark_request = (
            term in DIRECT_SCHEDULE_LANDMARK_TERMS
            and bool(destination_context)
            and not any(marker in text for marker in NEARBY_MARKERS)
        )
        if direct_landmark_request:
            continue
        if term in SHANGHAI_DISTRICT_TERMS or term in AREA_ANCHOR_TERMS:
            return {
                "anchor": term,
                "mode": "area",
                "query": infer_anchor_query_from_text(text),
                "exclude_anchor_from_schedule": True,
                "note": f"用户输入“{term}”更像区域/商圈锚点，系统会围绕该片区生成具体小地点，而不是把“{term}”直接当作一站。",
            }
    return None


def is_area_anchor_value(value: str) -> bool:
    """中文说明：判断当前值或状态是否满足指定条件。"""
    text = normalize_area_like_anchor(value)
    if not text:
        return False
    return text in SHANGHAI_DISTRICT_TERMS or text in AREA_ANCHOR_TERMS

_geocode_cache = {}
_geocode_detail_cache = {}
_amap_text_poi_cache = {}
_amap_business_poi_cache = {}
_route_distance_cache = {}
_amap_poi_cache = {}
_last_amap_request_at = 0.0
_amap_rate_limited_until = 0.0


# 高德对“品牌 + 分店/园区内点位”有时会命中同名市中心分店，
# 例如“芝乐坊餐厅（迪士尼小镇店）”被定位到南京西路附近，导致迪士尼小镇到城堡前广场显示二十多公里。
# 这里不是替代高德，而是对已知大型园区内锚点做低风险兜底：只要名称明确包含迪士尼/小镇/城堡等语义，先使用园区坐标。
DISNEY_COORD_OVERRIDES = [
    ("迪士尼小镇", "121.663650,31.144360", "上海市浦东新区迪士尼小镇"),
    ("迪士尼城堡", "121.657900,31.143500", "上海市浦东新区上海迪士尼乐园奇幻童话城堡附近"),
    ("城堡前广场", "121.658100,31.143800", "上海市浦东新区上海迪士尼乐园城堡前广场"),
    ("上海迪士尼度假区", "121.667850,31.144020", "上海市浦东新区上海迪士尼度假区"),
    ("迪士尼乐园", "121.667850,31.144020", "上海市浦东新区上海迪士尼乐园"),
    ("迪士尼", "121.667850,31.144020", "上海市浦东新区上海迪士尼度假区"),
]


def disney_coord_override(place_name: str) -> Optional[dict]:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    text = str(place_name or "").strip()
    if not text or "迪士尼" not in text:
        return None
    compact = normalize_place_text(text) if "normalize_place_text" in globals() else re.sub(r"\s+", "", text).lower()
    for key, coord, address in DISNEY_COORD_OVERRIDES:
        key_compact = normalize_place_text(key) if "normalize_place_text" in globals() else key.lower()
        if key_compact and key_compact in compact:
            return {"location": coord, "formatted_address": address, "poi_name": key, "source": "local_disney_coord_override"}
    return {"location": "121.667850,31.144020", "formatted_address": "上海市浦东新区上海迪士尼度假区", "poi_name": "上海迪士尼度假区", "source": "local_disney_coord_override"}


def is_area_anchor_schedule_self(place: str, anchor: str) -> bool:
    """Only remove the anchor itself from nearby/area routes, not every POI containing the anchor word.

    “迪士尼周围” should not render the generic anchor “迪士尼”, but it may render concrete nearby POIs
    such as “迪士尼小镇店/城堡前广场”. 旧逻辑用 same_route_place 会把这些具体点也删掉，最后只剩“待确认地点”。
    """
    place_text = str(place or "").strip()
    anchor_text = str(anchor or "").strip()
    if not place_text or not anchor_text:
        return False
    if same_anchor_identity(place_text, anchor_text):
        return True
    # “迪士尼周围”里的泛锚点本体应排除，但“迪士尼小镇店/城堡前广场”这类具体 POI 应保留。
    if normalize_place_text(anchor_text) == "迪士尼" and normalize_place_text(place_text) in {
        "迪士尼", "上海迪士尼", "迪士尼乐园", "上海迪士尼乐园", "上海迪士尼度假区"
    }:
        return True
    place_key = normalize_area_like_anchor(place_text)
    anchor_key = normalize_area_like_anchor(anchor_text)
    return bool(place_key and anchor_key and place_key == anchor_key and (is_area_anchor_value(place_key) or place_key in GENERIC_LOCATION_TERMS))

AMAP_TRANSIENT_LIMIT_INFOS = {
    "CUQPS_HAS_EXCEEDED_THE_LIMIT",
    "QPS_HAS_EXCEEDED_THE_LIMIT",
    "LOCAL_RATE_LIMIT_BACKOFF",
}


def get_amap_key() -> str:
    """读取高德 Web 服务 API Key。兼容 AMAP_API_KEY 和 GAODE_API_KEY 两种命名。"""
    return (os.getenv("AMAP_API_KEY") or os.getenv("GAODE_API_KEY") or "").strip()


def is_amap_transient_limit(info: str) -> bool:
    """中文说明：判断当前值或状态是否满足指定条件。"""
    return str(info or "").strip() in AMAP_TRANSIENT_LIMIT_INFOS


def amap_backoff_remaining() -> float:
    """Seconds until the local Amap backoff window ends.

    Older code returned LOCAL_RATE_LIMIT_BACKOFF immediately during this window.
    That made one transient QPS response poison all following geocode/distance calls in
    the same planning run, so the same POI was logged as failed repeatedly.
    """
    return max(0.0, float(_amap_rate_limited_until or 0.0) - time.time())


def amap_get_json(url: str, params: dict, timeout: int = 2) -> dict:
    """带本地节流和轻量重试的高德请求。

    修复点：
    - 不再在本地 backoff 期间直接返回 LOCAL_RATE_LIMIT_BACKOFF；可等待的短 backoff 会先等待。
    - 默认请求间隔从 0.05s 调到 0.35s，避免一次规划内 geocode / poi / direction 连续请求触发高德 QPS。
    - 瞬时 QPS 错误最多重试 1 次；失败后不把瞬时失败写入永久缓存。
    - AMAP_REQUEST_TIMEOUT_SECONDS 现在可以把默认 timeout 调大，而不是被 min() 永远压回 2 秒。
    """
    global _last_amap_request_at, _amap_rate_limited_until

    min_interval = float(os.getenv("AMAP_REQUEST_INTERVAL_SECONDS", "0.35"))
    max_retries = max(0, int(os.getenv("AMAP_MAX_RETRIES", "1")))
    backoff_seconds = float(os.getenv("AMAP_QPS_BACKOFF_SECONDS", "1.2"))
    max_backoff_wait = float(os.getenv("AMAP_MAX_BACKOFF_WAIT_SECONDS", "2.0"))
    configured_timeout = os.getenv("AMAP_REQUEST_TIMEOUT_SECONDS")
    if configured_timeout:
        try:
            timeout = max(1, min(float(configured_timeout), 8.0))
        except ValueError:
            timeout = max(1, timeout)

    last_data = {"status": "0", "info": "LOCAL_RATE_LIMIT_BACKOFF"}
    for attempt in range(max_retries + 1):
        remaining = amap_backoff_remaining()
        if remaining > 0:
            # 等短暂 backoff，而不是直接失败；过长 backoff 只等上限，避免拖垮整轮规划。
            time.sleep(min(remaining, max_backoff_wait))

        elapsed = time.time() - _last_amap_request_at
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        query = urllib.parse.urlencode(params, safe=",|")
        with urllib.request.urlopen(f"{url}?{query}", timeout=timeout) as response:
            _last_amap_request_at = time.time()
            data = json.loads(response.read().decode("utf-8"))

        last_data = data
        if not is_amap_transient_limit(data.get("info")):
            return data

        _amap_rate_limited_until = time.time() + backoff_seconds
        if attempt < max_retries:
            time.sleep(min(backoff_seconds, max_backoff_wait))

    return last_data


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
    """中文说明：在多个候选中选择最适合当前约束的结果。"""
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
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
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
    """中文说明：查询外部接口或本地数据，返回匹配候选。"""
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
    """中文说明：清洗并归一化数据，避免脏值进入后续规划。"""
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

    override = disney_coord_override(address)
    if override and override.get("location"):
        return str(override.get("location"))

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
                _geocode_cache[cache_key] = saved_coord
                return saved_coord
    except Exception:
        pass

    poi = None if area_like else choose_best_poi_for_place(address, amap_search_place_text(address, city=city, limit=5))
    if poi and poi.get("location"):
        coord = str(poi.get("location") or "").strip()
        _geocode_cache[cache_key] = coord
        if poi.get("formatted_address"):
            _geocode_detail_cache[(normalized, city)] = {
                "formatted_address": poi.get("formatted_address", ""),
                "province": poi.get("province", ""),
                "city": poi.get("city", ""),
                "district": poi.get("district", ""),
                "location": coord,
                "level": "POI",
                "poi_name": poi.get("name", ""),
                "poi_type": poi.get("type", ""),
                "source": "amap_place_text",
            }
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
    """中文说明：清洗并归一化数据，避免脏值进入后续规划。"""
    if isinstance(value, list):
        value = "".join(str(item) for item in value if item)
    text = str(value or "").strip()
    if text.lower() in {"nan", "none", "null", "[]"}:
        return ""
    return text


def join_amap_address_parts(*parts) -> str:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
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


def extract_amap_business_hours(poi: dict) -> dict:
    """Extract opening-hour fields returned by Amap POI 2.0 when available."""
    business = poi.get("business") if isinstance(poi.get("business"), dict) else {}
    return {
        "opentime_today": clean_amap_address_part(
            business.get("opentime_today")
            or poi.get("opentime_today")
            or poi.get("opentime")
        ),
        "opentime_week": clean_amap_address_part(
            business.get("opentime_week")
            or poi.get("opentime_week")
        ),
    }


def amap_search_place_business(place_name: str, city: str = "上海", limit: int = 3) -> list[dict]:
    """Query Amap POI 2.0 for concrete POI info including opening hours when returned."""
    key = get_amap_key()
    keyword = str(place_name or "").strip()
    if not key or not keyword:
        return []

    limit = max(1, min(int(limit or 3), 5))
    cache_key = (keyword, city, limit)
    if cache_key in _amap_business_poi_cache:
        return _amap_business_poi_cache[cache_key]

    try:
        data = amap_get_json(
            "https://restapi.amap.com/v5/place/text",
            {
                "key": key,
                "keywords": keyword,
                "region": city,
                "city_limit": "true",
                "show_fields": "business",
                "page_size": str(limit),
                "page_num": "1",
                "output": "JSON",
            },
            timeout=4,
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as e:
        print(f"⚠️ 高德POI营业时间搜索失败: {keyword} / {e}")
        _amap_business_poi_cache[cache_key] = []
        return []

    if data.get("status") != "1":
        info = data.get("info")
        print(f"⚠️ 高德POI营业时间搜索无结果: {keyword} / {info}")
        if not is_amap_transient_limit(info):
            _amap_business_poi_cache[cache_key] = []
        return []

    pois = []
    for poi in data.get("pois", []) or []:
        name = clean_amap_address_part(poi.get("name"))
        location = clean_amap_address_part(poi.get("location"))
        if not name or not location:
            continue
        hours = extract_amap_business_hours(poi)
        pois.append({
            "name": name,
            "address": clean_amap_address_part(poi.get("address")),
            "location": location,
            "type": clean_amap_address_part(poi.get("type")),
            "source": "amap_place_text_v5",
            **hours,
        })
    _amap_business_poi_cache[cache_key] = pois
    return pois


def choose_best_poi_for_place(place_name: str, pois: list[dict]) -> Optional[dict]:
    """中文说明：在多个候选中选择最适合当前约束的结果。"""
    key = normalize_place_text(place_name)
    if not key or not pois:
        return None
    scored = []
    raw_place = str(place_name or "")
    branch_tokens = [
        token for token in re.split(r"[（）()·•\s\-_/【】\[\]《》,，、]+", raw_place)
        if token and len(normalize_place_text(token)) >= 2
    ]
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
        poi_full_text = f"{poi_name} {address} {poi.get('district','')} {poi.get('type','')}"
        poi_full_key = normalize_place_text(poi_full_text)
        for token in branch_tokens:
            tk = normalize_place_text(token)
            if tk and tk in poi_full_key:
                score += 30
        if "迪士尼" in raw_place:
            if "迪士尼" in poi_full_text or "川沙" in poi_full_text or "浦东" in poi_full_text:
                score += 120
            if any(bad in poi_full_text for bad in ["南京西路", "静安", "太古汇", "人民广场", "淮海"]):
                score -= 180
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

    override = disney_coord_override(address)
    if override and override.get("location"):
        return {
            "formatted_address": override.get("formatted_address", ""),
            "location": override.get("location", ""),
            "poi_name": override.get("poi_name", ""),
            "level": "POI",
            "source": override.get("source", "local_override"),
        }

    try:
        row = _find_place(address)
        if row is not None:
            saved_address = clean_amap_address_part(row.get("amap_address", ""))
            saved_coord = clean_amap_address_part(row.get("amap_location", ""))
            saved_district = clean_amap_address_part(row.get("amap_district", ""))
            if saved_address or saved_coord:
                detail = {
                    "formatted_address": normalize_shanghai_address(saved_address) if saved_address else normalize_shanghai_address(address),
                    "location": saved_coord,
                    "district": saved_district,
                    "source": "place_table",
                }
                normalized_for_cache = normalize_shanghai_address(address)
                _geocode_detail_cache[(normalized_for_cache, city)] = detail
                if saved_coord:
                    _geocode_cache[(normalized_for_cache, city)] = saved_coord
                return detail
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
        if detail.get("location"):
            _geocode_cache[cache_key] = detail.get("location")
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
    if detail.get("location"):
        _geocode_cache[cache_key] = detail.get("location")
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
    """中文说明：构建当前流程需要的结构化对象或展示内容。"""
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


def set_schedule_place(item: dict, place: str) -> dict:
    """Update a schedule item to a new place without carrying stale display data."""
    copied = reset_schedule_place_fields(item)
    copied["place"] = place
    role = place_role(place)
    meta = schedule_place_type_metadata(place)
    copied.update(meta)
    copied["place_role"] = role
    display_type = str(meta.get("display_type") or "").strip()
    copied["purpose"] = (
        "正餐/核心用餐" if role == "meal"
        else "轻量补充/咖啡休息" if role == "light_food"
        else f"{display_type}/体验" if display_type and display_type not in {"景点", "餐饮"}
        else "顺路游玩/散步体验"
    )
    return copied


def enrich_schedule_addresses(schedule: list[dict]) -> list[dict]:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
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
    """从用户需求中抽取适合交给高德周边搜索的关键词。

    The input may be canonical planning text that contains negative requirements
    (e.g. "不要安排：咖啡").  We therefore compute a positive recall text and
    explicitly skip AMAP specs that only match negated terms.
    """
    base_positive = positive_requirement_text_for_matching(user_input)
    negative_terms = extract_negated_requirement_terms(user_input)
    positive_pieces = [base_positive]
    for value in [
        str((intent or {}).get("location") or ""),
        str((intent or {}).get("meal_pref") or ""),
        " ".join((intent or {}).get("place_keywords", []) or []),
    ]:
        if not _keyword_is_negated_for_recall(value, value, negative_terms, base_positive):
            positive_pieces.append(value)
    text = positive_requirement_text_for_matching(" ".join(positive_pieces)).lower()
    rules = [
        (["海底捞"], "海底捞", "restaurant", "hotpot"),
        (["火锅", "涮锅", "锅底"], "火锅", "restaurant", "hotpot"),
        (["咖啡", "coffee", "星巴克", "蜜雪冰城"], "咖啡", "cafe", "cafe"),
        (["下午茶"], "下午茶", "cafe", "cafe"),
        (["奶茶"], "奶茶", "cafe", "milk_tea"),
        (["小笼包", "小笼", "汤包"], "小笼包", "restaurant", "xiaolongbao"),
        (["生煎"], "生煎", "restaurant", "shengjian"),
        (["面馆", "吃面", "汤面", "拉面"], "面馆", "restaurant", "noodle"),
        (["韩料", "韩国料理", "韩式"], "韩国料理", "restaurant", "korean_cuisine"),
        (["江浙菜", "本帮菜", "上海菜"], "江浙菜", "restaurant", "jiangzhe_cuisine"),
        (["面包", "烘焙", "甜品"], "面包店", "cafe", "bakery"),
        (["看展", "艺术展", "美术馆", "画廊"], "美术馆", "attraction", "art_exhibition"),
        (["博物馆"], "博物馆", "attraction", "museum"),
        (["散步", "遛弯", "踏青", "公园", "户外"], "公园", "leisure", "park"),
        (["寺庙", "寺"], "寺庙", "attraction", "temple"),
        (["电影", "影院", "影城"], "电影院", "activity", "cinema"),
        (["泡汤", "汤泉", "温泉"], "汤泉", "activity", "spa_relax"),
    ]
    for triggers, keyword, place_type, sub_type in rules:
        if _keyword_is_negated_for_recall(keyword, place_type, negative_terms, base_positive):
            continue
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
    limit = max(1, min(_safe_int(limit, 5), 25))
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
    """中文说明：判断当前值或状态是否满足指定条件。"""
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
        "密室", "密室逃脱", "剧本杀", "推理馆", "ktv", "唱歌", "清吧",
        "酒吧", "livehouse", "美甲", "温泉", "spa", "购物", "逛街",
        "海底捞", "星巴克", "manner", "seesaw", "% arabica",
        "逛吃", "吃喝", "吃喝玩乐", "玩乐", "轻松逛吃", "休闲娱乐",
        "散心", "散心活动", "休闲活动", "放松", "活动",
    }
    category_keys = {normalize_place_text(term) for term in category_terms}
    if text in category_keys:
        return True
    soft_preference_terms = {"逛吃", "吃喝", "吃喝玩乐", "玩乐", "轻松", "休闲", "散心", "放松", "活动"}
    return bool(text) and any(term in text for term in soft_preference_terms) and not is_shanghai_area_location(location)


def is_concrete_location_anchor(location: str) -> bool:
    """Return True when a location can be treated as a route anchor instead of only a preference."""
    text = str(location or "").strip()
    if not text or is_invalid_anchor_text(text):
        return False
    cleaned = clean_location_hint_candidate(text)
    if not cleaned or is_invalid_anchor_text(cleaned):
        return False
    return cleaned not in GENERIC_LOCATION_TERMS and not is_category_like_location(cleaned)


def default_amap_search_spec(intent: dict, user_input: str = "") -> dict:
    """区域/附近搜索的默认 spec：需求阶段只二分类，具体标签等高德返回。"""
    explicit = infer_amap_search_spec(intent, user_input)
    if explicit:
        explicit = dict(explicit)
        explicit["place_type"] = simplified_requirement_place_type(explicit.get("keyword", ""), explicit.get("sub_type", ""), explicit.get("category", ""))
        explicit.pop("sub_type", None)
        return explicit

    text = positive_requirement_text_for_matching(
        f"{user_input} {(intent or {}).get('location', '')} {' '.join((intent or {}).get('place_keywords', []) or [])}"
    )
    if is_food_drink_requirement_text(text):
        if any(term in text for term in ["咖啡", "奶茶", "下午茶", "甜品", "面包"]):
            return {"keyword": "咖啡", "place_type": "restaurant"}
        return {"keyword": "餐厅", "place_type": "restaurant"}
    if any(term in text for term in ["ktv", "KTV", "唱K", "唱歌"]):
        return {"keyword": "KTV", "place_type": "attraction"}
    if any(term in text for term in ["电影", "影院", "影城"]):
        return {"keyword": "电影院", "place_type": "attraction"}
    if any(term in text for term in ["公园", "散步", "户外", "踏青"]):
        return {"keyword": "公园", "place_type": "attraction"}
    return {"keyword": "景点", "place_type": "attraction"}


def classify_amap_poi_spec(poi: dict, intent: dict, user_input: str = "") -> dict:
    """高德 POI 标签为最终权威；LLM/用户需求只提供搜索词，不覆盖高德 type。"""
    fallback_keyword = str(poi.get("keyword") or poi.get("name") or "").strip()
    return derive_amap_authoritative_spec(poi, fallback_keyword=fallback_keyword)


def persist_text_poi_with_mock(poi: dict, spec: dict) -> str:
    """把 place/text 返回的 POI 转成运行时 POI 结构，不写入 mock 表。"""
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
                print(f"⚠️ 区域锚点 POI 运行时注册失败，仅作为本次候选使用: {chosen_name} / {e}")
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
            print(f"⚠️ 用户点名高德 POI 运行时注册失败，仅作为本次候选使用: {chosen_name} / {e}")
        fixed["location"] = chosen_name
        fixed["fixed_destination"] = chosen_name
        fixed["active_destination_anchor"] = chosen_name
        fixed["center_anchor"] = chosen_name
        fixed["amap_resolved_original_location"] = location
        fixed["amap_resolved_location"] = chosen_name
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


def sync_resolved_poi_anchor_fields(state: AgentState, original_name: str, resolved_name: str) -> AgentState:
    """高德把用户口语地点解析成真实 POI 后，同步所有核心锚点字段。

    否则会出现：日志里显示已解析为真实 POI，但 collected_info / locked_places
    仍保留“密室逃脱1小时”这类原始词，最终 structured_plan 又被旧锚点拉回去。
    """
    original = str(original_name or "").strip()
    resolved = str(resolved_name or "").strip()
    if not resolved:
        return state
    updated = dict(state or {})
    collected = dict(updated.get("collected_info") or {})
    intent = dict(updated.get("intent") or {})
    for target in [collected, intent]:
        for key in ["location", "fixed_destination", "active_destination_anchor", "center_anchor"]:
            current = str(target.get(key) or "").strip()
            if not current or not original or same_route_place(current, original) or current == original:
                target[key] = resolved
        target["_location_explicit"] = True
        target["explicit_place_match"] = True
        target["amap_resolved_original_location"] = original
        target["amap_resolved_location"] = resolved
    locked = []
    for place in updated.get("locked_places") or []:
        p = str(place or "").strip()
        if not p:
            continue
        if original and (same_route_place(p, original) or p == original):
            locked.append(resolved)
        else:
            locked.append(p)
    if resolved:
        locked.append(resolved)
    updated["locked_places"] = unique_preserve_order(locked)
    updated["fixed_destination"] = resolved
    updated["active_destination_anchor"] = resolved
    updated["center_anchor"] = resolved
    updated["collected_info"] = collected
    updated["intent"] = intent
    return updated


def estimate_price_by_sub_type(sub_type: str, place_type: str) -> tuple[int, int]:
    """中文说明：估算距离、时间、价格或其他规划指标。"""
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


def stable_demo_ratio(name: str, salt: str = "") -> float:
    """Return a deterministic 0-1 demo ratio for one place and rule."""
    seed = f"{name}|{salt}"
    digest = hashlib.sha256(seed.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def amap_demo_booking_state(place_name: str, place_type: str, sub_type: str, amap_spec: dict) -> bool:
    """Mock whether an Amap-only place is bookable/needs booking by type."""
    name = str(place_name or "")
    raw_text = " ".join([
        name,
        str(sub_type or ""),
        str(place_type or ""),
        str(amap_spec.get("amap_display_tag") or ""),
        " ".join(str(x) for x in amap_spec.get("amap_tags") or []),
        str(amap_spec.get("amap_raw_type") or ""),
    ]).lower()
    if sub_type in {"park", "street_walk", "temple"} or any(k in raw_text for k in ["公园", "绿地", "步行街", "自然", "广场"]):
        return False
    if sub_type in {"museum", "art_exhibition"} or any(k in raw_text for k in ["博物馆", "美术馆", "展览", "纪念馆"]):
        return stable_demo_ratio(name, "booking:museum") < 0.75
    if place_type == "restaurant":
        if sub_type in {"cafe", "bakery", "xiaolongbao", "shengjian", "noodle"}:
            return stable_demo_ratio(name, "booking:light_food") < 0.25
        return stable_demo_ratio(name, "booking:restaurant") < 0.45
    if place_type in {"activity", "sports", "leisure"}:
        return stable_demo_ratio(name, "booking:activity") < 0.55
    return False


def is_placeholder_route_place(value: str) -> bool:
    """Return True for UI/explanation placeholders that must never enter schedule."""
    text = str(value or "").strip()
    if not text:
        return True
    compact = re.sub(r"\s+", "", text)
    placeholder_terms = {
        "作为起点",
        "以目的地作为起点",
        "未指定以目的地作为起点",
        "未指定作为起点",
        "未指定",
        "待确认地点",
    }
    if compact in placeholder_terms:
        return True
    if any(marker in compact for marker in ["作为起点", "未指定以目的地", "以目的地作为"]):
        return True
    return is_invalid_anchor_text(text)


def clean_route_place_candidates(values) -> list[str]:
    """Remove placeholders/category-only values before building final schedule."""
    cleaned = []
    for value in values or []:
        place = str(value or "").strip()
        if not place or is_placeholder_route_place(place):
            continue
        cleaned.append(place)
    return unique_preserve_order(cleaned)


def amap_demo_seat_count(place_name: str, place_type: str, sub_type: str, amap_spec: dict) -> int:
    """Mock final demo inventory for Amap-only POIs: either 0 or a positive count."""
    name = str(place_name or "")
    raw_text = " ".join([
        name,
        str(sub_type or ""),
        str(place_type or ""),
        str(amap_spec.get("amap_display_tag") or ""),
        " ".join(str(x) for x in amap_spec.get("amap_tags") or []),
    ]).lower()
    if sub_type in {"park", "street_walk", "temple"} or any(k in raw_text for k in ["公园", "绿地", "步行街", "自然", "广场"]):
        return 999

    ratio = stable_demo_ratio(name, "seat:zero")
    if place_type == "restaurant":
        zero_threshold = 0.16 if sub_type not in {"cafe", "bakery"} else 0.08
        if ratio < zero_threshold:
            return 0
        return 4 + int(stable_demo_ratio(name, "seat:restaurant") * 56)
    if place_type in {"activity", "sports", "leisure"}:
        if ratio < 0.08:
            return 0
        return 3 + int(stable_demo_ratio(name, "seat:activity") * 37)
    if sub_type in {"museum", "art_exhibition"}:
        if ratio < 0.06:
            return 0
        return 20 + int(stable_demo_ratio(name, "seat:museum") * 180)
    base = stable_mock_seat_count(name)
    return base if base > 0 else 12 + int(stable_demo_ratio(name, "seat:default") * 80)


def amap_demo_coupon_state(place_name: str, place_type: str, sub_type: str, amap_spec: dict) -> bool:
    """Mock whether an Amap-only POI has one displayable coupon."""
    name = str(place_name or "")
    raw_text = " ".join([
        name,
        str(sub_type or ""),
        str(place_type or ""),
        str(amap_spec.get("amap_display_tag") or ""),
        " ".join(str(x) for x in amap_spec.get("amap_tags") or []),
    ]).lower()
    if sub_type in {"park", "street_walk", "temple"} or any(k in raw_text for k in ["公园", "绿地", "步行街", "自然", "广场"]):
        return False
    if place_type == "restaurant":
        threshold = 0.65 if sub_type in {"hotpot", "korean_cuisine", "jiangzhe_cuisine"} else 0.45
        return stable_demo_ratio(name, "coupon:restaurant") < threshold
    if sub_type in {"cafe", "bakery", "xiaolongbao", "shengjian", "noodle"}:
        return stable_demo_ratio(name, "coupon:light_food") < 0.5
    if place_type in {"activity", "sports", "leisure"}:
        return stable_demo_ratio(name, "coupon:activity") < 0.35
    return False


def build_amap_place_record(poi: dict, spec: dict) -> dict:
    """把高德 POI 转成运行时地点记录；高德标签是最终权威，不被 spec 覆盖。"""
    amap_spec = derive_amap_authoritative_spec(
        poi,
        fallback_keyword=str((spec or {}).get("keyword") or poi.get("keyword") or poi.get("name") or ""),
        fallback_place_type=str((spec or {}).get("place_type") or ""),
    )
    place_type = amap_spec.get("place_type", "attraction")
    sub_type = amap_spec.get("sub_type", "scenic")
    low, high = estimate_price_by_sub_type(sub_type, place_type)
    keyword = amap_spec.get("keyword", "")
    poi_name = str(poi.get("name") or "")
    need_booking = amap_demo_booking_state(poi_name, place_type, sub_type, amap_spec)
    seat_count = amap_demo_seat_count(poi_name, place_type, sub_type, amap_spec)
    has_coupon = amap_demo_coupon_state(poi_name, place_type, sub_type, amap_spec)
    tag_list = unique_preserve_order([
        "高德POI", "附近搜索", str(keyword),
        *[str(x) for x in amap_spec.get("amap_tags") or []],
        "真实地图候选",
    ])
    tags = "、".join([t for t in tag_list if t])
    return {
        "placeName": poi["name"],
        "是否可以预约": need_booking,
        "是否需要预约": need_booking,
        # 地图只提供 POI 存在性。Demo 按高德类型生成模拟预约/库存/团购状态：
        # 同一地点多次查询一致，不同地点允许出现 0，也会有正数余位。
        "是否有余位": seat_count > 0,
        "余位信息": seat_count,
        "availability_status": "available" if seat_count > 0 else "no_seat",
        "_seat_count_explicit": True,
        "_amap_demo_status": True,
        "是否有团购": has_coupon,
        "最低价格": low,
        "最高价格": high,
        "地点类型": place_type,
        "primary_type": place_type,
        "sub_type": sub_type,
        "search_tags": tags,
        "source_note": f"高德POI搜索候选：{poi.get('address', '')}；高德标签={amap_spec.get('amap_display_tag') or '未返回'}；Demo 已按地点类型模拟预约、余位和团购状态。",
        "amap_location": poi.get("location", ""),
        "amap_address": poi.get("address", ""),
        "amap_distance_from_query_m": _safe_int(poi.get("distance_m"), 0),
        "amap_raw_type": amap_spec.get("amap_raw_type", ""),
        "amap_type_path": amap_spec.get("amap_type_path", []),
        "amap_tags": "、".join(amap_spec.get("amap_tags") or []),
        "amap_display_tag": amap_spec.get("amap_display_tag", ""),
        "amap_route_category": amap_spec.get("amap_route_category", ""),
        "amap_primary_type": amap_spec.get("amap_primary_type", place_type),
        "display_type": amap_spec.get("amap_display_tag") or ("餐饮" if place_type == "restaurant" else "景点"),
        "tag_source": amap_spec.get("tag_source", "amap"),
    }


def persist_amap_poi_with_mock(poi: dict, spec: dict, force_coupon: bool = False) -> str:
    """Register an Amap POI for this process only; never write it into mock Excel."""
    name = str(poi.get("name") or "").strip()
    if not name:
        return ""
    existing = _find_place(name)
    if existing is not None:
        return name

    record = build_amap_place_record(poi, spec)
    poi_name = name
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
    key = normalize_place_text(poi_name)
    _runtime_amap_place_records[key] = record
    _runtime_amap_poi_records[key] = dict(poi, **{
        "name": poi_name,
        "place_type": record.get("地点类型", spec.get("place_type", "")),
        "sub_type": record.get("sub_type", spec.get("sub_type", "")),
        "amap_raw_type": record.get("amap_raw_type", ""),
        "amap_display_tag": record.get("amap_display_tag", ""),
        "amap_route_category": record.get("amap_route_category", ""),
        "amap_primary_type": record.get("amap_primary_type", ""),
        "tag_source": record.get("tag_source", ""),
    })
    return poi_name


def maybe_add_nearby_amap_candidate(intent: dict, state: AgentState) -> dict:
    """Deprecated: POI selection now happens in structured_plan after anchor choice."""
    return dict(intent or {})


def amap_driving_distance(origin_coord: str, dest_coord: str) -> Optional[dict]:
    """调用高德驾车路径规划，返回距离/时间和真实路径轨迹。strategy=2 表示距离优先。"""
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
    step_polylines = [
        str(step.get("polyline") or "").strip()
        for step in (path.get("steps") or [])
        if isinstance(step, dict) and str(step.get("polyline") or "").strip()
    ]
    polyline = ";".join(step_polylines)
    result = {
        "distance_m": _safe_int(path.get("distance"), 0),
        "duration_s": _safe_int(path.get("duration"), 0),
        "polyline": polyline,
        "strategy": "driving_distance_first",
    }
    _route_distance_cache[cache_key] = result
    return result


def format_distance(meters: int) -> str:
    """中文说明：把结构化数据格式化为前端或用户可读内容。"""
    if meters <= 0:
        return "未知"
    if meters < 1000:
        return f"{meters}米"
    return f"{meters / 1000:.1f}公里"


def format_duration(seconds: int) -> str:
    """中文说明：把结构化数据格式化为前端或用户可读内容。"""
    if seconds <= 0:
        return "未知"
    minutes = max(1, round(seconds / 60))
    if minutes < 60:
        return f"{minutes}分钟"
    return f"{minutes // 60}小时{minutes % 60}分钟"


def suggest_transport(distance_m: int) -> str:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
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
    """中文说明：从文本或结构中抽取当前流程需要的关键信息。"""
    text = str(route_plan_text or "")
    match = re.search(r"用餐\((.*)\)\s*(?:→|,|，|$)", text)
    if not match:
        match = re.search(r"用餐\((.*)\)", text)
    if not match:
        return []
    names = re.split(r"[、,，/|]+", match.group(1))
    return [name.strip() for name in names if name.strip()]


def choose_nearest_place(anchor_name: str, candidate_names: list) -> Optional[dict]:
    """从候选地点中选距离 anchor 最近的一个。无高德 Key 或查询失败时返回 None。"""
    anchor_coord = amap_geocode(anchor_name)
    if not anchor_coord:
        return None

    best = None
    max_candidates = _safe_int(os.getenv("AMAP_MAX_MEAL_CANDIDATES", "2"), 2)
    for name in candidate_names[:max_candidates]:
        coord = amap_geocode(name)
        if not coord:
            continue
        route = amap_driving_distance(anchor_coord, coord)
        if not route:
            continue
        item = {
            "name": name,
            "coord": coord,
            "distance_m": route["distance_m"],
            "duration_s": route["duration_s"],
        }
        if best is None or item["distance_m"] < best["distance_m"]:
            best = item
    return best


def build_route_distance_info(departure: str, main_location: str, route_plan_text: str) -> str:
    """生成 出发地 -> 主地点 -> 最近用餐点 的高德距离说明。"""
    if not get_amap_key():
        return "未配置 AMAP_API_KEY 或 GAODE_API_KEY，当前无法调用高德地图计算真实转场距离；禁止在方案中输出任何具体距离、分钟数或费用估算，只能提示用户配置 Key 后再计算。"

    stops = [str(departure or "").strip(), str(main_location or "").strip()]
    meal_candidates = extract_meal_candidates(route_plan_text)
    nearest_meal = choose_nearest_place(main_location, meal_candidates) if meal_candidates else None
    if nearest_meal:
        stops.append(nearest_meal["name"])

    stops = [stop for stop in stops if stop]
    if len(stops) < 2:
        return "高德距离计算失败：缺少出发地或目标地点。"

    lines = ["高德真实距离参考（驾车距离优先；短距离可按建议改步行/骑行）："]
    total_m = 0
    failed = []
    for idx in range(len(stops) - 1):
        start, end = stops[idx], stops[idx + 1]
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
        lines.append(
            f"{idx + 1}. {start} -> {end}: "
            f"{format_distance(route['distance_m'])}，约{format_duration(route['duration_s'])}，"
            f"建议：{suggest_transport(route['distance_m'])}"
        )

    if nearest_meal:
        lines.append(
            f"用餐点已按距离从候选中优先选择：{nearest_meal['name']} "
            f"（距 {main_location} {format_distance(nearest_meal['distance_m'])}）。"
        )
    if total_m:
        lines.append(f"当前核心转场合计约 {format_distance(total_m)}。")
    if failed:
        lines.append("未成功计算：" + "；".join(failed))
    return "\n".join(lines)


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


def repair_structured_plan_distance_violations(state: AgentState) -> tuple[dict, list[str]]:
    """Backward-compatible no-op: distance is now reference-only, not a hard repair step."""
    return state.get("structured_plan") or {}, []


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
    started_at = time.time()
    # 这里不是强制 30 秒兜底，只是限制高德距离子流程自身不要无限重试。
    # 默认计算“出发地->第一站”和后续最多 3 段，覆盖常见 3-4 站半日路线。
    max_wall = float(os.getenv("AMAP_DISTANCE_WALL_LIMIT_SECONDS", "8"))
    max_segments = int(os.getenv("AMAP_DISTANCE_MAX_SEGMENTS", "4"))
    for idx in range(min(len(route_stops) - 1, max_segments)):
        if time.time() - started_at > max_wall:
            failed.append("高德距离计算超过本轮时间上限，已停止剩余路段，避免方案生成超时")
            break
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
            "polyline": route.get("polyline", ""),
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


def overlong_route_segments(segments: list[dict], max_m: Optional[int] = None) -> list[dict]:
    """Backward-compatible no-op: no fixed-kilometer overlong route blocking is applied."""
    return []


def schedule_index_for_place(schedule: list[dict], place_name: str) -> Optional[int]:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    target = normalize_place_text(place_name)
    if not target:
        return None
    for index, item in enumerate(schedule or []):
        if not isinstance(item, dict):
            continue
        place = str(item.get("place") or "")
        if normalize_place_text(place) == target or same_route_place(place, place_name):
            return index
    return None


def bridge_replacement_between(
        previous_anchor: str,
        next_anchor: str,
        original_place: str,
        intent: dict,
        max_m: int,
        force_indoor: bool = False,
        force_coupon: bool = False,
        exclude_places: Optional[set] = None,
) -> Optional[str]:
    """Find one replacement that is within max_m from both adjacent anchors."""
    if not previous_anchor or not next_anchor:
        return None
    exclude_keys = {normalize_place_text(p) for p in (exclude_places or set())}
    wanted_role = place_role(original_place)
    candidates = []

    for _, row in _df.iterrows():
        name = str(row.get("placeName", "") or "").strip()
        if not name:
            continue
        key = normalize_place_text(name)
        if not key or key in exclude_keys:
            continue
        if same_route_place(name, previous_anchor) or same_route_place(name, next_anchor):
            continue
        role = place_role(name)
        if wanted_role in {"meal", "light_food"} and role != wanted_role:
            continue
        if force_coupon and not bool(row.get("是否有团购", False)):
            continue
        d1 = route_distance_between_places(previous_anchor, name)
        if d1 is None or d1 > max_m:
            continue
        d2 = route_distance_between_places(name, next_anchor)
        if d2 is None or d2 > max_m:
            continue
        candidates.append((d1 + d2, name))

    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    spec = amap_search_spec_for_replacement(original_place, intent, force_indoor=force_indoor)
    pois = amap_search_pois_near(
        departure=next_anchor,
        keyword=spec["keyword"],
        radius=max(1000, max_m),
        limit=_safe_int(os.getenv("AMAP_POI_LIMIT", "5"), 5),
    )
    for poi in pois:
        name = str(poi.get("name") or "").strip()
        if not name or normalize_place_text(name) in exclude_keys:
            continue
        d1 = route_distance_between_places(previous_anchor, name)
        d2 = route_distance_between_places(name, next_anchor)
        if d1 is None or d2 is None or d1 > max_m or d2 > max_m:
            continue
        try:
            return persist_amap_poi_with_mock(poi, spec, force_coupon=force_coupon)
        except Exception as e:
            print(f"⚠️ 高德桥接候选运行时注册失败: {name} / {e}")
            return name
    return None


def enforce_route_segments_after_amap(
        state: AgentState,
        structured_plan: dict,
        departure: str,
) -> tuple[dict, list[str], str, list[dict], list[str]]:
    """Backward-compatible no-op: distance is reference-only; no segment repair is applied."""
    stops = [
        item.get("place")
        for item in (structured_plan.get("schedule") or [])
        if isinstance(item, dict) and item.get("place") and item.get("place") != "待确认地点"
    ]
    route_distance_info, route_segments, failed_segments = compute_route_segments(departure, stops) if stops else ("", [], [])
    return structured_plan, [], route_distance_info, route_segments, failed_segments


def parse_lnglat(location: str) -> Optional[tuple[float, float]]:
    """中文说明：解析输入或中间结果，转换为后续节点可用的数据。"""
    match = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$", str(location or ""))
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def _parse_static_map_size(size: str) -> tuple[int, int]:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    match = re.match(r"^(\d{2,4})\*(\d{2,4})$", str(size or ""))
    if not match:
        return 1024, 640
    return int(match.group(1)), int(match.group(2))


def _lat_to_mercator_y(lat: float) -> float:
    # Web mercator normalized y in radians. Clamp avoids infinite values.
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    lat = max(-85.0, min(85.0, float(lat)))
    rad = lat * 3.141592653589793 / 180.0
    return math.log(math.tan(3.141592653589793 / 4.0 + rad / 2.0))


def estimate_static_map_zoom(coords: list[tuple[float, float]], size: str = "1024*640") -> int:
    """Pick a conservative static-map zoom so every stop remains visible.

    Earlier logic used a rough degree-span table. When two stops were far apart,
    the map could crop one marker. This version computes the zoom from the
    longitude/mercator-latitude bounding box plus padding, then chooses the
    highest zoom that still fits inside the static image.
    """
    if len(coords) < 2:
        return 16

    width_px, height_px = _parse_static_map_size(size)
    usable_w = max(240, width_px * 0.72)
    usable_h = max(180, height_px * 0.72)
    padding = float(os.getenv("ROUTE_MAP_BBOX_PADDING", "1.45"))

    lngs = [float(c[0]) for c in coords]
    lats = [float(c[1]) for c in coords]
    lng_span = max(max(lngs) - min(lngs), 0.00008)
    mercs = [_lat_to_mercator_y(lat) for lat in lats]
    merc_span = max(max(mercs) - min(mercs), 0.00008)

    # Try from detailed to broad. Amap static map uses the same Web-Mercator
    # zoom intuition as most tiled maps, so this fit test is stable enough for
    # Shanghai city/near-suburb routes.
    for zoom in range(17, 4, -1):
        world_px = 256 * (2 ** zoom)
        x_px = (lng_span / 360.0) * world_px * padding
        y_px = (merc_span / (2 * 3.141592653589793)) * world_px * padding
        if x_px <= usable_w and y_px <= usable_h:
            return zoom
    return 5


def estimate_route_span_km(coords: list[tuple[float, float]]) -> float:
    """中文说明：估算距离、时间、价格或其他规划指标。"""
    if len(coords) < 2:
        return 0.0
    lngs = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    # 上海附近 1 纬度约111km，1经度约95km；用于选择静态图缩放说明。
    width_km = (max(lngs) - min(lngs)) * 95
    height_km = (max(lats) - min(lats)) * 111
    return max(width_km, height_km)


def estimate_static_map_size(coords: list[tuple[float, float]]) -> str:
    """中文说明：估算距离、时间、价格或其他规划指标。"""
    explicit = os.getenv("ROUTE_MAP_SIZE", "").strip()
    if explicit:
        return explicit
    span_km = estimate_route_span_km(coords)
    if span_km <= 3:
        return "900*560"
    if span_km <= 10:
        return "960*600"
    return "1024*640"


def short_static_map_label(index: int, place: str) -> str:
    """Keep marker text compact so distant routes still show every stop label."""
    cleaned = re.sub(r"\s+", "", str(place or "地点"))
    for token in ["上海市", "上海", "旗舰店", "官方直营店", "总店", "门店"]:
        cleaned = cleaned.replace(token, "")
    cleaned = re.sub(r"[（）()【】\[\]《》]", "", cleaned)
    return f"{index}-{cleaned[:8]}"


def static_map_marker_label(index: int, place: str) -> str:
    """高德静态图 marker 原生标签保持短编号，中文地名由前端 chips 展示。"""
    mode = os.getenv("ROUTE_MAP_MARKER_LABEL_MODE", "index").strip().lower()
    if mode == "name":
        return short_static_map_label(index, place)
    return str(index)


def parse_polyline_coords(polyline: str) -> list[tuple[float, float]]:
    """解析高德 direction 返回的 lng,lat;lng,lat 轨迹。"""
    coords = []
    for point in str(polyline or "").split(";"):
        parsed = parse_lnglat(point)
        if parsed:
            coords.append(parsed)
    return coords


def simplify_static_polyline_points(points: list[str], max_points: Optional[int] = None) -> list[str]:
    """限制静态地图 URL 里的轨迹点数量，避免 URL 过长导致图片加载失败。"""
    clean = []
    seen_previous = ""
    for point in points or []:
        text = str(point or "").strip()
        if not text or text == seen_previous or not parse_lnglat(text):
            continue
        clean.append(text)
        seen_previous = text
    if not clean:
        return []
    limit = max_points or _safe_int(os.getenv("ROUTE_MAP_MAX_POLYLINE_POINTS", "80"), 80)
    if len(clean) <= limit:
        return clean
    step = max(1, math.ceil(len(clean) / max(2, limit - 2)))
    sampled = [clean[0]] + clean[1:-1:step] + [clean[-1]]
    return sampled[:limit]


def route_polyline_points_from_segments(route_segments: list[dict]) -> list[str]:
    """从 route_segments 中提取真实高德驾车轨迹点。"""
    points = []
    for segment in route_segments or []:
        if not isinstance(segment, dict):
            continue
        points.extend([
            point.strip()
            for point in str(segment.get("polyline") or "").split(";")
            if point.strip()
        ])
    return simplify_static_polyline_points(points)


def route_endpoint_coords_from_segments(route_segments: list[dict]) -> dict[str, str]:
    """Reuse direction polyline endpoints as stop coordinates for lazy map rendering."""
    coords: dict[str, str] = {}
    for segment in route_segments or []:
        if not isinstance(segment, dict):
            continue
        parsed = parse_polyline_coords(segment.get("polyline", ""))
        if not parsed:
            continue
        start_name = str(segment.get("from") or "").strip()
        end_name = str(segment.get("to") or "").strip()
        if start_name:
            coords.setdefault(start_name, f"{parsed[0][0]:.6f},{parsed[0][1]:.6f}")
        if end_name:
            coords.setdefault(end_name, f"{parsed[-1][0]:.6f},{parsed[-1][1]:.6f}")
    return coords


def build_route_map_info(structured_plan: dict) -> dict:
    """Build a proxied Amap static map descriptor from final structured_plan.schedule."""
    if not get_amap_key():
        return {"available": False, "reason": "未配置高德 Web 服务 API Key，无法生成路线地图。"}

    schedule = (structured_plan or {}).get("schedule") or []
    route_segments = (structured_plan or {}).get("route_segments") or []
    segment_coords = route_endpoint_coords_from_segments(route_segments)
    markers = []
    coords = []
    for index, item in enumerate(schedule, start=1):
        if not isinstance(item, dict):
            continue
        place = str(item.get("place") or "").strip()
        coord = item.get("amap_location") or segment_coords.get(place) or amap_geocode(place)
        parsed = parse_lnglat(coord)
        if not place or not parsed:
            continue
        coords.append(parsed)
        markers.append({
            "index": index,
            "place": place,
            "label": short_static_map_label(index, place),
            "coord": f"{parsed[0]:.6f},{parsed[1]:.6f}",
            "time": item.get("time", ""),
        })

    if len(markers) < 2:
        return {"available": False, "reason": "结构化方案中可定位地点少于2个，暂不生成路线地图。", "markers": markers}

    route_polyline_points = route_polyline_points_from_segments(route_segments)
    path_points = route_polyline_points or [m["coord"] for m in markers]
    path_coords = [coord for point in path_points for coord in [parse_lnglat(point)] if coord]
    fit_coords = coords + path_coords

    # Use bounding-box center instead of average center, otherwise one far stop can be cropped.
    center_lng = (min(c[0] for c in fit_coords) + max(c[0] for c in fit_coords)) / 2
    center_lat = (min(c[1] for c in fit_coords) + max(c[1] for c in fit_coords)) / 2
    marker_color = os.getenv("ROUTE_MAP_MARKER_COLOR", "0xFF8A00").strip() or "0xFF8A00"
    marker_param = "|".join(
        f"large,{marker_color},{static_map_marker_label(m['index'], m['place'])}:{m['coord']}"
        for m in markers
    )
    route_line_weight = _safe_int(os.getenv("ROUTE_MAP_LINE_WEIGHT", "7"), 7)
    route_line_color = os.getenv("ROUTE_MAP_LINE_COLOR", "0x1677FF").strip() or "0x1677FF"
    path_param = f"{route_line_weight},{route_line_color},0.95,,:" + ";".join(path_points)
    size = estimate_static_map_size(fit_coords)
    zoom = estimate_static_map_zoom(fit_coords, size=size)
    span_km = estimate_route_span_km(fit_coords)
    base_params = {
        "key": get_amap_key(),
        "location": f"{center_lng:.6f},{center_lat:.6f}",
        "zoom": str(zoom),
        "size": size,
        "scale": "2",
    }
    params = {
        **base_params,
        "markers": marker_param,
        "paths": path_param,
    }
    amap_base_url = "https://restapi.amap.com/v3/staticmap?" + urllib.parse.urlencode(base_params, safe=",|:*;")
    amap_url = "https://restapi.amap.com/v3/staticmap?" + urllib.parse.urlencode(params, safe=",|:*;")
    return {
        "available": True,
        "provider": "amap_staticmap",
        "amap_url": amap_url,
        "amap_base_url": amap_base_url,
        "markers": markers,
        "marker_color": marker_color,
        "center": f"{center_lng:.6f},{center_lat:.6f}",
        "path_points": path_points,
        "route_line_weight": route_line_weight,
        "route_polyline_points": len(route_polyline_points),
        "zoom": zoom,
        "size": size,
        "span_km": round(span_km, 2),
        "fit_strategy": "bbox_conservative_zoom_with_route_polyline",
        "note": "高德静态地图优先使用 direction 返回的真实驾车轨迹绘制粗线；地图使用保守 bbox 缩放，优先保证路线和站点都在画面内。",
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
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    seen = set()
    result = []
    for item in items:
        key = str(item or "").strip()
        if key and key not in seen:
            result.append(key)
            seen.add(key)
    return result


def route_variant_seed(context: dict, salt: str = "") -> int:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
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
    runtime_row = _runtime_amap_place_records.get(raw_key)
    if runtime_row is not None:
        return runtime_row
    return None


def build_coupon_info(main_location: str, route_plan_text: str) -> dict:
    """从当前路线涉及的地点里读取团购券信息，返回前端可渲染的数据。"""
    names = unique_preserve_order([main_location] + extract_meal_candidates(route_plan_text))
    return build_coupon_info_for_places(names)


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
        sub_type = str(row.get("sub_type", "") or "").strip()
        has_coupon = bool(row.get("是否有团购", False))
        checked.append(f"{place_name}: {'有团购' if has_coupon else '无团购'}")
        if not has_coupon:
            continue

        display_detail = build_place_display_detail(place_name)
        display_name = display_detail.get("display_name") or place_name
        low = float(row.get("最低价格", 0) or 0)
        high = float(row.get("最高价格", 0) or low or 0)
        if sub_type in {"cafe", "bakery"}:
            discount = "满50减5"
        elif place_type == "restaurant" and (high >= 160 or low >= 80):
            discount = "满200减30" if stable_demo_ratio(place_name, "coupon:discount") < 0.5 else "满200减20"
        elif place_type == "restaurant":
            discount = "满100减20"
        elif high >= 180 or low >= 100:
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
            "note": "当前数据源显示有团购券，出行前需在美团/大众点评核验实时可用性。",
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
        summary = "团购券信息：当前路线涉及地点未发现可用团购券。"

    return {
        "items": coupons,
        "summary": summary,
        "checked_places": checked,
    }


def infer_coupon_theme(place_name: str, place_type: str) -> dict:
    """中文说明：根据上下文推断隐含状态或用户意图。"""
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
        summary = "团购券信息：最终方案涉及地点未发现可用团购券。"
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
    if any(term in raw_name for term in ["密室", "密室逃脱", "剧本杀", "推理馆", "游戏体验馆", "Xcape", "异时刻"]):
        return "non_food"

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
        # 高德 POI 的餐饮/非餐饮角色以高德 type 派生字段为准。
        amap_primary = str(row.get("amap_primary_type") or "").strip()
        amap_display = str(row.get("amap_display_tag") or "").strip()
        amap_raw = str(row.get("amap_raw_type") or "").strip()
        if amap_primary:
            if amap_primary != "restaurant":
                return "non_food"
            if _contains_any(f"{raw_name} {amap_display} {amap_raw}", ["咖啡", "奶茶", "茶饮", "甜品", "面包", "蛋糕", "下午茶", "冷饮"]):
                return "light_food"
            return "meal"
        place_type = str(row.get("地点类型", "")).strip()
        sub_type = str(row.get("sub_type", "") or "").strip()
        name = str(row.get("placeName", raw_name) or raw_name).lower()
        combined = f"{raw_name} {name}".lower()
        if place_type == "cafe" or sub_type in {"cafe", "cafe_general", "milk_tea", "bakery", "dessert"}:
            return "light_food"
        if place_type == "restaurant":
            if sub_type in {"cafe", "cafe_general", "milk_tea", "bakery", "dessert"} or any(term.lower() in combined for term in light_keywords):
                return "light_food"
            return "meal"
        # 地点表类型不是 restaurant/cafe，但复合展示名里明显含餐饮关键词时，仍按餐饮处理。
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




def is_food_place_role(role: str) -> bool:
    """中文说明：判断当前值或状态是否满足指定条件。"""
    return role in {"meal", "light_food"}





def route_place_category(place_name: str) -> str:
    """A stable category for adjacent duplicate-type guards.

    高德 POI 优先使用 amap_route_category / amap_display_tag，保证最终方案与高德标签一致。
    """
    name = str(place_name or "")
    row = _find_place(place_name)
    if row is not None:
        amap_category = str(row.get("amap_route_category") or "").strip()
        if amap_category:
            return amap_category
        amap_display = str(row.get("amap_display_tag") or "").strip()
        amap_raw = str(row.get("amap_raw_type") or "").strip()
        if amap_display or amap_raw:
            return amap_route_category_from_text(name, amap_raw, amap_display)
    sub_type = str(row.get("sub_type", "") or "").strip() if row is not None else ""
    place_type = str(row.get("地点类型", "") or "").strip() if row is not None else ""
    local_category = amap_route_category_from_text(name, "", "")
    if local_category and local_category not in {"scenic", "meal"}:
        return local_category
    role = place_role(place_name)
    if role in {"meal", "light_food"}:
        return role
    if sub_type and not sub_type.startswith("general"):
        return sub_type
    return place_type or "unknown"


def adjacent_same_type_conflict(candidate: str, previous: str) -> Optional[str]:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    if not candidate or not previous:
        return None
    c1, c2 = route_place_category(candidate), route_place_category(previous)
    if not c1 or c1 == "unknown" or c2 == "unknown" or c1 != c2:
        return None
    if c1 in {"meal", "light_food"}:
        return None
    return f"避免连续安排同一类型地点（{c1}）：{previous} → {candidate}"


def is_type_repeat_exempt_place(candidate: str, state: Optional[AgentState] = None,
                                intent: Optional[dict] = None) -> bool:
    if not candidate:
        return False
    if state is not None and is_locked_route_place(candidate, state, intent):
        return True
    required_places = []
    if state is not None:
        required_places.extend(state.get("_ordered_step_required_places") or [])
    required_places.extend((intent or {}).get("_ordered_step_required_places") or [])
    return any(same_route_place(candidate, place) or same_route_place(place, candidate) for place in required_places)


def repeated_subtype_conflict(candidate: str, existing_places: list[str],
                              state: Optional[AgentState] = None,
                              intent: Optional[dict] = None) -> Optional[str]:
    """整条路线级别禁止重复小类型：两个火锅、两个咖啡、两个公园、两个密室等都不再并存。"""
    category = normalized_route_category(route_place_category(candidate))
    if not category or category == "unknown":
        return None
    for old in existing_places or []:
        old_category = normalized_route_category(route_place_category(old))
        if old_category and old_category != "unknown" and old_category == category:
            return f"同一方案中不重复安排相同小类型地点（{category}）：{old} / {candidate}"
    return None


def normalized_route_category(category: str) -> str:
    """中文说明：清洗并归一化数据，避免脏值进入后续规划。"""
    aliases = {
        "shopping_mall": "shopping",
        "business_district": "shopping",
        "hotpot": "meal",
        "restaurant": "meal",
        "general_restaurant": "meal",
        "cafe": "light_food",
        "cafe_general": "light_food",
        "dessert": "light_food",
        "bakery": "light_food",
        "milk_tea": "light_food",
    }
    return aliases.get(str(category or "").strip(), str(category or "").strip())


def route_has_meal(places: list[str]) -> bool:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
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
        repeat_reason = repeated_subtype_conflict(place, result, state, intent)
        adjacent_reason = adjacent_same_type_conflict(place, result[-1]) if result else None
        type_reason = repeat_reason or adjacent_reason
        locked = state is not None and is_locked_route_place(place, state, intent)
        if (reason or type_reason) and (not locked or repeat_reason):
            if notes is not None:
                notes.append(f"已跳过“{place}”：{reason or type_reason}。")
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
        repeat_reason = repeated_subtype_conflict(place, result, state, intent)
        adjacent_reason = adjacent_same_type_conflict(place, result[-1]) if result else None
        type_reason = repeat_reason or adjacent_reason
        if reason and not locked:
            notes.append(f"最终餐饮节奏校验：已移除“{place}”，{reason}。")
            continue
        if type_reason and (not locked or repeat_reason):
            notes.append(f"最终类型节奏校验：已移除“{place}”，{type_reason}。")
            continue
        if reason and locked:
            notes.append(f"餐饮节奏提示：用户锁定地点“{place}”与前一餐饮点存在冲突，系统保留该锚点并优先调整其他地点。")
        if type_reason and locked:
            notes.append(f"类型节奏提示：用户锁定地点“{place}”与前一站类型相同，系统保留该锚点并优先调整其他地点。")
        result.append(place)

    result = preserve_route_anchors(result, state, intent)
    deduped = []
    for place in result:
        repeat_reason = repeated_subtype_conflict(place, deduped, state, intent)
        if repeat_reason:
            notes.append(f"最终小类型去重：已移除“{place}”，{repeat_reason}。")
            continue
        deduped.append(place)
    result = deduped
    return result, unique_preserve_order(notes)

def persist_amap_poi_if_needed(poi: dict, spec: dict) -> str:
    """Register a 高德 POI for this run; do not write the mock Excel table."""
    try:
        persist_amap_poi_with_mock(poi, spec)
    except Exception as e:
        print(f"⚠️ 高德 POI 运行时注册失败，仅作为本次结构化候选使用: {poi['name']} / {e}")
    return poi["name"]


def structured_plan_fast_mode() -> bool:
    """Keep structured planning inside the 30s target by avoiding sync map fan-out."""
    return os.getenv("STRUCTURED_PLAN_FAST_MODE", "1") == "1"


def sync_amap_complement_enabled() -> bool:
    """Allow expensive POI complement only when explicitly enabled."""
    return os.getenv("ENABLE_SYNC_AMAP_COMPLEMENT", "0") == "1" or not structured_plan_fast_mode()


def default_route_anchor_for_generation(state: AgentState, intent: dict, collected: dict,
                                        anchor_name: str, anchor_mode: str) -> tuple[str, str]:
    """Pick the single geography center used for schedule generation."""
    if anchor_name and anchor_mode in {"destination", "area", "nearby"}:
        return anchor_name, anchor_mode
    departure = str((collected or {}).get("fixed_departure") or (collected or {}).get("departure") or (intent or {}).get("departure") or "").strip()
    if departure and is_concrete_location_anchor(departure) and not is_placeholder_route_place(departure):
        return departure, "departure"
    has_ordered_specs = bool(ordered_search_specs_from_intent(intent))
    if has_ordered_specs:
        return os.getenv("DEFAULT_ROUTE_CENTER_ANCHOR", "人民广场"), "default_center"
    return os.getenv("DEFAULT_RANDOM_ROUTE_ANCHOR", "人民广场"), "random_default"


def _geo_cache_key(anchor: str, keyword: str, radius_m: int) -> str:
    return f"{normalize_place_text(anchor)}|{normalize_place_text(keyword)}|{int(radius_m)}"


def register_runtime_poi_candidate(poi: dict, spec: dict) -> str:
    """Make an Amap POI usable by later schedule steps without writing mock Excel."""
    name = str((poi or {}).get("name") or "").strip()
    if not name:
        return ""
    return persist_amap_poi_with_mock(poi, spec)


def prefetch_ordered_step_geofence_cache(state: AgentState, anchor: str, intent: dict,
                                         radius_m: Optional[int] = None) -> tuple[AgentState, list[str]]:
    """Prefetch same-anchor POI pools for ordered_steps; candidates stay in session memory."""
    if os.getenv("ENABLE_AMAP_POI_SEARCH", "1") != "1" or not get_amap_key() or not anchor:
        return state, []
    specs = ordered_search_specs_from_intent(intent)
    if not specs:
        return state, []
    radius_m = radius_m or _safe_int(os.getenv("ROUTE_CLUSTER_RADIUS_METERS", "10000"), 10000)
    limit = _safe_int(os.getenv("AMAP_GEOFENCE_PREFETCH_LIMIT", "2"), 2)
    limit = max(1, min(limit, 2))
    state = dict(state or {})
    cache = dict(state.get("geo_fence_poi_cache") or {})
    notes = []
    for spec in specs:
        keyword = str(spec.get("keyword") or "").strip()
        if not keyword:
            continue
        key = _geo_cache_key(anchor, keyword, radius_m)
        if key in cache:
            continue
        pois = amap_search_pois_near(anchor, keyword, radius=radius_m, limit=limit)
        pois = vary_candidates(pois, intent, f"geo_fence_prefetch:{anchor}:{keyword}")
        names = []
        for poi in pois:
            distance_m = _safe_int(poi.get("distance_m"), 0)
            if distance_m > radius_m:
                continue
            name = register_runtime_poi_candidate(poi, spec)
            if name:
                names.append(name)
        cache[key] = {
            "anchor": anchor,
            "keyword": keyword,
            "radius_m": radius_m,
            "places": unique_preserve_order(names),
        }
        notes.append(f"地理围栏预缓存：已围绕“{anchor}”为“{keyword}”缓存{len(names)}个候选。")
    state["geo_fence_poi_cache"] = cache
    return state, notes


def geofence_cached_places(state: AgentState, anchor: str, keyword: str, radius_m: int) -> list[str]:
    cache = (state or {}).get("geo_fence_poi_cache") or {}
    item = cache.get(_geo_cache_key(anchor, keyword, radius_m)) or {}
    return list(item.get("places") or [])


def geofence_cache_has_keyword(state: AgentState, anchor: str, keyword: str, radius_m: int) -> bool:
    cache = (state or {}).get("geo_fence_poi_cache") or {}
    return _geo_cache_key(anchor, keyword, radius_m) in cache


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

    ordered_count = len(sanitize_ordered_steps((intent or {}).get("ordered_steps") or []))
    if ordered_count > 1:
        min_stops = max(min_stops, ordered_count)
        max_stops = max(max_stops, min_stops)

    min_stops = max(2, min(min_stops, max_stops))
    max_stops = max(min_stops, max_stops)
    return min_stops, max_stops


def append_unique_route_places(base: list[str], additions: list[str], limit: Optional[int] = None) -> list[str]:
    """Append non-empty, non-duplicate places while preserving route order."""
    result = unique_preserve_order([p for p in base if p])
    for place in additions or []:
        place = str(place or "").strip()
        if not place:
            continue
        if any(same_route_place(place, old) or same_route_place(old, place) for old in result):
            continue
        result.append(place)
        if limit and len(result) >= limit:
            break
    return result


def row_seat_count(row) -> int:
    """读取余位数。明确填 0 时表示无余位，用于恢复异常处理测试。"""
    if row is None:
        return 0
    try:
        return _safe_int(row.get("余位信息", 0), 0)
    except Exception:
        return 0


def row_has_seat(row) -> bool:
    """统一余位判断。

    余位信息是库存测试的权威字段：明确填了 0 就是暂无余位；
    只有余位信息为空/无法解析时，才回退到“是否有余位”。
    这可以避免用户把迪士尼相关行余位改成 0 后，布尔列残留 True 仍显示有余位。
    """
    return row_availability_kind(row) == "available"


def row_availability_kind(row) -> str:
    """Return available / no_seat / unknown for route decisions."""
    if row is None:
        return "unknown"
    try:
        raw_status = str(row.get("availability_status", "") or "").strip().lower()
        if raw_status in {"unknown", "pending", "unverified", "待核验", "未知"}:
            return "unknown"
        if raw_status in {"available", "有余位", "可用"}:
            return "available"
        if raw_status in {"no_seat", "full", "sold_out", "已满", "无余位"}:
            return "no_seat"
        if bool(row.get("_amap_demo_status", False)):
            return "available" if row_seat_count(row) > 0 else "no_seat"
        source_note = str(row.get("source_note", "") or row.get("_source_file", ""))
        if "高德POI搜索候选" in source_note or "amap_poi" in source_note:
            return "unknown"
        explicit = bool(row.get("_seat_count_explicit", True))
        seats = row_seat_count(row)
        if explicit:
            return "available" if seats > 0 else "no_seat"
        if bool(row.get("是否有余位", False)) or seats > 0:
            return "available"
        return "unknown"
    except Exception:
        return "available" if row_seat_count(row) > 0 else "unknown"


def place_price_detail(place_name: str) -> dict:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    refresh_place_data_if_changed(False)
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





def estimate_schedule_budget(schedule: list[dict], num_people, requested_budget: str = "") -> dict:
    """Compute route-based budget after final schedule is known."""
    people = parse_people_count(num_people, None) or 1
    per_min = per_max = 0.0
    unknown_count = 0
    food_min = food_max = ticket_min = ticket_max = cafe_min = cafe_max = 0.0
    for item in schedule or []:
        if not isinstance(item, dict):
            continue
        low = float(item.get("price_min") or 0)
        high = float(item.get("price_max") or low or 0)
        if low <= 0 and high <= 0 and "待核验" in str(item.get("price_text") or ""):
            unknown_count += 1
            continue
        high = max(high, low)
        per_min += max(0.0, low); per_max += max(0.0, high)
        role = item.get("place_role") or place_role(str(item.get("place") or ""))
        if role == "meal":
            food_min += low; food_max += high
        elif role == "light_food":
            cafe_min += low; cafe_max += high
        else:
            ticket_min += low; ticket_max += high
    if per_min <= 0 and per_max <= 0 and unknown_count:
        text = "预算待核验"
    elif int(round(per_min)) == int(round(per_max)):
        text = f"预计人均约{int(round(per_max))}元，总计约{int(round(per_max * people))}元（{people}人）"
    else:
        text = f"预计人均约{int(round(per_min))}-{int(round(per_max))}元，总计约{int(round(per_min * people))}-{int(round(per_max * people))}元（{people}人）"
    if unknown_count and text != "预算待核验":
        text += f"；另有{unknown_count}个地点价格待核验"
    return {
        "per_person_min": int(round(per_min)), "per_person_max": int(round(per_max)),
        "total_min": int(round(per_min * people)), "total_max": int(round(per_max * people)),
        "num_people": people, "unknown_count": unknown_count, "text": text,
        "requested_budget": requested_budget or "",
        "breakdown": {
            "tickets_activities_min": int(round(ticket_min)), "tickets_activities_max": int(round(ticket_max)),
            "lunch_food_min": int(round(food_min)), "lunch_food_max": int(round(food_max)),
            "cafe_snacks_min": int(round(cafe_min)), "cafe_snacks_max": int(round(cafe_max)),
        }
    }


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
    raw_keys.extend((state or {}).get("exclude_anchor_keys_once") or [])
    raw_keys.extend(collected.get("exclude_anchor_keys_once") or [])
    raw_names = []
    raw_names.extend((state or {}).get("replaced_destinations") or [])
    raw_names.extend(collected.get("replaced_destinations") or [])
    raw_names.extend((state or {}).get("exclude_anchor_names_once") or [])
    raw_names.extend(collected.get("exclude_anchor_names_once") or [])
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
    """中文说明：判断当前值或状态是否满足指定条件。"""
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
    if not places or not get_amap_key() or os.getenv("ENABLE_SOFT_NEARBY_ORDER", "0") != "1":
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
        """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
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
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
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
        """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
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
            {"keyword": "咖啡", "place_type": "cafe", "sub_type": "cafe"},
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


def search_spec_for_activity_keyword(keyword: str) -> Optional[dict]:
    """只给高德检索关键词和二分类，不再在需求阶段指定小类型。"""
    key = str(keyword or "").strip()
    if not key:
        return None
    search_aliases = {
        "唱K": "KTV", "唱k": "KTV", "唱歌": "KTV", "量贩KTV": "KTV",
        "看电影": "电影院", "电影": "电影院", "影院": "电影院",
        "逛商场": "商场", "逛街": "商场", "购物": "商场",
        "喝咖啡": "咖啡", "下午茶": "下午茶", "吃饭": "餐厅",
    }
    search_keyword = search_aliases.get(key, key)
    place_type = simplified_requirement_place_type(search_keyword)
    return {"keyword": search_keyword, "place_type": place_type}


def find_requested_keyword_complement_places(anchor_name: str, existing_places: list[str], intent: dict,
                                             limit: int = 2, state: Optional[AgentState] = None) -> list[str]:
    """按用户原句顺序补齐活动点，如“密室→火锅→商场”。"""
    if os.getenv("ENABLE_AMAP_POI_SEARCH", "1") != "1" or not get_amap_key() or not anchor_name or limit <= 0:
        return []
    result = []
    existing_keys = {normalize_place_text(p) for p in existing_places or []}
    for spec in ordered_search_specs_from_intent(intent):
        if len(result) >= limit:
            break
        keyword = str(spec.get("keyword") or "").strip()
        if not keyword:
            continue
        radius_m = _safe_int(os.getenv("ROUTE_CLUSTER_RADIUS_METERS", "10000"), 10000)
        cached_names = geofence_cached_places(state or {}, anchor_name, keyword, radius_m)
        if not cached_names and not geofence_cache_has_keyword(state or {}, anchor_name, keyword, radius_m):
            pois = amap_search_pois_near(
                departure=anchor_name,
                keyword=keyword,
                radius=radius_m,
                limit=_safe_int(os.getenv("AMAP_ORDERED_STEP_POI_LIMIT", "5"), 5),
            )
            pois = vary_candidates(pois, intent, f"amap_ordered_keyword:{anchor_name}:{keyword}")
            cached_names = [register_runtime_poi_candidate(poi, spec) for poi in pois]
        for name in cached_names:
            name = str(name or "").strip()
            key = normalize_place_text(name)
            if not key or key in existing_keys:
                continue
            result.append(name)
            if state is not None:
                state["_ordered_step_required_places"] = unique_preserve_order(
                    list(state.get("_ordered_step_required_places") or []) + [name]
                )
            existing_keys.add(key)
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
    interest_names = [str(x) for x in ((intent or {}).get("interests") or [])]
    if any(x in interest_names for x in ["美食"]):
        wanted_roles = ["meal", "light_food", "non_food"]
    elif any(x in interest_names for x in ["咖啡"]):
        wanted_roles = ["light_food", "meal", "non_food"]
    elif any(x in interest_names for x in ["文化", "艺术", "展览", "自然", "散步", "购物", "拍照", "亲子"]):
        wanted_roles = ["non_food", "light_food", "meal"]

    def local_candidate_matches_anchor(row, name: str) -> bool:
        """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
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
            interest_terms = interest_keywords((intent or {}).get("interests") or [])
            fields_for_interest = f"{name} {row.get('search_tags','')} {row.get('地点类型','')} {row.get('sub_type','')}"
            interest_bonus = 45 if any(str(term).lower() in fields_for_interest.lower() for term in interest_terms) else 0
            price_penalty = float(row.get("最低价格", 0) or 0) / 40
        except (TypeError, ValueError):
            available_bonus = coupon_bonus = area_bonus = 0
            price_penalty = 0
        candidates.append(
            (wanted_roles.index(role), -(available_bonus + coupon_bonus + area_bonus + interest_bonus - price_penalty), name))

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
    for place in unique_preserve_order(raw_places):
        places.append(place)
    return places, notes


def place_is_indoor(place_name: str) -> bool:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
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
    if place_type in {"restaurant", "cafe", "shopping"}:
        return True
    return any(
        term in name for term in ["馆", "商场", "广场", "中心", "影院", "影城", "店", "室内", "剧院"]) or sub_type in {
        "museum", "art_exhibition", "cinema", "anime", "spa_relax", "shopping"}


def place_has_coupon(place_name: str) -> bool:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    row = find_place_exact_for_route(place_name)
    if row is None:
        row = _find_place(place_name)
    return bool(row is not None and row.get("是否有团购", False))


def place_min_price(place_name: str) -> float:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
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
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    candidates = _df.copy()
    if require_coupon:
        candidates = candidates[candidates["是否有团购"] == True]
    if role == "meal":
        candidates = candidates[candidates["地点类型"] == "restaurant"].copy()
        candidates = candidates[~candidates.get("sub_type", "").isin(
            ["cafe", "cafe_general", "milk_tea", "bakery", "dessert"])] if "sub_type" in candidates.columns else candidates
    elif role == "light_food":
        cafe_subtypes = ["cafe", "cafe_general", "milk_tea", "bakery", "dessert"]
        candidates = candidates[
            (candidates["地点类型"].isin(["cafe", "restaurant"])) & (candidates.get("sub_type", "").isin(cafe_subtypes) | (candidates["地点类型"] == "cafe"))].copy()
    elif role == "non_food":
        candidates = candidates[candidates["地点类型"].isin(["activity", "attraction", "leisure", "shopping", "sports"])].copy()
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
    """中文说明：在多个候选中选择最适合当前约束的结果。"""
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
            print(f"⚠️ 高德团购候选运行时注册失败: {poi.get('name')} / {e}")
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
        return {"keyword": "咖啡", "place_type": "cafe", "sub_type": "cafe"}
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
    """当库内候选无法满足固定距离硬限制时，从高德搜索近距离运行时 POI。"""
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
            print(f"⚠️ 高德近距离候选运行时注册失败: {name} / {e}")
    return None


def emergency_route_search_specs(intent: dict, force_indoor: bool = False, force_coupon: bool = False) -> list[dict]:
    """Specs for the final Amap-only rebuild when local/RAG candidates cannot satisfy 固定距离."""
    specs = []
    base = default_amap_search_spec(intent, " ".join([
        str((intent or {}).get("location") or ""),
        str((intent or {}).get("meal_pref") or ""),
        " ".join((intent or {}).get("place_keywords", []) or []),
    ]))
    if base:
        specs.append(base)
    if force_indoor:
        specs.extend([
            {"keyword": "商场", "place_type": "leisure", "sub_type": "shopping"},
            {"keyword": "咖啡", "place_type": "cafe", "sub_type": "cafe"},
            {"keyword": "餐厅", "place_type": "restaurant", "sub_type": "restaurant"},
        ])
    else:
        specs.extend([
            {"keyword": "咖啡", "place_type": "cafe", "sub_type": "cafe"},
            {"keyword": "餐厅", "place_type": "restaurant", "sub_type": "restaurant"},
            {"keyword": "公园", "place_type": "leisure", "sub_type": "park"},
            {"keyword": "商场", "place_type": "leisure", "sub_type": "shopping"},
            {"keyword": "景点", "place_type": "attraction", "sub_type": "scenic"},
        ])
    if force_coupon:
        for spec in specs:
            spec["force_coupon"] = True

    seen = set()
    result = []
    for spec in specs:
        key = (spec.get("keyword"), spec.get("place_type"), spec.get("sub_type"))
        if key in seen:
            continue
        seen.add(key)
        result.append(spec)
    return result


def find_emergency_amap_place_near(
        anchor: str,
        intent: dict,
        existing_places: set,
        max_m: int,
        force_indoor: bool = False,
        force_coupon: bool = False,
) -> Optional[str]:
    """Find one runtime Amap POI that is truly within max_m from anchor."""
    if os.getenv("ENABLE_AMAP_POI_SEARCH", "1") != "1" or not get_amap_key() or not anchor:
        return None
    existing_keys = {normalize_place_text(p) for p in existing_places if p}
    for spec in emergency_route_search_specs(intent, force_indoor=force_indoor, force_coupon=force_coupon):
        pois = amap_search_pois_near(
            departure=anchor,
            keyword=spec["keyword"],
            radius=max(1000, max_m),
            limit=_safe_int(os.getenv("AMAP_EMERGENCY_POI_LIMIT", "8"), 8),
        )
        pois = vary_candidates(pois, intent, f"amap_emergency:{anchor}:{spec['keyword']}")
        for poi in pois:
            name = str(poi.get("name") or "").strip()
            if not name or normalize_place_text(name) in existing_keys:
                continue
            if _safe_int(poi.get("distance_m"), 0) > max_m:
                continue
            if force_indoor and spec.get("sub_type") == "park":
                continue
            try:
                persisted = persist_amap_poi_with_mock(
                    poi,
                    spec,
                    force_coupon=force_coupon or bool(spec.get("force_coupon")),
                )
            except Exception as e:
                print(f"⚠️ 高德强制补点运行时注册失败: {name} / {e}")
                persisted = name
            verified_distance = route_distance_between_places(anchor, persisted)
            if verified_distance is not None and verified_distance <= max_m:
                return persisted
    return None


def rebuild_schedule_for_places(state: AgentState, structured_plan: dict, places: list[str], departure: str) -> dict:
    """Rebuild schedule after emergency Amap refill while preserving strict user start time."""
    collected = state.get("collected_info") or {}
    intent = state.get("intent") or {}
    places = unique_preserve_order([p for p in places if p])[:3]
    old_schedule = structured_plan.get("schedule") or []
    if not old_schedule:
        slots = build_schedule_slots(collected, intent, max(1, len(places)))
        old_schedule = [{"time": slot} for slot in slots]
    structured_plan["schedule"] = enrich_schedule_addresses(rebuild_schedule_with_places(old_schedule, places))
    structured_plan["places"] = unique_preserve_order([
        item.get("place")
        for item in structured_plan.get("schedule", [])
        if isinstance(item, dict) and item.get("place")
    ])
    structured_plan.setdefault("hard_constraints", {})["departure"] = departure
    return structured_plan


def schedule_place_names(structured_plan: dict) -> list[str]:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    return [
        item.get("place")
        for item in (structured_plan.get("schedule") or [])
        if isinstance(item, dict) and item.get("place")
    ]


def anchor_radius_violations(structured_plan: dict) -> list[dict]:
    """Backward-compatible no-op: anchor radius is not a hard constraint."""
    return []


def route_needs_emergency_rebuild(structured_plan: dict, route_segments: list[dict],
                                  failed_segments: list[str]) -> bool:
    """Backward-compatible no-op: distance is reference-only, so emergency hard rebuild is disabled."""
    return False


def force_rebuild_route_with_amap(state: AgentState, structured_plan: dict, departure: str) -> tuple[dict, list[str]]:
    """Backward-compatible no-op: no emergency hard-distance rebuild is applied."""
    return structured_plan, []


def apply_quick_adjustment_to_places(places: list[str], state: AgentState, intent: dict) -> tuple[list[str], list[str]]:
    """中文说明：把规则或用户修改应用到当前状态中。"""
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
        """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
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
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    if not get_amap_key() or not start or not end:
        return None
    start_coord = amap_geocode(start)
    end_coord = amap_geocode(end)
    if not start_coord or not end_coord:
        return None
    route = amap_driving_distance(start_coord, end_coord)
    return route["distance_m"] if route else None


def straight_line_distance_m(coord_a: str, coord_b: str) -> Optional[int]:
    """Fast radius check using coordinates; keeps route design independent from departure."""
    a = parse_lnglat(coord_a)
    b = parse_lnglat(coord_b)
    if not a or not b:
        return None
    lng1, lat1 = map(math.radians, a)
    lng2, lat2 = map(math.radians, b)
    d_lng = lng2 - lng1
    d_lat = lat2 - lat1
    h = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lng / 2) ** 2
    return int(round(6371000 * 2 * math.asin(min(1, math.sqrt(h)))))


def place_distance_from_anchor_m(anchor: str, place: str) -> Optional[int]:
    """Distance from planning anchor to candidate, not from user departure."""
    if not anchor or not place or same_route_place(anchor, place):
        return 0
    anchor_coord = amap_geocode(anchor)
    place_coord = amap_geocode(place)
    if not anchor_coord or not place_coord:
        return None
    return straight_line_distance_m(anchor_coord, place_coord)


def route_cluster_anchor_for_constraints(
        places: list[str],
        anchor_name: str,
        anchor_mode: str,
        intent: dict,
        collected: dict,
) -> tuple[str, str]:
    """Choose the route center for constraints. Destination/area wins; departure is fallback."""
    if anchor_name and anchor_mode in {"destination", "area", "nearby", "departure", "default_center", "random_default"}:
        return anchor_name, anchor_mode
    location = str((collected or {}).get("fixed_destination") or (collected or {}).get("location") or (intent or {}).get("location") or "").strip()
    if location and is_concrete_location_anchor(location) and not is_category_like_location(location):
        return location, "destination"
    return "", "none"


def replace_place_near_anchor(
        anchor: str,
        original_place: str,
        intent: dict,
        used_keys: set,
        radius_m: int,
        require_open_slot: str = "",
        state: Optional[AgentState] = None,
) -> Optional[tuple[str, str]]:
    """Find a nearby replacement around the route anchor, optionally requiring opening hours."""
    spec = amap_search_spec_for_replacement(original_place, intent)
    cached_names = geofence_cached_places(state or {}, anchor, spec["keyword"], radius_m)
    if cached_names:
        candidate_names = cached_names
    elif ordered_search_specs_from_intent(intent):
        return None
    else:
        pois = amap_search_pois_near(
            departure=anchor,
            keyword=spec["keyword"],
            radius=radius_m,
            limit=_safe_int(os.getenv("AMAP_POI_LIMIT", "5"), 5),
        )
        candidate_names = [register_runtime_poi_candidate(poi, spec) for poi in pois]
    for name in candidate_names:
        name = str(name or "").strip()
        if not name:
            continue
        key = normalize_place_text(name)
        if key in used_keys:
            continue
        if require_open_slot:
            open_status = check_place_open_for_slot(name, require_open_slot)
            if open_status.get("status") == "closed":
                continue
        distance_m = place_distance_from_anchor_m(anchor, name)
        return name, f"{name} 距离锚点约{format_distance(distance_m or 0)}"
    return None


def enforce_destination_radius_constraint(
        places: list[str],
        state: AgentState,
        intent: dict,
        anchor_name: str,
        anchor_mode: str,
        collected: dict,
) -> tuple[list[str], list[str], dict]:
    """Keep final route places around the selected geography anchor."""
    radius_m = _safe_int(os.getenv("ROUTE_CLUSTER_RADIUS_METERS", "10000"), 10000)
    if os.getenv("ENABLE_ROUTE_CLUSTER_RADIUS", "1") != "1" or radius_m <= 0:
        return places, [], {"enabled": False, "radius_m": radius_m}
    anchor, anchor_kind = route_cluster_anchor_for_constraints(places, anchor_name, anchor_mode, intent, collected)
    if not anchor:
        return places, ["10km范围约束未执行：没有可用目的地/出发地/默认中心锚点。"], {
            "enabled": False,
            "radius_m": radius_m,
            "anchor": "",
            "anchor_kind": anchor_kind,
        }

    kept = []
    if anchor_kind in {"destination", "area", "nearby"}:
        scope_note = "用户出发地只用于计算到第一站的转场。"
    elif anchor_kind == "departure":
        scope_note = "本次未识别到目的地，因此按出发地附近生成。"
    else:
        scope_note = "本次未识别到出发地/目的地，因此按默认中心附近生成。"
    notes = [
        f"路线范围约束：本次以“{anchor}”作为设计锚点，最终站点优先控制在{format_distance(radius_m)}内；{scope_note}"
    ]
    used_keys = set()
    checked = []
    replaced = []
    for place in places or []:
        place = str(place or "").strip()
        if not place:
            continue
        key = normalize_place_text(place)
        dist = place_distance_from_anchor_m(anchor, place)
        checked.append({"place": place, "distance_m": dist})
        if dist is None or dist <= radius_m or same_route_place(place, anchor):
            kept.append(place)
            used_keys.add(key)
            continue
        replacement = replace_place_near_anchor(anchor, place, intent, used_keys, radius_m, state=state)
        if replacement:
            new_place, why = replacement
            kept.append(new_place)
            used_keys.add(normalize_place_text(new_place))
            msg = f"10km范围替换：{place} 距离“{anchor}”约{format_distance(dist)}，超出范围；已替换为{why}。"
            notes.append(msg)
            replaced.append({"from": place, "to": new_place, "distance_m": dist, "reason": msg})
        else:
            notes.append(f"10km范围提醒：{place} 距离“{anchor}”约{format_distance(dist)}，暂未找到同类型近距离替换，保留但标记需人工核验。")
            kept.append(place)
            used_keys.add(key)
    return unique_preserve_order(kept), notes, {
        "enabled": True,
        "anchor": anchor,
        "anchor_kind": anchor_kind,
        "radius_m": radius_m,
        "checked": checked,
        "replacements": replaced,
    }


def enforce_adjacent_distance_limit(places: list[str], state: AgentState, intent: dict) -> tuple[list[str], list[str]]:
    """Backward-compatible no-op: route distance is reference-only and does not remove places."""
    return places, []


def estimate_queue_minutes(row) -> int:
    """中文说明：估算距离、时间、价格或其他规划指标。"""
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
    """保留 LLM 的地点类型，不再本地补语义类型或重排关键词。"""
    normalized = dict(intent or {})
    normalized["place_keywords"] = unique_preserve_order(normalized.get("place_keywords") or [])
    normalized["place_type_reason"] = "保留 LLM 输出的地点类型、关键词和 ordered_steps；本地不再补语义类型。"
    return normalized


def parse_people_count(value, default: Optional[int] = None) -> Optional[int]:
    """Parse Arabic or simple Chinese people counts from LLM output or user text."""
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
    ]
    for pattern in digit_patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))

    chinese_patterns = [
        r"([一二两俩三四五六七八九十])\s*(?:个)?人",
        r"([一二两俩三四五六七八九十])\s*(?:位|名)",
        r"一家([一二两俩三四五六七八九十])口",
    ]
    for pattern in chinese_patterns:
        match = re.search(pattern, text)
        if match:
            return CHINESE_NUMERAL_MAP.get(match.group(1), default)

    compact_text = re.sub(r"\s+", "", text)
    companion_patterns = [
        r"(我|我们)?(和|跟|带|陪)([一二两俩三四五六七八九十]|\d{1,2})(个)?(朋友|同学|同事)",
        r"(和|跟|带|陪)([一二两俩三四五六七八九十]|\d{1,2})(个)?(朋友|同学|同事)(一起)?",
    ]
    for pattern in companion_patterns:
        match = re.search(pattern, compact_text)
        if match:
            raw_count = next((g for g in match.groups() if g and (g.isdigit() or g in CHINESE_NUMERAL_MAP)), "")
            friend_count = int(raw_count) if raw_count.isdigit() else CHINESE_NUMERAL_MAP.get(raw_count, 0)
            if friend_count > 0:
                return friend_count + 1
    if any(word in compact_text for word in ["父母", "爸妈", "爸爸妈妈", "爹妈"]) and re.search(r"(?:我|我们)?(?:和|跟|带|陪)", compact_text):
        return 3
    if any(word in text for word in ["我和朋友", "和朋友", "跟朋友", "我俩", "我们俩"]):
        return 2
    if re.search(r"(?:和|跟|带|陪)(?:妈妈|爸爸|朋友|同学|同事|对象|男朋友|女朋友)", compact_text):
        return 2
    return default


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
    """把普通数值字段转成 int；人数语义必须单独调用 parse_people_count。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        match = re.search(r"-?\d+", str(value or ""))
        if match:
            try:
                return int(match.group(0))
            except (TypeError, ValueError):
                return default
        return default


def normalize_place_text(text: str) -> str:
    """中文说明：清洗并归一化数据，避免脏值进入后续规划。"""
    value = str(text or "").lower()
    for ch in " \t\r\n·•-_ /（）()【】[]《》<>，,。；;：:、|":
        value = value.replace(ch, "")
    return value


def place_aliases(place_name: str) -> list[str]:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
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
            if candidate and candidate not in GENERIC_LOCATION_TERMS and not is_invalid_anchor_text(candidate):
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
    """Clean route verbs accidentally captured with a location.

    注意：附近/周边/一带不再被简单当作噪音丢弃。真正的“附近”语义会由
    extract_spatial_anchor_from_user_text() 转成 area_anchor；这里只返回地点本体。
    """
    text = strip_anchor_edit_prefix(candidate)
    if not text:
        return ""
    text = re.split(r"[/／,，。！？；;\\n]", text, maxsplit=1)[0].strip()
    if re.fullmatch(r"(?:是|在|到)?(?:上午|早上|中午|下午|晚上|夜里|傍晚|晚点)(?:开始|出发|玩|吃|去)?", text):
        return ""
    text = re.sub(r"(换近一点|换便宜一点|换室内|换成室内|优先有团购|少走路|重新生成|再来一版)$", "", text).strip()
    # “密室逃脱1小时/密室逃脱（1小时”里的时长是玩法约束，不是地点名的一部分；
    # 先在地点候选层剥掉，减少拿“地点+时长”去查高德或写入 fixed_destination。
    text = re.sub(r"[（(【\[]?\s*(?:约|大概)?(?:\d{1,2}|半|一|二|两|俩|三|四|五|六|七|八|九|十)\s*(?:个)?小时\s*[）)】\]]?$", "", text).strip()
    text = normalize_area_like_anchor(text)
    for district in sorted(SHANGHAI_DISTRICT_TERMS, key=len, reverse=True):
        if text == district:
            return district
    # “临港大学城/松江大学城”等强后缀地点只能在指令壳清洗后保留。
    strong_suffixes = ("大学城", "动物园", "植物园", "迪士尼", "乐园", "博物馆", "美术馆", "展览馆", "古镇")
    if any(text.endswith(suffix) for suffix in strong_suffixes):
        return text
    text = re.sub(r"(轻松逛吃|逛吃|吃喝玩乐|吃喝|玩乐|玩|逛|走走|散步|吃饭|吃东西|看看|打卡)$", "", text).strip()
    return text
###
def extract_time_period_hint_from_user_text(user_input: str) -> Optional[str]:
    """中文说明：从文本或结构中抽取当前流程需要的关键信息。"""
    text = str(user_input or "")
    if any(w in text for w in ["上午", "早上", "早晨", "中午前"]):
        return "上午"
    if any(w in text for w in ["下午", "午后"]):
        return "下午"
    if any(w in text for w in ["晚上", "夜晚", "夜里", "傍晚", "晚点"]):
        return "晚上"
    return None


def extract_weather_hint_from_user_text(user_input: str) -> Optional[str]:
    """中文说明：从文本或结构中抽取当前流程需要的关键信息。"""
    text = str(user_input or "")
    rules = [
        ("下雨", ["下雨", "雨天", "避雨", "别淋雨"]),
        ("晴天", ["晴天", "好天气", "晒太阳"]),
        ("阴天", ["阴天", "多云"]),
        ("热", ["太热", "避暑", "别晒", "空调"]),
        ("冷", ["太冷", "保暖"]),
    ]
    for value, words in rules:
        if any(w in text for w in words):
            return value
    return None


# ==========================================
# 语义需求改写层（方案二）：LLM 先把灵活表达改成“正向召回文本 + 完整规划文本”
# ==========================================
_REQUIREMENT_REWRITE_CACHE: dict = {}


def _requirement_rewrite_cache_key(text: str, collected: Optional[dict] = None) -> str:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    collected = collected or {}
    relevant = {
        "text": str(text or ""),
        # 只把会影响“当前有效需求”理解的旧软偏好放进 key；不新增 session 持久字段。
        "meal_pref": collected.get("meal_pref"),
        "place_type": collected.get("place_type"),
        "place_keywords": collected.get("place_keywords") or [],
        "requirements": collected.get("requirements"),
    }
    return hashlib.sha256(json.dumps(relevant, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _all_preference_keyword_terms() -> list[str]:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    terms = []
    for words in PLACE_TYPE_KEYWORDS.values():
        terms.extend(words)
    terms.extend([
        "火锅", "海底捞", "咖啡", "咖啡馆", "下午茶", "小笼包", "生煎", "面馆",
        "韩料", "韩国料理", "江浙菜", "本帮菜", "日料", "西餐", "甜品", "奶茶",
        "烧烤", "烤肉", "商场", "购物", "电影", "影院", "公园", "展览", "看展",
    ])
    # 长词优先，避免“咖啡馆”被“咖啡”提前吞掉。
    return sorted(unique_preserve_order(terms), key=len, reverse=True)


def split_requirement_clauses(text: str) -> list[str]:
    """把用户口语按标点/连接词切成较短语义片段，用于识别否定片段。"""
    raw = str(text or "").strip()
    if not raw:
        return []
    # “而且/但是/不过/然后”常连接两个独立偏好，先切开。
    raw = re.sub(r"(而且|但是|但|不过|然后|另外|还有|并且)", r"，\1", raw)
    return [part.strip() for part in re.split(r"[，。！？；;\n]+", raw) if part.strip()]


def clause_has_negative_preference(clause: str) -> bool:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    compact = re.sub(r"\s+", "", str(clause or ""))
    if not compact:
        return False
    negative_patterns = [
        r"不想", r"不要", r"不需要", r"不用", r"别", r"避免", r"排除",
        r"不去", r"别去", r"不喝", r"不吃", r"不看", r"不逛", r"不安排",
        r"算了", r"跳过", r"去掉", r"换掉",
    ]
    return any(re.search(pattern, compact) for pattern in negative_patterns)


def extract_negated_requirement_terms(text: str) -> list[str]:
    """返回本轮被否定的业务词，只作为临时改写/剪枝依据，不写入持久状态。"""
    terms = []
    known_terms = _all_preference_keyword_terms()
    for clause in split_requirement_clauses(text):
        compact = re.sub(r"\s+", "", clause)
        if not clause_has_negative_preference(compact):
            continue
        for term in known_terms:
            if term and term in compact:
                terms.append(term)
        # 兜底：不要去/别去 + 具体地点名。
        for pattern in [
            r"(?:不要去|不去|别去|避开|排除|不要|不安排)([^，。！？；;、\n]{2,30})",
            r"([^，。！？；;、\n]{2,30})(?:不要了|不去了|换掉|算了)",
        ]:
            for match in re.finditer(pattern, clause):
                candidate = clean_location_hint_candidate(match.group(1))
                if (
                    candidate
                    and candidate not in GENERIC_LOCATION_TERMS
                    and candidate not in {"太累", "累", "太远", "太贵", "太晒", "走太多", "太赶", "安排", "需要避开", "避开", "需要", "安排需要避开"}
                    and (
                        any(term and term in candidate for term in known_terms)
                        or is_category_like_location(candidate)
                        or is_concrete_location_anchor(candidate)
                    )
                ):
                    terms.append(candidate)
    return unique_preserve_order(terms)


def positive_requirement_text_for_matching(text: str) -> str:
    """给关键词抽取/POI召回使用的文本：删除“不要/不想/避免”等否定片段。

    这个函数是规则兜底，不负责保存任何新字段。LLM 改写可用时会在上游产出
    positive_text；这里保证即便某处仍误传 raw/planning_text，也不会把“不要咖啡”
    里的“咖啡”当成正向关键词。
    """
    raw = str(text or "").strip()
    if not raw:
        return ""
    clauses = split_requirement_clauses(raw)
    if not clauses:
        return raw
    kept = []
    for clause in clauses:
        if clause_has_negative_preference(clause) and any(term in clause for term in _all_preference_keyword_terms()):
            continue
        kept.append(clause)
    cleaned = "，".join(kept).strip("，。；; ")
    return cleaned or ""


def _related_preference_terms(terms: list[str]) -> set[str]:
    """把“咖啡”扩展到同类召回词，用于清理旧的正向关键词；不作为持久字段。"""
    related = set()
    normalized_terms = [str(term or "").lower() for term in terms if str(term or "").strip()]
    for term in normalized_terms:
        related.add(term)
    for place_type, words in PLACE_TYPE_KEYWORDS.items():
        lowered_words = [str(w).lower() for w in words]
        if any(term in lowered_words or any(term in w or w in term for w in lowered_words) for term in normalized_terms):
            related.update(lowered_words)
            related.add(place_type)
    # 常见类簇补充：用户说“不想喝咖啡”时，旧的“咖啡馆/下午茶/甜品/奶茶/烘焙”不应继续作为正向召回。
    cafe_cluster = {"咖啡", "咖啡馆", "coffee", "cafe", "下午茶", "奶茶", "甜品", "面包", "烘焙", "蜜雪冰城"}
    if related & cafe_cluster:
        related.update(cafe_cluster)
    return related


def heuristic_rewrite_requirement_text(latest_text: str, collected: Optional[dict] = None) -> dict:
    """LLM 不可用/超时时的本地改写兜底。

    返回格式保持和 LLM 改写一致：positive_text / planning_text / negative_terms / fields。
    但 fields 默认空；本地兜底主要负责“不把否定词当正向需求”，不是完整结构化识别。
    """
    raw = strip_ui_preference_hints(str(latest_text or "")).strip()
    positive = positive_requirement_text_for_matching(raw)
    negated_terms = extract_negated_requirement_terms(raw)
    negative_note = ""
    if negated_terms:
        negative_note = "不要安排/需要避开：" + "、".join(negated_terms)
    parts = [part for part in [positive, negative_note] if part]
    planning_text = "；".join(parts) if parts else raw
    return {
        "raw_text": raw,
        # 如果本轮只有“不要/不想X”，positive_text 必须保持为空，不能回退到 raw，
        # 否则会再次把 X 抽成正向 place_keywords。
        "positive_text": positive if negated_terms else (positive or raw),
        "planning_text": planning_text or raw,
        "negative_terms": negated_terms,
        "fields": {},
        "source": "heuristic",
    }


def _coerce_list_field(value) -> list:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = re.split(r"[、,，;；/\s]+", str(value))
    return unique_preserve_order([str(item).strip() for item in raw_items if str(item).strip()])


def _is_empty_requirement_value(value) -> bool:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    return value in [None, "", [], {}]


def _set_if_missing(target: dict, key: str, value):
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    if not _is_empty_requirement_value(value) and _is_empty_requirement_value(target.get(key)):
        target[key] = value


def _merge_ordered_list(target: dict, key: str, values):
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    merged = unique_preserve_order(list(target.get(key) or []) + list(values or []))
    if merged:
        target[key] = merged


def canonical_sub_type_for_llm_subtype(keyword: str = "", llm_sub_type: str = "", category: str = "") -> str:
    """把大模型自由识别的小类型尽量映射成旧代码认识的 canonical subtype。"""
    text = f"{keyword} {llm_sub_type} {category}".lower()
    mapping = [
        (["ktv", "karaoke", "唱k", "唱歌", "量贩"], "ktv"),
        (["清吧", "酒吧", "bar", "鸡尾酒", "小酒馆", "喝一杯"], "bar"),
        (["livehouse", "live house", "音乐现场", "听歌", "听吧", "演出"], "livehouse"),
        (["美甲", "美睫", "nail"], "nail_beauty"),
        (["温泉", "泡汤", "汤泉", "洗浴", "汗蒸", "spa"], "spa_relax"),
        (["密室", "密室逃脱", "剧本杀", "推理馆"], "escape_room"),
        (["火锅", "涮锅"], "hotpot"),
        (["商场", "购物中心", "逛街", "购物"], "shopping_mall"),
        (["咖啡", "coffee", "下午茶"], "cafe"),
        (["公园", "散步", "走走"], "park"),
        (["电影院", "电影", "影院"], "cinema"),
        (["博物馆"], "museum"),
        (["美术馆", "展览", "艺术馆"], "art_exhibition"),
    ]
    for words, canonical in mapping:
        if any(word in text for word in words):
            return canonical
    return ""


def canonical_place_type_for_llm_category(keyword: str = "", llm_sub_type: str = "", category: str = "") -> str:
    """简化需求阶段类型：吃喝归 restaurant，其它统一归 attraction。

    具体小类型不再由 LLM 决定，最终以高德 POI type / amap_display_tag 为准。
    """
    if SIMPLIFY_USER_DEMAND_TYPES:
        return simplified_requirement_place_type(keyword, llm_sub_type, category)
    text = f"{keyword} {llm_sub_type} {category}".lower()
    if any(word in text for word in ["餐饮", "吃", "火锅", "烧烤", "日料", "西餐", "饭", "餐厅"]):
        return "restaurant"
    return "attraction"


def normalize_step_search_keyword(keyword: str = "", llm_sub_type: str = "", category: str = "") -> str:
    """生成高德/美团检索关键词，优先保留用户说法，其次使用 LLM 小类型。"""
    for value in [keyword, llm_sub_type, category]:
        cleaned = str(value or "").strip()
        if cleaned:
            return cleaned
    return ""


def sanitize_ordered_steps(value) -> list[dict]:
    """中文说明：清洗并归一化数据，避免脏值进入后续规划。"""
    steps = []
    raw_steps = value if isinstance(value, list) else []
    seen = set()
    for raw_step in raw_steps:
        if isinstance(raw_step, dict):
            keyword = str(raw_step.get("keyword") or raw_step.get("name") or raw_step.get("activity") or "").strip()
            place_type = str(raw_step.get("place_type") or "").strip()
            canonical_place_type = str(raw_step.get("canonical_place_type") or "").strip().lower()
            category = str(raw_step.get("category") or raw_step.get("activity_label") or "").strip()
            llm_sub_type = str(
                raw_step.get("llm_sub_type")
                or raw_step.get("llm_subtype")
                or raw_step.get("subtype")
                or raw_step.get("sub_type")
                or ""
            ).strip()
            canonical_sub_type = str(
                raw_step.get("canonical_sub_type")
                or raw_step.get("canonical_subtype")
                or ""
            ).strip().lower()
            search_keyword = str(raw_step.get("search_keyword") or "").strip()
            duration_min = _safe_int(raw_step.get("duration_min") or raw_step.get("duration_minutes"), 0)
            constraints = _coerce_list_field(raw_step.get("constraints") or raw_step.get("requirements") or [])
        else:
            keyword = str(raw_step or "").strip()
            place_type = ""
            canonical_place_type = ""
            category = ""
            llm_sub_type = ""
            canonical_sub_type = ""
            search_keyword = ""
            duration_min = 0
            constraints = []
        if not keyword:
            continue
        spec = search_spec_for_activity_keyword(keyword) or {}
        if not canonical_place_type:
            canonical_place_type = (
                place_type.lower() if place_type.lower() in CANONICAL_PLACE_TYPES else ""
            )
        if not canonical_place_type:
            canonical_place_type = spec.get("place_type", "") or canonical_place_type_for_llm_category(keyword, llm_sub_type, category)
        if not llm_sub_type:
            llm_sub_type = str(spec.get("sub_type") or "").strip()
        if not canonical_sub_type:
            canonical_sub_type = (
                spec.get("sub_type")
                or canonical_sub_type_for_llm_subtype(keyword, llm_sub_type, category)
            )
        if not search_keyword:
            search_keyword = normalize_step_search_keyword(keyword, llm_sub_type, category)
        if SIMPLIFY_USER_DEMAND_TYPES:
            # 需求阶段不再判断 KTV/公园/商场等细类型；只区分吃喝和非吃喝。
            canonical_place_type = simplified_requirement_place_type(keyword, llm_sub_type, category)
            place_type = canonical_place_type
            category = simplified_requirement_category(keyword, llm_sub_type, category)
            llm_sub_type = ""
            canonical_sub_type = ""
        normalized_key = (normalize_place_text(keyword), place_type, canonical_place_type, canonical_sub_type, normalize_place_text(llm_sub_type))
        if normalized_key in seen:
            continue
        seen.add(normalized_key)
        step = {"keyword": keyword}
        if category:
            step["category"] = category
        if place_type:
            step["place_type"] = place_type
        if canonical_place_type:
            step["canonical_place_type"] = canonical_place_type
        if llm_sub_type:
            step["llm_sub_type"] = llm_sub_type
        if canonical_sub_type:
            step["canonical_sub_type"] = canonical_sub_type
            step["sub_type"] = canonical_sub_type
        if search_keyword:
            step["search_keyword"] = search_keyword
        if duration_min > 0:
            step["duration_min"] = max(15, min(duration_min, 240))
        if constraints:
            step["constraints"] = constraints
        steps.append(step)
    return steps


def build_ordered_steps_from_keywords(keywords) -> list[dict]:
    """中文说明：构建当前流程需要的结构化对象或展示内容。"""
    steps = []
    for keyword in unique_preserve_order(keywords or []):
        spec = search_spec_for_activity_keyword(keyword) or {}
        step = {"keyword": str(keyword).strip()}
        if spec.get("place_type"):
            step["place_type"] = spec["place_type"]
        if spec.get("sub_type"):
            step["sub_type"] = spec["sub_type"]
        if step["keyword"]:
            steps.append(step)
    return sanitize_ordered_steps(steps)


def _ordered_step_compare_keys(steps) -> list[str]:
    """Return normalized keyword keys for comparing two ordered_steps lists."""
    result = []
    for step in sanitize_ordered_steps(steps or []):
        key = normalize_place_text(step.get("keyword", ""))
        if key:
            result.append(key)
    return result


def merge_ordered_steps_without_downgrade(richer_steps, candidate_steps) -> list[dict]:
    """Keep LLM ordered_steps when a later fast fallback is less complete.

    Typical case fixed here:
    - LLM rewrite: 密室逃脱 + 火锅 + 清吧, with duration_min on 密室逃脱
    - local fallback: 密室逃脱 + 火锅

    In that case the local fallback is only a subset, so it must not overwrite the
    richer LLM result. If both lists contain the same keywords, merge them while
    preserving detailed fields such as duration_min and constraints.
    """
    richer = sanitize_ordered_steps(richer_steps or [])
    candidate = sanitize_ordered_steps(candidate_steps or [])

    if not richer:
        return candidate
    if not candidate:
        return richer

    richer_keys = _ordered_step_compare_keys(richer)
    candidate_keys = _ordered_step_compare_keys(candidate)
    if not richer_keys:
        return candidate
    if not candidate_keys:
        return richer

    # Candidate has fewer steps and all of them are already in the richer LLM list:
    # treat it as a downgrade and keep the richer version.
    if len(candidate_keys) < len(richer_keys) and set(candidate_keys).issubset(set(richer_keys)):
        return richer

    # Same sequence of activities: merge fields and preserve richer details.
    if candidate_keys == richer_keys:
        richer_by_key = {
            normalize_place_text(step.get("keyword", "")): dict(step)
            for step in richer
            if step.get("keyword")
        }
        merged = []
        for step in candidate:
            key = normalize_place_text(step.get("keyword", ""))
            base = dict(richer_by_key.get(key) or {})
            for field, value in step.items():
                if value not in [None, "", [], {}]:
                    base[field] = value
            merged.append(base)
        return sanitize_ordered_steps(merged)

    # Candidate may contain a new activity not found by LLM; keep candidate.
    return candidate


def extract_ordered_steps_hint_from_user_text(text: str) -> list[dict]:
    raw = str(text or "")
    if not raw or not any(token in raw for token in ["先", "然后", "再", "之后", "接着", "结束后"]):
        return []
    keyword_aliases = [
        ("密室逃脱", ["密室逃脱", "密室", "剧本杀", "推理馆"]),
        ("火锅", ["火锅", "涮锅"]),
        ("清吧", ["清吧", "酒吧", "小酒馆", "喝一杯", "鸡尾酒", "bar", "Bar"]),
        ("商场", ["商场", "购物中心", "逛商场", "逛街", "购物"]),
    ]
    hits = []
    for keyword, aliases in keyword_aliases:
        positions = [raw.find(alias) for alias in aliases if raw.find(alias) >= 0]
        if positions:
            hits.append((min(positions), keyword))
    hits.sort(key=lambda item: item[0])
    return build_ordered_steps_from_keywords([keyword for _, keyword in hits])


def sync_ordered_steps_and_keywords(fields: dict) -> dict:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    data = dict(fields or {})
    ordered_steps = sanitize_ordered_steps(data.get("ordered_steps") or [])
    keywords_from_steps = [step["keyword"] for step in ordered_steps if step.get("keyword")]
    existing_keywords = _coerce_list_field(data.get("place_keywords") or [])
    if ordered_steps:
        data["ordered_steps"] = ordered_steps
        data["place_keywords"] = unique_preserve_order(keywords_from_steps + existing_keywords)
    elif existing_keywords:
        data["place_keywords"] = existing_keywords
    return data


def ordered_search_specs_from_intent(intent: dict) -> list[dict]:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    specs = []
    for step in sanitize_ordered_steps((intent or {}).get("ordered_steps") or []):
        keyword = step.get("keyword")
        spec = {
            "keyword": step.get("search_keyword") or keyword,
            "place_type": step.get("place_type"),
            "sub_type": step.get("canonical_sub_type") or step.get("sub_type"),
            "llm_sub_type": step.get("llm_sub_type"),
            "category": step.get("category"),
        }
        specs.append({k: v for k, v in spec.items() if v})
    return specs


def sanitize_structured_requirement_fields(fields: Optional[dict], positive_text: str = "", planning_text: str = "") -> dict:
    """校验 LLM 输出的结构化字段。

    这里是 LLM 与本地规则之间的安全边界：
    - 只允许现有字段名，避免模型发明字段。
    - place_type/interests 保留 LLM 原始输出；本地只额外给 canonical_place_type 做兼容映射。
    - sub_type 允许 LLM 自由表达；本地只尽量映射 canonical_sub_type，不再删除自由小类型。
    - location/fixed_destination 必须是真实可到访锚点；“散心/放松/咖啡”这类活动或品类会被丢掉。
    """
    raw = sync_ordered_steps_and_keywords(dict(fields or {}))
    cleaned = {}
    allowed_keys = {
        "departure", "location", "fixed_destination", "num_people", "date", "time_period", "start_time",
        "duration_hours", "weather", "budget", "group_type", "companion_notes", "transport_mode", "meal_pref",
        "requirements", "place_type", "canonical_place_type", "category", "sub_type", "llm_sub_type", "canonical_sub_type", "search_keyword",
        "area_anchor", "area_anchor_mode", "exclude_anchor_from_schedule",
        "place_keywords", "excluded_places",
        "avoid_places", "interests", "ordered_steps",
    }
    for key in allowed_keys:
        value = raw.get(key)
        if value in [None, "", [], {}]:
            continue
        cleaned[key] = value

    has_ordered_steps = bool(cleaned.get("ordered_steps"))
    if cleaned.get("category"):
        cleaned["category"] = str(cleaned.get("category") or "").strip()

    raw_place_type = str(cleaned.get("place_type") or "").strip()
    if raw_place_type:
        cleaned["place_type"] = raw_place_type
    canonical_place_type = str(cleaned.get("canonical_place_type") or "").strip().lower()
    if canonical_place_type:
        cleaned["canonical_place_type"] = canonical_place_type

    raw_sub_type = str(cleaned.get("llm_sub_type") or cleaned.get("sub_type") or "").strip()
    if raw_sub_type:
        cleaned["llm_sub_type"] = raw_sub_type
    canonical_sub_type = str(cleaned.get("canonical_sub_type") or "").strip().lower()
    if canonical_sub_type:
        cleaned["canonical_sub_type"] = canonical_sub_type
        cleaned["sub_type"] = canonical_sub_type
    else:
        cleaned.pop("sub_type", None)
        cleaned.pop("canonical_sub_type", None)
    if cleaned.get("search_keyword"):
        cleaned["search_keyword"] = str(cleaned.get("search_keyword") or "").strip()

    if cleaned.get("area_anchor"):
        area_anchor = clean_spatial_anchor_candidate(str(cleaned.get("area_anchor") or ""))
        if area_anchor and not is_invalid_anchor_text(area_anchor):
            cleaned["area_anchor"] = area_anchor
            cleaned["area_anchor_mode"] = str(cleaned.get("area_anchor_mode") or "area").strip() or "area"
            cleaned["exclude_anchor_from_schedule"] = bool(cleaned.get("exclude_anchor_from_schedule", True))
        else:
            cleaned.pop("area_anchor", None)
            cleaned.pop("area_anchor_mode", None)
            cleaned.pop("exclude_anchor_from_schedule", None)

    if cleaned.get("num_people") is not None:
        people = parse_people_count(cleaned.get("num_people"), None)
        if people is not None:
            cleaned["num_people"] = people
        else:
            cleaned.pop("num_people", None)

    if cleaned.get("duration_hours") is not None:
        duration = _safe_int(cleaned.get("duration_hours"), 0)
        if duration > 0:
            cleaned["duration_hours"] = duration
        else:
            cleaned.pop("duration_hours", None)

    if cleaned.get("time_period") not in [None, "上午", "下午", "晚上"]:
        cleaned.pop("time_period", None)

    if cleaned.get("group_type") not in [None, "情侣", "家庭", "朋友", "独行"]:
        cleaned.pop("group_type", None)

    if cleaned.get("companion_notes"):
        cleaned["companion_notes"] = str(cleaned.get("companion_notes") or "").strip()

    if cleaned.get("interests"):
        cleaned["interests"] = _coerce_list_field(cleaned.get("interests"))

    for key in ["place_keywords", "excluded_places", "avoid_places"]:
        if cleaned.get(key):
            cleaned[key] = _coerce_list_field(cleaned.get(key))

    if cleaned.get("ordered_steps"):
        cleaned["ordered_steps"] = sanitize_ordered_steps(cleaned.get("ordered_steps"))
        if not cleaned["ordered_steps"]:
            cleaned.pop("ordered_steps", None)

    if cleaned.get("requirements"):
        if isinstance(cleaned["requirements"], list):
            cleaned["requirements"] = "、".join(_coerce_list_field(cleaned["requirements"]))
        else:
            cleaned["requirements"] = str(cleaned["requirements"]).strip()

    for key in ["location", "fixed_destination"]:
        value = str(cleaned.get(key) or "").strip()
        if not value:
            cleaned.pop(key, None)
            continue
        value = clean_location_hint_candidate(value)
        if value and is_concrete_location_anchor(value):
            cleaned[key] = value
        else:
            cleaned.pop(key, None)
            if key == "fixed_destination":
                cleaned.pop("location", None)

    if cleaned.get("fixed_destination") and not cleaned.get("location"):
        cleaned["location"] = cleaned["fixed_destination"]
    if cleaned.get("location") and not cleaned.get("fixed_destination") and place_matches_text(cleaned["location"], f"{positive_text} {planning_text}"):
        cleaned["fixed_destination"] = cleaned["location"]

    negative_terms = extract_negated_requirement_terms(planning_text)
    if negative_terms:
        rewrite_info = {"negative_terms": negative_terms, "positive_text": positive_text}
        for key in ["meal_pref", "place_type", "sub_type", "canonical_sub_type", "llm_sub_type", "search_keyword", "location", "fixed_destination"]:
            if _value_matches_rewrite_negative(str(cleaned.get(key) or ""), rewrite_info, positive_text):
                cleaned.pop(key, None)
        if cleaned.get("place_keywords"):
            cleaned["place_keywords"] = [
                kw for kw in cleaned["place_keywords"]
                if not _value_matches_rewrite_negative(str(kw), rewrite_info, positive_text)
            ]
            if not cleaned["place_keywords"]:
                cleaned.pop("place_keywords", None)
        if cleaned.get("ordered_steps"):
            cleaned["ordered_steps"] = [
                step for step in cleaned["ordered_steps"]
                if not _value_matches_rewrite_negative(str(step.get("keyword") or ""), rewrite_info, positive_text)
            ]
            if not cleaned["ordered_steps"]:
                cleaned.pop("ordered_steps", None)

    return sync_ordered_steps_and_keywords(cleaned)


def rewrite_user_requirement_for_pipeline(latest_text: str, collected: Optional[dict] = None, allow_llm: bool = True) -> dict:
    """每轮意图识别前的语义改写入口。

    单人模式和多人模式共用这个函数：
    1. LLM 把用户口语改写成 positive_text / planning_text / negative_terms / fields。
    2. sanitize_structured_requirement_fields 清洗结构化字段。
    3. LLM 不可用时回退到 heuristic_rewrite_requirement_text。

    后续规则抽取优先看 positive_text，避免把“不要咖啡”里的咖啡当成正向偏好。
    """
    """把自然语言需求改写成两份临时文本。

    - positive_text：只保留正向要安排/要召回的内容，供规则抽取、关键词匹配、高德POI召回使用。
    - planning_text：保留完整有效需求，包含“不要/避免/不想”等限制，供最终规划提示词使用。

    注意：本函数不往 collected_info/session 写入新字段；返回值只在当前调用链里使用。
    """
    raw = strip_ui_preference_hints(str(latest_text or "")).strip()
    if not raw:
        return {"raw_text": "", "positive_text": "", "planning_text": "", "negative_terms": [], "source": "empty"}

    cache_key = _requirement_rewrite_cache_key(raw, collected)
    if cache_key in _REQUIREMENT_REWRITE_CACHE:
        return dict(_REQUIREMENT_REWRITE_CACHE[cache_key])

    fallback = heuristic_rewrite_requirement_text(raw, collected)
    use_llm = bool(allow_llm) and os.getenv("ENABLE_LLM_REQUIREMENT_REWRITE", "1") == "1" and llm is not None
    if not use_llm:
        _REQUIREMENT_REWRITE_CACHE[cache_key] = fallback
        return dict(fallback)

    try:
        collected = collected or {}
        prompt = ChatPromptTemplate.from_template("""
你是 LocalMate 的“需求改写器”，只负责把用户最新一句话改写成后续程序更容易处理的文本，不生成路线。

请严格输出 JSON，不要 Markdown，不要解释。JSON 字段：
- positive_text: 只保留用户想要安排、可以作为搜索关键词/偏好的正向内容。凡是“不想/不要/别/避免/不去/不喝/不吃/算了/换掉”等否定目标，不得出现在 positive_text。
- planning_text: 保留本轮完整有效需求，包含时间、人数、预算、偏好，以及明确的禁止/避开事项。要求简洁、无歧义。
- negative_terms: 数组，列出本轮被用户否定的地点、品类或偏好词；没有则 []。
- fields: 结构化字段对象；没有把握的字段填 null 或不写，不要猜。

fields 可用字段：
- departure: 出发地；只有用户明确说“从/出发地/起点”时填写。
- location: 具体目的地、商圈或可到访地标；“散心/放松/走走/吃饭/咖啡”这类活动或品类不要填 location。
- fixed_destination: 与 location 相同，只在用户明确点名具体地点或商圈时填写。
- area_anchor: 用户表达“某地附近/周边/一带/围绕某区域”时填写核心区域锚点，例如“陆家嘴”“人民广场”“中山公园”。此时通常不要把 area_anchor 直接当成 schedule 站点。
- area_anchor_mode: 只能表达语义，不做硬枚举；常见值 nearby/area/business_district。
- exclude_anchor_from_schedule: 如果 area_anchor 只是附近检索中心而不是用户点名要去的一站，填 true。
- num_people: 出行总人数，整数；只能来自明确人数表达或明确同行者关系推断。如未提及填 null。
- date, time_period, start_time, budget, group_type, companion_notes, transport_mode, meal_pref, requirements。
- companion_notes: 同行者细节备注，例如“5岁女儿”“父母同行”“两个朋友”；年龄、身份、关系都写在这里，用于最终方案展示，不用于本地重新计算人数。
- duration_hours: 整条路线总时长小时数。只有用户明确说“整个行程/总共/安排半天/路线4小时”时填写；如果用户只说某个活动“玩2小时/吃1小时/逛1小时”，不要填 duration_hours，应写入 ordered_steps[].duration_min。
- budget: 大致预算，只能来自金额表达，例如“人均100元”“总预算300元”；“人均200以内/人均200元”必须写 200，不要乘以人数；预算数字不能进入 num_people。
- place_type / canonical_place_type: 只做二分类：用户需求中涉及吃喝（吃饭、火锅、烧烤、咖啡、奶茶、酒吧等）时填 restaurant；除此之外一律填 attraction。不要输出 activity/cafe/shopping/leisure 等细类型。
- category: 只允许“餐饮”或“景点”。吃喝为“餐饮”，其它都为“景点”。
- llm_sub_type / canonical_sub_type / sub_type: 不要判断细类型，保持 null 或不写。具体小类型必须由高德 API 返回的 POI type 决定。
- search_keyword: 用于高德搜索的关键词，必须优先保留用户真实说法，例如“唱K”“清吧”“美甲”“温泉”“火锅”。
- place_keywords: 数组，只放正向召回关键词。
- ordered_steps: 数组，按用户原句顺序记录连续活动。每项格式为 {{"keyword":"唱K","category":"景点","place_type":"attraction","canonical_place_type":"attraction","search_keyword":"KTV","duration_min":60,"constraints":[]}}；吃喝项才填 restaurant/餐饮。
- excluded_places/avoid_places: 数组，只放用户明确不要去/避开的地点或品类。
- interests: 数组，保留大模型理解到的用户兴趣词，不受本地枚举限制。

改写原则：
1. “火锅是晚上，玩是下午，而且我们不想喝咖啡” → positive_text 应类似“火锅是晚上，玩是下午”；planning_text 应包含“不要安排咖啡/咖啡馆/下午茶类”。
2. 如果用户说“不想喝咖啡但奶茶可以”，positive_text 必须保留“奶茶可以”，negative_terms 只写“咖啡”。
3. 如果用户只是说“不要太累/少走路”，这是正向软约束，positive_text 也要保留“少走路/轻松”。
4. “散心/放松/遛弯/休闲/走走”是活动意图或氛围偏好，不是具体目的地；不要把它改写成“目的地=散心”。应写成“散心活动/轻松散步/休闲放松”等偏好。
5. 不要编造用户没说的新地点。
6. 人数必须按中文语义计算：“和两个朋友/跟2个朋友”表示用户本人 + 2 个朋友，即 3 人；“和朋友两个人”才是 2 人。同行者年龄、身份和关系写入 companion_notes，不要把年龄数字当人数。
7. 用户连续说多个活动时，ordered_steps 必须按原句顺序保留，但只做二分类。例如“先密室逃脱1小时，然后火锅不要太辣，再逛商场”应输出：
{{"ordered_steps":[{{"keyword":"密室逃脱","category":"景点","place_type":"attraction","canonical_place_type":"attraction","search_keyword":"密室逃脱","duration_min":60}},{{"keyword":"火锅","category":"餐饮","place_type":"restaurant","canonical_place_type":"restaurant","search_keyword":"火锅","constraints":["不要太辣"]}},{{"keyword":"商场","category":"景点","place_type":"attraction","canonical_place_type":"attraction","search_keyword":"商场"}}],"place_keywords":["密室逃脱","火锅","商场"]}}

数字角色约束：
1. 同一句话里出现多个数字时，必须先判断每个数字的角色。
2. “上午10点/14:00/晚上7点”是 start_time，不是人数。
3. “5岁女儿/6岁孩子”是 companion_notes，不是人数。
4. “玩2小时/逛1小时/吃饭1小时”是 ordered_steps[].duration_min，不是人数，也不是整条路线总时长。
5. “人均100元/预算200元”是 budget，不是人数；“人均200以内，2个人”仍然 budget=200，不要改成总预算400或800。
6. num_people 只能来自明确人数表达或同行关系推断；无法确定就填 null。
7. 输出前自检：num_people 不得等于年龄、时间点、预算金额、活动时长或 RAG/历史案例人数。

示例：
用户“周六早晨和父母去散心，口味清淡，避免辛辣”
fields 应包含 {{"date":"周六","time_period":"上午","group_type":"家庭","place_type":"attraction","canonical_place_type":"attraction","category":"景点","interests":["散步"],"requirements":["适合父母","口味清淡","避免辛辣"]}}，location/fixed_destination 必须为空。

用户“周六上午想去人民广场散步”
fields 应包含 {{"date":"周六","time_period":"上午","location":"人民广场","fixed_destination":"人民广场","place_type":"attraction","canonical_place_type":"attraction","category":"景点","interests":["散步"]}}。

用户“周日下午2点和两个朋友去密室逃脱1小时，然后想吃火锅，不要太辣，之后逛逛商场”
fields 应包含 {{"date":"周日","time_period":"下午","start_time":"14:00","num_people":3,"group_type":"朋友","meal_pref":"火锅","requirements":["不要太辣"],"ordered_steps":[{{"keyword":"密室逃脱","category":"景点","place_type":"attraction","canonical_place_type":"attraction","search_keyword":"密室逃脱","duration_min":60}},{{"keyword":"火锅","category":"餐饮","place_type":"restaurant","canonical_place_type":"restaurant","search_keyword":"火锅","constraints":["不要太辣"]}},{{"keyword":"商场","category":"景点","place_type":"attraction","canonical_place_type":"attraction","search_keyword":"商场"}}],"place_keywords":["密室逃脱","火锅","商场"]}}。

上一轮已收集软偏好（仅供理解上下文，不能直接照抄旧偏好）：
餐饮偏好={meal_pref}
地点类型={place_type}
地点关键词={place_keywords}
其他要求={requirements}

用户最新输入：{input}
""")
        result = (prompt | llm | StrOutputParser()).invoke({
            "input": raw,
            "meal_pref": collected.get("meal_pref") or "无",
            "place_type": collected.get("place_type") or "无",
            "place_keywords": "、".join(collected.get("place_keywords") or []) or "无",
            "requirements": collected.get("requirements") or "无",
        })
        cleaned = re.sub(r"```json|```", "", str(result or "")).strip()
        data = json.loads(cleaned)
        positive = str(data.get("positive_text") or "").strip()
        planning = str(data.get("planning_text") or "").strip()
        negative_terms = data.get("negative_terms") or []
        if not isinstance(negative_terms, list):
            negative_terms = [str(negative_terms)]
        # 双保险：LLM 若不小心把否定片段留在 positive_text，再用本地规则清掉。
        cleaned_positive = positive_requirement_text_for_matching(positive)
        negative_terms = unique_preserve_order(list(negative_terms) + fallback.get("negative_terms", []))
        positive = cleaned_positive if negative_terms else (cleaned_positive or fallback["positive_text"])
        rewritten = {
            "raw_text": raw,
            "positive_text": positive,
            "planning_text": planning or fallback["planning_text"],
            "negative_terms": negative_terms,
            "fields": sanitize_structured_requirement_fields(
                data.get("fields") or {},
                positive,
                planning or fallback["planning_text"],
            ),
            "source": "llm",
        }
    except Exception as exc:
        print(f"⚠️ LLM需求改写失败，使用本地兜底: {exc}")
        rewritten = fallback

    # 限制缓存大小，避免多人长时间运行时无限增长。
    max_items = int(os.getenv("REQUIREMENT_REWRITE_CACHE_MAX_ITEMS", "128"))
    if len(_REQUIREMENT_REWRITE_CACHE) >= max_items:
        first_key = next(iter(_REQUIREMENT_REWRITE_CACHE))
        _REQUIREMENT_REWRITE_CACHE.pop(first_key, None)
    _REQUIREMENT_REWRITE_CACHE[cache_key] = rewritten
    return dict(rewritten)


def effective_positive_text_from_rewrite(rewrite_info: Optional[dict], fallback_text: str = "") -> str:
    """Return the text that may be used for keyword extraction/POI recall.

    Crucial detail: when the latest user turn is purely negative (e.g. "不想喝咖啡"),
    positive_text is intentionally empty.  Callers must not use ``or raw_text`` because that
    reintroduces the negated keyword as a positive search term.
    """
    info = dict(rewrite_info or {})
    positive = str(info.get("positive_text") or "").strip()
    if positive:
        return positive_requirement_text_for_matching(positive)
    if info.get("negative_terms"):
        return ""
    return positive_requirement_text_for_matching(fallback_text)


def _value_matches_rewrite_negative(value: str, rewrite_info: Optional[dict], positive_text: str = "") -> bool:
    """Whether a transient value should be treated as a negated preference.

    This does not create persistent exclusion fields. It is a local guard used to prevent
    planning_text such as "不要安排：咖啡" from becoming location/place_keywords/AMAP keyword.
    If the same value appears in positive_text, we keep it, which supports expressions like
    "不想喝咖啡但奶茶可以".
    """
    raw = str(value or "").strip()
    if not raw:
        return False
    pos = str(positive_text or "").lower()
    raw_lower = raw.lower()
    if pos and raw_lower and raw_lower in pos:
        return False
    terms = list((rewrite_info or {}).get("negative_terms") or [])
    if not terms:
        return False
    related = _related_preference_terms(terms)
    if not related:
        return False
    return any(term and (term in raw_lower or raw_lower in term) for term in related)


def _keyword_is_negated_for_recall(keyword: str, place_type: str, negative_terms: list[str], positive_text: str = "") -> bool:
    """Filter a candidate search spec if it only came from a negated clause."""
    keyword_l = str(keyword or "").lower()
    place_type_l = str(place_type or "").lower()
    pos = str(positive_text or "").lower()
    if keyword_l and keyword_l in pos:
        return False
    related = _related_preference_terms(negative_terms or [])
    if not related:
        return False
    return bool(
        (keyword_l and any(term and (term in keyword_l or keyword_l in term) for term in related))
        or (place_type_l and place_type_l in related)
    )


def prune_collected_soft_preferences_by_rewrite(collected: dict, rewrite_info: dict, positive_hints: Optional[dict] = None) -> dict:
    """根据本轮改写临时清理旧的正向软偏好，不新增任何持久字段。

    场景：上一轮 place_keywords=['咖啡']，本轮用户说“不想喝咖啡”。如果没有这一步，
    即使 positive_text 不再含咖啡，旧的 collected_info 仍会把咖啡带入后续候选召回。
    """
    copied = dict(collected or {})
    negative_terms = list((rewrite_info or {}).get("negative_terms") or [])
    if not negative_terms:
        return copied
    related = _related_preference_terms(negative_terms)
    if not related:
        return copied

    positive_hints = dict(positive_hints or {})
    positive_text = str((rewrite_info or {}).get("positive_text") or "")

    def _matches_negative(value: str) -> bool:
        """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
        raw = str(value or "").strip()
        key = raw.lower()
        # If the current positive rewrite explicitly keeps this value, do not prune it.
        if positive_text and key and key in positive_text.lower():
            return False
        return bool(raw) and any(term and (term in key or key in term) for term in related)

    if copied.get("meal_pref") and _matches_negative(str(copied.get("meal_pref"))):
        copied.pop("meal_pref", None)

    # Category-like locations can be polluted by the negated clause, e.g. location=咖啡
    # from "想去陆家嘴；不要咖啡". Clear these transient anchors without adding
    # persistent exclusion fields.
    for loc_key in ["location", "fixed_destination", "active_destination_anchor", "center_anchor"]:
        if copied.get(loc_key) and _matches_negative(str(copied.get(loc_key))):
            copied.pop(loc_key, None)
            if loc_key in {"location", "fixed_destination", "active_destination_anchor"}:
                copied["_location_explicit"] = False

    old_keywords = [kw for kw in (copied.get("place_keywords") or []) if not _matches_negative(str(kw))]
    if old_keywords:
        copied["place_keywords"] = old_keywords
    else:
        copied.pop("place_keywords", None)

    place_type = str(copied.get("place_type") or "").lower()
    if place_type and _matches_negative(place_type):
        copied.pop("place_type", None)
    elif place_type:
        type_words = [str(w).lower() for w in PLACE_TYPE_KEYWORDS.get(place_type, [])]
        if type_words and any(word in related for word in type_words) and not positive_hints.get("place_type"):
            copied.pop("place_type", None)

    # 如果本轮 positive_text 明确给了新的正向偏好，后续 merge 会覆盖；这里不额外处理。
    return copied


def extract_meal_pref_hint_from_user_text(user_input: str) -> Optional[str]:
    # 只从正向文本中抽取，避免“不想喝咖啡”被识别成 meal_pref=咖啡。
    """中文说明：从文本或结构中抽取当前流程需要的关键信息。"""
    text = positive_requirement_text_for_matching(str(user_input or ""))
    rules = [
        "火锅", "海底捞", "咖啡", "下午茶", "小笼包", "生煎", "面馆",
        "韩料", "韩国料理", "江浙菜", "本帮菜", "日料", "西餐",
        "甜品", "奶茶", "烧烤", "烤肉"
    ]
    for word in rules:
        if word in text:
            return word
    return None


def extract_excluded_places_from_user_text(user_input: str) -> list[str]:
    """中文说明：从文本或结构中抽取当前流程需要的关键信息。"""
    text = str(user_input or "")
    results = []

    patterns = [
        r"(?:不要去|不去|别去|避开|排除|不要)([^，。！？；;、\n]{2,30})",
        r"([^，。！？；;、\n]{2,30})(?:不要了|不去了|换掉)",
    ]

    for pattern in patterns:
        for m in re.finditer(pattern, text):
            candidate = clean_location_hint_candidate(m.group(1))
            if candidate and candidate not in GENERIC_LOCATION_TERMS:
                results.append(candidate)

    return unique_preserve_order(results)


###
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
    """规则兜底抽取“从 X 出发/出发地换为 X/起点改成 X”。"""
    text = str(user_input or "")
    patterns = [
        r"(?:把|将|请把)?(?:出发地|起点|始发地)(?:换成|换为|改成|改为|换到|改到|设为|设置为|为|是|在|到)([^，。！？；;、/／|｜\s]{2,30})",
        r"(?:从)([^，。！？；;、/／|｜\s]{2,30}?)(?:出发|开始|走|$|，|。|！|？|；|;|\||｜)",
        r"([^，。！？；;、/／|｜\s]{2,30}?)(?:出发)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            candidate = clean_location_hint_candidate(match.group(1))
            if candidate and candidate not in GENERIC_LOCATION_TERMS and not is_invalid_anchor_text(candidate):
                return candidate
    return None


def extract_destination_hint_from_user_text(user_input: str) -> Optional[str]:
    """规则兜底抽取目的地；附近/区域锚点不作为具体目的地返回。"""
    text = str(user_input or "")
    if is_departure_update_only_text(text):
        return None
    spatial = extract_spatial_anchor_from_user_text(text)
    if spatial and spatial.get("exclude_anchor_from_schedule"):
        return None

    suffixes = "动物园|植物园|公园|古镇|乐园|大学城|景区|博物馆|美术馆|展览馆|广场|商场|街区|寺庙|寺|园区|步道|滨江|外滩|沙滩|海滩|海湾|新区|区"
    patterns = [
        rf"(?:更换|修改|调整|重新设置)?(?:目的地|终点|想去的地方|要去的地方)(?:换成|改成|改为|换为|换到|改到|为|是|到)([^，。！？；;、/／|｜\s]{{2,30}}(?:{suffixes})?)",
        rf"(?:目的地|终点)([^，。！？；;、/／|｜\s]{{2,30}}(?:{suffixes})?)",
        rf"(?:不去|不要去|换掉)[^，。！？；;|｜]{{0,18}}(?:了|啦)?[，,、\s]*(?:去|换成|改成|改为|换为|换到|改到)([^，。！？；;、/／|｜\s]{{2,30}}(?:{suffixes})?)",
        rf"(?:到|去到|目的地是|目的地为|目的地在|终点是|终点为|想去|要去|去|逛|玩|安排)([^，。！？；;、/／|｜\s]{{2,30}}(?:{suffixes})?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            candidate = clean_location_hint_candidate(match.group(1))
            if is_area_anchor_value(candidate) and candidate not in DIRECT_SCHEDULE_LANDMARK_TERMS:
                return None
            if (
                candidate
                and candidate not in GENERIC_LOCATION_TERMS
                and is_concrete_location_anchor(candidate)
                and not is_departure_only_mention(text, candidate)
            ):
                return candidate
    candidate = extract_location_hint_from_user_text(text)
    if (
        candidate
        and is_concrete_location_anchor(candidate)
        and (not is_area_anchor_value(candidate) or candidate in DIRECT_SCHEDULE_LANDMARK_TERMS)
    ):
        return candidate
    return None

###--------------------修改

def clear_saved_anchor_history_in_workflow(state: dict) -> dict:
    """清理旧版持久化锚点字段，避免后续多轮修改把旧起点/旧目的地锁回方案。"""
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


def update_locked_places_from_state(state: dict, latest_text: str = "") -> dict:
    """缓存用户明确点名的核心地点，并锁定/覆盖明确起点和终点。

    关键规则：
    - 后续“换近一点/换便宜一点/换室内/重新生成”不改出发地、目的地、单中心锚点。
    - 后续“目的地换成X/更换目的地为X/终点改为X”只覆盖目的地；旧目的地从 locked_places 中移除。
    - 后续“出发地换成X/起点改为X”只覆盖出发地。
    """
    state = clear_saved_anchor_history_in_workflow(state)
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

    latest_departure_hint = extract_departure_hint_from_user_text(text)
    latest_destination_hint = extract_destination_hint_from_user_text(text)

    if is_departure_edit_value(old_fixed_destination, text, latest_departure_hint or ""):
        # 兜底清理：上一轮若已把“出发地换为X”误存成目的地，本轮不能继续保留。
        old_fixed_destination = ""
        for key in ["fixed_destination", "active_destination_anchor"]:
            state.pop(key, None)
        for key in ["fixed_destination", "active_destination_anchor", "location"]:
            collected.pop(key, None)
        collected["_location_explicit"] = False
        intent_location = str(intent.get("location") or "").strip()
        if is_departure_edit_value(intent_location, text, latest_departure_hint or ""):
            intent.pop("location", None)

    departure_change = bool(latest_departure_hint) and user_requests_departure_change(text, old_fixed_departure)
    destination_change = bool(latest_destination_hint) and user_requests_destination_change(text, old_fixed_destination)

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
            if old_fixed_departure and same_route_place(str(place), old_fixed_departure):
                continue
            if same_route_place(str(place), new_destination):
                continue
            previous_destination_candidates.append(place)

        stale_anchor_names = []
        for place in previous_destination_candidates:
            name = str(place or "").strip()
            if not name:
                continue
            if same_route_place(name, new_destination):
                continue
            if old_fixed_departure and same_route_place(name, old_fixed_departure):
                continue
            stale_anchor_names.append(name)
        stale_anchor_names = list(dict.fromkeys(stale_anchor_names))
        # 旧目的地只用于本轮过滤，不再持久化到 session。
        state = set_transient_anchor_exclusions_in_state(state, stale_anchor_names)

        # 只保留原出发地和最新目的地；不要保留旧目的地。
        kept_locked = []
        for place in locked:
            if old_fixed_departure and same_route_place(str(place), old_fixed_departure):
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
            """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
            return any(same_route_place(str(value or ""), old) for old in stale_anchor_names)
        state["avoid_places"] = [p for p in (state.get("avoid_places") or []) if not _is_stale_anchor(str(p))]
        cleaned_previous = []
        for route in (state.get("previous_plan_places") or []):
            cleaned_previous.append([p for p in route if not _is_stale_anchor(str(p))])
        state["previous_plan_places"] = cleaned_previous
        if new_destination not in locked:
            locked.append(new_destination)
    elif old_fixed_destination and is_concrete_location_anchor(old_fixed_destination):
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
    if destination and collected.get("_location_explicit") and is_concrete_location_anchor(destination):
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
    user_mentioned_candidate = bool(candidate) and place_matches_text(candidate, user_text_for_this_turn)
    if (
        candidate
        and candidate != "待确认地点"
        and (intent.get("explicit_place_match") is True or collected.get("_location_explicit"))
        and user_mentioned_candidate
    ):
        if candidate not in locked:
            locked.append(candidate)

    persistent_anchor_keys = {normalize_place_text(p) for p in [
        collected.get("fixed_departure"), collected.get("fixed_destination"), collected.get("center_anchor")
    ] if p}
    replaced_keys = set(state.get("exclude_anchor_keys_once") or [])
    current_destination_key = normalize_place_text(collected.get("fixed_destination") or "")
    kept = []
    for place in locked:
        place_key = normalize_place_text(place)
        if place_key in replaced_keys and place_key != current_destination_key:
            continue
        if place_key not in persistent_anchor_keys and not place_matches_text(place, user_text_for_this_turn) and not same_route_place(place, destination):
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
            if old_fixed_departure and same_route_place(str(p), old_fixed_departure):
                strict_kept.append(p)
            elif current_destination and same_route_place(str(p), current_destination):
                strict_kept.append(current_destination)
        kept = strict_kept or ([current_destination] if current_destination else [])

    state["intent"] = intent
    state["locked_places"] = list(dict.fromkeys(kept))
    state["collected_info"] = collected
    return state

def is_destination_anchor_intent(intent: dict, collected: Optional[dict] = None) -> bool:
    """用户明确给了具体目的地时，后续路线应围绕该目的地；区域/附近锚点走 area_anchor 分支。"""
    if (collected or {}).get("area_anchor") or (intent or {}).get("area_anchor"):
        return False
    location = str((intent or {}).get("location") or "").strip()
    if is_area_anchor_value(location) and location not in DIRECT_SCHEDULE_LANDMARK_TERMS:
        return False
    if not is_concrete_location_anchor(location):
        return False
    if bool((collected or {}).get("_location_explicit")):
        return True
    if (intent or {}).get("amap_anchor_type") in {"explicit_poi"}:
        return True
    if (intent or {}).get("explicit_place_match") is True:
        return True
    return False


def planning_anchor_for_intent(intent: dict, collected: dict) -> tuple[str, str]:
    """返回规划锚点。

    destination=用户点名具体地点；departure=围绕出发地；area/nearby=围绕区域/附近但不把锚点放进 schedule。
    """
    area_anchor = str((collected or {}).get("area_anchor") or (intent or {}).get("area_anchor") or "").strip()
    if area_anchor:
        return area_anchor, str((collected or {}).get("area_anchor_mode") or (intent or {}).get("area_anchor_mode") or "area")
    fixed_destination = str((collected or {}).get("fixed_destination") or (collected or {}).get("active_destination_anchor") or "").strip()
    if fixed_destination and is_concrete_location_anchor(fixed_destination) and (
            not is_area_anchor_value(fixed_destination) or fixed_destination in DIRECT_SCHEDULE_LANDMARK_TERMS):
        return fixed_destination, "destination"
    collected_location = str((collected or {}).get("location") or "").strip()
    if bool((collected or {}).get("_location_explicit")) and is_concrete_location_anchor(collected_location) and (
            not is_area_anchor_value(collected_location) or collected_location in DIRECT_SCHEDULE_LANDMARK_TERMS):
        return collected_location, "destination"
    if is_destination_anchor_intent(intent, collected):
        return str((intent or {}).get("location") or "").strip(), "destination"
    departure = str((collected or {}).get("fixed_departure") or (collected or {}).get("departure") or (intent or {}).get("departure") or "").strip()
    if departure and is_concrete_location_anchor(departure) and not is_placeholder_route_place(departure):
        return departure, "departure"
    return "", "default_center"


def same_route_place(a: str, b: str) -> bool:
    """路线级地点等价判断，用于保证目的地锚点不重复又能作为终点校验。"""
    if not a or not b:
        return False
    return place_matches_text(a, b) or place_matches_text(b, a)


def user_requests_departure_change(latest_text: str, current_departure: str = "") -> bool:
    """Only unlock a saved departure when the latest message explicitly gives a new one."""
    text = str(latest_text or "")
    hint = extract_departure_hint_from_user_text(text)
    if not hint:
        return any(token in text for token in ["换出发地", "改出发地", "起点改", "起点换", "出发地改为", "出发地换成"])
    if not current_departure:
        return True
    return not same_route_place(hint, current_departure)


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





def set_transient_anchor_exclusions_in_state(state: dict, names: list[str]) -> dict:
    """旧锚点只在本轮用于排除，不长期保存为用户记忆。"""
    state = dict(state or {})
    clean_names = []
    clean_keys = []

    for name in names or []:
        text = str(name or "").strip()
        if not text:
            continue
        key = normalize_place_text(text)
        if key and key not in clean_keys:
            clean_keys.append(key)
            clean_names.append(text)

    if clean_names:
        state["exclude_anchor_names_once"] = clean_names
        state["exclude_anchor_keys_once"] = clean_keys

    collected = dict(state.get("collected_info") or {})
    if clean_names:
        collected["exclude_anchor_names_once"] = clean_names
        collected["exclude_anchor_keys_once"] = clean_keys
    state["collected_info"] = collected
    return state


def clear_route_artifacts_after_anchor_change(state: dict) -> dict:
    """起点/终点/区域锚点变化后，清掉依赖旧路线的运行时产物。"""
    state = dict(state or {})
    for key in [
        "final_plan",
        "structured_plan",
        "route_distance_info",
        "route_map",
        "coupon_info",
        "reservation_options",
        "weather_info",
        "rag_context",
        "attraction_info",
        "ticket_info",
        "route_plan",
    ]:
        state[key] = None
    return state


def apply_area_anchor_change_to_state(state: dict, collected: dict, spatial_anchor: dict) -> tuple[dict, dict, list[str]]:
    """区域/附近锚点覆盖旧具体目的地，并清理旧 center_anchor / locked_places。"""
    state = dict(state or {})
    collected = dict(collected or {})
    intent = dict(state.get("intent") or {})

    new_area_anchor = str((spatial_anchor or {}).get("anchor") or "").strip()
    if not new_area_anchor:
        return state, collected, []

    old_anchor_candidates = unique_preserve_order([
        collected.get("fixed_destination"),
        collected.get("active_destination_anchor"),
        collected.get("center_anchor"),
        collected.get("location"),
        state.get("fixed_destination"),
        state.get("active_destination_anchor"),
        state.get("center_anchor"),
        intent.get("location"),
    ])

    departure_anchor = str(
        collected.get("fixed_departure")
        or collected.get("departure")
        or state.get("fixed_departure")
        or ""
    ).strip()

    stale_anchor_names = []
    for name in old_anchor_candidates:
        text = str(name or "").strip()
        if not text:
            continue
        if departure_anchor and same_route_place(text, departure_anchor):
            continue
        if same_route_place(text, new_area_anchor):
            continue
        stale_anchor_names.append(text)
    stale_anchor_names = unique_preserve_order(stale_anchor_names)

    for key in ["fixed_destination", "active_destination_anchor", "center_anchor"]:
        state.pop(key, None)
        collected.pop(key, None)

    collected["area_anchor"] = new_area_anchor
    collected["area_anchor_mode"] = spatial_anchor.get("mode", "area")
    collected["area_anchor_note"] = spatial_anchor.get("note", "")
    collected["exclude_anchor_from_schedule"] = True
    collected["_area_anchor_explicit"] = True
    collected["_location_explicit"] = False
    collected["location"] = spatial_anchor.get("query") or "周边休闲活动"

    intent.pop("location", None)
    intent["area_anchor"] = new_area_anchor
    intent["area_anchor_mode"] = spatial_anchor.get("mode", "area")
    intent["exclude_anchor_from_schedule"] = True
    state["intent"] = intent

    def _is_stale_anchor(value: str) -> bool:
        """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
        text = str(value or "").strip()
        if not text:
            return False
        if departure_anchor and same_route_place(text, departure_anchor):
            return False
        if same_route_place(text, new_area_anchor):
            return False
        return any(same_route_place(text, old) for old in stale_anchor_names)

    state["locked_places"] = [
        p for p in (state.get("locked_places") or [])
        if not _is_stale_anchor(str(p))
    ]

    state["avoid_places"] = [
        p for p in (state.get("avoid_places") or [])
        if not _is_stale_anchor(str(p))
    ]

    cleaned_previous = []
    for route in (state.get("previous_plan_places") or []):
        cleaned_previous.append([
            p for p in route
            if not _is_stale_anchor(str(p))
        ])
    state["previous_plan_places"] = cleaned_previous

    if stale_anchor_names:
        state = set_transient_anchor_exclusions_in_state(state, stale_anchor_names)

    state["collected_info"] = collected
    state = clear_route_artifacts_after_anchor_change(state)
    return state, collected, stale_anchor_names

def update_latest_requirement_state(state: dict, latest_text: str = "") -> dict:
    """
    统一处理多轮需求更新：
    - 新出发地覆盖旧出发地
    - 新目的地覆盖旧目的地
    - 新区域/附近锚点覆盖旧目的地和旧 center_anchor
    - 未提及字段保留 collected_info 里的旧值
    """
    state = dict(state or {})
    collected = dict(state.get("collected_info") or {})
    intent = dict(state.get("intent") or {})
    locked = list(state.get("locked_places") or [])
    text = str(latest_text or state.get("latest_user_input") or state.get("user_input") or "").strip()

    if not text:
        state["collected_info"] = collected
        return state

    old_fixed_departure = str(
        state.get("fixed_departure")
        or collected.get("fixed_departure")
        or collected.get("departure")
        or ""
    ).strip()

    old_fixed_destination = str(
        state.get("fixed_destination")
        or collected.get("fixed_destination")
        or collected.get("location")
        or ""
    ).strip()

    latest_departure_hint = extract_departure_hint_from_user_text(text)
    latest_destination_hint = extract_destination_hint_from_user_text(text)

    # 1. 出发地变更
    departure_change = bool(latest_departure_hint) and user_requests_departure_change(text, old_fixed_departure)
    if departure_change:
        new_departure = str(latest_departure_hint).strip()
        state["fixed_departure"] = new_departure
        collected["fixed_departure"] = new_departure
        collected["departure"] = new_departure
        collected["_departure_explicit"] = True
        state = clear_route_artifacts_after_anchor_change(state)
    elif old_fixed_departure:
        state["fixed_departure"] = old_fixed_departure
        collected["fixed_departure"] = old_fixed_departure
        collected["departure"] = old_fixed_departure
        collected["_departure_explicit"] = True

    # 2. 目的地变更
    destination_change = bool(latest_destination_hint) and user_requests_destination_change(text, old_fixed_destination)
    if destination_change:
        new_destination = str(latest_destination_hint).strip()

        previous_destination_candidates = [
            old_fixed_destination,
            state.get("fixed_destination"),
            state.get("active_destination_anchor"),
            state.get("center_anchor"),
            collected.get("fixed_destination"),
            collected.get("active_destination_anchor"),
            collected.get("center_anchor"),
            collected.get("location"),
            intent.get("location"),
        ]

        for place in locked:
            if not place:
                continue
            if old_fixed_departure and same_route_place(str(place), old_fixed_departure):
                continue
            if same_route_place(str(place), new_destination):
                continue
            previous_destination_candidates.append(place)

        stale_anchor_names = []
        for place in previous_destination_candidates:
            name = str(place or "").strip()
            if not name:
                continue
            if same_route_place(name, new_destination):
                continue
            if old_fixed_departure and same_route_place(name, old_fixed_departure):
                continue
            stale_anchor_names.append(name)
        stale_anchor_names = unique_preserve_order(stale_anchor_names)

        if stale_anchor_names:
            state = set_transient_anchor_exclusions_in_state(state, stale_anchor_names)

        for key in ["area_anchor", "area_anchor_mode", "area_anchor_note"]:
            collected.pop(key, None)
            state.pop(key, None)

        collected["_area_anchor_explicit"] = False
        collected["exclude_anchor_from_schedule"] = False

        state["fixed_destination"] = new_destination
        state["active_destination_anchor"] = new_destination
        state["center_anchor"] = new_destination

        collected["fixed_destination"] = new_destination
        collected["active_destination_anchor"] = new_destination
        collected["center_anchor"] = new_destination
        collected["location"] = new_destination
        collected["_location_explicit"] = True

        intent["location"] = new_destination

        kept_locked = []
        for place in locked:
            if old_fixed_departure and same_route_place(str(place), old_fixed_departure):
                kept_locked.append(place)
        kept_locked.append(new_destination)
        locked = unique_preserve_order(kept_locked)

        def _is_stale_anchor(value: str) -> bool:
            """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
            return any(same_route_place(str(value or ""), old) for old in stale_anchor_names)

        state["avoid_places"] = [
            p for p in (state.get("avoid_places") or [])
            if not _is_stale_anchor(str(p))
        ]

        cleaned_previous = []
        for route in (state.get("previous_plan_places") or []):
            cleaned_previous.append([
                p for p in route
                if not _is_stale_anchor(str(p))
            ])
        state["previous_plan_places"] = cleaned_previous

        state = clear_route_artifacts_after_anchor_change(state)

    elif old_fixed_destination and is_concrete_location_anchor(old_fixed_destination):
        state["fixed_destination"] = old_fixed_destination
        state["active_destination_anchor"] = old_fixed_destination
        collected["fixed_destination"] = old_fixed_destination
        collected["active_destination_anchor"] = old_fixed_destination
        collected["location"] = old_fixed_destination
        collected["_location_explicit"] = True
        if old_fixed_destination not in locked:
            locked.append(old_fixed_destination)

    # 4. 重新计算 locked_places
    persistent_anchor_keys = {
        normalize_place_text(p)
        for p in [
            collected.get("fixed_departure"),
            collected.get("fixed_destination"),
            collected.get("center_anchor"),
        ]
        if p
    }

    replaced_keys = set(state.get("exclude_anchor_keys_once") or [])
    current_destination_key = normalize_place_text(collected.get("fixed_destination") or "")

    kept = []
    for place in locked:
        place_key = normalize_place_text(place)
        if not place_key:
            continue
        if place_key in replaced_keys and place_key != current_destination_key:
            continue
        if place_key in persistent_anchor_keys:
            kept.append(place)

    state["intent"] = intent
    state["locked_places"] = unique_preserve_order(kept)
    state["latest_user_input"] = text
    state["collected_info"] = collected
    return state

###↓
def apply_fixed_anchor_guards(extracted: dict, collected: dict, state: AgentState) -> tuple[dict, dict, list[str]]:
    """
    兼容旧调用点，但真实锚点合并逻辑统一走 update_latest_requirement_state。
    """
    extracted = dict(extracted or {})
    collected = dict(collected or {})
    state = dict(state or {})

    merged_collected = dict(collected)
    for key, value in extracted.items():
        if value is not None:
            merged_collected[key] = value

    temp_state = {
        **state,
        "collected_info": merged_collected,
    }

    before = dict(state.get("collected_info") or {})
    latest_text = str(state.get("latest_user_input") or state.get("user_input") or "")
    updated_state = update_latest_requirement_state(temp_state, latest_text)
    updated_collected = dict(updated_state.get("collected_info") or {})

    for key in [
        "departure",
        "location",
        "area_anchor",
        "area_anchor_mode",
        "exclude_anchor_from_schedule",
        "fixed_departure",
        "fixed_destination",
        "active_destination_anchor",
        "center_anchor",
    ]:
        if updated_collected.get(key) is not None:
            extracted[key] = updated_collected.get(key)

    notes = []
    if before.get("fixed_departure") != updated_collected.get("fixed_departure") and updated_collected.get("fixed_departure"):
        notes.append(f"固定出发地已更新/保留：{updated_collected.get('fixed_departure')}。")
    if before.get("fixed_destination") != updated_collected.get("fixed_destination") and updated_collected.get("fixed_destination"):
        notes.append(f"固定目的地已更新/保留：{updated_collected.get('fixed_destination')}。")
    if before.get("area_anchor") != updated_collected.get("area_anchor") and updated_collected.get("area_anchor"):
        notes.append(f"区域/附近锚点已更新：{updated_collected.get('area_anchor')}。")

    state.update({
        "fixed_departure": updated_state.get("fixed_departure", state.get("fixed_departure")),
        "fixed_destination": updated_state.get("fixed_destination", state.get("fixed_destination")),
        "active_destination_anchor": updated_state.get("active_destination_anchor", state.get("active_destination_anchor")),
        "center_anchor": updated_state.get("center_anchor", state.get("center_anchor")),
        "locked_places": updated_state.get("locked_places", state.get("locked_places") or []),
        "intent": updated_state.get("intent", state.get("intent")),
    })

    return extracted, updated_collected, notes

####↑
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
    positive_user_input = positive_requirement_text_for_matching(user_input)
    latest_text = str(state.get("latest_user_input") or "")
    positive_latest_text = positive_requirement_text_for_matching(latest_text)
    collected = state.get("collected_info", {}) or {}

    fixed_departure = str(collected.get("fixed_departure") or state.get("fixed_departure") or "").strip()
    fixed_destination = str(collected.get("fixed_destination") or state.get("fixed_destination") or "").strip()

    latest_departure_hint = extract_departure_hint_from_user_text(positive_latest_text) if positive_latest_text else None
    latest_destination_hint = extract_destination_hint_from_user_text(positive_latest_text) if positive_latest_text else None
    departure_hint = latest_departure_hint if (latest_departure_hint and user_requests_departure_change(positive_latest_text, fixed_departure)) else None
    destination_hint = latest_destination_hint if (latest_destination_hint and user_requests_destination_change(positive_latest_text, fixed_destination)) else None

    # 没有显式修改时，保留已锁定的锚点；只有首轮/无锁定时才从完整输入兜底抽取。
    if not departure_hint and not fixed_departure:
        departure_hint = extract_departure_hint_from_user_text(positive_user_input)
    if not destination_hint:
        if fixed_destination:
            destination_hint = fixed_destination
        else:
            destination_hint = extract_destination_hint_from_user_text(positive_user_input)

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
    if fixed.get("location") and is_departure_only_mention(positive_user_input or user_input, str(fixed.get("location"))):
        fixed["location"] = ""
    if fixed.get("location") and not is_concrete_location_anchor(str(fixed.get("location"))):
        fixed["location"] = ""
    if not is_concrete_location_anchor(str(fixed.get("location") or "")):
        if not is_concrete_location_anchor(str(collected.get("location") or "")):
            collected.pop("location", None)
        collected["_location_explicit"] = False

    collected_people = parse_people_count(collected.get("num_people"), None)
    intent_people = parse_people_count(fixed.get("num_people"), None)
    explicit_people = parse_people_count(user_input, None) if not (collected_people or intent_people) else None
    final_people = collected_people or intent_people or explicit_people
    if final_people:
        fixed["num_people"] = final_people

    explicit_place = None
    if not destination_hint or _find_place(destination_hint) is not None:
        explicit_place = find_explicit_place_in_user_input(positive_user_input)
        if explicit_place and (
                is_departure_only_mention(positive_user_input, explicit_place.get("alias", ""))
                or is_departure_only_mention(positive_user_input, explicit_place.get("place_name", ""))
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
        unmatched = detect_unmatched_specific_place(positive_user_input)
        if unmatched:
            fixed["location"] = unmatched
            fixed["explicit_place_match"] = False
            fixed["explicit_place_note"] = (
                f"用户点名“{unmatched}”，但当前 mock 库未找到该地点；规划时必须提示需接入商家/地图 API 核验，不要改成其他店冒充。"
            )

    original_location_before_amap = str(fixed.get("location") or "").strip()
    fixed = resolve_unmatched_location_with_amap(fixed, state)
    if fixed.get("amap_anchor_type") == "explicit_poi" and fixed.get("location"):
        synced_state = sync_resolved_poi_anchor_fields(state, original_location_before_amap, str(fixed.get("location")))
        synced_collected = dict(synced_state.get("collected_info") or {})
        collected.update(synced_collected)
        for key in ["location", "fixed_destination", "active_destination_anchor", "center_anchor"]:
            fixed[key] = str(fixed.get("location") or "").strip()
        fixed["_location_explicit"] = True
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
    """中文说明：从文本或结构中抽取当前流程需要的关键信息。"""
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
        preferred_types = ["leisure", "attraction", "activity", "shopping"]
    elif place_type == "activity":
        preferred_types = ["activity", "leisure", "attraction", "shopping"]
    elif place_type == "attraction":
        preferred_types = ["attraction", "leisure", "activity", "shopping"]
    elif place_type == "cafe":
        preferred_types = ["cafe", "restaurant", "leisure"]
    elif place_type == "shopping":
        preferred_types = ["shopping", "leisure", "attraction"]

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
        """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
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
def extract_budget_hint_from_user_text(user_input: str) -> Optional[str]:
    """规则级抽取预算，避免首轮只依赖 LLM。"""
    return None
    text = str(user_input or "")
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return None
    total_match = re.search(r"(?:总预算|预算总共|总共|一共)(?:约|大概|控制在|不超过|少于|低于|以内)?(\d{2,5})\s*(?:元|块)?(?:以内|以下|左右)?", compact)
    if total_match:
        suffix = "以内" if re.search(r"(?:以内|以下|不超过|少于|低于|控制在)", compact) else "左右"
        return f"总预算{int(total_match.group(1))}元{suffix}"
    pp_match = re.search(r"(?:人均|每人|单人|一个人)(?:约|大概|控制在|不超过|少于|低于|以内)?(\d{2,5})\s*(?:元|块)?(?:以内|以下|左右)?", compact)
    if pp_match:
        suffix = "以内" if re.search(r"(?:以内|以下|不超过|少于|低于|控制在)", compact) else "左右"
        return f"人均{int(pp_match.group(1))}元{suffix}"
    generic_match = re.search(r"(?:预算|消费|花费|价格|价位)?(?:约|大概|控制在|不超过|少于|低于)?(\d{2,5})\s*(?:元|块)(?:以内|以下|左右)?", compact)
    if generic_match:
        amount = int(generic_match.group(1))
        suffix = "以内" if re.search(r"(?:以内|以下|不超过|少于|低于|控制在)", compact) else "左右"
        return f"人均{amount}元{suffix}"
    if any(word in compact for word in ["便宜", "省钱", "低预算", "学生党"]):
        return "人均100元以内"
    return None


def extract_date_hint_from_user_text(user_input: str) -> Optional[str]:
    """规则级抽取日期/星期，作为 LLM 漏抽兜底。"""
    text = str(user_input or "")
    patterns = [
        r"(20\d{2}[-/.年]\d{1,2}[-/.月]\d{1,2}[日号]?)",
        r"(\d{1,2}\s*月\s*\d{1,2}\s*[日号]?)",
        r"(今天|明天|后天|今晚|本周末|这周末|周末|周六|周日|周天|星期六|星期日|星期天|礼拜六|礼拜天|周一|周二|周三|周四|周五|星期一|星期二|星期三|星期四|星期五)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return re.sub(r"\s+", "", m.group(1))
    return None


def extract_duration_hint_from_user_text(user_input: str) -> Optional[int]:
    """中文说明：从文本或结构中抽取当前流程需要的关键信息。"""
    text = str(user_input or "")
    if any(word in text for word in ["半天", "半日"]):
        return 5
    m = re.search(r"(\d{1,2})\s*(?:个)?小时", text)
    if m:
        return _safe_int(m.group(1), 5)
    m = re.search(r"([一二两俩三四五六七八九十])\s*(?:个)?小时", text)
    if m:
        return CHINESE_NUMERAL_MAP.get(m.group(1), 5)
    return None




def strip_ui_preference_hints(user_input: str) -> str:
    """Remove UI-only Interests/Pace hints before extracting departure/destination.

    The hints remain available through extract_ui_preferences_from_text(); this
    function prevents them from becoming part of a POI or route anchor.
    """
    text = str(user_input or "")
    text = re.sub(r"[（(]\s*(?:兴趣偏好|Interests?|节奏偏好|Pace)[:：][^）)]*[）)]", "", text, flags=re.I)
    text = re.sub(r"(?:兴趣偏好|Interests?)[:：][^；;。\n]*", "", text, flags=re.I)
    text = re.sub(r"(?:节奏偏好|Pace)[:：][^；;。\n]*", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" ，,。；;｜|")
    return text.strip()


INTEREST_KEYWORD_MAP = {
    "美食": ["餐厅", "美食", "吃饭", "小吃", "火锅", "江浙菜", "本帮菜"],
    "文化": ["博物馆", "历史", "文化", "纪念馆", "老街", "古镇"],
    "购物": ["商场", "购物", "广场", "商圈", "百联", "万达", "环球港"],
    "艺术": ["美术馆", "艺术", "展览", "画廊", "艺术中心"],
    "自然": ["公园", "森林", "湿地", "郊野", "自然", "草坪"],
    "拍照": ["拍照", "出片", "打卡", "景观", "滨江", "街区"],
    "展览": ["展览", "看展", "美术馆", "博物馆", "艺术展"],
    "咖啡": ["咖啡", "咖啡馆", "下午茶", "甜品", "面包", "烘焙"],
    "散步": ["散步", "citywalk", "步道", "街道", "滨江", "公园"],
    "亲子": ["亲子", "家庭", "儿童", "乐园", "动物园", "科技馆"],
}

PACE_LABEL_MAP = {
    "relaxed": "Relaxed", "relax": "Relaxed", "放松": "Relaxed", "轻松": "Relaxed",
    "balanced": "Balanced", "balance": "Balanced", "正常": "Balanced", "均衡": "Balanced",
    "packed": "Packed", "pack": "Packed", "紧凑": "Packed", "特种兵": "Packed", "充实": "Packed",
}


def extract_ui_preferences_from_text(user_input: str) -> dict:
    """Parse frontend-injected preference hints without relying on another LLM call."""
    text = str(user_input or "")
    interests = []
    for pattern in [r"兴趣偏好[:：]([^；;。\n]+)", r"Interests?[:：]([^；;。\n]+)"]:
        m = re.search(pattern, text, flags=re.I)
        if m:
            parts = re.split(r"[,，、/\s]+", m.group(1))
            for part in parts:
                name = part.strip().lstrip("#")
                if name in INTEREST_KEYWORD_MAP and name not in interests:
                    interests.append(name)
    # Also accept explicit Chinese words from natural language, but only when they are clearly preference-like.
    # 自然语言兴趣必须基于正向文本；“不想喝咖啡”不能产生 interests=["咖啡"]。
    positive_text = positive_requirement_text_for_matching(text)
    if any(token in positive_text for token in ["兴趣", "偏好", "想", "喜欢", "优先", "适合"]):
        for name in INTEREST_KEYWORD_MAP:
            if name in positive_text and name not in interests:
                interests.append(name)

    return {"interests": interests}


def interest_keywords(interests) -> list[str]:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    result = []
    for name in interests or []:
        result.extend(INTEREST_KEYWORD_MAP.get(str(name).strip(), []))
    return unique_preserve_order(result)


def build_interest_match_notes(interests, schedule: list[dict]) -> list[str]:
    """中文说明：构建当前流程需要的结构化对象或展示内容。"""
    if not interests:
        return []
    route_text = " ".join(
        f"{item.get('place','')} {item.get('display_name','')} {item.get('purpose','')} {item.get('place_role','')}"
        for item in schedule or [] if isinstance(item, dict)
    )
    notes = []
    for name in interests:
        keys = INTEREST_KEYWORD_MAP.get(str(name), [])
        matched = [kw for kw in keys if kw and kw.lower() in route_text.lower()]
        if matched:
            notes.append(f"{name}：路线里包含“{matched[0]}”相关体验，已尽量贴合该兴趣。")
        else:
            notes.append(f"{name}：当前区域候选有限，已作为软偏好参与排序；可继续要求更偏{name}。")
    return notes


def extract_group_type_from_user_text(user_input: str) -> Optional[str]:
    """中文说明：从文本或结构中抽取当前流程需要的关键信息。"""
    text = str(user_input or "")
    if any(w in text for w in ["父母", "爸妈", "长辈", "家庭", "一家", "亲子", "孩子", "小孩"]):
        return "家庭"
    if any(w in text for w in ["情侣", "对象", "男朋友", "女朋友"]):
        return "情侣"
    if any(w in text for w in ["朋友", "同学", "同事"]):
        return "朋友"
    if any(w in text for w in ["一个人", "独自", "自己"]):
        return "独行"
    return None


REQUIREMENT_FIELD_LABELS = {
    "departure": "出发地",
    "location": "目的地/地点偏好",
    "area_anchor": "区域/附近锚点",
    "num_people": "人数",
    "date": "日期",
    "time_period": "时间段",
    "start_time": "出发时间",
    "duration_hours": "出行时长",
    "weather": "天气偏好",
    "budget": "预算",
    "group_type": "同行类型",
    "transport_mode": "交通方式",
    "meal_pref": "餐饮偏好",
    "requirements": "其他要求",
    "place_type": "地点类型",
    "canonical_place_type": "标准地点类型",
    "category": "大模型地点大类",
    "sub_type": "地点细分类",
    "llm_sub_type": "大模型地点细分类",
    "canonical_sub_type": "标准地点细分类",
    "search_keyword": "搜索关键词",
    "place_keywords": "地点关键词",
    "ordered_steps": "有序活动步骤",
    "excluded_places": "排除地点",
    "interests": "兴趣偏好",
}


def _split_requirement_labels(value) -> list[str]:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    return unique_preserve_order([
        part.strip()
        for part in re.split(r"[、,，;；]+", str(value or ""))
        if part.strip() and not part.strip().startswith("本轮语义改写：")
    ])


def extract_latest_requirement_field_hints(latest_text: str) -> tuple[dict, set[str]]:
    """Extract only fields explicitly present in the newest user message.

    latest_text 应优先传入 LLM 改写后的 positive_text；这里仍做一次否定片段剥离作为兜底。
    """
    text = positive_requirement_text_for_matching(strip_ui_preference_hints(str(latest_text or "")))
    hints = {}
    explicit_fields = set()

    people = parse_people_count(text, None)
    if people is not None:
        hints["num_people"] = people

    for key, func in [
        ("date", extract_date_hint_from_user_text),
        ("duration_hours", extract_duration_hint_from_user_text),
        ("group_type", extract_group_type_from_user_text),
        ("start_time", extract_start_time_hint_from_user_text),
        ("time_period", extract_time_period_hint_from_user_text),
        ("weather", extract_weather_hint_from_user_text),
        ("meal_pref", extract_meal_pref_hint_from_user_text),
        ("transport_mode", extract_transport_mode_from_user_text),
    ]:
        try:
            value = func(text)
        except Exception:
            value = None
        if value:
            hints[key] = value

    requirements = None
    if any(token in text for token in ["希望", "需要", "不要", "别", "优先", "避免", "改成", "换成"]):
        # Allow the LLM extractor to carry a custom soft requirement through the
        # revision guard even when it is outside the local rule vocabulary.
        explicit_fields.add("requirements")

    excluded_places = extract_excluded_places_from_user_text(text)
    if excluded_places:
        hints["excluded_places"] = excluded_places
        hints["avoid_places"] = excluded_places

    departure = extract_departure_hint_from_user_text(text)
    destination = extract_destination_hint_from_user_text(text)
    spatial_anchor = extract_spatial_anchor_from_user_text(text)
    if departure:
        hints["departure"] = departure
    if destination and is_concrete_location_anchor(str(destination)):
        hints["location"] = destination
    elif spatial_anchor and spatial_anchor.get("anchor"):
        hints["area_anchor"] = spatial_anchor.get("anchor")
        hints["area_anchor_mode"] = spatial_anchor.get("mode", "area")
        hints["exclude_anchor_from_schedule"] = bool(spatial_anchor.get("exclude_anchor_from_schedule", True))
        hints["location"] = spatial_anchor.get("query") or "周边休闲活动"

    ui_prefs = extract_ui_preferences_from_text(latest_text)
    if ui_prefs.get("interests"):
        hints["interests"] = ui_prefs["interests"]

    explicit_fields.update(hints)
    return hints, explicit_fields


def _requirement_change(field: str, before, after) -> Optional[dict]:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    if before == after:
        return None
    return {
        "field": field,
        "label": REQUIREMENT_FIELD_LABELS.get(field, field),
        "before": before,
        "after": after,
    }


def merge_latest_requirement_changes(state: dict, latest_text: str = "") -> dict:
    """Apply one revision turn as a field-level patch over the previous requirements.

    The newest message is the only source of changed fields. Omitted fields retain
    their previous values. This is shared by single-user and collaboration modes.

    中文说明：
    - 每一轮用户输入都先经过 LLM 语义改写，再提取结构化字段。
    - 本轮明确提到的字段覆盖旧值；没提到的字段沿用旧值。
    - requirements 这类软约束会累积合并，避免第二句补充把第一句偏好覆盖掉。
    - 多人模式 @Agent 后会按群聊顺序逐条调用这里，从前到后合并每位群友需求。
    """
    state = dict(state or {})
    text = str(latest_text or state.get("latest_user_input") or "").strip()
    if not text:
        return state
    if state.get("_latest_requirement_merged_text") == text:
        return state

    previous_source = str(state.get("requirement_change_source_text") or "")
    same_cycle = previous_source == text
    previous_cycle_changes = list(state.get("latest_requirement_changes") or []) if same_cycle else []
    before = dict(state.get("collected_info") or {})

    # 每轮字段级合并前先做语义改写。后续目的地/类别/偏好抽取只看改写后的正向文本，
    # 避免“想和父母去散心”这类口语被规则误锁成 fixed_destination=散心。
    cached_rewrite = state.get("_latest_rewrite_info") if state.get("_latest_rewrite_source_text") == text else None
    if isinstance(cached_rewrite, dict):
        rewrite_info = cached_rewrite
    else:
        rewrite_info = rewrite_user_requirement_for_pipeline(text, before, allow_llm=True)
        state["_latest_rewrite_info"] = rewrite_info
        state["_latest_rewrite_source_text"] = text
    positive_text = effective_positive_text_from_rewrite(rewrite_info, text)
    planning_text = str(rewrite_info.get("planning_text") or text).strip()
    structured_fields = sanitize_structured_requirement_fields(
        rewrite_info.get("fields") or {},
        positive_text,
        planning_text,
    )
    if not structured_fields.get("ordered_steps"):
        ordered_hint = extract_ordered_steps_hint_from_user_text(" ".join([planning_text, positive_text, text]))
        if ordered_hint:
            structured_fields["ordered_steps"] = ordered_hint

    state["latest_user_input"] = text
    collected = dict(state.get("collected_info") or {})
    rule_hints, explicit_fields = extract_latest_requirement_field_hints(positive_text)
    # 出发地/目的地/区域锚点只信 LLM rewrite fields。规则层可以补时间、人数、
    # 兴趣等低风险字段，但不能从自然语言或拼接说明文本里发明路线锚点。
    hard_anchor_keys = {
        "departure", "fixed_departure", "location", "fixed_destination",
        "active_destination_anchor", "area_anchor", "area_anchor_mode", "exclude_anchor_from_schedule",
    }
    for key in hard_anchor_keys:
        rule_hints.pop(key, None)
        explicit_fields.discard(key)
    if "budget" not in structured_fields:
        rule_hints.pop("budget", None)
        explicit_fields.discard("budget")
    hints = sync_ordered_steps_and_keywords({**rule_hints, **structured_fields})
    explicit_fields.update(structured_fields)
    raw_excluded_places = extract_excluded_places_from_user_text(text)
    if raw_excluded_places:
        hints["excluded_places"] = raw_excluded_places
        hints["avoid_places"] = raw_excluded_places
        explicit_fields.update({"excluded_places", "avoid_places"})
    collected = prune_collected_soft_preferences_by_rewrite(collected, rewrite_info, hints)

    for key, value in hints.items():
        if key in {"excluded_places", "avoid_places"}:
            collected[key] = unique_preserve_order(list(collected.get(key) or []) + list(value or []))
        elif key == "requirements":
            collected[key] = value
        elif key == "ordered_steps":
            collected[key] = sanitize_ordered_steps(value)
        else:
            collected[key] = value

    structured_destination = str(collected.get("fixed_destination") or collected.get("location") or "").strip()
    if structured_destination and is_concrete_location_anchor(structured_destination):
        collected["fixed_destination"] = structured_destination
        collected["active_destination_anchor"] = structured_destination
        collected["location"] = structured_destination
        collected["center_anchor"] = structured_destination
        collected["_location_explicit"] = True
        state["fixed_destination"] = structured_destination
        state["active_destination_anchor"] = structured_destination
        locked = list(state.get("locked_places") or [])
        if structured_destination not in locked:
            locked.append(structured_destination)
        state["locked_places"] = unique_preserve_order(locked)

    collected = prune_collected_soft_preferences_by_rewrite(collected, rewrite_info, hints)
    if collected.get("avoid_places"):
        state["avoid_places"] = unique_preserve_order(
            list(state.get("avoid_places") or []) + list(collected.get("avoid_places") or [])
        )

    tracked_fields = list(REQUIREMENT_FIELD_LABELS)
    changes = []
    for field in tracked_fields:
        change = _requirement_change(field, before.get(field), collected.get(field))
        if change:
            changes.append(change)

    merged_changes = {}
    for change in previous_cycle_changes + changes:
        merged_changes[change["field"]] = change

    state["collected_info"] = collected
    state["requirement_change_source_text"] = text
    state["_latest_requirement_merged_text"] = text
    state["_latest_rewrite_info"] = rewrite_info
    state["_latest_rewrite_source_text"] = text
    state["latest_requirement_explicit_fields"] = sorted(
        set(state.get("latest_requirement_explicit_fields") or []) | explicit_fields
        if same_cycle else explicit_fields
    )
    state["latest_requirement_changes"] = list(merged_changes.values())
    if changes and not same_cycle:
        history = list(state.get("requirement_change_log") or [])
        history.append({"input": text, "changes": changes})
        state["requirement_change_log"] = history[-20:]
    return state


def filter_unmentioned_revision_extractions(extracted: dict, state: dict) -> dict:
    """多轮修改保护：用户本轮没提到的字段，不允许 LLM/规则误抽后覆盖旧值。"""
    """Block LLM guesses from overwriting fields omitted in a revision message."""
    if not (state or {}).get("requirement_revision_mode"):
        return dict(extracted or {})
    explicit_fields = set((state or {}).get("latest_requirement_explicit_fields") or [])
    protected_fields = set(REQUIREMENT_FIELD_LABELS) | {
        "avoid_places", "area_anchor_mode", "exclude_anchor_from_schedule",
    }
    return {
        key: value
        for key, value in dict(extracted or {}).items()
        if key not in protected_fields or key in explicit_fields
    }


def summarize_latest_requirement_changes(state: dict) -> str:
    """把本轮需求变化整理成可展示文本，方便前端/日志说明方案为什么重新生成。"""
    changes = list((state or {}).get("latest_requirement_changes") or [])
    if not changes:
        return ""
    parts = []
    for change in changes:
        before = change.get("before")
        after = change.get("after")
        label = change.get("label") or change.get("field")
        parts.append(f"{label}：{before if before not in [None, ''] else '未指定'} → {after if after not in [None, ''] else '未指定'}")
    return "；".join(parts)



def summarize_collected_info_for_followup(collected: dict) -> str:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    pieces = []
    if collected.get("date") or collected.get("start_time") or collected.get("time_period"):
        time_bits = [str(collected.get("date") or "").strip(), str(collected.get("start_time") or collected.get("time_period") or "").strip()]
        pieces.append("时间" + "".join([b for b in time_bits if b]))
    if collected.get("num_people"):
        pieces.append(f"{collected.get('num_people')}人")
    dest = collected.get("location") or collected.get("area_anchor") or collected.get("fixed_destination")
    if dest:
        pieces.append(f"想去/围绕{dest}")
    if collected.get("departure"):
        pieces.append(f"从{collected.get('departure')}出发")
    return "、".join(str(p) for p in pieces if p) or "你刚才说的部分需求"


def build_missing_followup_question(collected: dict, missing_followups: list[str]) -> str:
    """根据缺失字段生成追问话术；只在初次收集信息阶段使用。"""
    label_map = {
        "departure": "出发地",
        "destination_or_area": "想去的具体地点/区域",
        "budget": "人均预算",
        "transport_mode": "交通方式",
        "requirements": "其他要求",
    }
    labels = [label_map.get(k, k) for k in missing_followups]
    known = summarize_collected_info_for_followup(collected)
    missing_text = "、".join(labels)
    return f"我已识别到：{known}。还差：{missing_text}。请只补充这些信息；没有特别要求可以说“其余你看着办”。"

def build_canonical_planning_input(collected: dict, latest_text: str = "") -> str:
    """只用当前有效需求状态构造规划输入，不再把历史 user_input 当事实来源。"""
    c = collected or {}
    return (
        f"用户最新输入：{latest_text or '无'}。"
        f"当前有效需求："
        f"目的地/偏好={c.get('location') or '未指定'}；"
        f"区域/附近锚点={c.get('area_anchor') or '无'}；"
        f"出发地点={c.get('departure') or '未指定，以目的地作为起点'}；"
        f"出行人数={c.get('num_people') or 2}人；"
        f"出行日期={c.get('date') or '本周末'}；"
        f"出行时间段={c.get('time_period') or '下午'}；"
        f"出发时间={c.get('start_time') or '未指定'}；"
        f"天气={c.get('weather') or '天气正常'}；"
        f"预算={c.get('budget') or '预算未指定，以实际路线估算为准'}；"
        f"同行者备注={c.get('companion_notes') or '无'}；"
        f"交通方式={c.get('transport_mode') or '公交地铁'}；"
        f"餐饮偏好={c.get('meal_pref') or '中餐'}；"
        f"地点类型={c.get('place_type') or '未指定'}；"
        f"标准地点类型={c.get('canonical_place_type') or '未指定'}；"
        f"大模型地点大类={c.get('category') or '未指定'}；"
        f"标准地点细分类={c.get('canonical_sub_type') or c.get('sub_type') or '未指定'}；"
        f"大模型地点细分类={c.get('llm_sub_type') or '未指定'}；"
        f"搜索关键词={c.get('search_keyword') or '未指定'}；"
        f"地点关键词={'、'.join(c.get('place_keywords') or []) or '无'}；"
        f"有序活动步骤={json.dumps(c.get('ordered_steps') or [], ensure_ascii=False)}；"
        f"排除地点={'、'.join(c.get('excluded_places') or c.get('avoid_places') or []) or '无'}；"
        f"其他要求={c.get('requirements') or '无'}；"
        f"兴趣偏好={'、'.join(c.get('interests') or []) or '无'}；"
        f"出行时长={c.get('duration_hours') or 5}小时。"
    )

def collect_required_info_for_api(state: AgentState) -> AgentState:
    """
    API 版本的信息收集节点：
    - 不使用 input() 阻塞等待
    - 信息不齐全时不再追问，直接用默认值补齐并继续生成方案

    中文说明：
    - 这是单人/多人进入完整规划前的“信息收集节点”。
    - 先调用 merge_latest_requirement_changes 合并本轮新输入。
    - 再用 LLM extractor + 本地规则补漏，更新 collected_info。
    - 如果还缺出发地/目的地/预算/交通等关键字段，直接填默认值；不返回 need_info。
    """
    extract_prompt = ChatPromptTemplate.from_template("""
请从用户输入中提取以下信息，以 JSON 格式返回（只返回 JSON，不加任何说明和 Markdown 标记）：
- departure: 完整出发地点（尽量包含城市/区/地铁站/小区/地标，如"上海人民广场地铁站"，如未提及填 null）
- location: 用户明确想去的具体目的地、商圈、店铺或可到访地标（如"海底捞""共青森林公园""人民广场"）；"散心/放松/走走/吃饭/咖啡/看展"这类活动或品类不要填 location，如未提及填 null
- num_people: 出行总人数，整数；只能来自明确人数表达或明确同行者关系推断。如未提及填 null
- date: 出行日期（具体日期或描述，如"本周六""5月20日"，如未提及填 null）
- time_period: 出行时间段，只能从 ["上午", "下午", "晚上"] 中选择；如果用户说具体几点，按上午/下午/晚上归类
- start_time: 具体出发时间（如"14:00""下午两点""晚上7点"，如未提及填 null）
- duration_hours: 整条路线总时长小时数。只有用户明确说“整个行程/总共/安排半天/路线4小时”时填写；如果用户只说某个活动“玩2小时/吃1小时/逛1小时”，不要填 duration_hours
- weather: 出行天气或天气偏好，如"晴天""下雨""阴天""热""冷"，如未提及填 null
- budget: 大致预算（人均金额整数或描述，如"人均100元""200以内"，如未提及填 null）；“人均200以内/人均200元”必须写 200，不要乘以人数；预算数字不能进入 num_people
- group_type: 人群类型，从 [情侣, 家庭, 朋友, 独行] 中选择；如未提及填 null
- companion_notes: 同行者细节备注，如"5岁女儿""父母同行""两个朋友"，如未提及填 null；年龄、身份、关系都写在这里，不用于计算人数
- transport_mode: 交通方式偏好，如"公交地铁""自驾""打车""步行优先"，如未提及填 null
- meal_pref: 餐饮偏好，如"火锅""咖啡""江浙菜"，如未提及填 null
- requirements: 用户的其他要求，如"室内""少走路""有团购""不要太累""附近逛一下"，如未提及填 null
- place_type: 大模型自由识别的地点主类型，不受枚举限制；如未提及填 null
- canonical_place_type: 如果能映射到代码认识的标准类型就填写，如 restaurant、cafe、activity、shopping、leisure、attraction；不能映射填 null
- category: 大模型理解的中文大类，如"娱乐""餐饮""购物""丽人""休闲""酒吧"；如未提及填 null
- llm_sub_type: 大模型自由识别的地点细分类，如"KTV""清吧""美甲""温泉""密室逃脱"，不受枚举限制；如未提及填 null
- canonical_sub_type: 如果能映射到代码认识的标准类型就填写，如 hotpot、escape_room、shopping_mall、spa_relax；不能映射填 null
- search_keyword: 用于高德/美团搜索的关键词，优先保留用户真实说法，如"唱K""清吧""美甲"；如未提及填 null
- sub_type: 为兼容旧逻辑，只在能确定 canonical_sub_type 时填写同样的标准值；不能映射时填 null
- place_keywords: 从用户原话中提取地点偏好关键词数组，如 ["郊区", "踏青", "户外"]；没有则填 []
- excluded_places: 用户明确说不要去/避开的地点数组，如 ["迪士尼", "EKA"]；没有则填 []                                                      

数字角色约束：
1. 同一句话里出现多个数字时，必须先判断每个数字的角色。
2. “上午10点/14:00/晚上7点”是 start_time，不是人数。
3. “5岁女儿/6岁孩子”是 companion_notes，不是人数。
4. “玩2小时/逛1小时/吃饭1小时”是某个活动的停留时长，不是人数，也不是整条路线总时长。
5. “人均100元/预算200元”是 budget，不是人数；“人均200以内，2个人”仍然 budget=200，不要改成总预算400或800。
6. num_people 只能来自明确人数表达或同行关系推断；无法确定就填 null。
7. 输出前自检：num_people 不得等于年龄、时间点、预算金额、活动时长或 RAG/历史案例人数。


用户输入: {input}
""")

    current_input = state.get("user_input", "")
    latest_input = str(state.get("latest_user_input") or current_input or "")
    state = merge_latest_requirement_changes(state, latest_input)
    # UI 选中的 Interests/Pace 只用于偏好排序，不允许进入地点/起终点抽取。
    latest_plain = strip_ui_preference_hints(str(state.get("latest_user_command") or latest_input))
    latest_for_llm = latest_plain or latest_input
    collected = dict(state.get("collected_info") or {})
    cached_rewrite = state.get("_latest_rewrite_info") if state.get("_latest_rewrite_source_text") == latest_for_llm else None
    if isinstance(cached_rewrite, dict):
        rewrite_info = cached_rewrite
    else:
        rewrite_info = rewrite_user_requirement_for_pipeline(latest_for_llm, collected)
        state["_latest_rewrite_info"] = rewrite_info
        state["_latest_rewrite_source_text"] = latest_for_llm
    latest_positive = effective_positive_text_from_rewrite(rewrite_info, latest_for_llm)
    latest_planning = rewrite_info.get("planning_text") or latest_for_llm

    structured_fields = sanitize_structured_requirement_fields(
        rewrite_info.get("fields") or {},
        latest_positive,
        latest_planning,
    )
    if not structured_fields.get("ordered_steps"):
        ordered_hint = extract_ordered_steps_hint_from_user_text(" ".join([latest_planning, latest_positive, latest_for_llm]))
        if ordered_hint:
            structured_fields["ordered_steps"] = ordered_hint

    # 主路径信任 rewrite_user_requirement_for_pipeline() 产出的结构化 fields。
    # 旧 extractor 只在 LLM 改写没有给出任何字段时兜底，避免二次抽取把人数、目的地和顺序重新覆盖。
    extracted = dict(structured_fields)
    if not extracted and llm is not None:
        extract_chain = extract_prompt | llm | StrOutputParser()
        result = extract_chain.invoke({"input": latest_positive})
        cleaned = re.sub(r"```json|```", "", result).strip()
        try:
            extracted = sanitize_structured_requirement_fields(json.loads(cleaned), latest_positive, latest_planning)
        except json.JSONDecodeError:
            print(f"⚠️ 信息提取解析失败: {result}")
            extracted = {}

    if extracted.get("num_people") is not None:
        extracted["num_people"] = parse_people_count(extracted.get("num_people"), extracted.get("num_people"))
    else:
        explicit_people = parse_people_count(latest_for_llm, None)
        if explicit_people is not None:
            extracted["num_people"] = explicit_people

    # 规则级兜底：把首轮用户已说出的预算/日期/时长/人群/软约束全部并入，不等追问。
    date_hint = extract_date_hint_from_user_text(latest_positive)
    duration_hint = extract_duration_hint_from_user_text(latest_positive)
    group_type_hint = extract_group_type_from_user_text(latest_positive)
    _set_if_missing(extracted, "date", date_hint)
    _set_if_missing(extracted, "duration_hours", duration_hint)
    _set_if_missing(extracted, "group_type", group_type_hint)
    time_period_hint = extract_time_period_hint_from_user_text(latest_positive)
    weather_hint = extract_weather_hint_from_user_text(latest_positive)
    meal_pref_hint = extract_meal_pref_hint_from_user_text(latest_positive)
    excluded_places_hint = extract_excluded_places_from_user_text(latest_for_llm)

    _set_if_missing(extracted, "time_period", time_period_hint)

    _set_if_missing(extracted, "weather", weather_hint)

    _set_if_missing(extracted, "meal_pref", meal_pref_hint)

    if excluded_places_hint:
        _merge_ordered_list(extracted, "excluded_places", excluded_places_hint)
        _merge_ordered_list(extracted, "avoid_places", excluded_places_hint)

    ui_prefs = extract_ui_preferences_from_text(latest_input)
    if ui_prefs.get("interests") and not extracted.get("interests"):
        extracted["interests"] = ui_prefs["interests"]

    spatial_anchor = None
    skip_words = ["看着办", "随便", "都行", "你决定", "你来定", "随机", "无所谓"]
    if any(word in str(extracted.get("location") or "") for word in skip_words):
        extracted["location"] = None

    start_time_hint = extract_start_time_hint_from_user_text(latest_positive)
    transport_mode_hint = extract_transport_mode_from_user_text(latest_positive)
    if extracted.get("location") is not None:
        cleaned_location = clean_location_hint_candidate(str(extracted.get("location") or ""))
        if cleaned_location != str(extracted.get("location") or ""):
            extracted["location"] = cleaned_location or None
        if extracted.get("location") and not is_concrete_location_anchor(str(extracted.get("location"))):
            extracted["location"] = None
    if start_time_hint and not extracted.get("start_time"):
        extracted["start_time"] = start_time_hint
        if any(token in start_time_hint for token in ["晚上", "夜里", "傍晚"]):
            extracted["time_period"] = "晚上"
        elif any(token in start_time_hint for token in ["下午"]):
            extracted["time_period"] = "下午"
        elif any(token in start_time_hint for token in ["上午", "早上"]):
            extracted["time_period"] = "上午"

    # 合并已有信息与新提取信息（非 null 值覆盖旧值）
    _set_if_missing(extracted, "transport_mode", transport_mode_hint)
    extracted = sync_ordered_steps_and_keywords(extracted)

    extracted = filter_unmentioned_revision_extractions(extracted, state)
    collected = prune_collected_soft_preferences_by_rewrite(collected, rewrite_info, extracted)
    if latest_planning:
        if not extracted.get("requirements") and collected.get("requirements"):
            extracted["requirements"] = collected.get("requirements")
    for key in [
        "departure", "location", "num_people", "date", "time_period", "start_time",
        "duration_hours", "weather", "budget", "group_type", "companion_notes", "transport_mode",
        "meal_pref", "requirements", "interests",
        "place_type", "canonical_place_type", "category", "sub_type", "llm_sub_type", "canonical_sub_type", "search_keyword",
        "place_keywords", "ordered_steps", "excluded_places", "avoid_places", "fixed_destination",
        "area_anchor", "area_anchor_mode", "exclude_anchor_from_schedule"
    ]:
        new_val = extracted.get(key)
        if new_val is not None:
            if key == "requirements":
                collected[key] = new_val
            elif key == "ordered_steps":
                collected[key] = sanitize_ordered_steps(new_val)
            else:
                collected[key] = new_val
            if key == "departure" and str(new_val).strip():
                collected["_departure_explicit"] = True
            if key == "location" and is_concrete_location_anchor(str(new_val)) and (
                    not is_area_anchor_value(str(new_val)) or str(new_val) in DIRECT_SCHEDULE_LANDMARK_TERMS
            ) and not collected.get("_area_anchor_explicit"):
                collected["_location_explicit"] = True
                collected["fixed_destination"] = str(new_val).strip()
                collected["active_destination_anchor"] = str(new_val).strip()
                collected["center_anchor"] = str(new_val).strip()
    if collected.get("avoid_places"):
        state["avoid_places"] = unique_preserve_order(
            list(state.get("avoid_places") or []) + list(collected.get("avoid_places") or [])
        )
    print(f"📝 当前已收集信息: {collected}")

    has_destination_or_area = (
        bool(collected.get("area_anchor"))
        or is_concrete_location_anchor(str(collected.get("location") or ""))
        or is_concrete_location_anchor(str(collected.get("fixed_destination") or ""))
    )
    missing_followups = []
    if not collected.get("departure"):
        missing_followups.append("departure")
    if not has_destination_or_area:
        missing_followups.append("destination_or_area")
    for field in ["budget", "transport_mode"]:
        if not collected.get(field):
            missing_followups.append(field)
    if missing_followups:
        # 产品要求：不再进行缺项追问。用户没提到的字段直接用默认值，让 Agent 立即生成方案。
        print(f"ℹ️ 缺失字段将使用默认值: {missing_followups}")

    defaults = {
        "departure": None,
        "num_people": 2,
        "date": "本周末",
        "time_period": "下午",
        "start_time": None,
        "duration_hours": 5,
        "weather": "天气正常",
        "budget": "预算未指定，以实际路线估算为准",
        "group_type": "朋友",
        "transport_mode": "公交地铁",
        "interests": [],
    }
    for key, value in defaults.items():
        if not collected.get(key):
            collected[key] = value
    collected.setdefault("_departure_explicit", False)
    collected.setdefault("_location_explicit", False)

    print("✅ 信息收集结束：缺失项已按默认值处理")
    collected = prune_collected_soft_preferences_by_rewrite(collected, rewrite_info, extracted)
    return {
        **state,
        "user_input": latest_input,
        "collected_info": collected,
        "info_complete": True,
        "pending_question": None,
        "missing_followups": [],
    }



def fast_collect_required_info_for_api(state: AgentState) -> AgentState:
    """LLM extractor 慢或不可用时的规则兜底收集节点。

    这个路径不应生成不同业务语义，只是尽量用同一套字段和缺项判断快速兜底。
    所以这里也要维护和 collect_required_info_for_api 一致的 requirements 合并、目的地判断等规则。
    """
    current_input = str((state or {}).get("user_input") or "")
    latest_input = str((state or {}).get("latest_user_input") or current_input or "")

    # merge_latest_requirement_changes 已经完成 LLM 语义改写和字段级合并，并会把
    # rewrite_info 缓存在 _latest_rewrite_info。fast collector 不应再生成一套更粗的
    # fallback 字段覆盖它，而是优先复用缓存结果。
    state = merge_latest_requirement_changes(state, latest_input)
    latest_plain = strip_ui_preference_hints(str((state or {}).get("latest_user_command") or latest_input)) or latest_input
    collected = dict((state or {}).get("collected_info") or {})
    merged_ordered_steps_before_fast = sanitize_ordered_steps(collected.get("ordered_steps") or [])

    cached_rewrite = (
        state.get("_latest_rewrite_info")
        if state.get("_latest_rewrite_source_text") == latest_input
        else None
    )
    if isinstance(cached_rewrite, dict) and cached_rewrite:
        rewrite_info = cached_rewrite
    else:
        # 只有没有 LLM 缓存时才使用本地 heuristic，避免重复改写和语义降级。
        rewrite_info = rewrite_user_requirement_for_pipeline(latest_plain, collected, allow_llm=False)

    latest_positive = effective_positive_text_from_rewrite(rewrite_info, latest_plain)
    latest_planning = rewrite_info.get("planning_text") or latest_plain
    extracted = sanitize_structured_requirement_fields(
        rewrite_info.get("fields") or {},
        latest_positive,
        latest_planning,
    )
    if not extracted.get("ordered_steps"):
        ordered_hint = extract_ordered_steps_hint_from_user_text(" ".join([latest_planning, latest_positive, latest_plain]))
        if ordered_hint:
            extracted["ordered_steps"] = ordered_hint

    if not extracted.get("num_people"):
        people = parse_people_count(latest_positive, None)
        if people is not None:
            extracted["num_people"] = people
    for key, func in [
        ("date", extract_date_hint_from_user_text),
        ("duration_hours", extract_duration_hint_from_user_text),
        ("group_type", extract_group_type_from_user_text),
        ("start_time", extract_start_time_hint_from_user_text),
        ("time_period", extract_time_period_hint_from_user_text),
        ("weather", extract_weather_hint_from_user_text),
        ("meal_pref", extract_meal_pref_hint_from_user_text),
        ("transport_mode", extract_transport_mode_from_user_text),
    ]:

        try:
            value = func(latest_positive)
        except Exception:
            value = None
        if value:
            if key == "requirements":
                if not extracted.get("requirements"):
                    extracted["requirements"] = value
            else:
                _set_if_missing(extracted, key, value)
    excluded_places_hint = extract_excluded_places_from_user_text(latest_plain)
    if excluded_places_hint:
        extracted["excluded_places"] = excluded_places_hint
        extracted["avoid_places"] = excluded_places_hint

    if extracted.get("start_time"):
        st = str(extracted["start_time"])
        if "晚上" in st or "夜" in st:
            extracted["time_period"] = "晚上"
        elif "上午" in st or "早" in st:
            extracted["time_period"] = "上午"
        else:
            extracted["time_period"] = "下午"

    spatial = None
    ui_prefs = extract_ui_preferences_from_text(latest_input)
    if ui_prefs.get("interests"):
        extracted["interests"] = ui_prefs["interests"]

    extracted = sync_ordered_steps_and_keywords(extracted)

    # 保护 LLM 已识别出的 ordered_steps：如果本地 fallback 只给出更少的子集，
    # 不允许它覆盖大模型结果；如果关键词相同，则合并字段并保留 duration_min。
    protected_ordered_steps = merge_ordered_steps_without_downgrade(
        merged_ordered_steps_before_fast,
        extracted.get("ordered_steps") or [],
    )
    if protected_ordered_steps:
        extracted["ordered_steps"] = protected_ordered_steps
        extracted["place_keywords"] = unique_preserve_order(
            [step.get("keyword") for step in protected_ordered_steps if step.get("keyword")]
            + _coerce_list_field(extracted.get("place_keywords") or [])
            + _coerce_list_field(collected.get("place_keywords") or [])
        )

    extracted = filter_unmentioned_revision_extractions(extracted, state)
    collected = prune_collected_soft_preferences_by_rewrite(collected, rewrite_info, extracted)
    if latest_planning:
        if not extracted.get("requirements") and collected.get("requirements"):
            extracted["requirements"] = collected.get("requirements")
    for key, value in extracted.items():
        if value is not None:
            if key == "requirements":
                collected[key] = value
            elif key == "ordered_steps":
                collected[key] = sanitize_ordered_steps(value)
            else:
                collected[key] = value
            if key == "departure" and str(value).strip():
                collected["_departure_explicit"] = True
            if key == "location" and is_concrete_location_anchor(str(value)) and (
                    not is_area_anchor_value(str(value)) or str(value) in DIRECT_SCHEDULE_LANDMARK_TERMS
            ) and not collected.get("_area_anchor_explicit"):
                collected["_location_explicit"] = True
                collected["fixed_destination"] = str(value).strip()
                collected["active_destination_anchor"] = str(value).strip()
                collected["center_anchor"] = str(value).strip()

    collected = prune_collected_soft_preferences_by_rewrite(collected, rewrite_info, extracted)
    has_destination_or_area = (
        bool(collected.get("area_anchor"))
        or is_concrete_location_anchor(str(collected.get("location") or ""))
        or is_concrete_location_anchor(str(collected.get("fixed_destination") or ""))
    )
    missing_followups = []
    if not collected.get("departure"):
        missing_followups.append("departure")
    if not has_destination_or_area:
        missing_followups.append("destination_or_area")
    for field in ["budget", "transport_mode"]:
        if not collected.get(field):
            missing_followups.append(field)
    if missing_followups:
        # fast fallback 与完整路径保持一致：不追问，直接填默认值继续规划。
        print(f"ℹ️ 缺失字段将使用默认值 [fast]: {missing_followups}")

    defaults = {
        "departure": None,
        "num_people": 2,
        "date": "本周末",
        "time_period": "下午",
        "start_time": None,
        "duration_hours": 5,
        "weather": "天气正常",
        "budget": "预算未指定，以实际路线估算为准",
        "group_type": "朋友",
        "transport_mode": "公交地铁",
        "interests": [],
    }
    for key, value in defaults.items():
        if not collected.get(key):
            collected[key] = value
    collected.setdefault("_departure_explicit", False)
    collected.setdefault("_location_explicit", False)
    return {
        **state,
        "user_input": latest_input,
        "collected_info": collected,
        "info_complete": True,
        "pending_question": None,
        "missing_followups": [],
        "fast_collect_fallback": True,
    }
def parse_intent(state: AgentState) -> AgentState:
    """把 collected_info 转成后续规划节点使用的 intent。

    FAST_INTENT_PARSE=1 时主要信任前面的 collected_info，不再让 LLM 重新自由抽取，
    这样可以降低耗时，也减少多轮状态被后续大模型改乱的风险。
    """
    if os.getenv("FAST_INTENT_PARSE", "1") == "1":
        collected = state.get("collected_info", {}) or {}
        collected = prune_collected_soft_preferences_by_rewrite(
            dict(collected),
            heuristic_rewrite_requirement_text(state.get("user_input", ""), collected),
            {},
        )
        intent = {
            "group_type": collected.get("group_type", "朋友"),
            "date": collected.get("date", "本周末"),
            "time_period": collected.get("time_period", "下午"),
            "weather": collected.get("weather", "天气正常"),
            "departure": collected.get("departure") or "",
            "location": collected.get("location", ""),
            "duration_hours": collected.get("duration_hours", 5),
            "meal_pref": collected.get("meal_pref", "中餐"),
            "num_people": collected.get("num_people", 2),
            "budget": collected.get("budget", "人均200元以内"),
            "companion_notes": collected.get("companion_notes", ""),

            "place_type": collected.get("place_type", "attraction"),
            "canonical_place_type": collected.get("canonical_place_type", ""),
            "category": collected.get("category", ""),
            "sub_type": collected.get("sub_type", ""),
            "llm_sub_type": collected.get("llm_sub_type", ""),
            "canonical_sub_type": collected.get("canonical_sub_type", ""),
            "search_keyword": collected.get("search_keyword", ""),
            "place_keywords": collected.get("place_keywords", []) or [],
            "ordered_steps": sanitize_ordered_steps(collected.get("ordered_steps") or []),
            "excluded_places": collected.get("excluded_places", []) or [],
            "avoid_places": collected.get("avoid_places", []) or state.get("avoid_places", []) or [],

            "start_time": collected.get("start_time"),
            "transport_mode": collected.get("transport_mode", "公交地铁"),
            
            "requirements": collected.get("requirements", ""),
            "interests": collected.get("interests", []) or [],
            "area_anchor": collected.get("area_anchor", ""),
            "area_anchor_mode": collected.get("area_anchor_mode", ""),
            "exclude_anchor_from_schedule": collected.get("exclude_anchor_from_schedule", False),
        }
        start_time_hint = (
            extract_start_time_hint_from_user_text(state.get("latest_user_input", ""))
            or extract_start_time_hint_from_user_text(positive_requirement_text_for_matching(state.get("user_input", "")))
        )
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
        intent = normalize_intent_place_type(intent, positive_requirement_text_for_matching(state.get("user_input", "")))
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
- duration_hours: 整条路线总时长小时数。只有用户明确说“整个行程/总共/安排半天/路线4小时”时填写；如果用户只说某个活动“玩2小时/吃1小时/逛1小时”，不要填 duration_hours
- meal_pref: 餐饮偏好关键词，如未提及填"中餐"
- num_people: 出行总人数，整数；只能来自明确人数表达或明确同行者关系推断。如未提及填 null
- budget: 大致预算描述；“人均200以内/人均200元”必须写 200，不要乘以人数；预算数字不能进入 num_people
- companion_notes: 同行者细节备注，如"5岁女儿""父母同行""两个朋友"，如未提及填空字符串；年龄、身份、关系都写在这里，不用于计算人数
- place_type: 偏好的地点类型，只能从 [attraction, restaurant, cafe, activity, leisure, shopping, sports] 中选择，如未提及填"attraction"
  - 如果用户说“郊区/近郊/远郊/踏青/户外/散步/放松”，不要输出 suburban/outdoor，优先填 "leisure"
  - 如果用户说“露营/团建/手作/体验/亲子活动”，填 "activity"
  - 如果用户说“公园/古镇/博物馆/美术馆/展馆/景区”，填 "attraction"
- place_keywords: 从用户原话中提取地点偏好关键词数组，如 ["郊区", "踏青", "户外"]；没有则填 []
- excluded_places: 用户明确说不要去/避开的地点数组，如 ["迪士尼", "EKA"]；没有则填 []
- avoid_places: 同 excluded_places；没有则填 []                                              

数字角色约束：
1. 同一句话里出现多个数字时，必须先判断每个数字的角色。
2. “上午10点/14:00/晚上7点”是 start_time，不是人数。
3. “5岁女儿/6岁孩子”是 companion_notes，不是人数。
4. “玩2小时/逛1小时/吃饭1小时”是某个活动的停留时长，不是人数，也不是整条路线总时长。
5. “人均100元/预算200元”是 budget，不是人数；“人均200以内，2个人”仍然 budget=200，不要改成总预算400或800。
6. num_people 只能来自明确人数表达或同行关系推断；无法确定就填 null。
7. 输出前自检：num_people 不得等于年龄、时间点、预算金额、活动时长或 RAG/历史案例人数。

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
            "budget": collected.get("budget", "适中"),
            "place_type": "attraction"
        }

    print(f"📋 意图解析结果: {intent}")
    collected = state.get("collected_info", {}) or {}
    positive_user_input_for_matching = positive_requirement_text_for_matching(state.get("user_input", ""))
    transport_mode_hint = extract_transport_mode_from_user_text(positive_user_input_for_matching)
    if transport_mode_hint:
        collected["transport_mode"] = transport_mode_hint
    for key in [
        "departure", "location", "date", "time_period", "start_time", "weather",
        "num_people", "budget", "duration_hours", "group_type", "companion_notes", "transport_mode",
        "meal_pref", "requirements", "place_type", "canonical_place_type", "category", "sub_type", "llm_sub_type",
        "canonical_sub_type", "search_keyword", "place_keywords", "ordered_steps",
        "excluded_places", "avoid_places"
    ]:
        if collected.get(key):
            intent[key] = collected[key]
    start_time_hint = (
        extract_start_time_hint_from_user_text(state.get("latest_user_input", ""))
        or extract_start_time_hint_from_user_text(positive_user_input_for_matching)
    )
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
    intent = normalize_intent_place_type(intent, positive_user_input_for_matching)
    intent = reconcile_intent_with_rules(intent, state)
    print(f"📍 地点类型归一结果: {intent.get('place_type')} / {intent.get('place_keywords', [])}")
    return {**state, "intent": intent}


def rag_retrieval(state: AgentState) -> AgentState:
    """检索历史案例作为风格/经验参考。

    注意：RAG 只提供参考表达和历史路线经验，最终地点事实仍以结构化地点表、高德和本轮用户约束为准。
    """
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

    if os.getenv("ENABLE_RAG_RETRIEVAL", "1") != "1":
        print("📚 RAG 检索已按配置跳过；结构化规划仍按 mock/高德/规则执行。")
        return {**state, "rag_context": ""}

    docs = retriever.invoke(query)

    max_chars = int(os.getenv("RAG_CONTEXT_MAX_CHARS", "1800"))
    context_blocks = []
    used_chars = 0
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", doc.metadata.get("doc_id", ""))
        content = str(doc.page_content or "").strip()
        remain = max(0, max_chars - used_chars)
        if remain <= 0:
            break
        if len(content) > remain:
            content = content[:remain] + "…"
        used_chars += len(content)
        context_blocks.append(
            f"【参考案例{idx}｜source={source}｜chunk={chunk_id}】\n{content}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    print(f"📚 RAG 检索完成，共召回 {len(docs)} 条相关案例，传入上下文约 {used_chars} 字")
    return {**state, "rag_context": context}


def tool_dispatch(state: AgentState) -> AgentState:
    """Legacy node: route drafts are no longer generated before schedule."""
    raise RuntimeError("tool_dispatch is deprecated; build_structured_plan owns schedule generation.")
def extract_tool_place_name(tool_text: str, fallback: str = "") -> str:
    """中文说明：从文本或结构中抽取当前流程需要的关键信息。"""
    text = str(tool_text or "").strip()
    if "|" in text:
        return text.split("|", 1)[0].strip() or fallback
    match = re.search(r"(?:未找到|找到)?\s*([^，。|]+?)(?:\s*的|\s*\|)", text)
    return (match.group(1).strip() if match else str(fallback or "").strip())


def find_available_alternative_for_unavailable(original_place: str, intent: dict, state: AgentState) -> Optional[str]:
    """Find a nearby/same-type available backup for an unavailable final stop.

    The replacement policy is user-facing: if a scheduled place has no seats or a
    very long queue, the executable route should not keep it as-is. We first look
    for same-type places in the local place table, prefer candidates connected to
    the latest destination/area anchor, and use cached/limited distance checks to
    make the backup as close as possible without causing a slow plan.
    """
    refresh_place_data_if_changed(False)
    original_key = normalize_place_text(original_place)
    collected = state.get("collected_info") or {}
    place_type = str((intent or {}).get("place_type") or "").strip()
    original_category = normalized_route_category(route_place_category(original_place))
    cached_pool = []
    for pool in ((state or {}).get("geo_fence_poi_cache") or {}).values():
        if isinstance(pool, dict):
            cached_pool.extend(pool.get("places") or [])
    for name in unique_preserve_order(cached_pool):
        key = normalize_place_text(name)
        if not key or key == original_key or same_route_place(name, original_place):
            continue
        if original_category and original_category != "unknown":
            if normalized_route_category(route_place_category(name)) != original_category:
                continue
        row = _find_place(name)
        if row is not None and not row_has_seat(row):
            continue
        return name
    # if ordered_search_specs_from_intent(intent):
    #     return None
    original_row = _find_place(original_place)
    if original_row is not None:
        place_type = str(original_row.get("地点类型") or place_type or "attraction")

    anchor = (
        collected.get("fixed_destination")
        or collected.get("area_anchor")
        or collected.get("center_anchor")
        or collected.get("location")
        or (intent or {}).get("location")
        or collected.get("departure")
        or (intent or {}).get("departure")
        or ""
    )
    anchor_key = normalize_place_text(anchor)
    original_searchable = ""
    original_district_tokens = []
    if original_row is not None:
        original_searchable = normalize_place_text(" ".join([
            str(original_row.get("placeName", "") or ""),
            str(original_row.get("search_tags", "") or ""),
            str(original_row.get("amap_address", "") or ""),
            str(original_row.get("amap_district", "") or ""),
            str(original_row.get("source_note", "") or ""),
        ]))
        district = str(original_row.get("amap_district", "") or "").strip()
        if district:
            original_district_tokens.append(normalize_place_text(district))
        for token in ["迪士尼", "陆家嘴", "松江", "嘉定", "徐家汇", "南京东路", "人民广场", "大学城"]:
            if token in str(original_row.get("placeName", "")) or token in str(original_row.get("amap_address", "")):
                original_district_tokens.append(normalize_place_text(token))

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
            str(row.get("amap_district", "") or ""),
            str(row.get("source_note", "") or ""),
        ]))
        score = 0.0
        if anchor_key and (anchor_key in searchable or searchable in anchor_key):
            score += 120
        for token in original_district_tokens:
            if token and token in searchable:
                score += 70
        # Share meaningful tokens with the original place/area. This helps keep
        # Disney-area backups near Disney instead of jumping to the city center.
        shared_tokens = 0
        for token in significant_place_tokens(original_place)[:4]:
            if token and token not in {"餐厅", "店", "咖啡", "火锅"} and token in searchable:
                shared_tokens += 1
        score += shared_tokens * 30
        if place_has_coupon(name):
            score += 10
        if place_is_indoor(name):
            score += 5
        try:
            score -= float(row.get("最低价格", 0) or 0) / 60
        except (TypeError, ValueError):
            pass
        candidates.append({"name": name, "score": score, "distance_m": None})

    if candidates:
        candidates.sort(key=lambda item: item["score"], reverse=True)
        max_distance_checks = max(0, _safe_int(os.getenv("AVAILABILITY_BACKUP_DISTANCE_CHECKS", "0"), 0))
        # Only check a few top-scored candidates to keep generation fast.
        for item in candidates[:max_distance_checks]:
            try:
                item["distance_m"] = route_distance_between_places(original_place, item["name"])
            except Exception:
                item["distance_m"] = None
        def rank(item: dict):
            """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
            dist = item.get("distance_m")
            # Prefer a known close candidate; otherwise use semantic/area score.
            dist_penalty = (float(dist) / 1000.0) if isinstance(dist, int) and dist > 0 else 8.0
            return (-float(item.get("score") or 0), dist_penalty)
        candidates.sort(key=rank)
        return candidates[0]["name"]

    # Amap only proves that a POI exists. Without a realtime inventory provider,
    # a map search result cannot be advertised as an available replacement.
    return None

def build_exception_event(kind: str, original_place: str, backup_place: Optional[str], state: AgentState, reason: str) -> dict:
    """中文说明：构建当前流程需要的结构化对象或展示内容。"""
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
    """Legacy node: final schedule checks now run inside build_structured_plan."""
    raise RuntimeError("exception_handler is deprecated; schedule conflicts are emitted after schedule generation.")
def route_distance_planner(state: AgentState) -> AgentState:
    """为结构化路线计算转场距离、耗时和地图信息。"""
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

    # 仅新增统计变量和安全默认值，不改变路线生成/校验/地图生成顺序。
    route_segments = []
    failed_segments = []
    route_segments_generation_seconds = 0.0
    route_map_generation_seconds = 0.0
    amap_route_generation_seconds = 0.0
    feasibility_report = (
        state.get("feasibility_report")
        or ((structured_plan or {}).get("feasibility_report") if isinstance(structured_plan, dict) else None)
        or {
            "intensity_score": 0,
            "warnings": [],
            "suggestions": [],
        }
    )

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
            "feasibility_report": feasibility_report,
            "route_segments_generation_seconds": 0.0,
            "route_map_generation_seconds": 0.0,
            "map_generation_seconds": 0.0,
            "amap_route_generation_seconds": 0.0,
        }

    route_plan_text = ""

    schedule_stops = [
        item.get("place")
        for item in structured_plan.get("schedule", [])
        if isinstance(item, dict) and item.get("place")
    ]
    if schedule_stops:
        route_segments_started_at = time.perf_counter()
        route_distance_info, route_segments, failed_segments = compute_route_segments(departure, schedule_stops)
        route_segments_generation_seconds = time.perf_counter() - route_segments_started_at

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

    feasibility_report = (
        evaluate_route_feasibility(structured_plan, intent, route_segments)
        if structured_plan
        else {
            "intensity_score": 0,
            "warnings": ["结构化方案为空，无法评估路线强度。"],
            "suggestions": [],
        }
    )
    if structured_plan:
        structured_plan["feasibility_report"] = feasibility_report
        structured_plan.setdefault("tool_facts", {})
        structured_plan["tool_facts"]["route_distance_info"] = route_distance_info
        structured_plan["tool_facts"]["coupon_summary"] = coupon_info.get("summary", "")

    if structured_plan:
        route_map_started_at = time.perf_counter()
        try:
            route_map = build_route_map_info(structured_plan)
        except Exception as exc:
            route_map = build_route_map_placeholder(structured_plan)
            route_map["reason"] = f"路线地图数据生成失败: {exc}"
        finally:
            route_map_generation_seconds = time.perf_counter() - route_map_started_at
    else:
        route_map = {"available": False, "reason": "结构化方案为空，未生成地图。"}

    # 统计口径专用：供 app_api.py 从“方案展示耗时”里整体扣除。
    # 不改变真实执行流程，只改变后端/前端展示的耗时数值。
    amap_route_generation_seconds = route_segments_generation_seconds + route_map_generation_seconds

    coupon_info = filter_coupon_info_for_schedule(coupon_info, structured_plan)
    final_plan = render_schedule_narrative_plan(state, structured_plan, coupon_info)
    validation_report = validate_generated_plan(final_plan, {**state, "structured_plan": structured_plan}, coupon_info)
    reservation_options = build_reservation_options(structured_plan)
    validation_report = add_reservation_consistency_checks(validation_report, structured_plan, reservation_options)
    print(f"📏 高德距离参考完成:\n{route_distance_info}")
    print(f"🎟️ 团购券校验完成:\n{coupon_info.get('summary')}")
    print(f"🗺️ 路线地图状态: {route_map.get('note') or route_map.get('reason')}")
    return {
        **state,
        "route_plan": route_plan_text,
        "route_distance_info": route_distance_info,
        "route_map": route_map,
        "route_segments_generation_seconds": round(route_segments_generation_seconds, 2),
        "route_map_generation_seconds": round(route_map_generation_seconds, 2),
        "map_generation_seconds": round(route_map_generation_seconds, 2),
        "amap_route_generation_seconds": round(amap_route_generation_seconds, 2),
        "coupon_info": coupon_info,
        "feasibility_report": feasibility_report,
        "structured_plan": structured_plan or state.get("structured_plan"),
        "final_plan": final_plan,
        "validation_report": validation_report,
        "reservation_options": reservation_options,
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
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
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
- 明确写一句：当前路线涉及地点未发现可用团购券。
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
                + "\n\n⚠️ 团购券核验：当前路线涉及地点未发现可用团购券；"
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
    """中文说明：合并多来源数据，同时尽量保留原有顺序和约束。"""
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
    """中文说明：解析输入或中间结果，转换为后续节点可用的数据。"""
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
    """中文说明：把结构化数据格式化为前端或用户可读内容。"""
    minutes = max(0, min(minutes, 23 * 60 + 59))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"



def place_duration_profile(place_name: str) -> dict:
    """Return a type-aware stay-duration profile for a route stop.

    目标不是给每个点平均分时间，而是按“用户真实会停留多久”分配：
    - 迪士尼/游乐园/大型动物园等长体验点获得更长时间；
    - citywalk/街区/商圈打卡获得较短时间；
    - 正餐/咖啡/展馆/影院等按各自典型停留时长处理。

    target/min/max 都是分钟。allocate_dynamic_stop_minutes() 会在这些范围内
    根据总时长做压缩或扩展，保证整条路线仍贴合 4-6 小时窗口。
    """
    name = str(place_name or "")
    lowered = name.lower()
    role = place_role(place_name)
    row = _find_place(place_name)
    sub_type = str(row.get("sub_type", "") or "").strip() if row is not None else ""
    place_type = str(row.get("地点类型", "") or "").strip() if row is not None else ""
    tags = str(row.get("search_tags", "") or "") if row is not None else ""
    combined = f"{name} {sub_type} {place_type} {tags}".lower()

    def profile(target: int, min_minutes: int, max_minutes: int, category: str, reason: str, priority: int) -> dict:
        """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
        target = int(target)
        min_minutes = int(min_minutes)
        max_minutes = int(max_minutes)
        if max_minutes < min_minutes:
            max_minutes = min_minutes
        target = max(min_minutes, min(max_minutes, target))
        return {
            "target": target,
            "min": min_minutes,
            "max": max_minutes,
            "category": category,
            "reason": reason,
            # priority 越大，额外时间越优先加给它；压缩时间时越晚被压缩。
            "priority": int(priority),
        }

    if role == "meal":
        return profile(95, 75, 125, "meal", "正餐需要点餐、用餐和休息，保留中等偏长停留。", 7)
    if role == "light_food":
        return profile(45, 30, 65, "light_food", "咖啡/甜品主要用于补给和休息，停留不宜过长。", 3)

    # 明确长时间游玩：游乐园、迪士尼、探险乐园、大型亲子/主题项目。
    if sub_type == "theme_park" or any(k in combined for k in [
        "迪士尼", "乐园", "游乐", "主题公园", "森林探险", "探险乐园", "欢乐谷", "海昌", "玛雅海滩"
    ]):
        return profile(190, 150, 240, "theme_park", "游乐园/主题项目排队和体验时间长，因此作为长停留核心点。", 10)

    # 大型自然/动物/植物类，通常比普通公园更久。
    if any(k in combined for k in ["动物园", "野生动物", "植物园", "森林公园", "郊野公园", "湿地", "辰山", "顾村"]):
        return profile(150, 110, 210, "large_nature", "大型公园/动物园/植物园需要较长游玩和步行时间。", 9)

    # 普通公园、滨江绿地、自然休闲。
    if sub_type == "park" or any(k in combined for k in ["公园", "草坪", "绿地", "滨江", "江边", "森林"]):
        return profile(110, 75, 160, "park", "公园类适合慢走拍照，但半日路线中不宜无限拉长。", 6)

    # city walk / 街区 / 步行街：更像路过、拍照、轻逛。
    if sub_type == "street_walk" or any(k in combined for k in [
        "citywalk", "城市漫步", "步行街", "大学路", "武康路", "安福路", "多伦路", "甜爱路", "街区", "老街", "步道"
    ]):
        return profile(60, 35, 85, "street_walk", "街区/city walk 以轻逛和打卡为主，停留比大型景点短。", 4)

    # 展馆、博物馆、美术馆：中等偏长，但通常比游乐园短。
    if sub_type in {"museum", "art_exhibition"} or any(k in combined for k in ["博物馆", "美术馆", "艺术馆", "展览", "展馆", "画廊"]):
        return profile(100, 70, 150, "museum_exhibition", "展馆类需要完整参观动线，分配中等偏长时间。", 7)

    if sub_type == "cinema" or any(k in combined for k in ["影院", "影城", "电影"]):
        return profile(130, 105, 170, "cinema", "电影按完整片长和入退场时间计算。", 8)

    if sub_type == "spa_relax" or any(k in combined for k in ["汤泉", "温泉", "泡汤", "洗浴"]):
        return profile(150, 120, 210, "spa_relax", "泡汤/汤泉是慢节奏体验，适合长停留。", 9)

    if any(k in combined for k in ["ktv", "KTV".lower(), "唱歌", "量贩", "歌城"]):
        return profile(120, 90, 180, "ktv", "KTV 通常按小时计费和体验，至少预留一段完整时间。", 8)

    if sub_type == "shopping" or any(k in combined for k in ["商场", "广场", "购物中心", "印象城", "合生汇", "万达", "环球港", "百联"]):
        return profile(70, 45, 110, "shopping", "商场/购物中心适合休息和轻逛，时间控制在中等。", 5)

    if place_type in {"sports"} or any(k in combined for k in ["运动", "攀岩", "射箭", "保龄球", "骑行"]):
        return profile(110, 80, 160, "sports", "运动体验需要准备、体验和缓冲时间。", 7)

    if place_type == "activity":
        return profile(90, 60, 130, "activity", "活动体验按中等停留处理。", 6)

    if place_type in {"attraction", "leisure"}:
        return profile(80, 50, 120, "attraction", "普通景点/休闲点按轻中度停留处理。", 5)

    return profile(65, 40, 100, "general", "普通地点按较轻停留处理。", 4)


def place_duration_weight(place_name: str) -> int:
    """Backward-compatible weight used by older code paths."""
    return int(place_duration_profile(place_name).get("target", 80))


def _distribute_duration_diff(durations: list[int], profiles: list[dict], diff: int) -> list[int]:
    """Distribute duration diff while respecting min/max and place priorities."""
    if not durations or diff == 0:
        return durations

    # diff > 0: 给更需要长时间的点加时间；diff < 0: 优先从低优先级/可压缩点扣时间。
    if diff > 0:
        order = sorted(range(len(durations)), key=lambda i: profiles[i].get("priority", 0), reverse=True)
    else:
        order = sorted(range(len(durations)), key=lambda i: profiles[i].get("priority", 0))

    step = 5
    guard = 0
    while diff != 0 and guard < 2000:
        changed = False
        for i in order:
            if diff == 0:
                break
            if diff > 0:
                room = int(profiles[i].get("max", durations[i])) - durations[i]
                if room <= 0:
                    continue
                delta = min(step, diff, room)
                durations[i] += delta
                diff -= delta
                changed = True
            else:
                room = durations[i] - int(profiles[i].get("min", 35))
                if room <= 0:
                    continue
                delta = min(step, -diff, room)
                durations[i] -= delta
                diff += delta
                changed = True
        if not changed:
            break
        guard += 1

    # 处理非 5 分钟倍数或极端情况下仍未归零的差值，最后放到可调整空间最大的点。
    if diff != 0:
        candidates = []
        for i, value in enumerate(durations):
            if diff > 0:
                room = int(profiles[i].get("max", value)) - value
            else:
                room = value - int(profiles[i].get("min", 35))
            if room > 0:
                candidates.append((room, profiles[i].get("priority", 0), i))
        if candidates:
            candidates.sort(reverse=(diff > 0))
            i = candidates[0][2]
            if diff > 0:
                delta = min(diff, int(profiles[i].get("max", durations[i])) - durations[i])
                durations[i] += delta
            else:
                delta = min(-diff, durations[i] - int(profiles[i].get("min", 35)))
                durations[i] -= delta
    return durations


def allocate_dynamic_stop_minutes(places: list[str], total_minutes: int) -> list[int]:
    """Allocate stop durations by POI type instead of averaging.

    Examples:
    - 迪士尼/游乐园会明显长于 city walk；
    - city walk/街区/商圈轻逛会较短；
    - 正餐、咖啡、展馆、影院、汤泉各有独立停留区间；
    - 最终总和仍严格贴合 total_minutes。
    """
    places = [str(p or "").strip() for p in places if str(p or "").strip()]
    if not places:
        return [max(60, int(total_minutes or 300))]

    total_minutes = max(60, int(total_minutes or 300))
    profiles = [place_duration_profile(p) for p in places]
    durations = [int(p["target"]) for p in profiles]

    # 如果总目标时长与用户窗口不一致，只在 min/max 内按优先级调整。
    durations = _distribute_duration_diff(durations, profiles, total_minutes - sum(durations))

    current = sum(durations)
    if current != total_minutes:
        # 极端情况：所有点的 min 之和仍超过总时长，按比例压缩到至少 30 分钟。
        if current > total_minutes:
            floor = 30
            reducible = sum(max(0, d - floor) for d in durations)
            need = current - total_minutes
            if reducible > 0:
                for i, d in enumerate(list(durations)):
                    if need <= 0:
                        break
                    cut = min(d - floor, round(need * max(0, d - floor) / reducible))
                    durations[i] -= max(0, cut)
                # 精确补差。
                while sum(durations) > total_minutes:
                    i = max(range(len(durations)), key=lambda idx: durations[idx])
                    if durations[i] <= floor:
                        break
                    durations[i] -= 1
            else:
                durations = [max(floor, total_minutes // len(durations)) for _ in durations]
        elif current < total_minutes:
            # 如果还有剩余但 max 都满了，就加给最高优先级点；这是比丢时间更合理的兜底。
            i = max(range(len(durations)), key=lambda idx: profiles[idx].get("priority", 0))
            durations[i] += total_minutes - current

    # 最后一轮保证分钟数严格相等。
    diff = total_minutes - sum(durations)
    if diff and durations:
        i = max(range(len(durations)), key=lambda idx: profiles[idx].get("priority", 0)) if diff > 0 else min(range(len(durations)), key=lambda idx: profiles[idx].get("priority", 0))
        durations[i] = max(30, durations[i] + diff)
    return [int(max(30, d)) for d in durations]

def ordered_step_matches_place(step: dict, place: str) -> bool:
    """Return True when an ordered_step likely corresponds to a final route place."""
    if not isinstance(step, dict):
        return False
    place_text = str(place or "")
    place_key = normalize_place_text(place_text)
    if not place_key:
        return False

    tokens = [
        step.get("keyword"),
        step.get("search_keyword"),
        step.get("llm_sub_type"),
        step.get("canonical_sub_type"),
        step.get("sub_type"),
        step.get("category"),
    ]
    for token in tokens:
        token_text = str(token or "").strip()
        token_key = normalize_place_text(token_text)
        if token_key and (token_key in place_key or place_key in token_key):
            return True

    # Category fallback: after Amap / place-table resolution, final place may be a
    # concrete POI name that does not literally contain the user keyword.
    try:
        route_cat = schedule_place_type_metadata(place).get("route_category") or route_place_category(place)
    except Exception:
        route_cat = route_place_category(place)
    step_text = " ".join(str(x or "") for x in tokens)
    if route_cat == "bar" and any(w in step_text for w in ["清吧", "酒吧", "bar", "喝一杯"]):
        return True
    if route_cat in {"hotpot", "bbq"} and any(w in step_text for w in ["火锅", "烤肉", "烧烤"]):
        return True
    if route_cat == "escape_room" and any(w in step_text for w in ["密室", "剧本杀", "推理馆"]):
        return True
    if route_cat == "ktv" and any(w in step_text for w in ["唱K", "唱歌", "KTV", "ktv"]):
        return True
    return False


def explicit_stop_minutes_for_places(places: list[str], collected: dict, intent: dict) -> list[int]:
    """Map ordered_steps[].duration_min to the final route places.

    The result is aligned with places: 0 means no explicit duration for that stop.
    """
    steps = sanitize_ordered_steps(
        (intent or {}).get("ordered_steps")
        or (collected or {}).get("ordered_steps")
        or []
    )
    result = []
    used_step_indexes = set()
    for place in places or []:
        explicit_minutes = 0
        for idx, step in enumerate(steps):
            if idx in used_step_indexes:
                continue
            duration = _safe_int(step.get("duration_min") or step.get("duration_minutes"), 0)
            if duration <= 0:
                continue
            if ordered_step_matches_place(step, place):
                explicit_minutes = max(15, min(duration, 240))
                used_step_indexes.add(idx)
                break
        result.append(explicit_minutes)
    return result


def apply_explicit_stop_durations(
    default_durations: list[int],
    explicit_durations: list[int],
    total_minutes: int,
    min_flexible_minutes: int = 30,
) -> list[int]:
    """Apply user-explicit stop durations before dynamic allocation.

    Explicit durations are treated as locked first. Remaining time is distributed
    across flexible stops according to the original dynamic durations.
    """
    count = len(default_durations or [])
    if count <= 0:
        return []

    total_minutes = max(1, _safe_int(total_minutes, sum(default_durations or []) or 1))
    explicit = []
    for i in range(count):
        value = explicit_durations[i] if i < len(explicit_durations or []) else 0
        value = _safe_int(value, 0)
        explicit.append(max(15, min(value, 240)) if value > 0 else 0)

    locked_indexes = [i for i, value in enumerate(explicit) if value > 0]
    flexible_indexes = [i for i in range(count) if i not in locked_indexes]
    if not locked_indexes:
        return [int(max(15, d)) for d in default_durations]

    result = [0] * count
    for i in locked_indexes:
        result[i] = explicit[i]

    locked_total = sum(result)
    if not flexible_indexes:
        return [int(max(15, d)) for d in result]

    remaining = total_minutes - locked_total
    if remaining <= 0:
        # 总时长已经被显式时长用完：保留显式时长，把非显式站点压到最小展示时长。
        for i in flexible_indexes:
            result[i] = min_flexible_minutes
        return [int(max(15, d)) for d in result]

    # 如果剩余时间不足以给每个非显式站点 30 分钟，则按可用剩余均分，
    # 避免时间总长被无意义撑大。
    flexible_floor = min_flexible_minutes if remaining >= min_flexible_minutes * len(flexible_indexes) else max(15, remaining // len(flexible_indexes))
    default_flexible_total = sum(max(1, default_durations[i]) for i in flexible_indexes)
    assigned = 0
    for i in flexible_indexes:
        if default_flexible_total > 0:
            portion = round(remaining * max(1, default_durations[i]) / default_flexible_total)
        else:
            portion = remaining // len(flexible_indexes)
        result[i] = max(flexible_floor, portion)
        assigned += result[i]

    # 修正总和，优先调非显式站点，不动用户明确指定的时长。
    diff = total_minutes - sum(result)
    guard = 0
    while diff != 0 and flexible_indexes and guard < 1000:
        guard += 1
        if diff > 0:
            target = max(flexible_indexes, key=lambda idx: result[idx])
            result[target] += 1
        else:
            reducible = [idx for idx in flexible_indexes if result[idx] > flexible_floor]
            if not reducible:
                break
            target = max(reducible, key=lambda idx: result[idx])
            result[target] -= 1
        diff = total_minutes - sum(result)

    return [int(max(15, d)) for d in result]


def build_schedule_slots(
    collected: dict,
    intent: dict,
    count: int,
    places: Optional[list[str]] = None,
    explicit_durations: Optional[list[int]] = None,
) -> list[str]:
    """生成时间槽。传入 places 时按地点类型动态分配；显式 duration_min 优先。"""
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
    if places:
        durations = allocate_dynamic_stop_minutes(list(places)[:count], total_minutes)
    else:
        base = total_minutes // count
        remainder = total_minutes % count
        durations = [base + (1 if index < remainder else 0) for index in range(count)]

    if explicit_durations:
        durations = apply_explicit_stop_durations(
            durations,
            list(explicit_durations)[:count],
            total_minutes,
        )

    slots = []
    cursor = start_minutes
    for duration in durations[:count]:
        end = min(cursor + duration, 23 * 60 + 59)
        slots.append(f"{format_time_minutes(cursor)}-{format_time_minutes(end)}")
        cursor = end
    return slots


def duration_minutes_from_slot(slot: str) -> int:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    m = re.match(r"(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})", str(slot or ""))
    if not m:
        return 0
    h1, m1, h2, m2 = map(int, m.groups())
    return max(0, h2 * 60 + m2 - (h1 * 60 + m1))


def build_availability_event(place: str, status: str, message: str, backup: str = "", level: str = "info") -> dict:
    # Keep the schema stable, but sanitize the user-facing message so the frontend
    # never exposes implementation words such as “mock”.
    """中文说明：构建当前流程需要的结构化对象或展示内容。"""
    try:
        visible_message = sanitize_user_visible_inventory_text(message)
    except NameError:
        visible_message = str(message or "")
    return {
        "type": status,
        "place_name": place,
        "original_place": place,
        "backup_place": backup or "",
        "display_level": level,
        "message": visible_message,
    }

def event_message(event) -> str:
    """Safely extract a display message from an event that may be dict or legacy string."""
    if isinstance(event, dict):
        return str(event.get("message") or event.get("detail") or event.get("summary") or event)
    return str(event or "")


def normalize_event_item(event, default_level: str = "warning") -> dict:
    """Normalize legacy string events and structured dict events into one schema.

    Older code paths and unique_preserve_order() could turn event dicts into strings.
    Frontend/rendering code expects .get(), so every event that enters
    route_logic_validation.exception_events should be a dict.
    """
    if isinstance(event, dict):
        item = dict(event)
        item.setdefault("type", item.get("status") or "event")
        item.setdefault("display_level", item.get("level") or default_level)
        item.setdefault("message", event_message(item))
        return item
    message = str(event or "").strip()
    return {
        "type": "legacy_event",
        "place_name": "",
        "original_place": "",
        "backup_place": "",
        "display_level": default_level,
        "message": message,
    }


def unique_event_preserve_order(events: list, default_level: str = "warning") -> list[dict]:
    """De-duplicate event records without converting dicts to strings."""
    seen = set()
    result = []
    for event in events or []:
        item = normalize_event_item(event, default_level=default_level)
        message = str(item.get("message") or "").strip()
        if not message:
            continue
        key = (str(item.get("type") or ""), str(item.get("place_name") or ""), message)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def sanitize_user_visible_inventory_text(text: str) -> str:
    """Remove implementation/source words from messages shown in the UI.

    The UI should tell users the operational state (available, no seats, replaced),
    not that the state came from a mock spreadsheet. Internal logs and debug fields
    may still use the word mock, but all event/status messages pass through here.
    """
    value = str(text or "")
    replacements = {
        "当前 mock 状态显示": "当前余位状态显示",
        "mock 状态显示": "余位状态显示",
        "mock 数据中": "当前数据中",
        "mock 数据": "当前数据",
        "mock 表中": "当前地点库中",
        "mock 表": "当前地点库",
        "mock 库": "当前地点库",
        "未接入 mock 余位库存": "暂未接入实时余位",
        "demo mock": "演示数据",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value.strip()


def user_visible_availability_event(place: str, status: str, message: str, backup: str = "", level: str = "info") -> dict:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    event = build_availability_event(place, status, sanitize_user_visible_inventory_text(message), backup=backup, level=level)
    event["message"] = sanitize_user_visible_inventory_text(event.get("message", ""))
    return event


def should_replace_unavailable_place(place: str, status: dict, state: AgentState, intent: dict) -> bool:
    """Unavailable final stops should be replaced even when the original was an anchor.

    Earlier versions skipped replacement for locked destinations to avoid disobeying
    the user. For the current product behavior, a place with no seats should not stay
    in the executable schedule; it should be surfaced as a conflict and replaced by
    a nearby/same-type backup when possible.
    """
    return str((status or {}).get("status") or "") in {"no_seat", "queue_long"}



def availability_status_for_place(place: str) -> dict:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    refresh_place_data_if_changed(False)
    row = find_place_exact_for_route(place)
    if row is None:
        row = _find_place(place)
    if row is None:
        return {
            "status": "unknown",
            "level": "info",
            "has_seat": None,
            "seat_count": 0,
            "queue_minutes": 0,
            "message": f"{place} 暂未查到实时余位，建议出发前二次核验。",
        }
    availability_kind = row_availability_kind(row)
    if availability_kind == "unknown":
        return {
            "status": "unknown",
            "level": "info",
            "has_seat": None,
            "seat_count": 0,
            "queue_minutes": 0,
            "message": f"{place} 是地图补充候选，暂未接入实时余位，不按“有余位”展示，建议出发前二次核验。",
        }
    seat_count = row_seat_count(row)
    has_seat = availability_kind == "available"
    queue_minutes = estimate_queue_minutes(row)
    if not has_seat:
        return {
            "status": "no_seat",
            "level": "warning",
            "has_seat": False,
            "seat_count": seat_count,
            "queue_minutes": queue_minutes,
            "message": f"{place} 当前余位状态显示暂无余位。",
        }
    if queue_minutes >= _safe_int(os.getenv("QUEUE_LONG_WARNING_MINUTES", "45"), 45):
        return {
            "status": "queue_long",
            "level": "warning",
            "has_seat": True,
            "seat_count": seat_count,
            "queue_minutes": queue_minutes,
            "message": f"{place} 有余位，但预计排队约 {queue_minutes} 分钟，可能偏久。",
        }
    return {
        "status": "available",
        "level": "success",
        "has_seat": True,
        "seat_count": seat_count,
        "queue_minutes": queue_minutes,
        "message": f"{place} 当前有余位（剩余 {seat_count}），可正常前往。",
    }


def parse_time_token_to_minutes(token: str) -> Optional[int]:
    text = str(token or "").strip()
    match = re.search(r"(\d{1,2})(?::(\d{2}))?", text)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    if hour > 24 or minute >= 60:
        return None
    if hour == 24:
        hour, minute = 23, 59
    return hour * 60 + minute


def parse_open_intervals(open_text: str) -> list[tuple[int, int]]:
    """Parse common Amap opening-hour strings into minute intervals."""
    text = str(open_text or "").strip()
    if not text:
        return []
    if any(token in text for token in ["全天", "24小时", "00:00-23:59", "00:00-24:00"]):
        return [(0, 23 * 60 + 59)]
    if any(token in text for token in ["暂停", "休息", "不营业", "关闭", "歇业"]):
        return []
    intervals = []
    normalized = text.replace("－", "-").replace("—", "-").replace("~", "-").replace("至", "-")
    for start, end in re.findall(r"(\d{1,2}(?::\d{2})?)\s*-\s*(\d{1,2}(?::\d{2})?)", normalized):
        s = parse_time_token_to_minutes(start)
        e = parse_time_token_to_minutes(end)
        if s is None or e is None:
            continue
        if e <= s:
            intervals.append((s, 23 * 60 + 59))
            intervals.append((0, e))
        else:
            intervals.append((s, e))
    return intervals


def slot_minutes(slot: str) -> Optional[tuple[int, int]]:
    match = re.match(r"\s*(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})", str(slot or ""))
    if not match:
        return None
    start = parse_time_token_to_minutes(match.group(1))
    end = parse_time_token_to_minutes(match.group(2))
    if start is None or end is None:
        return None
    return start, end


def open_intervals_cover_slot(intervals: list[tuple[int, int]], slot: str) -> bool:
    parsed_slot = slot_minutes(slot)
    if not parsed_slot or not intervals:
        return False
    start, end = parsed_slot
    return any(start >= s and end <= e for s, e in intervals)


def best_business_poi_for_place(place: str) -> Optional[dict]:
    pois = amap_search_place_business(place, limit=3)
    if not pois:
        return None
    exact = choose_best_poi_for_place(place, pois)
    return exact or pois[0]


def check_place_open_for_slot(place: str, slot: str) -> dict:
    """Return whether Amap provided enough opening-hour data and whether the slot fits."""
    if os.getenv("ENABLE_AMAP_OPENING_HOURS", "1") != "1":
        return {"status": "disabled", "message": "营业时间约束未开启。"}
    poi = best_business_poi_for_place(place)
    if not poi:
        return {"status": "unknown", "message": f"{place} 未查询到高德营业时间，出行前需二次核验。"}
    open_text = str(poi.get("opentime_today") or poi.get("opentime_week") or "").strip()
    if not open_text:
        return {
            "status": "unknown",
            "source": poi.get("source", "amap_place_text_v5"),
            "poi_name": poi.get("name", place),
            "message": f"{place} 高德未返回营业时间字段，未做硬替换，出行前需核验。",
        }
    intervals = parse_open_intervals(open_text)
    if not intervals:
        return {
            "status": "unknown",
            "source": poi.get("source", "amap_place_text_v5"),
            "poi_name": poi.get("name", place),
            "opening_hours": open_text,
            "message": f"{place} 高德营业时间为“{open_text}”，格式无法稳定解析，出行前需核验。",
        }
    if open_intervals_cover_slot(intervals, slot):
        return {
            "status": "open",
            "source": poi.get("source", "amap_place_text_v5"),
            "poi_name": poi.get("name", place),
            "opening_hours": open_text,
            "message": f"{place} 在计划时间 {slot} 内高德营业时间匹配：{open_text}。",
        }
    return {
        "status": "closed",
        "source": poi.get("source", "amap_place_text_v5"),
        "poi_name": poi.get("name", place),
        "opening_hours": open_text,
        "message": f"{place} 计划时间 {slot} 与高德营业时间“{open_text}”冲突。",
    }


def apply_schedule_business_hour_checks(
        schedule: list[dict],
        state: AgentState,
        intent: dict,
        cluster_info: dict,
) -> tuple[list[dict], list[dict], list[str], dict]:
    """Check Amap opening hours after 10km route-radius constraints."""
    if os.getenv("ENABLE_AMAP_OPENING_HOURS", "1") != "1":
        return schedule, [], ["营业时间约束未开启。"], {"enabled": False}
    anchor = str((cluster_info or {}).get("anchor") or "").strip()
    radius_m = _safe_int((cluster_info or {}).get("radius_m") or os.getenv("ROUTE_CLUSTER_RADIUS_METERS", "10000"), 10000)
    checked_schedule = []
    events = []
    notes = ["营业时间约束：已在10km范围约束之后查询高德 POI 2.0 营业时间；能解析到冲突时尝试替换为锚点附近营业候选。"]
    checked = []
    replacements = []
    used_keys = {normalize_place_text(str(item.get("place") or "")) for item in schedule or [] if isinstance(item, dict)}
    for item in schedule or []:
        if not isinstance(item, dict):
            continue
        current = dict(item)
        place = str(current.get("place") or "").strip()
        slot = str(current.get("time") or "")
        status = check_place_open_for_slot(place, slot)
        current["business_hours_status"] = status
        checked.append({"place": place, "time": slot, **status})
        if status.get("status") == "closed" and anchor:
            used_keys.discard(normalize_place_text(place))
            replacement = replace_place_near_anchor(anchor, place, intent, used_keys, radius_m, require_open_slot=slot, state=state)
            if replacement:
                new_place, why = replacement
                new_status = check_place_open_for_slot(new_place, slot)
                current = set_schedule_place(current, new_place)
                current["replaced_from"] = place
                current["replacement_reason"] = "营业时间冲突"
                current["business_hours_status"] = new_status
                used_keys.add(normalize_place_text(new_place))
                msg = f"营业时间替换：{status.get('message')} 已替换为{why}，并重新校验营业状态。"
                event = build_availability_event(place, "business_hours_replaced", msg, backup=new_place, level="warning")
                events.append(event)
                notes.append(msg)
                replacements.append({"from": place, "to": new_place, "time": slot, "reason": msg})
            else:
                msg = f"营业时间冲突未替换：{status.get('message')} 10km内暂未找到可解析为营业中的同类型候选。"
                events.append(build_availability_event(place, "business_hours_conflict", msg, level="warning"))
                notes.append(msg)
                used_keys.add(normalize_place_text(place))
        else:
            if status.get("status") == "open":
                notes.append(status.get("message", "营业时间已校验。"))
            elif status.get("status") == "unknown":
                notes.append(status.get("message", "高德未返回营业时间，需核验。"))
            used_keys.add(normalize_place_text(str(current.get("place") or place)))
        checked_schedule.append(current)
    return checked_schedule, events, unique_preserve_order(notes), {
        "enabled": True,
        "source": "amap_place_text_v5_business",
        "priority_after": "route_cluster_radius_10km",
        "checked": checked,
        "replacements": replacements,
        "summary": "已查询高德 POI 营业时间；可解析冲突会尝试在10km范围内替换，未返回/不可解析的营业时间会展示为需核验。",
    }


def apply_schedule_availability_checks(schedule: list[dict], state: AgentState, intent: dict) -> tuple[list[dict], list[dict], list[str]]:
    """对最终 schedule 做余位/排队检查，并在必要时替换为可用备选。"""
    checked_schedule = []
    events = []
    notes = []
    used = set()
    for item in schedule or []:
        if not isinstance(item, dict):
            continue
        current = dict(item)
        place = str(current.get("place") or "").strip()
        status = availability_status_for_place(place)
        if should_replace_unavailable_place(place, status, state, intent):
            backup = find_available_alternative_for_unavailable(place, intent, state)
            if backup and normalize_place_text(backup) not in used and not same_route_place(backup, place):
                level_reason = "暂无余位" if status["status"] == "no_seat" else f"排队约{status['queue_minutes']}分钟"
                event = build_availability_event(
                    place,
                    status["status"],
                    f"{place} 当前{level_reason}，已为你切换为附近同类型备选“{backup}”。",
                    backup=backup,
                    level="warning",
                )
                events.append(event)
                notes.append(event["message"])
                current = set_schedule_place(current, backup)
                current["replaced_from"] = place
                current["replacement_note"] = event["message"]
                current["replacement_reason"] = level_reason
                price_detail = place_price_detail(backup)
                current["price_text"] = price_detail.get("price_text", "价格待核验")
                current["price_min"] = price_detail.get("price_min", 0)
                current["price_max"] = price_detail.get("price_max", 0)
                status = availability_status_for_place(backup)
                current["availability_status"] = status
            else:
                events.append(build_availability_event(place, status["status"], status["message"], level=status["level"]))
                notes.append(status["message"])
                current["availability_status"] = status
        else:
            events.append(build_availability_event(place, status["status"], status["message"], level=status["level"]))
            current["availability_status"] = status
        used.add(normalize_place_text(str(current.get("place") or place)))
        checked_schedule.append(current)
    return checked_schedule, events, notes


CENTER_CROWD_KEYWORDS = {
    "陆家嘴", "外滩", "南京东路", "南京西路", "人民广场", "豫园", "新天地", "淮海路",
    "徐家汇", "静安寺", "武康路", "安福路", "巨鹿路", "长乐路", "愚园路", "五角场",
    "环球港", "迪士尼", "上海迪士尼", "上海博物馆", "上海动物园",
}
SUBURBAN_CROWD_KEYWORDS = {
    "松江", "松江区", "嘉定", "嘉定区", "青浦", "青浦区", "奉贤", "奉贤区",
    "金山", "金山区", "崇明", "崇明区", "临港", "南汇", "宝山", "宝山区",
    "郊野", "佘山", "朱家角", "枫泾", "海湾", "滴水湖",
}

def infer_crowd_context(structured_plan: dict, state: Optional[AgentState] = None) -> dict:
    """根据最终路线和锚点生成通用人流量提示，供前端 Tips 展示。

    这不是实时客流接口；它是基于上海常识、区域类型和周末场景的低风险提示。
    如以后接入实时热力/排队 API，只需要替换这里，不需要改前端结构。
    """
    structured_plan = structured_plan or {}
    state = state or {}
    hard = structured_plan.get("hard_constraints") or {}
    schedule = [item for item in (structured_plan.get("schedule") or []) if isinstance(item, dict)]
    text = " ".join([
        str(hard.get("planning_anchor") or ""),
        str(hard.get("area_anchor") or ""),
        str(hard.get("departure") or ""),
        str(hard.get("destination") or ""),
        " ".join(str(item.get("place") or "") for item in schedule),
        " ".join(str(item.get("display_name") or "") for item in schedule),
    ])
    central_hits = [kw for kw in CENTER_CROWD_KEYWORDS if kw and kw in text]
    suburban_hits = [kw for kw in SUBURBAN_CROWD_KEYWORDS if kw and kw in text]
    weekend_like = any(token in str(hard.get("date") or "") for token in ["周六", "周日", "周末", "星期六", "星期日", "星期天"])
    if central_hits:
        level = "high"
        label = "人流偏多"
        base = "、".join(central_hits[:2])
        tip = f"人流量提示：{base} 属于市中心/热门商圈或高热度景点，{('周末' if weekend_like else '出行当天')}可能人比较多；建议把拍照和排队型项目放在前半段，吃饭尽量提前到店或选择可预约地点。"
    elif suburban_hits:
        level = "low"
        label = "人流相对少"
        base = "、".join(suburban_hits[:2])
        tip = f"人流量提示：路线主要在 {base} 等近郊/非核心商圈活动，通常比市中心更松弛；但郊区点位间距可能更大，建议优先确认末班车、打车可达性和返程时间。"
    else:
        level = "medium"
        label = "人流中等"
        tip = "人流量提示：这条路线不是纯热门景点堆叠，预计人流中等；如果遇到节假日或临时展览，仍建议提前看营业和排队情况。"
    return {
        "level": level,
        "label": label,
        "tip": tip,
        "central_hits": central_hits[:5],
        "suburban_hits": suburban_hits[:5],
        "source": "local_rule_crowd_context",
    }


def schedule_item_need_booking(place: str) -> bool:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    row = find_place_exact_for_route(place)
    if row is None:
        row = _find_place(place)
    if row is None:
        return False
    try:
        return bool(row.get("是否需要预约", False))
    except Exception:
        return False


def _safe_story_value(value) -> str:
    """把地点表里的空值/NaN 清理成可给 LLM 使用的短文本。"""
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _first_place_row_value(row, names: list[str]) -> str:
    """兼容不同地点表列名：找到第一个存在且非空的字段。"""
    if row is None:
        return ""
    for name in names:
        try:
            value = _safe_story_value(row.get(name, ""))
        except Exception:
            value = ""
        if value:
            return value
    return ""


def build_stop_story_payload(item: dict, index: int, total: int) -> dict:
    """整理单个站点的文案素材，供 LLM 按地点独立生成种草文案。"""
    place = str(item.get("display_name") or item.get("place") or "").strip()
    row = find_place_exact_for_route(place)
    if row is None:
        row = _find_place(place)
    role = str(item.get("place_role") or place_role(place) or "")
    min_price = _first_place_row_value(row, ["最低价格", "min_price", "price_min"])
    max_price = _first_place_row_value(row, ["最高价格", "max_price", "price_max"])
    price_text = _safe_story_value(item.get("price_text", ""))
    if not price_text and (min_price or max_price):
        price_text = f"{min_price or max_price}-{max_price or min_price}元/人"
    payload = {
        "index": index + 1,
        "total": total,
        "time": _safe_story_value(item.get("time", "")),
        "activity_name": _safe_story_value(item.get("purpose") or item.get("activity") or item.get("title") or ""),
        "place": place,
        "type": role or _first_place_row_value(row, ["primary_type", "sub_type"]),
        "sub_type": _first_place_row_value(row, ["sub_type"]),
        "rating": _first_place_row_value(row, ["评分", "rating", "大众点评评分"]),
        "price_text": price_text,
        "business_hours": _first_place_row_value(row, ["营业时间", "business_hours", "开放时间", "open_hours"]),
        "tags": _first_place_row_value(row, ["search_tags", "interest_tags", "特色标签", "tags"]),
        "signature_items": _first_place_row_value(row, ["招牌菜", "推荐菜", "展览信息", "镇馆之宝", "特色项目"]),
        "review_snippets": _first_place_row_value(row, ["点评片段", "review_snippets", "reviews", "用户点评"]),
        "duration_min": item.get("duration_min", 0),
        "address": _safe_story_value(item.get("address") or _first_place_row_value(row, ["amap_address", "地址", "address"])),
        "need_booking": bool(schedule_item_need_booking(place)),
        "availability": _safe_story_value(item.get("availability_message") or item.get("status_text") or ""),
    }
    return {k: v for k, v in payload.items() if v not in ("", None, [])}





def _narrative_signature(text: str) -> str:
    """压缩文案特征，用于判断两个地点文案是否明显重复。"""
    return re.sub(r"[^\w\u4e00-\u9fff]", "", str(text or "").lower())


def _narrative_too_similar(text: str, previous_texts: list[str]) -> bool:
    """避免不同地点复用同一段话或只改很少几个字。"""
    sig = _narrative_signature(text)
    if len(sig) < 20:
        return True
    for prev in previous_texts or []:
        prev_sig = _narrative_signature(prev)
        if not prev_sig:
            continue
        if sig == prev_sig:
            return True
        overlap = len(set(sig) & set(prev_sig)) / max(1, min(len(set(sig)), len(set(prev_sig))))
        if overlap > 0.86 and (sig[:28] == prev_sig[:28] or sig[-28:] == prev_sig[-28:]):
            return True
    return False


def _narrative_matches_stop(text: str, item: dict) -> bool:
    """检查文案是否至少贴合当前地点类型，避免餐厅被写成公园散步。"""
    content = str(text or "")
    place = str(item.get("display_name") or item.get("place") or "")
    short_name = re.split(r"[（(]", place)[0].strip()
    role = str(item.get("place_role") or place_role(place) or "")
    purpose = str(item.get("purpose") or "")
    if short_name and short_name[:4] in content:
        return True
    hay = f"{place}{purpose}{role}"
    if role == "meal" or any(k in hay for k in ["餐厅", "饭", "正餐", "印度", "火锅", "小笼", "生煎", "面馆"]):
        return any(k in content for k in ["吃", "餐", "菜", "口味", "坐下来", "咖喱", "烤", "饭", "补能"])
    if role == "light_food" or any(k in hay for k in ["咖啡", "甜品", "奶茶", "面包"]):
        return any(k in content for k in ["咖啡", "饮品", "甜品", "坐一会", "休息", "补能", "下午茶"])
    if any(k in hay for k in ["公园", "森林", "草坪", "滨江", "步道", "散步"]):
        return any(k in content for k in ["公园", "散步", "树", "草坪", "慢慢走", "放松", "晒太阳", "拍照"])
    if any(k in hay for k in ["街", "路", "citywalk", "逛"]):
        return any(k in content for k in ["街", "路", "逛", "小店", "建筑", "街景", "citywalk"])
    return True


def generate_one_stop_narrative_with_llm(item: dict, index: int, total: int, hard: dict, crowd_context: dict, previous_texts: list[str]) -> str:
    """为单个地点单独调用 LLM 写文案，避免多地点批量生成时互相复制。"""
    payload = build_stop_story_payload(item, index, total)
    prompt = ChatPromptTemplate.from_template("""
你是一位资深本地生活推荐官，擅长用生动、有故事感的语言描述景点和餐厅。请根据以下行程信息，只为【当前这一个地点】生成一段详细种草文案。

当前地点 JSON：{item}
整条路线约束：{hard}
人流提示：{crowd}
前面已经写过的文案，禁止复用表达和画面：{previous}

输出格式要求：
第一行写：时间 & 活动名称 & 地点名称。
第一段写场景化描写，用温度、气味、光线、声音、人群氛围等细节营造身临其境的感觉。
第二段写核心亮点和结构化信息。
- 如果是餐厅：介绍招牌菜或推荐吃法；如果没有菜品字段，只能根据地点类型写用餐氛围，不要编造具体菜名。可写评分、人均、营业时间，但只在当前地点 JSON 明确提供这些字段时写。
- 如果是美术馆/博物馆：介绍展览、镇馆之宝或适合看的内容；没有展览字段时不要编造展名。可写评分、开放时间、票价，但只在当前地点 JSON 明确提供这些字段时写。
- 如果是游乐园/体验活动：介绍一个适合体验的项目或玩法；没有项目字段时只写玩法氛围。可写评分、开放时间、票价，但只在当前地点 JSON 明确提供这些字段时写。
- 如果是自然景点/公园/街区：介绍观景位、步行难度、拍照或休息点；没有明确数据时不要编造精确入口、路线或门票。
第三段写真实用户声音：只有当前地点 JSON 提供 review_snippets / 点评片段时，才引用一句并标注“网友说”；没有点评片段就跳过这一段。
第四段可用一句话总结为什么值得去。

整体约束：
1. 每个地点总字数控制在 200-300 个中文字符之间，语言轻松有趣，可以适当使用 emoji，但不要过度。
2. 必须贴合当前地点风格，餐厅不要写成公园，公园不要写成餐厅，咖啡甜品不要写成正餐。
3. 必须基于当前地点 JSON 和路线约束生成；字段缺失就跳过，不要编造评分、人均、营业时间、招牌菜、展览、点评、门票、团购、优惠券。
4. 禁止复用前面地点的句式、画面和例子；不同地点必须独立生成。
5. 禁止出现：这一站、第一站、最后一站、核心游玩、顺路游玩、状态OK、无需预约、当前没有查到、按普通到店/购票方式、热门时段人会比较多、建议先拍照再慢慢体验。
6. 只返回文案正文，不要 JSON，不要 Markdown 标题符号。
""")
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({
        "item": json.dumps(payload, ensure_ascii=False),
        "hard": json.dumps(hard or {}, ensure_ascii=False),
        "crowd": json.dumps(crowd_context or {}, ensure_ascii=False),
        "previous": json.dumps((previous_texts or [])[-4:], ensure_ascii=False),
    })
    text = re.sub(r"\s+", " ", str(raw or "")).strip().strip('"').strip("'")
    text = re.sub(r"^文案[:：]\s*", "", text).strip()
    return text[:320]


def parse_json_object_from_llm_text(text: str) -> dict:
    """Parse a JSON object from an LLM response that may contain fences or prose."""
    raw = re.sub(r"```(?:json)?|```", "", str(text or "")).strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(raw[start:end + 1])
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def generate_route_narratives_once_with_llm(schedule: list[dict], hard: dict, crowd_context: dict) -> dict:
    """Call the LLM once after the complete schedule exists, then split stop copy by delimiter."""
    if os.getenv("ENABLE_LLM_ROUTE_NARRATIVES", "0") != "1" or llm is None:
        return {}
    valid_items = [item for item in (schedule or []) if isinstance(item, dict)]
    if not valid_items:
        return {}
    total = len(valid_items)
    delimiter = "<<<STOP>>>"
    payload = [build_stop_story_payload(item, idx, total) for idx, item in enumerate(valid_items)]
    prompt = ChatPromptTemplate.from_template("""
你是一位资深本地生活路线文案编辑，擅长把已经确定的半日路线写成适合放在时间轴卡片里的短文案。请根据完整 schedule，为每个站点各写一段自然、有画面感的中文文案。

完整 schedule JSON：{schedule}
整条路线约束：{hard}
人流提示：{crowd}

写作风格要求：
第一句写场景化描写，用温度、气味、光线、声音、人群氛围等细节营造身临其境的感觉。
第二句写核心亮点和结构化信息。
- 如果是餐厅：介绍招牌菜或推荐吃法；如果没有菜品字段，只能根据地点类型写用餐氛围，不要编造具体菜名。可写评分、人均、营业时间，但只在当前地点 JSON 明确提供这些字段时写。
- 如果是美术馆/博物馆：介绍展览、镇馆之宝或适合看的内容；没有展览字段时不要编造展名。可写评分、开放时间、票价，但只在当前地点 JSON 明确提供这些字段时写。
- 如果是游乐园/体验活动：介绍一个适合体验的项目或玩法；没有项目字段时只写玩法氛围。可写评分、开放时间、票价，但只在当前地点 JSON 明确提供这些字段时写。
- 如果是自然景点/公园/街区：介绍观景位、步行难度、拍照或休息点；没有明确数据时不要编造精确入口、路线或门票。
第三句可用一句话总结为什么值得去。

硬性要求：
1. 只调用当前输入 JSON 中已有的信息；不要编造评分、人均、营业时间、招牌菜、展览名、门票、团购券、优惠券。
2. 每个站点只写对应地点，不要把 A 地点内容写到 B 地点。
3. 每段 80-100 个中文字符，适合放在前端时间轴卡片里。
4. 每段内部不允许换行。
5. 文案可以轻松、有生活感，可以适当使用 emoji，但不要过度。
6. 禁止出现：这一站、第一站、最后一站、核心游玩、顺路游玩、状态OK、无需预约、当前没有查到、按普通到店方式。
7. 只返回 N 段文案。
8. 每段对应一个站点，并严格按照 schedule 顺序输出。
9. 段与段之间只用一个 <<<STOP>>> 分隔。
10. 不要编号，不要标题，不要 Markdown，不要 JSON。
""")
    try:
        raw = (prompt | llm | StrOutputParser()).invoke({
            "schedule": json.dumps(payload, ensure_ascii=False),
            "hard": json.dumps(hard or {}, ensure_ascii=False),
            "crowd": json.dumps(crowd_context or {}, ensure_ascii=False),
        })
        result = {}
        previous_texts = []
        raw_text = re.sub(r"```(?:text)?|```", "", str(raw or "")).strip()
        parts = [re.sub(r"\s+", " ", part).strip().strip('"').strip("'") for part in raw_text.split(delimiter)]
        parts = [part for part in parts if part]
        if len(parts) != total:
            print(f"⚠️ 一次性路线文案分段数量不匹配：期望 {total} 段，实际 {len(parts)} 段；缺失站点使用本地兜底文案。")
        for idx, text in enumerate(parts[:total]):
            item = valid_items[idx]
            if len(text) < 60 or _narrative_too_similar(text, previous_texts) or not _narrative_matches_stop(text, item):
                continue
            result[str(idx + 1)] = text[:320]
            previous_texts.append(text)
        if len(result) < total:
            missing = [str(i + 1) for i in range(total) if str(i + 1) not in result]
            print(f"⚠️ 一次性路线文案缺失/不合格站点 {missing}，这些站点使用本地兜底文案。")
        return result
    except Exception as exc:
        print(f"⚠️ 一次性路线文案生成失败，使用本地兜底文案: {exc}")
        return {}


def generate_stop_narratives_with_llm(schedule: list[dict], hard: dict, crowd_context: dict) -> dict:
    """Generate optional LLM stop copy; default is fully local and calls no LLM."""
    if os.getenv("ENABLE_LLM_STOP_NARRATIVES", "0") == "1" and llm is not None:
        result = {}
        previous_texts = []
        valid_items = [item for item in (schedule or []) if isinstance(item, dict)]
        total = len(valid_items)
        for idx, item in enumerate(valid_items):
            try:
                text = generate_one_stop_narrative_with_llm(
                    item,
                    idx,
                    total,
                    hard,
                    crowd_context,
                    previous_texts,
                )
                text = re.sub(r"\s+", " ", str(text or "")).strip().strip('"').strip("'")
                if len(text) < 35 or _narrative_too_similar(text, previous_texts) or not _narrative_matches_stop(text, item):
                    print(f"⚠️ 单站 LLM 文案不合格，站点 {idx + 1} 使用本地兜底文案。")
                    continue
                result[str(idx + 1)] = text[:320]
                previous_texts.append(text)
            except Exception as exc:
                print(f"⚠️ 单站 LLM 文案生成失败，站点 {idx + 1} 使用本地兜底文案: {exc}")
        return result
    if os.getenv("ENABLE_LLM_ROUTE_NARRATIVES", "0") != "1":
        return {}
    return generate_route_narratives_once_with_llm(schedule, hard, crowd_context)


def stop_card_narrative(item: dict, index: int, total: int, crowd_context: dict) -> str:
    """Generate Xiaohongshu-style stop copy around 120-180 Chinese chars for timeline cards."""
    place = str(item.get("display_name") or item.get("place") or "这一站").strip()
    purpose = str(item.get("purpose") or "行程地点").strip()
    role = str(item.get("place_role") or "").strip()
    name_hint = re.split(r"[（(]", place)[0].strip() or "这一站"
    duration = _safe_int(item.get("duration_min"), 0)
    duration_text = f"留个{duration}分钟左右" if duration else "时间不用卡太死"
    crowd_tail = "人多的时候就把节奏放慢一点，挑自己喜欢的角落停留就好。" if crowd_context.get("level") == "high" else "不用赶场，慢慢走反而更容易逛出舒服的感觉。"

    if role == "meal":
        copy = (
            f"🍜 {name_hint}适合把节奏先放下来，{duration_text}，坐下来好好吃点东西再继续走。"
            f"和父母同行的话，这种不赶路的安排会舒服很多，点菜也可以优先选清淡稳妥的口味。{crowd_tail}"
        )
    elif role == "light_food":
        copy = (
            f"☕ {name_hint}很适合当半路的小暂停，{duration_text}，点杯喝的、坐一会儿，刚好把体力续上。"
            f"不用把这里逛得很满，聊聊天、看看照片、顺手拍几张就很够。{crowd_tail}"
        )
    elif any(token in f"{place}{purpose}" for token in ["公园", "滨江", "散步", "citywalk", "步道", "草坪", "郊野", "森林"]):
        copy = (
            f"🌿 {name_hint}适合慢慢散心，{duration_text}，边走边看树影、街景和路边小店。"
            f"不需要每个地方都打卡，随手拍几张、累了就停一停，反而更像一个真正放松的周末。{crowd_tail}"
        )
    elif any(token in f"{place}{purpose}" for token in ["展", "美术馆", "博物馆", "艺术", "馆"]):
        copy = (
            f"🎨 {name_hint}适合慢慢看，{duration_text}，先挑最感兴趣的展区逛，不用把所有内容都塞满。"
            f"细节、光线和空间感都可以多拍几张，整段路线会更有记忆点。{crowd_tail}"
        )
    elif index == 0:
        copy = (
            f"✨ 从{name_hint}开始会比较好进入状态，{duration_text}，先熟悉一下周边，再慢慢往后走。"
            f"开头别排太紧，拍几张照片、看看环境，后面的吃饭和散步都会更舒服。{crowd_tail}"
        )
    elif index == total - 1:
        copy = (
            f"🌙 {name_hint}放在后半段很适合收住节奏，{duration_text}，逛完就可以顺势返程。"
            f"如果还有体力，可以再补几张照片；如果累了，也能直接结束，不会有被行程推着走的感觉。{crowd_tail}"
        )
    else:
        copy = (
            f"🧭 {name_hint}适合换个场景继续逛，{duration_text}，不用把周边全部走完。"
            f"抓住一个最想体验的点就好，剩下的时间留给走路、休息和临时发现的小惊喜。{crowd_tail}"
        )
    if len(copy) < 120:
        copy += " 建议把这一站当作路线里的节奏点，按现场状态灵活停留，不用为了打卡把时间卡得太死。"
    return copy[:300]

def build_route_overview_copy(structured_plan: dict, state: Optional[AgentState] = None) -> str:
    """One Xiaohongshu-style overview shown above the Itinerary timeline."""
    structured_plan = structured_plan or {}
    hard = structured_plan.get("hard_constraints") or {}
    schedule = [item for item in (structured_plan.get("schedule") or []) if isinstance(item, dict)]
    collected = (state or {}).get("collected_info") or {}
    people = parse_people_count(hard.get("num_people") or collected.get("num_people"), None) or 1
    departure_raw = hard.get("departure") or collected.get("departure") or ""
    departure = clean_anchor_for_display(departure_raw)
    if not bool(hard.get("departure_explicit") or collected.get("_departure_explicit")):
        departure = ""
    anchor = clean_anchor_for_display(
        hard.get("overview_anchor")
        or hard.get("destination")
        or collected.get("fixed_destination")
        or collected.get("location")
        or hard.get("area_anchor")
        or collected.get("area_anchor")
        or hard.get("planning_anchor")
        or (schedule[0].get("place") if schedule else "")
    ) or "上海"
    budget = str(
        hard.get("budget_estimate_text")
        or (hard.get("budget_estimate") or {}).get("text")
        or "预算待定"
    ).strip()
    date_text = str(hard.get("date") or "本周末").strip()
    time_period = str(hard.get("time_period") or "半日").strip()
    stops = len(schedule)
    start_part = f"{people}个人从{departure}出发，" if departure else f"{people}个人，"
    stop_part = f"{stops or 3}个地方" if stops else "几个地方"
    return (
        f"✨ {date_text}{time_period}就这样慢慢逛吧：{start_part}围着{anchor}走一圈，"
        f"不用做很复杂的攻略，按舒服的节奏把{stop_part}串起来。"
        f"{budget}，整体更适合边走边聊、拍拍照、累了就停一会儿。"
    )



def _safe_display_value(value, fallback: str = "未明确") -> str:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return fallback
    if any(marker in text for marker in ["保留用户明确", "重新生成", "不同路线", "当前偏好", "上一版"]):
        return fallback
    return text


def summarize_availability_for_explanation(events: list) -> str:
    """Make step 5 reflect real final availability events instead of a generic sentence."""
    clean = []
    for ev in events or []:
        if not ev:
            continue
        msg = sanitize_user_visible_inventory_text(ev.get("message") if isinstance(ev, dict) else ev).strip()
        if msg and msg not in clean:
            clean.append(msg)
    if not clean:
        return "余位、预约和团购状态已按最终路线检查；未发现阻断性满位。"
    warnings = [m for m in clean if any(k in m for k in ["暂无余位", "已满", "排队", "切换", "未接入", "待核验", "核验"])]
    ok = [m for m in clean if m not in warnings]
    picked = (warnings[:3] + ok[:2])[:4]
    return "；".join(picked)


def build_planning_explanation(structured_plan: dict, state: Optional[AgentState] = None) -> dict:
    """Build a user-facing planning rationale summary.

    这是给前端展示的“可解释规划依据”，不是模型原始隐藏推理。
    它只包含稳定、可审计的决策摘要：识别到的约束、候选筛选、路线排序、
    时间分配、预算/余位/距离校验和异常处理。
    """
    structured_plan = structured_plan or {}
    state = state or {}
    hard = structured_plan.get("hard_constraints") or {}
    collected = state.get("collected_info") or {}
    intent = state.get("intent") or {}
    schedule = [item for item in (structured_plan.get("schedule") or []) if isinstance(item, dict)]
    validation = structured_plan.get("route_logic_validation") or {}
    feasibility = structured_plan.get("feasibility_report") or state.get("feasibility_report") or {}
    route_segments = structured_plan.get("route_segments") or []
    availability_events = structured_plan.get("availability_events") or validation.get("availability_events") or []
    exception_events = state.get("exception_events") or validation.get("exception_events") or []
    interests = hard.get("interests") or collected.get("interests") or []
    interest_notes = structured_plan.get("interest_match_notes") or hard.get("interest_match_notes") or build_interest_match_notes(interests, schedule)
    requirement_change_text = summarize_latest_requirement_changes(state)
    budget_estimate = structured_plan.get("budget_estimate") or hard.get("budget_estimate") or {}
    budget_text = hard.get("budget_estimate_text") or budget_estimate.get("text") or hard.get("budget") or "预算待核验"
    explicit_departure = bool(hard.get("departure_explicit") or collected.get("_departure_explicit"))
    departure = _safe_display_value(hard.get("departure") or collected.get("departure") or intent.get("departure"), "未指定") if explicit_departure else "未指定"
    anchor = _safe_display_value(
        hard.get("overview_anchor")
        or hard.get("destination")
        or collected.get("fixed_destination")
        or collected.get("location")
        or hard.get("area_anchor")
        or collected.get("area_anchor")
        or hard.get("planning_anchor")
        or (schedule[0].get("place") if schedule else ""),
        "上海"
    )
    places = [str(item.get("display_name") or item.get("place") or "").strip() for item in schedule]
    places = [p for p in places if p]
    place_line = " → ".join(places[:5]) if places else "待生成"

    duration_samples = []
    for item in schedule[:4]:
        name = str(item.get("display_name") or item.get("place") or "").strip()
        dur = item.get("duration_min")
        reason = str(item.get("duration_reason") or "按地点类型动态分配停留时间").strip()
        if name:
            duration_samples.append(f"{name}：约{dur}分钟（{reason}）" if dur else f"{name}：{reason}")

    distance_line = "已生成转场参考" if route_segments else "高德距离暂未完整返回，前端仅展示已核验片段，不编造距离"
    if route_segments:
        summaries = [str(seg.get("summary") or "").strip() for seg in route_segments if isinstance(seg, dict) and seg.get("summary")]
        if summaries:
            distance_line = "；".join(summaries[:3])

    conflict_messages = []
    conflict_messages.extend(str(x) for x in (validation.get("adjustment_conflicts") or []) if x)
    conflict_messages.extend(str(x) for x in (feasibility.get("warnings") or []) if x)
    for event in list(exception_events or []) + list(availability_events or []):
        if not event:
            continue
        if isinstance(event, dict):
            if event.get("display_level") not in {"warning", "error"} and not event.get("backup"):
                continue
            message = str(event.get("message") or "").strip()
        else:
            message = str(event or "").strip()
        if message:
            conflict_messages.append(message)
    conflict_messages = unique_preserve_order([
        msg for msg in conflict_messages
        if msg and not any(ok_text in msg for ok_text in ["当前有余位", "可正常前往", "状态OK"])
    ])
    conflict_line = (
        "本次风险/冲突：" + "；".join(conflict_messages[:3])
        if conflict_messages
        else "本次未发现营业时间、余位、距离或路线约束冲突；出发前仍建议核验实时交通和门店状态。"
    )

    steps = [
        {
            "title": "1. 需求识别",
            "detail": (
                f"识别到人数 {hard.get('num_people') or collected.get('num_people') or '未明确'}，"
                f"时间 {hard.get('date') or collected.get('date') or '本周末'} {hard.get('time_period') or collected.get('time_period') or ''}，"
                f"出发地 {departure}，规划锚点 {anchor}。"
                + (f" 本轮按最新群聊意见替换：{requirement_change_text}。" if requirement_change_text else "")
            ),
        },
        {
            "title": "2. 候选筛选",
            "detail": (
                f"优先围绕“{anchor}”筛选具体 POI，过滤旧目的地、不可用地点和过泛区域词；"
                f"兴趣偏好：{('、'.join(map(str, interests)) if interests else '未额外选择')}。"
            ),
        },
        {
            "title": "3. 路线排序",
            "detail": (
                f"当前路线为：{place_line}。排序时保留用户明确地点，并避免连续出现同一类型地点；"
                "同时避免同一方案中重复出现相同小类型地点。"
            ),
        },
        {
            "title": "4. 时间分配",
            "detail": "；".join(duration_samples) if duration_samples else "按地点类型、总时长和路线强度动态分配停留时间。",
        },
        {
            "title": "5. 预算与状态校验",
            "detail": (
                f"预算按最终路线逐站价格估算：{budget_text}。"
                f"{summarize_availability_for_explanation(availability_events)}"
            ),
        },
        {
            "title": "6. 转场与风险提示",
            "detail": conflict_line,
        },
    ]

    return {
        "title": "方案生成逻辑",
        "disclaimer": "这里展示的是可解释的规划依据摘要，不是模型原始隐藏推理。",
        "summary": f"本次方案先锁定最新出发地/目的地，再按兴趣、地点类型、预算、余位和转场可行性筛选路线。",
        "steps": steps,
        "interest_match_notes": interest_notes[:4] if isinstance(interest_notes, list) else [],
    }

def enrich_structured_plan_ui_fields(structured_plan: dict, state: Optional[AgentState] = None) -> dict:
    """给前端补充展示字段：人流提示、预约策略、每站叙事。

    这些字段只增强 UI，不改变原有 route / coupon / reservation 核心能力。
    """
    structured_plan = dict(structured_plan or {})
    schedule = [dict(item) for item in (structured_plan.get("schedule") or []) if isinstance(item, dict)]
    crowd_context = infer_crowd_context({**structured_plan, "schedule": schedule}, state)
    hard_for_budget = dict(structured_plan.get("hard_constraints") or {})
    budget_estimate = estimate_schedule_budget(schedule, hard_for_budget.get("num_people") or (state or {}).get("collected_info", {}).get("num_people"), hard_for_budget.get("budget", ""))
    generated_narratives = generate_stop_narratives_with_llm(schedule, hard_for_budget, crowd_context)
    collected = (state or {}).get("collected_info") or {}
    latest_anchor = clean_anchor_for_display(
        collected.get("fixed_destination")
        or hard_for_budget.get("destination")
        or collected.get("location")
        or collected.get("area_anchor")
        or hard_for_budget.get("area_anchor")
        or hard_for_budget.get("planning_anchor")
        or (schedule[0].get("place") if schedule else "")
    )
    if latest_anchor:
        hard_for_budget["overview_anchor"] = latest_anchor
    interests = hard_for_budget.get("interests") or collected.get("interests") or []
    hard_for_budget["interests"] = interests
    has_reservable = False
    for idx, item in enumerate(schedule):
        place = str(item.get("place") or "").strip()
        if place:
            item.update(schedule_place_type_metadata(place))
            # 修正已有 schedule 里可能沿用旧类型的问题，保证最终类型和高德标签一致。
            item["place_role"] = place_role(place)
            if item.get("display_type") and item.get("place_role") not in {"meal", "light_food"}:
                item["purpose"] = f"{item.get('display_type')}/体验" if item.get("display_type") not in {"景点", "餐饮"} else item.get("purpose", "顺路游玩/散步体验")
        need_booking = schedule_item_need_booking(place)
        item["need_booking"] = bool(need_booking)
        has_reservable = has_reservable or bool(need_booking)
        if place:
            reservation_info = build_reservation_info(place)
            item["can_reserve"] = bool(reservation_info.get("need_booking", False))
            item["has_coupon"] = bool(reservation_info.get("has_coupon", False))
            item["coupon_discount"] = reservation_info.get("discount", "暂无团购券")
            item["seat_count"] = reservation_info.get("seat_count", 0)
            item["has_seat"] = reservation_info.get("has_seat", False)
            item["queue_minutes"] = reservation_info.get("queue_minutes", 0)
        item["crowd_hint"] = crowd_context.get("tip", "")
        gen_text = generated_narratives.get(str(idx + 1)) or generated_narratives.get(str(idx))
        if gen_text and (
            _narrative_too_similar(gen_text, [
                schedule[j].get("narrative", "")
                for j in range(idx)
                if isinstance(schedule[j], dict)
            ])
            or not _narrative_matches_stop(gen_text, item)
        ):
            gen_text = ""

        item["narrative"] = gen_text or stop_card_narrative(item, idx, len(schedule), crowd_context)
        schedule[idx] = item
    structured_plan["schedule"] = schedule
    structured_plan["budget_estimate"] = budget_estimate
    structured_plan["route_overview_copy"] = build_route_overview_copy({**structured_plan, "schedule": schedule, "hard_constraints": {**hard_for_budget, "budget_estimate_text": budget_estimate.get("text"), "budget_estimate": budget_estimate}}, state)
    structured_plan["crowd_context"] = crowd_context
    structured_plan["crowd_tip"] = crowd_context.get("tip", "")
    structured_plan["interest_match_notes"] = build_interest_match_notes(interests, schedule)
    structured_plan["planning_explanation"] = build_planning_explanation(structured_plan, state)
    structured_plan["has_reservable_places"] = bool(has_reservable)
    structured_plan["reservation_prompt_policy"] = "show" if has_reservable else "hide"
    hard = dict(structured_plan.get("hard_constraints") or {})
    user_budget = hard.get("requested_budget") or hard.get("budget") or budget_estimate.get("requested_budget", "")
    hard["requested_budget"] = user_budget
    hard["budget_estimate"] = budget_estimate
    hard["budget_estimate_text"] = budget_estimate.get("text") or "预算待核验"
    hard["budget"] = user_budget or hard.get("budget") or ""
    hard["route_overview_copy"] = structured_plan.get("route_overview_copy", "")
    hard["overview_anchor"] = hard_for_budget.get("overview_anchor", "")
    hard["interests"] = interests
    hard["interest_match_notes"] = structured_plan.get("interest_match_notes", [])
    hard["crowd_context"] = crowd_context
    hard["planning_explanation"] = structured_plan.get("planning_explanation", {})
    hard["has_reservable_places"] = bool(has_reservable)
    hard["reservation_prompt_policy"] = structured_plan["reservation_prompt_policy"]
    structured_plan["hard_constraints"] = hard
    return structured_plan


def resolved_core_place_for_schedule(intent: dict, anchor_name: str, raw_places: list[str]) -> str:
    """Return a tool-resolved POI that must remain visible in the strict schedule."""
    amap_candidate = intent.get("amap_poi_candidate") or {}
    preferred_core_place = str(amap_candidate.get("name") or intent.get("location") or "").strip()
    if (
            intent.get("explicit_place_match") is True
            and preferred_core_place
            and not is_category_like_location(preferred_core_place)
            and not is_area_anchor_schedule_self(preferred_core_place, anchor_name)
            and not any(same_route_place(preferred_core_place, place) for place in raw_places)
    ):
        return preferred_core_place
    return ""


def build_structured_plan(state: AgentState) -> AgentState:
    """生成前端时间轴使用的结构化方案。

    这是前端渲染的核心数据源：places/schedule/hard_constraints/group_considerations
    都在这里确定。最终文案不能反过来覆盖 structured_plan。
    """
    """Build the final schedule, route validation, and user-visible conflict events."""
    intent = dict(state.get("intent", {}) or {})
    if state.get("route_variant_seed"):
        intent["route_variant_seed"] = state.get("route_variant_seed")
    collected = dict(state.get("collected_info", {}) or {})
    for container in (collected, intent):
        for key in ["departure", "fixed_departure", "location", "fixed_destination", "active_destination_anchor", "center_anchor"]:
            if is_placeholder_route_place(container.get(key, "")):
                container.pop(key, None)
    state = {**state, "collected_info": collected, "intent": intent}
    if not (collected.get("ordered_steps") or intent.get("ordered_steps")):
        ordered_hint = extract_ordered_steps_hint_from_user_text(
            " ".join([str(state.get("latest_user_input") or ""), str(state.get("user_input") or "")])
        )
        if ordered_hint:
            collected["ordered_steps"] = ordered_hint
            collected["place_keywords"] = [step.get("keyword") for step in ordered_hint if step.get("keyword")]
            intent["ordered_steps"] = ordered_hint
            intent["place_keywords"] = list(collected.get("place_keywords") or [])
    weather_info = state.get("weather_info") or {}
    route_plan = state.get("route_plan", "") or ""
    coupon_info = state.get("coupon_info") or {}
    locked_places = clean_route_place_candidates(state.get("locked_places") or [])
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
    elif anchor_mode in {"area", "nearby"}:
        route_logic_mode = "nearby_area_anchor"
    elif departure_explicit:
        route_logic_mode = "single_anchor_departure"
    else:
        route_logic_mode = "open_default"
    area_anchor_mode = anchor_mode in {"area", "nearby"}
    departure_only_mode = bool(collected.get("_departure_explicit")) and not is_destination_anchor_intent(intent,
                                                                                                          collected) and not area_anchor_mode
    generation_anchor, generation_anchor_mode = default_route_anchor_for_generation(
        state, intent, collected, anchor_name, anchor_mode
    )
    if ordered_search_specs_from_intent(intent):
        state, geofence_notes = prefetch_ordered_step_geofence_cache(
            state,
            generation_anchor,
            planning_context,
            _safe_int(os.getenv("ROUTE_CLUSTER_RADIUS_METERS", "10000"), 10000),
        )
    else:
        geofence_notes = []
    if area_anchor_mode:
        raw_places = []
        route_plan_candidates = []
        local_pool_places = nearby_existing_places_from_local_pool(
            anchor_name,
            [anchor_name],
            planning_context,
            limit=min_route_stops,
        )
        raw_places.extend(local_pool_places)
        route_plan_places = []
        if len(raw_places) < min_route_stops:
            route_plan_places = nearby_route_plan_places(
                anchor_name,
                route_plan_candidates,
                raw_places + [anchor_name],
                limit=max(0, min_route_stops - len(raw_places)),
            )
            raw_places.extend(route_plan_places)
        amap_places = []
        if len(raw_places) < min_route_stops and sync_amap_complement_enabled() and not has_ordered_specs:
            needed = max(0, min_route_stops - len(raw_places))
            amap_places = find_nearby_complement_places(anchor_name, raw_places + [anchor_name], planning_context, limit=needed)
            raw_places.extend(amap_places[:needed])
        structure_anchor_note = (
            f"用户表达的是“{anchor_name}”附近/区域需求；本次只把它作为搜索锚点，"
            "最终路线必须由周边具体小地点组成，不把该区域名直接塞进 schedule。"
        )
        if local_pool_places or route_plan_places or amap_places:
            structure_anchor_note += f" 已找到候选：{'、'.join(local_pool_places + route_plan_places + amap_places)}。"
    elif departure_only_mode:
        raw_places = unique_preserve_order(list(locked_places))
        route_plan_candidates = []
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
        if len(raw_places) < min_route_stops and sync_amap_complement_enabled() and not has_ordered_specs:
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
        route_plan_candidates = []
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
        if len(raw_places) < max(1, min_route_stops - 1) and sync_amap_complement_enabled() and not has_ordered_specs:
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
            list(locked_places) + ([] if is_category_like_location(intent.get("location")) else [intent.get("location", "")]))
        raw_places = clean_route_place_candidates(raw_places)
        if len(raw_places) < min_route_stops:
            raw_places.extend(nearby_existing_places_from_local_pool(
                generation_anchor,
                raw_places,
                planning_context,
                limit=max(0, min_route_stops - len(raw_places)),
            ))
        if len(raw_places) < min_route_stops and sync_amap_complement_enabled() and not has_ordered_specs:
            raw_places.extend(find_nearby_complement_places(
                generation_anchor,
                raw_places,
                planning_context,
                limit=max(0, min_route_stops - len(raw_places)),
            ))
        structure_anchor_note = f"未检测到明确目的地/出发地，已按默认锚点“{generation_anchor}”附近优先规划。"

    # A generic preference such as "想喝咖啡" can become one concrete nearby POI.
    # Keep that resolved core place in the strict schedule candidate pool even
    # when local/RAG candidates have already filled min stops.
    preferred_core_place = resolved_core_place_for_schedule(intent, anchor_name, raw_places)
    if preferred_core_place and anchor_mode in {"destination", "area", "nearby"}:
        raw_places = [preferred_core_place] + raw_places
        structure_anchor_note += f" 已保留工具解析出的核心地点“{preferred_core_place}”。"
    ordered_anchor = generation_anchor if generation_anchor else ""
    ordered_step_count = len(ordered_search_specs_from_intent(intent))
    if ordered_step_count and ordered_anchor:
        ordered_places = find_requested_keyword_complement_places(
            ordered_anchor,
            raw_places + [anchor_name],
            planning_context,
            limit=min(max_route_stops, ordered_step_count),
            state=state,
        )
        if ordered_places:
            raw_places = unique_preserve_order(ordered_places + raw_places)
            structure_anchor_note += f" 已围绕“{ordered_anchor}”按用户有序活动步骤补齐：{'、'.join(ordered_places)}。"
    if geofence_notes:
        structure_anchor_note += " " + " ".join(geofence_notes)
    raw_places = filter_replaced_destinations_from_places(clean_route_place_candidates(raw_places), state, anchor_name)

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
        elif route_logic_mode == "nearby_area_anchor":
            structure_anchor_note = (
                f"用户明确的是“{anchor_name}”附近/片区；系统围绕它找具体 POI，锚点本身不作为行程站点。"
                f"{structure_anchor_note}"
            )
        else:
            structure_anchor_note = (
                f"已按默认锚点“{generation_anchor or anchor_name or '上海人民广场'}”生成附近候选；"
                "餐饮/咖啡候选不再使用固定名单，并会清理连续咖啡或连续正餐。"
            )
    if anchor_mode != "destination" and sync_amap_complement_enabled() and not has_ordered_specs and (
            len(raw_places) < 2 or os.getenv("ENABLE_STRUCTURED_COMPLEMENT_SEARCH", "0") == "1"):
        complement_anchor = generation_anchor or anchor_name or os.getenv("DEFAULT_ROUTE_CENTER_ANCHOR", "人民广场")
        raw_places.extend(find_nearby_complement_places(complement_anchor, raw_places, planning_context, limit=max(1, min_route_stops - len(raw_places))))
    places, structure_notes = sanitize_structured_places(clean_route_place_candidates(raw_places), intent)
    places = clean_route_place_candidates(places)
    places = filter_replaced_destinations_from_places(places, state, anchor_name)
    structure_notes.insert(0, structure_anchor_note)
    if len(places) < 2 and sync_amap_complement_enabled() and not has_ordered_specs:
        refill_anchor = places[-1] if places else (anchor_name or intent.get("location", ""))
        refill_places = find_nearby_complement_places(refill_anchor, places + [anchor_name], planning_context, limit=max(1, min_route_stops - len(places)))
        if refill_places:
            places, refill_notes = sanitize_structured_places(clean_route_place_candidates(places + refill_places), intent)
            places = clean_route_place_candidates(places)
            structure_notes.extend(refill_notes)
            structure_notes.append(
                f"清理连续正餐/重复地点后路线不足，已围绕“{refill_anchor}”补充近距离候选：{'、'.join(refill_places)}。")
    places, adjustment_notes = apply_quick_adjustment_to_places(places, state, intent)
    places = clean_route_place_candidates(places)
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
    places = clean_route_place_candidates(places)
    places = filter_replaced_destinations_from_places(places, state, anchor_name)
    structure_notes.extend(min_route_notes)

    if ordered_step_count > 1:
        structure_notes.append("有序活动步骤：已保留用户表达顺序，不再用距离排序打乱这些关键词地点。")
    else:
        places, soft_order_notes = soft_optimize_route_order(places, state, intent)
        places = clean_route_place_candidates(places)
        structure_notes.extend(soft_order_notes)

    places, final_food_notes = final_sanitize_route_places(places, state, intent)
    places = clean_route_place_candidates(places)
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
    places = clean_route_place_candidates(places)
    structure_notes.extend(final_food_notes)

    if not places and not sync_amap_complement_enabled():
        static_fallback = generation_anchor or anchor_name
        if static_fallback and not is_placeholder_route_place(static_fallback):
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
    if not places and sync_amap_complement_enabled() and not has_ordered_specs:
        fallback_anchor = generation_anchor or anchor_name or os.getenv("DEFAULT_ROUTE_CENTER_ANCHOR", "人民广场")
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
    places = clean_route_place_candidates(places)
    structure_notes.extend(final_food_notes)

    places, radius_notes, route_radius_constraint = enforce_destination_radius_constraint(
        places,
        state,
        intent,
        generation_anchor or anchor_name,
        generation_anchor_mode or anchor_mode,
        collected,
    )
    structure_notes.extend(radius_notes)
    places, final_food_notes = final_sanitize_route_places(places, state, intent)
    structure_notes.extend(final_food_notes)

    if area_anchor_mode and anchor_name:
        before_anchor_filter = list(places)
        places = [p for p in places if not is_area_anchor_schedule_self(p, anchor_name)]
        if len(before_anchor_filter) != len(places):
            structure_notes.append(f"已从最终路线中移除区域/附近锚点“{anchor_name}”，只保留周边具体地点。")
        if len(places) < min_route_stops:
            places, area_refill_notes = ensure_minimum_route_places(
                places, state, intent, anchor_name, anchor_mode, min_route_stops, max_route_stops
            )
            places = [p for p in places if not is_area_anchor_schedule_self(p, anchor_name)]
            structure_notes.extend(area_refill_notes)
        if not places:
            # Last resort for nearby intent: use concrete local POIs that contain the anchor word,
            # but never the bare anchor itself. This prevents “待确认地点” for inputs like “迪士尼周围”.
            anchor_key = normalize_place_text(anchor_name)
            concrete = []
            for _, row in _df.iterrows():
                name = str(row.get("placeName", "") or "").strip()
                if not name or is_area_anchor_schedule_self(name, anchor_name):
                    continue
                if anchor_key and anchor_key in normalize_place_text(name):
                    concrete.append(name)
                if len(concrete) >= min_route_stops:
                    break
            if concrete:
                places = concrete[:max_route_stops]
                structure_notes.append(f"附近锚点“{anchor_name}”周边候选不足，已使用地点表中带该锚点的具体 POI：{'、'.join(places)}。")

    route_places_for_schedule = clean_route_place_candidates(places)[:max_route_stops]
    explicit_durations = explicit_stop_minutes_for_places(route_places_for_schedule, collected, intent)
    slots = build_schedule_slots(
        collected,
        intent,
        max(1, len(route_places_for_schedule)),
        route_places_for_schedule,
        explicit_durations=explicit_durations,
    )
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
        duration_profile = place_duration_profile(place)
        slot = slots[min(index, len(slots) - 1)]
        explicit_duration = explicit_durations[index] if index < len(explicit_durations) else 0
        schedule.append({
            "time": slot,
            "duration_min": duration_minutes_from_slot(slot),
            "duration_category": duration_profile.get("category", "general"),
            "duration_reason": (
                f"用户明确指定该活动约{explicit_duration}分钟，已优先按该时长排程。"
                if explicit_duration > 0
                else duration_profile.get("reason", "已按地点类型动态分配停留时间。")
            ),
            "duration_source": "user_explicit" if explicit_duration > 0 else "dynamic_profile",
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
    schedule, business_events, business_notes, business_hours_constraint = apply_schedule_business_hour_checks(
        schedule,
        state,
        intent,
        route_radius_constraint,
    )
    if business_notes:
        structure_notes.extend(business_notes)
    schedule, availability_events, availability_notes = apply_schedule_availability_checks(schedule, state, intent)
    if availability_notes:
        structure_notes.extend(availability_notes)
    warning_availability_events = [
        e for e in availability_events
        if isinstance(e, dict) and e.get("display_level") == "warning"
    ]
    warning_business_events = [
        e for e in business_events
        if isinstance(e, dict) and e.get("display_level") == "warning"
    ]
    exception_events = unique_event_preserve_order(list(exception_events) + warning_business_events + warning_availability_events)
    schedule_exception_conflicts = [
        event_message(event)
        for event in warning_business_events + warning_availability_events
        if event_message(event)
    ]
    schedule_places = [item["place"] for item in schedule]
    places = unique_preserve_order(schedule_places)
    effective_departure = clean_anchor_for_display(collected.get("departure") or intent.get("departure") or "")
    if not bool(collected.get("_departure_explicit")):
        effective_departure = ""
    if anchor_mode == "destination" and not bool(collected.get("_departure_explicit")):
        route_start_anchor = anchor_name or (places[0] if places else "")
        structure_notes.append(
            f"用户只明确目的地，未明确出发地；系统以目的地“{route_start_anchor}”作为周边路线锚点，不在文案中伪造出发地。"
        )
    adjustment_conflicts = detect_adjustment_conflicts(places, state, intent)
    if schedule_exception_conflicts:
        adjustment_conflicts = unique_preserve_order(list(adjustment_conflicts) + schedule_exception_conflicts)
    if adjustment_conflicts:
        structure_notes.extend(adjustment_conflicts)

    route_logic_validation = {
        "ok": not adjustment_conflicts,
        "notes": structure_notes,
        "adjustment_conflicts": adjustment_conflicts,
        "exception_events": exception_events,
        "availability_events": availability_events,
        "business_hours_events": business_events,
        "route_radius_constraint": route_radius_constraint,
        "business_hours_constraint": business_hours_constraint,
        "geo_fence_poi_cache": {
            "enabled": bool((state or {}).get("geo_fence_poi_cache")),
            "notes": geofence_notes,
            "pool_count": len((state or {}).get("geo_fence_poi_cache") or {}),
        },
        "rules": [
            "structured_plan.schedule 是最终文案唯一主路线",
            "同一条 4-6 小时路线最多安排一顿正餐",
            "最终 schedule 前会再次清理连续正餐、连续咖啡/甜品/轻食以及连续同类型活动点",
            "允许正餐后接散步/咖啡/轻量体验",
            "高德 POI 真实检索到的附近地点可以进入 structured_plan，但只保存在本次运行缓存，不写入 mock 表",
            "快捷按钮必须改动 structured_plan，而不是只改文案",
            "最终路线默认至少包含 2-3 个真实地点；少走路模式也不得退化为单点方案",
            "存在目的地/区域锚点时，出发地只用于计算到第一站的转场；没有目的地时才按出发地附近生成",
            "目的地/区域/出发地/默认中心 10km 范围约束优先于营业时间约束",
            "高德 POI 2.0 返回可解析营业时间时，会在10km约束后校验营业冲突并尝试近距离替换",
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
            "budget": collected.get("budget") or intent.get("budget"),
            "companion_notes": collected.get("companion_notes") or intent.get("companion_notes"),
            "duration_hours": intent.get("duration_hours"),
            "transport_mode": collected.get("transport_mode") or intent.get("transport_mode") or "公交地铁",
            "place_type": collected.get("place_type") or intent.get("place_type"),
            "canonical_place_type": collected.get("canonical_place_type") or intent.get("canonical_place_type"),
            "category": collected.get("category") or intent.get("category"),
            "llm_sub_type": collected.get("llm_sub_type") or intent.get("llm_sub_type"),
            "canonical_sub_type": collected.get("canonical_sub_type") or intent.get("canonical_sub_type") or collected.get("sub_type") or intent.get("sub_type"),
            "search_keyword": collected.get("search_keyword") or intent.get("search_keyword"),
            "ordered_steps": sanitize_ordered_steps(collected.get("ordered_steps") or intent.get("ordered_steps") or []),
            "place_keywords": unique_preserve_order(collected.get("place_keywords") or intent.get("place_keywords") or []),
            "planning_anchor": generation_anchor or anchor_name,
            "planning_anchor_mode": generation_anchor_mode or anchor_mode,
            "area_anchor": collected.get("area_anchor") or intent.get("area_anchor"),
            "area_anchor_mode": collected.get("area_anchor_mode") or intent.get("area_anchor_mode"),
            "exclude_anchor_from_schedule": bool(collected.get("exclude_anchor_from_schedule") or intent.get("exclude_anchor_from_schedule")),
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
            "latest_requirement_changes": state.get("latest_requirement_changes") or [],
        },
        "tool_facts": {
            "attraction_info": state.get("attraction_info", ""),
            "ticket_info": state.get("ticket_info", ""),
            "route_plan": "最终路线以 structured_plan.schedule 为准：" + " -> ".join(schedule_places),
            "route_distance_info": state.get("route_distance_info", ""),
            "coupon_summary": (coupon_info or {}).get("summary", ""),
            "weather_info": weather_info.get("summary", ""),
        },
        "availability_events": availability_events,
        "notes": {
            "exception": state.get("exception"),
            "exception_events": exception_events,
            "availability_events": availability_events,
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
            "不得安排连续两顿正餐、连续两家同类正餐店或连续两个同类型活动点",
            "区域/附近锚点只用于检索周边 POI，不直接渲染成路线站点",
            "每站时间按地点类型动态分配，不平均切分",
            "余位/排队/替换原因必须通过 availability_events 暴露给前端",
        ],
    }
    structured = enrich_structured_plan_ui_fields(structured, state)
    print("🧩 结构化方案已完整生成:")
    print(json.dumps(structured, ensure_ascii=False, indent=2))
    return {**state, "structured_plan": structured}




# fast_plan_for_api_timeout 已移除：不再为了 30 秒目标强制生成简化方案。

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
        type_reason = adjacent_same_type_conflict(schedule_places[idx + 1], schedule_places[idx])
        if type_reason:
            issues.append(f"structured_plan 中存在连续同类型地点：{type_reason}")

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
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
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
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
    return {
        "nearer": "换近一点",
        "cheaper": "换便宜一点",
        "indoor": "换室内",
        "coupon": "优先有团购",
        "less_walk": "少走路",
    }.get(str(mode or ""), str(mode or "调整需求"))


def quick_mode_default_explanation(mode: str, coupon_info: dict) -> str:
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
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
    """中文说明：构建当前流程需要的结构化对象或展示内容。"""
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
    """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
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
    budget = hard.get("budget") or collected.get("budget") or "预算待定"
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
            lines.append(f"• {event_message(event)}")
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
    ])
    if structured_plan.get("has_reservable_places"):
        lines.append("出发前建议再核验营业时间、余位和实时交通；路线里有需要预约的地点，前端会只在对应站点显示预订入口。想换近一点、换便宜一点、换室内、优先有团购或少走路，直接继续说就行。")
    else:
        lines.append("出发前建议再核验营业时间、余位和实时交通；本路线没有检测到必须预约的地点，所以不额外提示预订。想换近一点、换便宜一点、换室内、优先有团购或少走路，直接继续说就行。")
    return "\n".join(lines).strip()

def has_blocking_distance_conflict(structured_plan: dict) -> bool:
    """Distance is reference-only now, so normal rendering is never blocked by a fixed km rule."""
    return False


def build_distance_conflict_plan(state: AgentState, structured_plan: dict) -> str:
    """Backward-compatible fallback; normally unused because distance no longer blocks rendering."""
    return render_fast_plan_text(state, structured_plan, state.get("coupon_info") or {})


def render_schedule_narrative_plan(state: AgentState, structured_plan: dict, coupon_info: dict) -> str:
    """Build final_plan from schedule narratives only; no extra LLM rendering."""
    collected = state.get("collected_info", {}) or {}
    hard = (structured_plan or {}).get("hard_constraints", {}) or {}
    schedule = (structured_plan or {}).get("schedule", []) or []
    num_people = hard.get("num_people") or collected.get("num_people") or "未知"
    date_text = hard.get("date") or collected.get("date") or "本周末"
    budget = hard.get("requested_budget") or collected.get("budget") or hard.get("budget") or "预算待核验"
    lines = [
        f"本次按 {num_people}人 出行规划。",
        f"{date_text}路线已按最终 schedule 生成，预算参考：{budget}。",
        "",
    ]
    validation = (structured_plan or {}).get("route_logic_validation") or {}
    conflicts = validation.get("adjustment_conflicts") or []
    if conflicts:
        lines.append("冲突/替换提示：")
        for conflict in conflicts[:4]:
            lines.append(f"• {conflict}")
        lines.append("")
    for item in schedule:
        if not isinstance(item, dict):
            continue
        time_slot = str(item.get("time") or "时间待定")
        place = str(item.get("display_name") or item.get("place") or "待确认地点")
        purpose = str(item.get("purpose") or "行程地点")
        narrative = str(item.get("narrative") or "").strip()
        address = str(item.get("address") or "").strip()
        transport = ((item.get("transport_from_previous") or {}).get("summary") or "").strip()
        lines.append(f"• {time_slot}｜{place}｜{purpose}")
        if narrative:
            lines.append(narrative)
        if address:
            lines.append(f"地址：{address}")
        if transport:
            lines.append(f"转场：{transport}")
        lines.append("")
    coupon_summary = (coupon_info or {}).get("summary") or ""
    if coupon_summary:
        lines.append(coupon_summary)
    return "\n".join(lines).strip()


def result_formatter(state: AgentState) -> AgentState:
    """Legacy node: final_plan now comes from schedule[i].narrative."""
    raise RuntimeError("result_formatter is deprecated; final_plan is rendered from schedule narratives in route_distance_planner.")
def enrich_new_places(state: AgentState) -> AgentState:
    """Do not persist generated places into mock Excel; schedule uses runtime POI records."""
    print("ℹ️ 已跳过自动入库新地点；高德/模型新地点仅作为本次运行候选参与方案。")
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
    "route_distance_planner",
}


def timed_workflow_node(node_name: str, func):
    """给 LangGraph 节点包一层耗时统计，便于定位方案生成慢在哪个节点。"""
    def wrapper(state: AgentState) -> AgentState:
        """中文说明：执行本模块中的辅助处理逻辑，服务于路线规划主流程。"""
        started = time.perf_counter()
        refresh_place_data_if_changed(False)
        result = func(state)
        elapsed = time.perf_counter() - started
        timings = dict((result or {}).get("node_timings") or (state or {}).get("node_timings") or {})
        timings[node_name] = round(elapsed, 2)
        label = "工具调用耗时" if node_name in TOOL_TIMING_NODES else "节点耗时"
        print(f"⏱️ {label} [{node_name}]: {elapsed:.2f}s")
        return {**result, "node_timings": timings}

    return wrapper


def build_plan_workflow():
    """构建主规划工作流。

    信息收集在 API 层先跑，进入这里时默认已经有 collected_info。
    节点顺序：意图归一 → 天气 → RAG → 结构化方案/最终异常检测 → 距离/轻量文案。
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("parse_intent", timed_workflow_node("parse_intent", parse_intent))
    workflow.add_node("weather_lookup", timed_workflow_node("weather_lookup", weather_lookup))
    workflow.add_node("rag_retrieval", timed_workflow_node("rag_retrieval", rag_retrieval))
    workflow.add_node("route_distance_planner", timed_workflow_node("route_distance_planner", route_distance_planner))
    workflow.add_node("build_structured_plan", timed_workflow_node("build_structured_plan", build_structured_plan))
    workflow.add_node("enrich_new_places", timed_workflow_node("enrich_new_places", enrich_new_places))

    workflow.set_entry_point("parse_intent")
    workflow.add_edge("parse_intent", "weather_lookup")
    workflow.add_edge("weather_lookup", "rag_retrieval")
    workflow.add_edge("rag_retrieval", "build_structured_plan")
    workflow.add_edge("build_structured_plan", "route_distance_planner")
    workflow.add_edge("route_distance_planner", "enrich_new_places")
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
    """中文说明：初始化模型、数据或运行时依赖。"""
    global plan_workflow, book_workflow
    plan_workflow = build_plan_workflow()
    book_workflow = build_book_workflow()
    print("✅ Workflow 构建完成")


# ==========================================
# 6. 命令行主入口
# ==========================================
from workflow.pure_utils import _safe_int, normalize_place_text, unique_preserve_order
from workflow.amap_tools import format_distance, format_duration, parse_lnglat, short_static_map_label, static_map_marker_label
from workflow.intent_parser import clamp_duration_hours
from workflow.narratives import parse_json_object_from_llm_text, truncate_text
from workflow.route_validation import has_blocking_distance_conflict


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
