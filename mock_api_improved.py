# mock_api.py
# 本地模拟工具层：
# - 从 Excel 地点表读取地点、价格、余位、预约、团购等模拟数据。
# - 提供给 agent_workflow_improved.py 的 LangChain tools：search_attraction/check_ticket/plan_route/book_order。
# - 它不是 HTTP API，而是“本地商家/景点数据库 + 模拟商家接口”。
import os
import re
import hashlib
import time
from pathlib import Path
import pandas as pd
from langchain_core.tools import tool

def _unique_files(items):
    result = []
    seen = set()
    for item in items:
        if not item:
            continue
        key = str(item)
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


# 启动时加载地点表。
# 默认只读取当前标准表 all_place_mock_cleaned.xlsx，避免同时把旧 all_place_mock.xlsx
# 合并进来造成过期地点、旧类型或旧余位混入。
# 如需显式读取其他文件，可设置 PLACE_DATA_FILE；如需临时合并旧表/影院表，可设置
# INCLUDE_FALLBACK_PLACE_FILES=1 / INCLUDE_CINEMA_MOCK=1。
def _build_excel_files() -> list[str]:
    """确定本次启动要读取哪些 Excel 地点表。"""
    explicit = os.getenv("PLACE_DATA_FILE", "").strip()
    if explicit:
        files = [explicit]
    elif Path("all_place_mock_cleaned.xlsx").exists():
        files = ["all_place_mock_cleaned.xlsx"]
    else:
        files = ["all_place_mock.xlsx"]

    if os.getenv("INCLUDE_FALLBACK_PLACE_FILES", "0") == "1":
        files.extend(["all_place_mock_cleaned.xlsx", "all_place_mock.xlsx"])
    if os.getenv("INCLUDE_CINEMA_MOCK", "0") == "1":
        files.append("cinema_mock_status.xlsx")
    return _unique_files(files)


_EXCEL_FILES = _build_excel_files()

# 每个文件对应的 sheet 名
_SHEET_MAP = {
    "all_place_mock_cleaned.xlsx": "合并地点表",
    "all_place_mock.xlsx": "合并地点表",
    "cinema_mock_status.xlsx": "places_extracted",
}

# 每个文件对应的编号前缀
_PREFIX_MAP = {
    "all_place_mock_cleaned.xlsx": "P",
    "all_place_mock.xlsx": "P",
    "cinema_mock_status.xlsx": "ZLY",
}


def _parse_price_col(val):
    """加载时统一清洗价格列"""
    if pd.isna(val):
        return 0.0
    val_str = str(val).strip()
    if not val_str or val_str in ("免费", "free", "0"):
        return 0.0
    if "/" in val_str:
        val_str = val_str.split("/")[0].strip()
    try:
        return float(val_str)
    except ValueError:
        return 0.0

def _normalize_bool(val) -> bool:
    """兼容 Excel 中的 True/False、是/否、1/0、空值。"""
    if pd.isna(val):
        return False
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in {"true", "1", "是", "有", "需要", "yes", "y"}


def _normalize_place_type(val) -> str:
    """归一到代码主流程可识别的一级类型。

    一级类型用于路线约束和粗筛：restaurant / cafe / attraction / activity /
    leisure / shopping / sports。更细的类别（milk_tea、museum、theme_park 等）
    统一放到 sub_type。这样 Excel 可以更细，后端也不会把蜜雪冰城误当正餐。
    """
    s = str(val or "").strip().lower()
    mapping = {
        # 正餐
        "餐厅": "restaurant", "美食": "restaurant", "饭店": "restaurant", "火锅": "restaurant",
        "小笼包": "restaurant", "生煎": "restaurant", "面馆": "restaurant", "韩料": "restaurant",
        "韩国料理": "restaurant", "江浙菜": "restaurant", "本帮菜": "restaurant", "restaurant": "restaurant",
        "meal": "restaurant", "hotpot": "restaurant", "xiaolongbao": "restaurant", "shengjian": "restaurant",
        "noodle": "restaurant", "jiangzhe_cuisine": "restaurant", "korean_cuisine": "restaurant",
        "japanese_cuisine": "restaurant", "western_cuisine": "restaurant", "general_food": "restaurant",
        # 轻食/咖啡/甜品/奶茶，不能再并到 restaurant
        "咖啡": "cafe", "咖啡馆": "cafe", "下午茶": "cafe", "甜品": "cafe", "面包": "cafe",
        "茶饮": "cafe", "奶茶": "cafe", "烘焙": "cafe", "cafe": "cafe", "coffee": "cafe",
        "cafe_dessert": "cafe", "milk_tea": "cafe", "bakery": "cafe", "dessert": "cafe", "light_food": "cafe",
        # 景点/文化类
        "景点": "attraction", "景区": "attraction", "展馆": "attraction", "博物馆": "attraction",
        "美术馆": "attraction", "公园": "attraction", "古镇": "attraction", "街区": "attraction",
        "寺庙": "attraction", "temple": "attraction", "museum": "attraction", "art_exhibition": "attraction",
        "theme_park": "attraction", "park": "attraction", "tourist_attraction": "attraction", "art_center": "attraction",
        # 活动体验
        "活动": "activity", "体验": "activity", "露营": "activity", "团建": "activity", "亲子": "activity",
        "电影": "activity", "影院": "activity", "影城": "activity", "二次元": "activity", "泡汤": "activity",
        "汤泉": "activity", "cinema": "activity", "activity_attraction": "activity", "entertainment_experience": "activity",
        "entertainment": "activity", "anime": "activity", "spa_relax": "activity", "ktv": "activity",
        # 购物/商圈单独一级，避免和 leisure 混在一起
        "购物": "shopping", "商场": "shopping", "商圈": "shopping", "shopping": "shopping",
        "shopping_mall": "shopping", "business_district": "shopping",
        # 休闲/户外/街区漫步
        "休闲": "leisure", "郊区": "leisure", "近郊": "leisure", "远郊": "leisure", "户外": "leisure",
        "踏青": "leisure", "散步": "leisure", "遛弯": "leisure", "街道": "leisure", "大学路": "leisure",
        "citywalk": "leisure", "城市漫步": "leisure", "outdoor": "leisure", "suburban": "leisure",
        "suburb": "leisure", "nature": "leisure", "street": "leisure", "street_walk": "leisure",
        # 运动
        "运动": "sports", "徒步": "sports", "骑行": "sports", "sports": "sports",
    }
    return mapping.get(s, s or "unknown")


SUB_TYPE_RULES = [
    ("milk_tea", ["蜜雪冰城", "奶茶", "茶饮", "甜啦啦", "bobo", "甜啦啦bobo", "喜茶", "奈雪", "茶百道", "古茗", "霸王茶姬"]),
    ("cafe", ["咖啡", "coffee", "星巴克", "% arabica", "arabica", "manner", "seesaw", "coffee cube", "kuddo", "tim hortons", "tims"]),
    ("dessert", ["甜品", "冰淇淋", "糖水", "蛋糕", "cheesecake", "芝乐坊", "布丁", "芋泥"]),
    ("bakery", ["面包", "烘焙", "西饼", "蛋糕", "贝果", "吐司", "欧包"]),
    ("hotpot", ["海底捞", "火锅", "湊湊", "凑凑", "小龙坎", "哥老官", "锅底", "涮锅"]),
    ("xiaolongbao", ["小笼", "小笼包", "南翔", "来来", "佳家汤包", "万寿斋"]),
    ("shengjian", ["生煎", "小杨生煎", "大壶春"]),
    ("noodle", ["面馆", "面", "拉面", "汤面", "小桃面馆"]),
    ("jiangzhe_cuisine", ["江浙", "杭帮", "本帮", "半亩田", "上海菜", "苏浙"]),
    ("korean_cuisine", ["韩式", "韩国", "烤肉", "部队锅", "韩料", "韩国料理"]),
    ("japanese_cuisine", ["日料", "寿司", "居酒屋", "拉面"]),
    ("western_cuisine", ["西餐", "披萨", "牛排", "德国汽车餐厅"]),
    ("temple", ["寺", "宝山寺", "龙华寺", "静安寺", "南山寺"]),
    ("street_walk", ["路", "街", "大道", "大学路", "武康路", "安福路", "多伦路", "滨江", "步道", "绿道", "citywalk", "散步", "遛弯"]),
    ("art_exhibition", ["美术馆", "艺术", "画廊", "展览", "外滩美术馆", "浦东美术馆"]),
    ("museum", ["博物馆", "纪念馆", "历史"]),
    ("cinema", ["影院", "影城", "电影", "cinema"]),
    ("anime", ["二次元", "百联zx", "animate", "谷子", "动漫"]),
    ("spa_relax", ["汤泉", "泡汤", "温泉", "浅山", "洗浴"]),
    ("theme_park", ["迪士尼", "乐园", "游乐", "欢乐谷", "海昌"]),
    ("park", ["公园", "世纪公园", "鲁迅公园", "顾村", "森林", "湿地", "郊野", "踏青", "草坪"]),
    ("shopping_mall", ["商场", "百联", "万达", "合生汇", "环球港", "印象城", "购物中心"]),
    ("business_district", ["商圈", "陆家嘴", "南京东路", "南京西路", "新天地", "人民广场"]),
]


INTEREST_TAG_RULES = {
    "美食": ["餐厅", "美食", "吃饭", "火锅", "小笼", "生煎", "面", "烤肉", "江浙", "本帮", "日料", "西餐"],
    "文化": ["博物馆", "纪念馆", "历史", "古镇", "寺", "文化", "书店", "图书馆", "老街", "石库门"],
    "购物": ["商场", "广场", "百联", "万达", "环球港", "合生汇", "购物", "市集", "商业街", "步行街", "shopping", "mall"],
    "艺术": ["美术馆", "艺术", "画廊", "展", "剧场", "音乐厅", "创意", "设计", "演艺"],
    "自然": ["公园", "森林", "湿地", "郊野", "草坪", "植物", "动物园", "湖", "滨江", "绿地", "户外"],
    "拍照": ["打卡", "外滩", "陆家嘴", "夜景", "天台", "红砖", "建筑", "小镇", "街区", "滨江", "出片"],
    "展览": ["展览", "展馆", "美术馆", "博物馆", "艺术馆", "画廊", "馆"],
    "咖啡": ["咖啡", "coffee", "manner", "seesaw", "arabica", "甜品", "下午茶", "面包", "烘焙", "奶茶", "蜜雪冰城", "milk_tea", "cafe"],
    "散步": ["散步", "citywalk", "路", "街", "步道", "滨江", "绿道", "老街", "步行街", "公园"],
    "亲子": ["亲子", "儿童", "家庭", "乐园", "动物园", "科技馆", "自然博物馆", "海洋", "迪士尼", "游乐"],
}

PACE_TAG_RULES = {
    "Relaxed": ["咖啡", "甜品", "下午茶", "电影", "影院", "汤泉", "泡汤", "书店", "室内", "休闲", "餐厅", "奶茶", "面包"],
    "Balanced": ["美食", "文化", "艺术", "散步", "公园", "商圈", "博物馆", "展览", "购物"],
    "Packed": ["公园", "乐园", "游乐", "徒步", "骑行", "城市漫步", "citywalk", "展览", "主题乐园"],
}

def _merge_tag_text(*parts) -> str:
    result = []
    for part in parts:
        for item in re.split(r"[、,，;；|/\s]+", str(part or "")):
            item = item.strip()
            if item and item.lower() not in {"nan", "none"} and item not in result:
                result.append(item)
    return "、".join(result)

def _infer_sub_type(place_name: str, place_type: str) -> str:
    """根据地点名和一级类型推断细分类，如 cafe/park/hotpot/museum。"""
    text = str(place_name or "").lower()
    normalized_type = _normalize_place_type(place_type)
    for sub_type, keywords in SUB_TYPE_RULES:
        if any(keyword.lower() in text for keyword in keywords):
            return sub_type
    return {
        "restaurant": "general_food",
        "cafe": "cafe_general",
        "attraction": "general_attraction",
        "activity": "general_activity",
        "leisure": "general_leisure",
        "shopping": "shopping_mall",
        "sports": "general_sports",
    }.get(normalized_type, "general")


def _build_search_tags(place_name: str, place_type: str, sub_type: str) -> str:
    """为地点生成搜索标签，后续本地候选筛选会用这些标签匹配用户兴趣。"""
    place_type = _normalize_place_type(place_type)
    tags = {place_type, sub_type}
    for rule_sub_type, keywords in SUB_TYPE_RULES:
        if rule_sub_type == sub_type:
            tags.update(keywords)
    name = str(place_name or "")
    if "路" in name or "街" in name:
        tags.update(["散步", "遛弯", "citywalk", "街道"])
    if place_type == "restaurant":
        tags.update(["吃饭", "美食", "餐厅", "正餐"])
    if place_type == "cafe" or sub_type in {"cafe", "milk_tea", "bakery", "dessert", "cafe_general"}:
        tags.update(["咖啡", "甜品", "奶茶", "下午茶", "轻食", "补给"])
    if place_type == "shopping" or sub_type in {"shopping_mall", "business_district"}:
        tags.update(["购物", "商场", "商圈", "逛街"])
    if sub_type in {"art_exhibition", "museum"}:
        tags.update(["看展", "艺术展", "展览", "文化"])
    if sub_type in {"park", "street_walk"}:
        tags.update(["散步", "踏青", "休闲", "户外"])
    return "、".join(sorted(str(tag) for tag in tags if tag))


def _infer_interest_tags(place_name: str, place_type: str, sub_type: str, search_tags: str = "") -> str:
    text = f"{place_name} {place_type} {sub_type} {search_tags}".lower()
    hits = []
    for tag, words in INTEREST_TAG_RULES.items():
        if any(str(w).lower() in text for w in words):
            hits.append(tag)
    place_type = _normalize_place_type(place_type)
    if not hits:
        if place_type == "restaurant":
            hits.append("美食")
        elif place_type == "cafe":
            hits.append("咖啡")
        elif place_type == "shopping":
            hits.append("购物")
        elif place_type == "attraction":
            hits.append("文化")
        elif place_type == "leisure":
            hits.append("散步")
    return "、".join(dict.fromkeys(hits))

def _infer_pace_tags(place_name: str, place_type: str, sub_type: str, search_tags: str = "") -> str:
    text = f"{place_name} {place_type} {sub_type} {search_tags}".lower()
    hits = []
    for tag, words in PACE_TAG_RULES.items():
        if any(str(w).lower() in text for w in words):
            hits.append(tag)
    if not hits:
        hits.append("Balanced")
    return "、".join(dict.fromkeys(hits))


def _is_blank_value(value) -> bool:
    """Return True for empty/NaN-like Excel cells."""
    if pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def _series_all_blank(series) -> bool:
    try:
        return series.apply(_is_blank_value).all()
    except Exception:
        return True


def _clean_text_col(val) -> str:
    if pd.isna(val):
        return ""
    text = str(val).strip()
    if text.lower() in {"nan", "none", "null", "[]"}:
        return ""
    return text


def _apply_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Bridge the current cleaned schema and older Chinese/English schemas.

    The latest all_place_mock_cleaned.xlsx intentionally keeps only:
    placeName, primary_type, sub_type, search_tags, interest_tags, pace_tags,
    amap_address, amap_location, amap_district and Chinese business columns.
    Older code still expects 地点类型, so this loader maps primary_type to 地点类型
    and keeps both in sync for backward compatibility.
    """
    alias_groups = {
        "placeID": ["Place ID", "PlaceID", "place_id", "ID", "id"],
        "placeName": ["Place name", "PlaceName", "place_name", "name", "地点名称", "名称"],
        "是否需要预约": ["need_booking", "NeedBooking", "Need booking", "需要预约", "is_booking_required"],
        "是否有余位": ["has_seat", "HasSeat", "Has seat", "有余位", "available", "is_available"],
        "余位信息": ["seat_count", "SeatCount", "Seats", "availability_count", "余位", "库存"],
        "是否有团购": ["has_group_buy", "HasGroupBuy", "Has group buy", "group_buy", "是否有优惠券", "是否有券"],
        "最低价格": ["Price min", "PlaceMin", "price_min", "min_price", "最低消费"],
        "最高价格": ["Price max", "PlaceMax", "price_max", "max_price", "最高消费"],
        "地点类型": ["Place type", "PlaceType", "place_type", "type", "类型"],
        "primary_type": ["Primary_type", "PrimaryType", "primary type", "main_type", "一级类型"],
        "sub_type": ["Subtype", "SubType", "subtype", "sub type", "二级类型"],
        "search_tags": ["SearchTags", "search tags", "search_tag", "搜索标签", "关键词"],
        "interest_tags": ["InterestTags", "interest tags", "兴趣标签"],
        "pace_tags": ["PaceTags", "pace tags", "节奏标签"],
        "amap_address": ["AMAPAddress", "AmapAddress", "amap address", "高德地址"],
        "amap_location": ["AMAPLocation", "AmapLocation", "amap location", "高德坐标", "经纬度"],
        "amap_district": ["AMAPDistrict", "AmapDistrict", "amap district", "amap_verified_district", "AMAPVerifiedDistrict", "verified_district", "adname", "高德区县", "区县"],
    }
    for canonical, aliases in alias_groups.items():
        if canonical not in df.columns:
            for alias in aliases:
                if alias in df.columns:
                    df[canonical] = df[alias]
                    break
        else:
            # Fill blank canonical cells from aliases without overwriting existing values.
            for alias in aliases:
                if alias in df.columns:
                    mask = df[canonical].apply(_is_blank_value)
                    if mask.any():
                        df.loc[mask, canonical] = df.loc[mask, alias]
    return df


def _primary_type_source(row) -> str:
    """Pick the authoritative primary type from the latest schema.

    Preference order:
    1. primary_type from the cleaned table;
    2. legacy 地点类型 / PlaceType;
    3. sub_type, mapped through _normalize_place_type;
    4. inferred sub_type from placeName.
    """
    for key in ["primary_type", "地点类型", "sub_type"]:
        value = row.get(key, "")
        if not _is_blank_value(value):
            normalized = _normalize_place_type(value)
            if normalized and normalized != "unknown":
                return normalized
    inferred_sub = _infer_sub_type(row.get("placeName", ""), "")
    normalized = _normalize_place_type(inferred_sub)
    return normalized if normalized and normalized != "unknown" else "attraction"


def stable_mock_seat_count(place_name: str) -> int:
    """给新地点稳定生成一个模拟余位数。

    同一个地点名会得到稳定结果；不同地点不会永远是 0 或 unknown，便于演示余位逻辑。
    """
    """Generate deterministic demo inventory for a map POI.

    A POI name always receives the same count across reloads. Around 20% of new
    map POIs intentionally receive zero seats so the unavailable path remains
    testable; the remaining POIs receive a positive simulated count.
    """
    digest = hashlib.sha256(str(place_name or "").encode("utf-8", errors="ignore")).hexdigest()
    seed = int(digest[:16], 16)
    if seed % 10 < 2:
        return 0
    return 8 + (seed % 173)


def _apply_amap_mock_availability(df: pd.DataFrame) -> pd.DataFrame:
    """给高德新增地点补模拟余位/预约字段，避免新地点全都显示“已满/未知”。"""
    """Backfill demo inventory for new or legacy Amap rows with unknown seats."""
    if df.empty:
        return df
    source_note = df["source_note"].fillna("").astype(str)
    status = df["availability_status"].fillna("").astype(str).str.strip().str.lower()
    seat_numeric = pd.to_numeric(df["余位信息"], errors="coerce")
    amap_mask = source_note.str.contains("高德POI搜索候选|amap_poi", na=False, regex=True)
    unknown_status = status.isin({"", "unknown", "pending", "unverified", "待核验", "未知"})
    needs_mock = amap_mask & (seat_numeric.isna() | unknown_status)
    if not needs_mock.any():
        return df

    counts = df.loc[needs_mock, "placeName"].map(stable_mock_seat_count).astype(int)
    df.loc[needs_mock, "余位信息"] = counts
    df.loc[needs_mock, "是否有余位"] = counts.gt(0)
    df.loc[needs_mock, "availability_status"] = counts.map(lambda count: "available" if count > 0 else "no_seat")
    return df


def _load_all() -> pd.DataFrame:
    """读取所有配置的地点表，并统一字段、类型、标签、余位和价格。"""
    frames = []
    required_cols = [
        "placeID", "placeName", "是否需要预约", "是否有余位", "余位信息", "是否有团购", "availability_status",
        "最低价格", "最高价格", "地点类型", "primary_type", "sub_type", "search_tags", "interest_tags", "pace_tags",
        "amap_address", "amap_location", "amap_district", "source_note",
        # kept as optional backward-compatible fields; absence should not break latest table
        "amap_verified_name", "amap_verified_status", "amap_verified_address", "amap_verified_location",
    ]

    for f in _EXCEL_FILES:
        if not Path(f).exists():
            continue
        sheet = _SHEET_MAP.get(Path(f).name, _SHEET_MAP.get(f, 0))
        df = pd.read_excel(f, sheet_name=sheet)
        df["_source_file"] = f
        df = _apply_column_aliases(df)

        for col in required_cols:
            if col not in df.columns:
                df[col] = None

        df = _apply_amap_mock_availability(df)

        # If the latest table only has verified columns, bridge them into the runtime columns.
        # If the latest table already has amap_address/location/district, keep them unchanged.
        if _series_all_blank(df["amap_address"]) and "amap_verified_address" in df.columns:
            df["amap_address"] = df["amap_verified_address"]
        if _series_all_blank(df["amap_location"]) and "amap_verified_location" in df.columns:
            df["amap_location"] = df["amap_verified_location"]

        if df["placeID"].isna().all() or _series_all_blank(df["placeID"]):
            prefix = _PREFIX_MAP.get(Path(f).name, "P")
            df["placeID"] = [f"{prefix}{idx + 1:04d}" for idx in range(len(df))]

        df["placeID"] = df["placeID"].astype(str).str.strip()
        df["placeName"] = df["placeName"].astype(str).str.strip()
        df["是否需要预约"] = df["是否需要预约"].apply(_normalize_bool)
        df["是否有余位"] = df["是否有余位"].apply(_normalize_bool)
        df["是否有团购"] = df["是否有团购"].apply(_normalize_bool)

        # 余位信息是库存测试的权威字段：明确 0 = 无余位；明确 >0 = 有余位；空值才回退布尔列。
        seat_numeric = pd.to_numeric(df["余位信息"], errors="coerce")
        df["_seat_count_explicit"] = seat_numeric.notna()
        df["余位信息"] = seat_numeric.fillna(0).astype(int)
        df.loc[df["_seat_count_explicit"] & (df["余位信息"] <= 0), "是否有余位"] = False
        df.loc[df["余位信息"] > 0, "是否有余位"] = True

        df["最低价格"] = df["最低价格"].apply(_parse_price_col)
        df["最高价格"] = df["最高价格"].apply(_parse_price_col)
        df.loc[df["最高价格"] < df["最低价格"], "最高价格"] = df["最低价格"]

        # Latest schema: primary_type is authoritative.  Legacy code: 地点类型 is what many functions read.
        # Keep both synchronized so all downstream code captures the same type.
        df["primary_type"] = df.apply(_primary_type_source, axis=1)
        df["地点类型"] = df["primary_type"]

        df["sub_type"] = df.apply(
            lambda row: str(row.get("sub_type") or "").strip()
            if str(row.get("sub_type") or "").strip() not in {"", "nan", "None"}
            else _infer_sub_type(row.get("placeName", ""), row.get("primary_type", "")),
            axis=1,
        )
        df["search_tags"] = df.apply(
            lambda row: _merge_tag_text(
                row.get("search_tags", ""),
                _build_search_tags(row.get("placeName", ""), row.get("primary_type", ""), row.get("sub_type", "")),
            ),
            axis=1,
        )
        df["interest_tags"] = df.apply(
            lambda row: _merge_tag_text(
                row.get("interest_tags", ""),
                row.get("interest_tags_refined", ""),
                _infer_interest_tags(row.get("placeName", ""), row.get("primary_type", ""), row.get("sub_type", ""), row.get("search_tags", "")),
            ),
            axis=1,
        )
        df["pace_tags"] = df.apply(
            lambda row: _merge_tag_text(
                row.get("pace_tags", ""),
                row.get("pace_tags_refined", ""),
                _infer_pace_tags(row.get("placeName", ""), row.get("primary_type", ""), row.get("sub_type", ""), row.get("search_tags", "")),
            ),
            axis=1,
        )
        # Merge refined tags back into search_tags so matching code benefits without schema changes elsewhere.
        df["search_tags"] = df.apply(
            lambda row: "、".join(dict.fromkeys(
                [x for x in str(row.get("search_tags", "")).split("、") if x] +
                [x for x in str(row.get("interest_tags", "")).split("、") if x] +
                [x for x in str(row.get("pace_tags", "")).split("、") if x]
            )),
            axis=1,
        )

        for col in ["amap_address", "amap_location", "amap_district"]:
            df[col] = df[col].apply(_clean_text_col)

        df = df[df["placeName"].ne("") & df["placeID"].ne("")]
        frames.append(df)

    if not frames:
        raise FileNotFoundError("未找到可读取的地点状态表：all_place_mock_cleaned.xlsx / all_place_mock.xlsx / cinema_mock_status.xlsx")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["placeID"], keep="first")
    combined.set_index("placeID", inplace=True)
    return combined

def _excel_signature() -> tuple:
    sig = []
    for f in _EXCEL_FILES:
        path = Path(f)
        if path.exists():
            try:
                stat = path.stat()
                sig.append((str(path.resolve()), round(stat.st_mtime, 3), stat.st_size))
            except OSError:
                sig.append((str(path), 0, 0))
    return tuple(sig)


def reload_place_data_if_changed(force: bool = False) -> bool:
    """运行中检测 Excel 是否变化；有变化时刷新 _df，避免服务不重启时读旧地点表。"""
    """Reload Excel-backed mock data after the sheet is edited.

    Without this, editing all_place_mock.xlsx while uvicorn is running leaves the
    old in-memory _df in place, so availability tests can still show stale seats.
    """
    global _df, _PLACE_DATA_SIGNATURE
    current = _excel_signature()
    if not force and current == _PLACE_DATA_SIGNATURE:
        return False
    _df = _load_all()
    _PLACE_DATA_SIGNATURE = _excel_signature()
    return True


def place_data_signature() -> str:
    reload_place_data_if_changed(False)
    return repr(_PLACE_DATA_SIGNATURE)


def _row_availability_status(row) -> str:
    """Return available / no_seat / unknown without inventing realtime inventory."""
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
        source_note = str(row.get("source_note", "") or "")
        if "高德POI搜索候选" in source_note or "amap_poi" in source_note:
            return "unknown"
        explicit = bool(row.get("_seat_count_explicit", True))
        seat_count = int(row.get("余位信息", 0) or 0)
        if explicit:
            return "available" if seat_count > 0 else "no_seat"
        if bool(row.get("是否有余位", False)) or seat_count > 0:
            return "available"
        return "unknown"
    except Exception:
        return "available" if bool(row.get("是否有余位", False)) else "unknown"


def _row_has_seat(row) -> bool:
    return _row_availability_status(row) == "available"


_df = _load_all()
_PLACE_DATA_SIGNATURE = _excel_signature()

def _compact_name(value: str) -> str:
    return re.sub(r"[\s·•\-_/（）()【】\[\]《》<>]+", "", str(value or "").lower())


def _place_aliases(place_name: str) -> list[str]:
    raw = str(place_name or "").strip()
    base = re.split(r"[（(]", raw)[0].strip()
    no_city_base = re.sub(r"^上海市?", "", base).strip()
    simplified = re.sub(r"(上海市?|火锅|餐厅|饭店|咖啡馆|咖啡|园区|公园|美术馆|博物馆|店)$", "", base).strip()
    aliases = [raw, base, no_city_base, simplified]
    if "EKA" in raw.upper():
        aliases.extend(["EKA", "EKA园区", "EKA天物空间"])

    seen = set()
    result = []
    for alias in aliases:
        key = _compact_name(alias)
        if key and key not in seen:
            seen.add(key)
            result.append(alias)
    return result


def _find_place(name: str):
    """在本地地点表中查找地点，支持别名、短名和适度模糊匹配。"""
    reload_place_data_if_changed(False)
    """按名称安全模糊匹配，返回第一条记录。

    改进点：
    - 过滤“景区/地点/上海”等泛词，避免误命中第一条；
    - 使用 re.escape，避免地点名中的括号、+、? 被当作正则；
    - 同时支持“全称包含关键词”和“关键词包含全称”的轻量匹配。
    """
    if name is None:
        return None

    raw = str(name).strip()
    compact = raw.replace(" ", "")
    generic_terms = {
        "景区", "地点", "活动", "室内", "上海", "周末", "附近", "郊区", "近郊", "远郊",
        "户外", "踏青", "公园", "休闲", "放松", "朋友", "情侣", "家庭", "独行",
        # 区域词不是具体 POI。否则“松江/浦东/嘉定”会模糊命中某个带区名的店，
        # 导致路线、团购和预约都串到错误地点。
        "黄浦", "黄浦区", "徐汇", "徐汇区", "长宁", "长宁区", "静安", "静安区",
        "普陀", "普陀区", "虹口", "虹口区", "杨浦", "杨浦区", "浦东", "浦东新区",
        "闵行", "闵行区", "宝山", "宝山区", "嘉定", "嘉定区", "金山", "金山区",
        "松江", "松江区", "青浦", "青浦区", "奉贤", "奉贤区", "崇明", "崇明区",
    }
    if not compact or compact in generic_terms:
        return None

    raw_key = _compact_name(raw)
    exact_scores = []
    for _, row in _df.iterrows():
        place_name = str(row.get("placeName", "")).strip()
        if not place_name:
            continue
        place_key = _compact_name(place_name)
        aliases = [_compact_name(alias) for alias in _place_aliases(place_name)]
        for alias_key in aliases:
            if len(alias_key) < 2 or alias_key in generic_terms:
                continue
            if raw_key == alias_key:
                exact_scores.append((1000 + len(alias_key), row))
            elif alias_key in raw_key or raw_key in alias_key:
                exact_scores.append((len(alias_key), row))
    if exact_scores:
        exact_scores.sort(key=lambda item: item[0], reverse=True)
        return exact_scores[0][1]

    pattern = re.escape(raw)
    matches = _df[_df["placeName"].astype(str).str.contains(pattern, na=False, regex=True)]
    if not matches.empty:
        return matches.iloc[0]

    # 反向包含：用户只输入了带商圈/分店的长名时，尝试找短名。
    for _, row in _df.iterrows():
        place_name = str(row.get("placeName", "")).strip()
        if place_name and place_name in raw:
            return row

    return None

@tool
def search_attraction(location: str, date: str) -> str:
    """查询景点/场馆模拟余位和预约信息，返回给 agent_workflow 的 tool_dispatch。"""
    row = _find_place(location)
    if row is None:
        return f"未找到 {location} 的相关信息"
    seat_count = int(row["余位信息"]) if pd.notna(row["余位信息"]) else 0
    availability_status = _row_availability_status(row)
    if availability_status == "available":
        has_seat = "有余位"
        seat_info = seat_count if seat_count else "余位数量待核验"
    elif availability_status == "no_seat":
        has_seat = "已满"
        seat_info = seat_count
    else:
        has_seat = "余位待核验"
        seat_info = "未接入实时余位信息"
    need_book = "需要预约" if row["是否需要预约"] else "无需预约"
    return f"{row['placeName']} | {has_seat} | {seat_info} | {need_book}"

@tool
def check_ticket(attraction: str, date: str, num_people: int) -> str:
    """查询某地点是否有团购/门票价格，并按人数估算费用。"""
    """查询门票库存与价格"""
    row = _find_place(attraction)
    if row is None:
        return f"未找到 {attraction} 的价格信息"

    def parse_price(val):
        """将价格字段统一转为数字，支持 '128/95'、'128 / 95'、纯数字等格式"""
        if pd.isna(val):
            return 0
        val_str = str(val).strip()
        if not val_str or val_str in ("0", "免费", "free"):
            return 0
        # 如果包含 '/'，取第一个数字
        if "/" in val_str:
            val_str = val_str.split("/")[0].strip()
        try:
            return float(val_str)
        except ValueError:
            return 0

    low = parse_price(row["最低价格"])
    high = parse_price(row["最高价格"])

    group_buy = "有团购" if row["是否有团购"] else "无团购"
    total_low = int(low) * num_people
    total_high = int(high) * num_people
    price_str = f"{total_low}-{total_high}元" if (total_low or total_high) else "免费"

    return f"{row['placeName']} | {group_buy} | {num_people}人预估: {price_str}"

@tool
def plan_route(attractions: str, meal_pref: str, total_hours: int) -> str:
    """生成一个轻量路线草稿；最终路线仍由 structured_plan 决定。"""
    """基于时间窗口规划路线，自动匹配同类型餐厅"""
    main_row = _find_place(attractions)
    if main_row is not None and str(main_row.get("地点类型", "")).strip() == "restaurant":
        return f"推荐路线: 用餐({main_row['placeName']}) → 周边轻量休闲/散步 → 返程, 预计总时长 {total_hours}h"

    restaurants = _df[_df["地点类型"] == "restaurant"]
    if meal_pref:
        filtered = restaurants[restaurants["placeName"].str.contains(str(meal_pref), na=False, regex=False)]
        restaurants = filtered if not filtered.empty else restaurants
    # 不再固定取表格前几家，避免每次都出现同一批咖啡/斋堂候选。
    available = restaurants[restaurants.apply(_row_has_seat, axis=1)].copy()
    if not available.empty and "团购" in str(meal_pref):
        sort_cols = [col for col in ["是否有团购", "最低价格"] if col in available.columns]
        if sort_cols:
            ascending = [False if col == "是否有团购" else True for col in sort_cols]
            available = available.sort_values(by=sort_cols, ascending=ascending)
    if not any(term in str(meal_pref).lower() for term in ["咖啡", "coffee", "cafe", "下午茶"]):
        non_cafe = available[
            ~available["placeName"].astype(str).str.contains("咖啡|coffee|cafe|COFFEE|星巴克|Arabica", na=False, regex=True)
        ]
        if not non_cafe.empty:
            available = non_cafe
    if not available.empty:
        seed_text = f"{attractions}|{meal_pref}|{total_hours}"
        offset = int(hashlib.sha256(seed_text.encode("utf-8", errors="ignore")).hexdigest()[:8], 16) % len(available)
        available = available.iloc[[offset]]
    meal_options = "、".join(available["placeName"].tolist()) if not available.empty else "附近餐饮"
    return f"推荐路线: {attractions} → 用餐({meal_options}) → 休闲, 预计总时长 {total_hours}h"

@tool
def book_order(attraction: str, date: str, num_people: int, meal: str) -> str:
    """模拟预订/下单动作，供前端预订按钮演示使用。"""
    """执行预订，检查是否需要预约"""
    row = _find_place(attraction)
    need_book = row is not None and row["是否需要预约"]
    order_id = f"ORD-{abs(hash(attraction + date)) % 100000:05d}"
    note = "⚠️ 该地点需提前预约，请确认已完成预约" if need_book else "无需预约，直接前往"
    return f"✅ 订单 {order_id} | {attraction} | {date} | {num_people}人 | 餐厅: {meal} | {note}"

def add_new_place(record: dict, target_file: str = "all_place_mock.xlsx"):
    """把自动生成的新地点写回 Excel；默认主流程关闭自动入库，避免固化模型幻觉。"""
    """将新地点追加到内存 _df 并持久化到指定 Excel 文件"""
    global _df

    target_path = Path(target_file)
    target_name = target_path.name
    prefix = _PREFIX_MAP.get(target_name, _PREFIX_MAP.get(target_file, "NEW"))
    sheet = _SHEET_MAP.get(target_name, _SHEET_MAP.get(target_file, "合并地点表"))

    # cleaned 主表启用后，动态 sidecar 默认不会加载进 _df。写回时仍要先读取
    # sidecar 自身，否则每次新增高德 POI 都会覆盖掉此前持久化的记录。
    persisted = pd.DataFrame()
    if target_path.exists():
        persisted = pd.read_excel(target_path, sheet_name=sheet)
        for col in ["source_note", "availability_status", "余位信息", "是否有余位"]:
            if col not in persisted.columns:
                persisted[col] = None
        persisted = _apply_amap_mock_availability(persisted)

    existing_ids = {str(idx).strip() for idx in _df.index}
    if "placeID" in persisted.columns:
        existing_ids.update(str(idx).strip() for idx in persisted["placeID"].dropna())
    next_index = 1
    while f"{prefix}{next_index:04d}" in existing_ids:
        next_index += 1
    new_id = f"{prefix}{next_index:04d}"

    record["placeID"] = new_id
    record["_source_file"] = target_file
    new_row = pd.DataFrame([record]).set_index("placeID")
    _df = pd.concat([_df, new_row])

    persisted_record = {key: value for key, value in record.items() if key != "_source_file"}
    subset = pd.concat([persisted, pd.DataFrame([persisted_record])], ignore_index=True)
    subset = subset.drop_duplicates(subset=["placeID"], keep="last")
    subset.to_excel(
        target_file,
        sheet_name=sheet,
        index=False
    )
    global _PLACE_DATA_SIGNATURE
    _PLACE_DATA_SIGNATURE = _excel_signature()
    print(f"📥 新地点已入库: {record['placeName']} ({new_id}) → {target_file}")
