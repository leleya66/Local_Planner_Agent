# mock_api.py
import os
import re
import hashlib
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


# 启动时加载多个文件，合并到一个 _df。默认回到原始 all_place_mock.xlsx；
# cleaned 表只在用户显式设置 PLACE_DATA_FILE 时使用。
_EXCEL_FILES = _unique_files([
    os.getenv("PLACE_DATA_FILE", "all_place_mock.xlsx"),
    "all_place_mock.xlsx",
    "cinema_mock_status.xlsx",
])

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
    s = str(val or "").strip().lower()
    mapping = {
        "景点": "attraction",
        "景区": "attraction",
        "展馆": "attraction",
        "博物馆": "attraction",
        "美术馆": "attraction",
        "公园": "attraction",
        "古镇": "attraction",
        "街区": "attraction",
        "餐厅": "restaurant",
        "美食": "restaurant",
        "饭店": "restaurant",
        "火锅": "restaurant",
        "小笼包": "restaurant",
        "生煎": "restaurant",
        "面馆": "restaurant",
        "韩料": "restaurant",
        "韩国料理": "restaurant",
        "江浙菜": "restaurant",
        "本帮菜": "restaurant",
        "面包": "restaurant",
        "甜品": "restaurant",
        "咖啡": "restaurant",
        "咖啡馆": "restaurant",
        "下午茶": "restaurant",
        "活动": "activity",
        "体验": "activity",
        "露营": "activity",
        "团建": "activity",
        "亲子": "activity",
        "电影": "activity",
        "影院": "activity",
        "影城": "activity",
        "二次元": "activity",
        "泡汤": "activity",
        "汤泉": "activity",
        "休闲": "leisure",
        "郊区": "leisure",
        "近郊": "leisure",
        "远郊": "leisure",
        "户外": "leisure",
        "踏青": "leisure",
        "散步": "leisure",
        "遛弯": "leisure",
        "街道": "leisure",
        "大学路": "leisure",
        "citywalk": "leisure",
        "城市漫步": "leisure",
        "商圈": "leisure",
        "outdoor": "leisure",
        "suburban": "leisure",
        "suburb": "leisure",
        "nature": "leisure",
        "运动": "sports",
        "徒步": "sports",
        "骑行": "sports",
        "cinema": "activity",
        "activity_attraction": "activity",
        "entertainment_experience": "activity",
        "entertainment": "activity",
        "cafe_dessert": "restaurant",
        "shopping_mall": "leisure",
        "business_district": "leisure",
        "street": "leisure",
        "park": "attraction",
        "tourist_attraction": "attraction",
        "art_center": "attraction",
        "temple": "attraction",
        "寺庙": "attraction",
    }
    return mapping.get(s, s or "unknown")


SUB_TYPE_RULES = [
    ("hotpot", ["海底捞", "火锅", "湊湊", "凑凑", "小龙坎", "哥老官", "锅底", "涮锅"]),
    ("xiaolongbao", ["小笼", "小笼包", "南翔", "来来", "佳家汤包", "万寿斋"]),
    ("shengjian", ["生煎", "小杨生煎", "大壶春"]),
    ("noodle", ["面馆", "面", "拉面", "汤面", "小桃面馆"]),
    ("cafe", ["咖啡", "coffee", "星巴克", "% arabica", "arabica", "manner", "seesaw", "coffee cube", "kuddo"]),
    ("bakery", ["面包", "烘焙", "西饼", "蛋糕", "贝果"]),
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
    ("theme_park", ["迪士尼", "乐园", "游乐"]),
    ("park", ["公园", "世纪公园", "鲁迅公园", "顾村", "森林", "湿地", "郊野", "踏青", "草坪"]),
    ("shopping", ["商场", "百联", "万达", "合生汇", "环球港", "商圈"]),
]


def _infer_sub_type(place_name: str, place_type: str) -> str:
    text = str(place_name or "").lower()
    for sub_type, keywords in SUB_TYPE_RULES:
        if any(keyword.lower() in text for keyword in keywords):
            return sub_type
    return {
        "restaurant": "general_food",
        "attraction": "general_attraction",
        "activity": "general_activity",
        "leisure": "general_leisure",
        "sports": "general_sports",
    }.get(place_type, "general")


def _build_search_tags(place_name: str, place_type: str, sub_type: str) -> str:
    tags = {place_type, sub_type}
    for rule_sub_type, keywords in SUB_TYPE_RULES:
        if rule_sub_type == sub_type:
            tags.update(keywords)
    name = str(place_name or "")
    if "路" in name or "街" in name:
        tags.update(["散步", "遛弯", "citywalk", "街道"])
    if place_type == "restaurant":
        tags.update(["吃饭", "美食", "餐厅"])
    if sub_type in {"art_exhibition", "museum"}:
        tags.update(["看展", "艺术展", "展览"])
    if sub_type in {"park", "street_walk"}:
        tags.update(["散步", "踏青", "休闲", "户外"])
    return "、".join(sorted(str(tag) for tag in tags if tag))


def _load_all() -> pd.DataFrame:
    frames = []
    required_cols = [
        "placeID", "placeName", "是否需要预约", "是否有余位", "余位信息", "是否有团购",
        "最低价格", "最高价格", "地点类型", "primary_type", "sub_type", "search_tags"
    ]

    for f in _EXCEL_FILES:
        if not Path(f).exists():
            continue
        sheet = _SHEET_MAP.get(Path(f).name, _SHEET_MAP.get(f, 0))
        df = pd.read_excel(f, sheet_name=sheet)
        df["_source_file"] = f

        column_aliases = {
            "Place ID": "placeID",
            "Place name": "placeName",
            "Price min": "最低价格",
            "Price max": "最高价格",
            "Place type": "地点类型",
        }
        for old, new in column_aliases.items():
            if new not in df.columns and old in df.columns:
                df[new] = df[old]

        for col in required_cols:
            if col not in df.columns:
                df[col] = None

        if df["placeID"].isna().all():
            prefix = _PREFIX_MAP.get(Path(f).name, "P")
            df["placeID"] = [f"{prefix}{idx + 1:04d}" for idx in range(len(df))]

        df["placeID"] = df["placeID"].astype(str).str.strip()
        df["placeName"] = df["placeName"].astype(str).str.strip()
        df["是否需要预约"] = df["是否需要预约"].apply(_normalize_bool)
        df["是否有余位"] = df["是否有余位"].apply(_normalize_bool)
        df["是否有团购"] = df["是否有团购"].apply(_normalize_bool)
        df["余位信息"] = pd.to_numeric(df["余位信息"], errors="coerce").fillna(0).astype(int)
        # 余位信息大于0时强制视为有余位，避免 Excel 中“是否有余位=无/False”但余位数为300+的矛盾。
        df.loc[df["余位信息"] > 0, "是否有余位"] = True
        df["最低价格"] = df["最低价格"].apply(_parse_price_col)
        df["最高价格"] = df["最高价格"].apply(_parse_price_col)
        df.loc[df["最高价格"] < df["最低价格"], "最高价格"] = df["最低价格"]
        df["地点类型"] = df["地点类型"].apply(_normalize_place_type)
        df["primary_type"] = df["地点类型"]
        df["sub_type"] = df.apply(
            lambda row: str(row.get("sub_type") or "").strip()
            if str(row.get("sub_type") or "").strip() not in {"", "nan", "None"}
            else _infer_sub_type(row.get("placeName", ""), row.get("地点类型", "")),
            axis=1,
        )
        df["search_tags"] = df.apply(
            lambda row: _build_search_tags(row.get("placeName", ""), row.get("地点类型", ""), row.get("sub_type", "")),
            axis=1,
        )
        df = df[df["placeName"].ne("") & df["placeID"].ne("")]
        frames.append(df)

    if not frames:
        raise FileNotFoundError("未找到可读取的地点状态表：all_place_mock_cleaned.xlsx / all_place_mock.xlsx / cinema_mock_status.xlsx")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["placeID"], keep="first")
    combined.set_index("placeID", inplace=True)
    return combined

_df = _load_all()

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
    """查询景点/场馆实时座位与排队信息"""
    row = _find_place(location)
    if row is None:
        return f"未找到 {location} 的相关信息"
    seat_count = int(row["余位信息"]) if pd.notna(row["余位信息"]) else 0
    has_seat = "有余位" if (bool(row["是否有余位"]) or seat_count > 0) else "已满"
    seat_info = seat_count if seat_count else "无余位信息"
    need_book = "需要预约" if row["是否需要预约"] else "无需预约"
    return f"{row['placeName']} | {has_seat} | {seat_info} | {need_book}"

@tool
def check_ticket(attraction: str, date: str, num_people: int) -> str:
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
    """基于时间窗口规划路线，自动匹配同类型餐厅"""
    main_row = _find_place(attractions)
    if main_row is not None and str(main_row.get("地点类型", "")).strip() == "restaurant":
        return f"推荐路线: 用餐({main_row['placeName']}) → 周边轻量休闲/散步 → 返程, 预计总时长 {total_hours}h"

    restaurants = _df[_df["地点类型"] == "restaurant"]
    if meal_pref:
        filtered = restaurants[restaurants["placeName"].str.contains(str(meal_pref), na=False, regex=False)]
        restaurants = filtered if not filtered.empty else restaurants
    # 不再固定取表格前几家，避免每次都出现同一批咖啡/斋堂候选。
    available = restaurants[(restaurants["是否有余位"] == True) | (restaurants["余位信息"].fillna(0).astype(int) > 0)].copy()
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
    """执行预订，检查是否需要预约"""
    row = _find_place(attraction)
    need_book = row is not None and row["是否需要预约"]
    order_id = f"ORD-{abs(hash(attraction + date)) % 100000:05d}"
    note = "⚠️ 该地点需提前预约，请确认已完成预约" if need_book else "无需预约，直接前往"
    return f"✅ 订单 {order_id} | {attraction} | {date} | {num_people}人 | 餐厅: {meal} | {note}"

def add_new_place(record: dict, target_file: str = "all_place_mock.xlsx"):
    """将新地点追加到内存 _df 并持久化到指定 Excel 文件"""
    global _df

    prefix = _PREFIX_MAP.get(target_file, "NEW")

    # 统计该前缀已有多少条，避免 ID 冲突
    existing = [idx for idx in _df.index if str(idx).startswith(prefix)]
    new_id = f"{prefix}{(len(existing) + 1):04d}"

    record["placeID"] = new_id
    record["_source_file"] = target_file
    new_row = pd.DataFrame([record]).set_index("placeID")
    _df = pd.concat([_df, new_row])

    # 只把属于 target_file 的行写回对应文件
    sheet = _SHEET_MAP[target_file]
    subset = _df[_df["_source_file"] == target_file].reset_index()
    subset.drop(columns=["_source_file"]).to_excel(
        target_file,
        sheet_name=sheet,
        index=False
    )
    print(f"📥 新地点已入库: {record['placeName']} ({new_id}) → {target_file}")
