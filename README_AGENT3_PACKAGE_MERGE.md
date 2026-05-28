# Agent 3 package merge notes

本目录已参考 `localmate_meituan_improved_package/localmate_meituan_improved` 做兼容合并，但没有整包覆盖当前可运行版本。

## 已合并内容

- `all_place_mock.xlsx`
  - 已改回作为默认地点状态表读取，不再默认使用 `all_place_mock_cleaned.xlsx`。
  - 已补充 `primary_type`、`sub_type`、`search_tags` 三列，用于区分火锅、小笼包、咖啡、街道散步、艺术展、泡汤、寺庙、影院、二次元等细粒度标签。
  - `mock_api_improved.py` 仍支持通过 `PLACE_DATA_FILE` 显式切换数据源，但默认读取原始 `all_place_mock.xlsx`，并继续合并 `cinema_mock_status.xlsx`。
- `build_chroma_db_improved.py`
  - 替换为 package 版本，保留 `.env`、`planning_cases/`、`chroma_db/` 的运行方式。
- `extract_guides_to_jsonl.py`
  - 新增可选脚本，用于把 Word 攻略粗抽取为 `itinerary_cases_from_word.jsonl`。
- `eval_product_quality_rules.py`
  - 新增产品级规则评测脚本，用于检查最终方案是否满足人数、地点、团购、格式等硬约束。
- `agent_workflow_improved.py`
  - 保留当前 Agent 3 的多人讨论、满意度追问、团购预约、高德距离等功能。
  - 新增严格 `structured_plan -> 小红书文案渲染` 链路和 `validation_report` 产品规则校验。
  - 新增高德周边 POI 搜索：用户从出发地搜索“海底捞/火锅/咖啡/小笼包/看展/散步”等需求时，会优先尝试从高德筛选附近真实地点，并可写入 `all_place_mock.xlsx` 作为候选。
- `app_api.py`
  - `/plan` 响应新增 `structured_plan` 和 `validation_report`。
  - session 增加 TTL，默认 3600 秒；线程池改为全局复用。
- `index.html`
  - 新增产品规则校验结果展示。
  - 新增“开启多人偏好收集”开关，默认不强制进入多人讨论。
  - 新增快捷调整按钮：换近一点、换便宜一点、换室内、优先有团购、少走路。
  - 用户文本不再通过 `innerHTML` 直接插入，降低 XSS 风险。

## 未直接覆盖的内容

- 没有直接用 package 的 `app_api.py` 覆盖当前文件，因为 package 版本默认端口是 8001，并且多人讨论需要手动勾选；当前项目要求继续使用 8032 并保留已有对话流程。
- 没有直接用 package 的 `agent_workflow_improved.py` 覆盖当前文件，因为当前文件已经包含之前新增的多人讨论、满意度反馈、高德距离、团购预约和地点识别修复。
- 没有直接用 package 的 `index.html` 覆盖当前前端，因为当前前端已包含动态 A/B/C/D/E 多人身份、团购预约按钮等功能。

## 推荐运行

```powershell
cd E:\dev\projects\localmate_beginner_mvp\classmate_rag\agent_3
conda activate classmate_agentsecond
$env:PYTHONIOENCODING="utf-8"
$env:PLACE_DATA_FILE="all_place_mock.xlsx"
$env:ENABLE_AMAP_POI_SEARCH="1"
python -m uvicorn app_api:app --host 127.0.0.1 --port 8032
```

浏览器打开：

```text
http://127.0.0.1:8032
```

高德周边 POI 搜索可选参数：

```powershell
$env:AMAP_POI_RADIUS_METERS="12000"
$env:AMAP_POI_ACCEPT_RADIUS_METERS="15000"
$env:AMAP_POI_TARGET_FILE="all_place_mock.xlsx"
```

## 可选脚本

重建向量库：

```powershell
python build_chroma_db_improved.py
```

产品级规则评测：

```powershell
python eval_product_quality_rules.py
```

Word 攻略粗结构化抽取：

```powershell
python extract_guides_to_jsonl.py
```
