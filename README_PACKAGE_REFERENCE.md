# LocalMate / 美团黑客松改进版运行说明

这套文件是“可替换运行版”，文件名保持与你原项目一致：

- `app_api.py`：FastAPI 服务入口
- `agent_workflow_improved.py`：核心 Agent / LangGraph 工作流
- `mock_api_improved.py`：地点状态表、团购、预约 mock tools
- `build_chroma_db_improved.py`：Word/PDF 攻略 Case-RAG 向量库构建
- `auto_generate_eval_ragas_improved.py`：保留你的 RAGAS 评测脚本
- `eval_product_quality_rules.py`：新增产品级规则评测脚本
- `extract_guides_to_jsonl.py`：新增 Word 攻略粗结构化抽取工具
- `index.html`：安全版前端
- `all_place_mock_cleaned.xlsx`：清洗增强后的地点状态表

## 一、替换方式

把本文件夹内的文件复制到你的运行目录，例如：

```powershell
cd E:\dev\projects\localmate_beginner_mvp\classmate_rag\agent_run_improved
copy /Y path\to\localmate_meituan_improved\*.py .
copy /Y path\to\localmate_meituan_improved\index.html .
copy /Y path\to\localmate_meituan_improved\all_place_mock_cleaned.xlsx .
```

保留你的 `planning_cases/` 目录和 `.env` 文件。

## 二、环境变量

`.env` 至少需要：

```env
DASHSCOPE_API_KEY=你的百炼DashScope Key
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION=localmate_planning_cases
PLACE_DATA_FILE=all_place_mock_cleaned.xlsx
```

可选高德：

```env
AMAP_API_KEY=你的高德Web服务Key
```

如果没有高德 Key，系统仍能运行，只是不输出具体距离和分钟数。

## 三、重建向量库

先把 Word/PDF 攻略放到 `planning_cases/`，再运行：

```powershell
conda activate classmate_rag
$env:PYTHONIOENCODING="utf-8"
python build_chroma_db_improved.py
```

## 四、启动服务

```powershell
conda activate classmate_rag
$env:PYTHONIOENCODING="utf-8"
$env:PLACE_DATA_FILE="all_place_mock_cleaned.xlsx"
python -m uvicorn app_api:app --host 127.0.0.1 --port 8001
```

浏览器打开：

```text
http://127.0.0.1:8001
```

## 五、产品级评测

RAGAS 只能评估“检索问答”，不能评估最终路线质量。新增脚本用于检查最终方案是否满足人数、地点、团购、距离等约束：

```powershell
python eval_product_quality_rules.py
```

输出：`product_quality_eval_report.csv`

## 六、Word 攻略结构化抽取

```powershell
python extract_guides_to_jsonl.py
```

输出：`itinerary_cases_from_word.jsonl`。这是粗抽取，建议人工审核后再用于正式 Planner。

## 七、本版核心设计原则

1. Word 攻略只作为相似案例与风格参考，不作为实时事实。
2. 表格状态负责价格、余位、预约、团购等结构化事实/Mock 状态。
3. 最终路线先生成 `structured_plan`，再渲染成小红书文案。
4. 没有高德 Key 时禁止编造距离、分钟数、打车费。
5. 没有结构化 coupon_info 时禁止编造团购券。
6. 模型生成后做规则校验，前端会展示 `validation_report`。
