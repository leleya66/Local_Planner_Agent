# auto_generate_eval_ragas_improved.py
# 作用：
# 1. 从 planning_cases 读取 docx/pdf 文档
# 2. 按 chunk 自动生成 RAGAS 测试题
# 3. 对每道题执行检索 + RAG 回答
# 4. 分指标调用 RAGAS，并打印详细进度
#
# 推荐先这样跑小样本：
#   $env:MAX_CHUNKS_FOR_QA="5"
#   $env:QUESTIONS_PER_CHUNK="1"
#   python -u auto_generate_eval_ragas_improved.py
#
# 跑通后再扩大：
#   $env:MAX_CHUNKS_FOR_QA="20"
#   python -u auto_generate_eval_ragas_improved.py

import os
import re
import glob
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset, Features, Sequence, Value

from langchain_chroma import Chroma
from langchain_dashscope import DashScopeEmbeddings, ChatDashScope
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# RAGAS
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


# ============================================================
# 0. 可调参数
# ============================================================
load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "./planning_cases")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "localmate_planning_cases")

# 先默认小一点，避免你看到“卡住”
MAX_CHUNKS_FOR_QA = int(os.getenv("MAX_CHUNKS_FOR_QA", "10"))
QUESTIONS_PER_CHUNK = int(os.getenv("QUESTIONS_PER_CHUNK", "1"))

CHUNK_SIZE = int(os.getenv("EVAL_CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("EVAL_CHUNK_OVERLAP", "140"))

RETRIEVER_K = int(os.getenv("RETRIEVER_K", "4"))
RETRIEVER_FETCH_K = int(os.getenv("RETRIEVER_FETCH_K", "30"))
RETRIEVER_SEARCH_TYPE = os.getenv("RETRIEVER_SEARCH_TYPE", "similarity")

# 出题模型、回答模型、裁判模型
GEN_MODEL = os.getenv("GEN_MODEL", "qwen-plus")
RAG_MODEL = os.getenv("RAG_MODEL", "qwen-plus")
EVAL_MODEL = os.getenv("EVAL_MODEL", "qwen-plus")

# 方便调试：只生成测试集，不跑 RAGAS
SKIP_RAGAS = os.getenv("SKIP_RAGAS", "0") == "1"

# 输出文件
OUT_TEST_JSON = "ragas_test_dataset_improved.json"
OUT_DEBUG_INPUTS = "ragas_debug_inputs_improved.csv"
OUT_REPORT = "ragas_evaluation_report_improved.csv"
OUT_REPORT_DEBUG = "ragas_evaluation_report_improved_debug.csv"
OUT_BADCASES = "ragas_badcases_top20_improved.csv"


def log(msg: str) -> None:
    """统一打印，flush=True 防止 PowerShell 里看起来像卡住。"""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe_msg = str(msg).encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(safe_msg, flush=True)


def now_s() -> float:
    return time.perf_counter()


def cost_s(start: float) -> str:
    return f"{time.perf_counter() - start:.1f}s"


def clean_json_text(text: str) -> str:
    """清理模型可能返回的 Markdown 包裹。"""
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_array(text: str) -> List[Dict[str, Any]]:
    """
    尽量从模型输出里解析 JSON 数组。
    如果模型多输出了说明文字，就截取第一个 [ 到最后一个 ]。
    """
    cleaned = clean_json_text(text)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        left = cleaned.find("[")
        right = cleaned.rfind("]")
        if left == -1 or right == -1 or right <= left:
            raise
        obj = json.loads(cleaned[left : right + 1])

    if isinstance(obj, dict):
        obj = [obj]

    if not isinstance(obj, list):
        raise ValueError("模型返回的不是 JSON 数组")

    return obj


def safe_short(text: str, n: int = 120) -> str:
    text = str(text).replace("\n", " ").replace("\r", " ").strip()
    return text[:n] + ("..." if len(text) > n else "")


# ============================================================
# 1. 检查环境
# ============================================================
def check_env() -> None:
    log("=" * 80)
    log("0. 环境检查")
    log("=" * 80)

    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise ValueError("⚠️ 未找到 DASHSCOPE_API_KEY，请检查 .env 文件")

    log(f"✅ DASHSCOPE_API_KEY 已读取: {key[:8]}***")
    log(f"📂 DATA_DIR: {DATA_DIR}")
    log(f"📦 CHROMA_DIR: {CHROMA_DIR}")
    log(f"📚 CHROMA_COLLECTION: {CHROMA_COLLECTION}")
    log(f"🧩 MAX_CHUNKS_FOR_QA: {MAX_CHUNKS_FOR_QA}")
    log(f"🧩 QUESTIONS_PER_CHUNK: {QUESTIONS_PER_CHUNK}")
    log(f"🔎 RETRIEVER_SEARCH_TYPE/K: {RETRIEVER_SEARCH_TYPE}/{RETRIEVER_K}")
    log(f"🧠 GEN_MODEL/RAG_MODEL/EVAL_MODEL: {GEN_MODEL}/{RAG_MODEL}/{EVAL_MODEL}")
    log(f"⏭️ SKIP_RAGAS: {SKIP_RAGAS}")


# ============================================================
# 2. 读取资料并切分 chunk
# ============================================================
def load_source_documents(folder_path: str) -> List[Any]:
    log("=" * 80)
    log("1. 读取 planning_cases 原始资料")
    log("=" * 80)

    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"资料目录不存在: {folder_path}")

    file_paths = sorted(
        glob.glob(str(folder / "*.docx")) + glob.glob(str(folder / "*.pdf"))
    )

    log(f"🔍 找到文件数: {len(file_paths)}")
    for p in file_paths[:20]:
        log(f"  - {Path(p).name}")
    if len(file_paths) > 20:
        log(f"  ... 还有 {len(file_paths) - 20} 个文件")

    if not file_paths:
        raise FileNotFoundError(f"在 {folder_path} 下没有找到 docx/pdf 文件")

    documents = []

    for idx, file_path in enumerate(file_paths, start=1):
        start = now_s()
        name = Path(file_path).name
        log(f"📄 [{idx}/{len(file_paths)}] 正在读取: {name}")

        try:
            if file_path.lower().endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif file_path.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                continue

            docs = loader.load()
            for d in docs:
                d.metadata["source"] = name

            documents.extend(docs)
            log(f"   ✅ 读取完成: {len(docs)} 页/段，用时 {cost_s(start)}")

        except Exception as e:
            log(f"   ❌ 读取失败: {name} | {repr(e)}")

    log(f"✅ 原始 Document 总数: {len(documents)}")
    return documents


def split_documents(documents: List[Any]) -> List[Any]:
    log("=" * 80)
    log("2. 切分文档 chunk")
    log("=" * 80)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n#",
            "\n##",
            "\n一、",
            "\n二、",
            "\n三、",
            "\n四、",
            "\n五、",
            "\n行程",
            "\n路线",
            "\n",
            "。",
            "！",
            "？",
            "；",
            "，",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    # 给每个 chunk 添加稳定元数据，方便 debug
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_id"] = f"{source}::chunk_{i}"

    # 过滤太短 chunk
    chunks = [c for c in chunks if len(c.page_content.strip()) >= 80]

    log(f"✂️ 切分后 chunk 数: {len(chunks)}")
    log(f"🔢 本次用于出题 chunk 数: {min(MAX_CHUNKS_FOR_QA, len(chunks))}")

    for i, c in enumerate(chunks[:5], start=1):
        log(
            f"  样例 chunk {i}: source={c.metadata.get('source')} "
            f"len={len(c.page_content)} text={safe_short(c.page_content, 80)}"
        )

    return chunks


# ============================================================
# 3. 初始化 RAG 检索和回答链
# ============================================================
def init_rag():
    log("=" * 80)
    log("3. 初始化 Chroma 检索器和 RAG 回答链")
    log("=" * 80)

    if not Path(CHROMA_DIR).exists():
        raise FileNotFoundError(
            f"找不到向量库目录 {CHROMA_DIR}，请先运行 build_chroma_db_improved.py"
        )

    start = now_s()
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    try:
        collection_count = vectorstore._collection.count()
        log(f"📚 Chroma collection={CHROMA_COLLECTION} 文档数: {collection_count}")
        if collection_count == 0:
            raise RuntimeError(
                f"Chroma collection '{CHROMA_COLLECTION}' 是空的。"
                "请确认 build_chroma_db_improved.py 写入的 collection 名称一致。"
            )
    except AttributeError:
        pass

    if RETRIEVER_SEARCH_TYPE == "mmr":
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": RETRIEVER_K,
                "fetch_k": RETRIEVER_FETCH_K,
                "lambda_mult": 0.65,
            },
        )
    else:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVER_K},
        )

    rag_llm = ChatDashScope(model=RAG_MODEL, temperature=0.05)

    rag_prompt = ChatPromptTemplate.from_template(
        """
你是一个严谨的 RAG 问答助手。
请只根据【参考资料】回答【问题】。

要求：
1. 如果参考资料能回答，请用完整中文句子直接回答，不要只输出孤立短语。
2. 回答要尽量复用参考资料中的原文关键词，避免改写成资料外的新说法。
3. 不要加入参考资料以外的地点、价格、交通、时间。
4. 如果参考资料没有答案，请回答：抱歉，资料中未找到相关信息。
5. 回答不要写小红书风格，不要扩写，不要编造。

回答格式示例：
- 问“具体地址是什么？”时，答“菲林公园的具体地址是徐汇区嘉善路 140 弄 1 号甲 C 区弄堂内。”
- 问“具体门牌号是多少？”时，答“小陶面馆位于徐汇区嘉善路的具体门牌号是 222 号。”
- 问“时间段是什么？”时，答“EKA 园区夜景拍摄的具体时间段是 17:30–18:00。”
- 问“费用是多少？”时，答“上海 EKA 天物空间的入园费用是免费入园。”

【参考资料】：
{context}

【问题】：
{question}
"""
    )

    chain = rag_prompt | rag_llm | StrOutputParser()

    log(f"✅ RAG 初始化完成，用时 {cost_s(start)}")
    log(f"🔎 Retriever: search_type={RETRIEVER_SEARCH_TYPE}, k={RETRIEVER_K}, fetch_k={RETRIEVER_FETCH_K}")

    return retriever, chain, embeddings


def format_docs_for_context(docs: List[Any]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", doc.metadata.get("chunk_index", "unknown"))
        text = doc.page_content.strip()
        parts.append(f"[资料{i} | source={source} | chunk={chunk_id}]\n{text}")
    return "\n\n".join(parts)


# ============================================================
# 4. 自动生成测试集
# ============================================================
def generate_test_dataset(chunks: List[Any]) -> List[Dict[str, Any]]:
    log("=" * 80)
    log("4. 自动生成 RAGAS 测试集")
    log("=" * 80)

    selected_chunks = chunks[:MAX_CHUNKS_FOR_QA]
    log(f"📚 准备处理 chunk 数: {len(selected_chunks)}")

    llm_generator = ChatDashScope(model=GEN_MODEL, temperature=0.1)

    gen_prompt = ChatPromptTemplate.from_template(
        """
你是一个严谨的 RAG 系统评测数据构造专家。
请根据下面【参考文档片段】生成 {num} 个中文问答对，用来测试 RAG 系统。

强制要求：
1. 问题必须能完全根据【参考文档片段】回答。
2. ground_truth 必须完全基于文档片段，不允许补充常识或猜测。
3. evidence 必须摘录文档片段中的原文短句，用来证明 ground_truth。
4. 不要问文档没有明确给出的门牌号、精确票价、营业时间、地铁口等细节。
5. 问题不要使用“上述路线”“该活动”“这个地方”这种脱离上下文后不清楚的指代。
6. 只输出 JSON 数组，不要 Markdown，不要解释。

JSON 格式：
[
  {{
    "question": "问题",
    "ground_truth": "标准答案",
    "evidence": "文档中的原文证据"
  }}
]

【参考文档片段】：
{content}
"""
    )

    chain = gen_prompt | llm_generator | StrOutputParser()
    all_qa_pairs: List[Dict[str, Any]] = []

    for idx, chunk in enumerate(selected_chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        chunk_index = chunk.metadata.get("chunk_index", idx - 1)
        content = chunk.page_content.strip()[:1800]

        log("-" * 80)
        log(
            f"🧠 出题进度 [{idx}/{len(selected_chunks)}] "
            f"source={source} chunk={chunk_index} len={len(content)}"
        )
        log(f"   片段预览: {safe_short(content, 100)}")

        start = now_s()
        raw = ""

        try:
            log("   ⏳ 正在调用出题模型...",)
            raw = chain.invoke({"content": content, "num": QUESTIONS_PER_CHUNK})
            log(f"   ✅ 出题模型返回，用时 {cost_s(start)}，返回长度={len(raw)}")
            log(f"   返回预览: {safe_short(raw, 160)}")

            qa_pairs = parse_json_array(raw)
            accepted = 0

            for q_i, item in enumerate(qa_pairs, start=1):
                question = str(item.get("question", "")).strip()
                ground_truth = str(item.get("ground_truth", "")).strip()
                evidence = str(item.get("evidence", "")).strip()

                if not question or not ground_truth:
                    log(f"   ⚠️ 第 {q_i} 条缺少 question/ground_truth，跳过")
                    continue

                # 证据不强制完全匹配，但会标记，便于后续排查
                evidence_exact_match = bool(evidence and evidence in content)

                record = {
                    "question": question,
                    "ground_truth": ground_truth,
                    "reference": ground_truth,  # 兼容新版 RAGAS
                    "evidence": evidence,
                    "evidence_exact_match": evidence_exact_match,
                    "source": source,
                    "chunk_index": chunk_index,
                    "source_chunk_preview": safe_short(content, 300),
                }
                all_qa_pairs.append(record)
                accepted += 1

                log(
                    f"   ✅ 接收 QA {accepted}: {safe_short(question, 60)} | "
                    f"evidence_exact_match={evidence_exact_match}"
                )

            if accepted == 0:
                log("   ⚠️ 本 chunk 没有接收到有效 QA")

        except Exception as e:
            log(f"   ❌ 出题失败: {repr(e)}")
            if raw:
                log(f"   模型原始返回: {safe_short(raw, 500)}")
            log(traceback.format_exc())

    log("=" * 80)
    log(f"✅ 测试集生成完成，共 {len(all_qa_pairs)} 条 QA")

    with open(OUT_TEST_JSON, "w", encoding="utf-8") as f:
        json.dump(all_qa_pairs, f, ensure_ascii=False, indent=2)

    log(f"📁 已保存测试集: {OUT_TEST_JSON}")
    return all_qa_pairs


# ============================================================
# 5. 执行 RAG：检索 + 回答
# ============================================================
def run_rag_for_dataset(
    qa_dataset: List[Dict[str, Any]],
    retriever: Any,
    rag_chain: Any,
) -> Dict[str, List[Any]]:
    log("=" * 80)
    log("5. 执行 RAG 检索与回答")
    log("=" * 80)

    data_samples = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
        "reference": [],  # 兼容新版 RAGAS
        "source": [],
        "chunk_index": [],
        "evidence": [],
        "retrieved_sources": [],
    }

    total = len(qa_dataset)

    for i, item in enumerate(qa_dataset, start=1):
        query = item["question"]
        ground_truth = item["ground_truth"]

        log("-" * 80)
        log(f"🔄 RAG 进度 [{i}/{total}]")
        log(f"   Q: {query}")
        log(f"   GT: {safe_short(ground_truth, 100)}")

        try:
            # 1. 检索
            t1 = now_s()
            log("   🔎 正在检索 Chroma...")
            retrieved_docs = retriever.invoke(query)
            log(f"   ✅ 检索完成，用时 {cost_s(t1)}，召回 {len(retrieved_docs)} 条")

            contexts = [doc.page_content for doc in retrieved_docs]
            retrieved_sources = []
            for j, doc in enumerate(retrieved_docs, start=1):
                source = doc.metadata.get("source", "unknown")
                chunk_id = doc.metadata.get("chunk_id", doc.metadata.get("chunk_index", "unknown"))
                retrieved_sources.append(f"{source}::{chunk_id}")
                log(
                    f"      [{j}] source={source} chunk={chunk_id} "
                    f"text={safe_short(doc.page_content, 90)}"
                )

            # 2. 回答
            t2 = now_s()
            log("   🤖 正在调用 RAG 回答模型...")
            context_text = format_docs_for_context(retrieved_docs)
            answer = rag_chain.invoke({"context": context_text, "question": query})
            log(f"   ✅ 回答完成，用时 {cost_s(t2)}")
            log(f"   A: {safe_short(answer, 180)}")

            # 3. 收集
            data_samples["question"].append(query)
            data_samples["answer"].append(answer)
            data_samples["contexts"].append(contexts)
            data_samples["ground_truth"].append(ground_truth)
            data_samples["reference"].append(ground_truth)
            data_samples["source"].append(item.get("source", ""))
            data_samples["chunk_index"].append(item.get("chunk_index", ""))
            data_samples["evidence"].append(item.get("evidence", ""))
            data_samples["retrieved_sources"].append(" | ".join(retrieved_sources))

        except Exception as e:
            log(f"   ❌ RAG 运行失败: {repr(e)}")
            log(traceback.format_exc())

    if not data_samples["question"]:
        raise RuntimeError("没有成功收集到任何 RAG 样本，无法评估")

    df_debug = pd.DataFrame(
        {
            "question": data_samples["question"],
            "answer": data_samples["answer"],
            "ground_truth": data_samples["ground_truth"],
            "source": data_samples["source"],
            "chunk_index": data_samples["chunk_index"],
            "evidence": data_samples["evidence"],
            "retrieved_sources": data_samples["retrieved_sources"],
        }
    )
    df_debug.to_csv(OUT_DEBUG_INPUTS, index=False, encoding="utf-8-sig")
    log(f"📁 RAG 输入/输出调试表已保存: {OUT_DEBUG_INPUTS}")

    return data_samples


# ============================================================
# 6. RAGAS 评估：逐指标打印进度
# ============================================================
def evaluate_with_ragas_one_by_one(
    data_samples: Dict[str, List[Any]],
    embeddings: Any,
) -> Optional[pd.DataFrame]:
    log("=" * 80)
    log("6. 调用 RAGAS 评估")
    log("=" * 80)

    if SKIP_RAGAS:
        log("⏭️ SKIP_RAGAS=1，跳过 RAGAS 评估")
        return None

    # RAGAS 不一定需要 source/chunk_index/evidence 这些调试列
    dataset_dict = {
        "question": data_samples["question"],
        "answer": data_samples["answer"],
        "contexts": data_samples["contexts"],
        "ground_truth": data_samples["ground_truth"],
        "reference": data_samples["reference"],
    }

    dataset_features = Features(
        {
            "question": Value("string"),
            "answer": Value("string"),
            "contexts": Sequence(Value("string")),
            "ground_truth": Value("string"),
            "reference": Value("string"),
        }
    )
    dataset = Dataset.from_dict(dataset_dict, features=dataset_features)

    eval_llm = LangchainLLMWrapper(ChatDashScope(model=EVAL_MODEL, temperature=0.1))
    eval_embeddings = LangchainEmbeddingsWrapper(embeddings)

    metrics = [
        ("context_recall", context_recall),
        ("context_precision", context_precision),
        ("faithfulness", faithfulness),
        ("answer_relevancy", answer_relevancy),
    ]

    final_df = pd.DataFrame(
        {
            "question": data_samples["question"],
            "answer": data_samples["answer"],
            "ground_truth": data_samples["ground_truth"],
            "source": data_samples["source"],
            "chunk_index": data_samples["chunk_index"],
            "evidence": data_samples["evidence"],
            "retrieved_sources": data_samples["retrieved_sources"],
        }
    )

    summary_scores: Dict[str, float] = {}

    for idx, (metric_name, metric_obj) in enumerate(metrics, start=1):
        log("-" * 80)
        log(f"⚖️ RAGAS 指标进度 [{idx}/{len(metrics)}]: {metric_name}")
        log("   这一阶段会调用裁判模型，样本多时会比较慢。")

        start = now_s()

        try:
            result = evaluate(
                dataset=dataset,
                metrics=[metric_obj],
                llm=eval_llm,
                embeddings=eval_embeddings,
            )

            log(f"   ✅ {metric_name} 完成，用时 {cost_s(start)}")
            log(f"   指标结果: {result}")

            df_metric = result.to_pandas()

            # 找到该指标列
            if metric_name in df_metric.columns:
                score_col = metric_name
            else:
                # 兼容某些版本列名略有不同
                possible_cols = [
                    c for c in df_metric.columns
                    if c not in ("question", "answer", "contexts", "ground_truth", "reference")
                ]
                score_col = possible_cols[-1] if possible_cols else None

            if score_col:
                final_df[metric_name] = df_metric[score_col].values
                try:
                    summary_scores[metric_name] = float(pd.to_numeric(final_df[metric_name]).mean())
                except Exception:
                    pass
            else:
                log(f"   ⚠️ 没找到 {metric_name} 的分数列，df columns={list(df_metric.columns)}")

        except Exception as e:
            log(f"   ❌ {metric_name} 评估失败: {repr(e)}")
            log(traceback.format_exc())
            final_df[metric_name] = None

    log("=" * 80)
    log("📊 RAGAS 综合评估报告")
    log("=" * 80)
    if summary_scores:
        log(str({k: round(v, 4) for k, v in summary_scores.items()}))
    else:
        log("⚠️ 没有成功计算出任何指标")

    final_df.to_csv(OUT_REPORT, index=False, encoding="utf-8-sig")
    log(f"📁 详细评估结果已保存: {OUT_REPORT}")

    # debug 版：把 contexts 合并为字符串，方便 Excel 打开看
    debug_df = final_df.copy()
    debug_df["contexts_joined"] = [
        "\n\n---\n\n".join(ctxs) if isinstance(ctxs, list) else str(ctxs)
        for ctxs in data_samples["contexts"]
    ]
    debug_df.to_csv(OUT_REPORT_DEBUG, index=False, encoding="utf-8-sig")
    log(f"📁 Debug 评估结果已保存: {OUT_REPORT_DEBUG}")

    # Badcase：按已有指标求平均，低分排前面
    metric_cols = [m[0] for m in metrics if m[0] in final_df.columns]
    if metric_cols:
        score_df = final_df.copy()
        for col in metric_cols:
            score_df[col] = pd.to_numeric(score_df[col], errors="coerce")
        score_df["avg_score"] = score_df[metric_cols].mean(axis=1)
        badcases = score_df.sort_values("avg_score", ascending=True).head(20)
        badcases.to_csv(OUT_BADCASES, index=False, encoding="utf-8-sig")
        log(f"📁 Top20 Badcases 已保存: {OUT_BADCASES}")

    return final_df


# ============================================================
# 7. 主入口
# ============================================================
def main() -> None:
    total_start = now_s()

    try:
        check_env()

        docs = load_source_documents(DATA_DIR)
        chunks = split_documents(docs)

        retriever, rag_chain, embeddings = init_rag()

        qa_dataset = generate_test_dataset(chunks)

        if not qa_dataset:
            log("⚠️ 测试集为空，终止。")
            return

        data_samples = run_rag_for_dataset(qa_dataset, retriever, rag_chain)

        evaluate_with_ragas_one_by_one(data_samples, embeddings)

        log("=" * 80)
        log(f"🎉 全部流程结束，总用时 {cost_s(total_start)}")
        log("=" * 80)

    except KeyboardInterrupt:
        log("\n⛔ 用户手动中断。已经生成的中间文件会保留。")
    except Exception as e:
        log("=" * 80)
        log(f"❌ 程序异常退出: {repr(e)}")
        log("=" * 80)
        log(traceback.format_exc())


if __name__ == "__main__":
    main()
