# build_chroma_db_improved.py
# 重新构建 Case-RAG 向量库：Word/PDF 攻略 -> 清洗 chunk -> Chroma
# 适合本项目：RAG 只作为“相似案例参考”，最终事实仍以结构化地点状态表为准。

import glob
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_dashscope import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = Path(os.getenv("PLANNING_CASES_DIR") or os.getenv("DATA_DIR") or "./planning_cases")
PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR") or os.getenv("CHROMA_DIR") or "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "localmate_planning_cases")
RESET_DB = os.getenv("RESET_CHROMA_DB", "1") == "1"
EMBEDDING_MODEL = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v2")
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "20"))

if not os.getenv("DASHSCOPE_API_KEY"):
    raise ValueError("⚠️ 未找到 DASHSCOPE_API_KEY，请检查 .env 文件")


def normalize_text(text: str) -> str:
    text = (text or "").replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def infer_doc_tags(name: str, text: str) -> List[str]:
    combined = f"{name} {text[:800]}".lower()
    tag_rules = {
        "sunny": ["晴天", "sunny"],
        "rainy": ["雨天", "rainy", "室内"],
        "high_budget": ["高预算", "300plus", "纪念日", "庆祝"],
        "low_budget": ["低预算", "学生党", "省钱"],
        "movie": ["电影", "影院", "cinema", "movie"],
        "family": ["父母", "长辈", "家庭", "亲子"],
        "sports": ["运动", "保龄球", "羽毛球", "射箭"],
        "leisure": ["躺平", "休闲", "放松", "解压"],
        "food": ["美食", "餐厅", "小吃", "火锅"],
    }
    tags = []
    for tag, words in tag_rules.items():
        if any(w.lower() in combined for w in words):
            tags.append(tag)
    return sorted(set(tags))


def load_source_documents(data_dir: Path) -> List[Document]:
    data_dir.mkdir(parents=True, exist_ok=True)
    file_paths = sorted(glob.glob(str(data_dir / "*.docx")) + glob.glob(str(data_dir / "*.pdf")))
    if not file_paths:
        raise FileNotFoundError(f"⚠️ {data_dir} 下没有找到 .docx 或 .pdf 文件")

    docs: List[Document] = []
    for file_path in file_paths:
        path = Path(file_path)
        print(f"📄 读取: {path.name}")
        try:
            loader = Docx2txtLoader(str(path)) if path.suffix.lower() == ".docx" else PyPDFLoader(str(path))
            loaded = loader.load()
            full_preview = "\n".join(d.page_content for d in loaded[:2])
            tags = infer_doc_tags(path.name, full_preview)
            for i, doc in enumerate(loaded):
                text = normalize_text(doc.page_content)
                if len(text) < 80:
                    continue
                doc.page_content = text
                doc.metadata.update({
                    "source": path.name,
                    "source_path": str(path),
                    "source_index": i,
                    "doc_tags": ",".join(tags),
                    "data_nature": "case_rag_reference_not_realtime_fact",
                })
                docs.append(doc)
        except Exception as e:
            print(f"   ❌ 读取失败: {path.name} | {e}")
    return docs


def stable_doc_id(source: str, chunk_index: int, text: str) -> str:
    digest = hashlib.md5(f"{source}-{chunk_index}-{text[:240]}".encode("utf-8")).hexdigest()[:12]
    return f"{Path(source).stem}-{chunk_index:04d}-{digest}"


def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", "700")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "140")),
        separators=[
            "\n\n行程", "\n行程", "\n一、", "\n二、", "\n三、", "\n四、", "\n五、",
            "\n为什么推荐", "\n具体玩法说明", "\n整体时间", "\n地点简介", "\nTips", "\n注意事项",
            "\n\n", "\n", "。", "；", "，", " "
        ],
    )
    chunks = splitter.split_documents(docs)
    cleaned: List[Document] = []
    for idx, chunk in enumerate(chunks):
        text = normalize_text(chunk.page_content)
        if len(text) < 80:
            continue
        source = chunk.metadata.get("source", "unknown")
        chunk_id = stable_doc_id(source, idx, text)
        chunk.page_content = text
        chunk.metadata.update({"chunk_id": chunk_id, "chunk_index": idx, "doc_id": chunk_id})
        cleaned.append(chunk)
    return cleaned


def main() -> None:
    print(f"📂 读取攻略资料目录: {DATA_DIR}")
    raw_docs = load_source_documents(DATA_DIR)
    print(f"✅ 原始 Document 数: {len(raw_docs)}")
    chunks = split_documents(raw_docs)
    print(f"✂️ 切分后 chunk 数: {len(chunks)}")

    if RESET_DB and PERSIST_DIR.exists():
        print(f"🧹 清空旧向量库: {PERSIST_DIR}")
        shutil.rmtree(PERSIST_DIR)
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = DashScopeEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(collection_name=COLLECTION_NAME, persist_directory=str(PERSIST_DIR), embedding_function=embeddings)

    print(f"📦 写入 Chroma collection={COLLECTION_NAME}")
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        ids = [doc.metadata["chunk_id"] for doc in batch]
        vectorstore.add_documents(documents=batch, ids=ids)
        print(f"👉 已写入 {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)}")
    if hasattr(vectorstore, "persist"):
        vectorstore.persist()

    manifest = {
        "data_dir": str(DATA_DIR),
        "persist_dir": str(PERSIST_DIR),
        "collection_name": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "num_raw_docs": len(raw_docs),
        "num_chunks": len(chunks),
        "sources": sorted({doc.metadata.get("source", "") for doc in raw_docs}),
        "note": "Case-RAG 只作为相似攻略参考；价格/余位/团购等事实以结构化地点状态表或实时API为准。",
    }
    manifest_path = PERSIST_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("✅ 向量数据库重建完成")
    print(f"📝 manifest: {manifest_path}")


if __name__ == "__main__":
    main()
