"""InsurWiki-RAG 配置：零依赖加载 .env，统一配置入口。"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


_load_dotenv(ROOT / ".env")


def env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


class Config:
    # LLM（GLM Coding Plan · Anthropic 兼容端点）
    LLM_BASE_URL = env("LLM_BASE_URL", "https://open.bigmodel.cn/api/anthropic")
    LLM_API_KEY = env("LLM_API_KEY")
    LLM_MODEL = env("LLM_MODEL", "glm-5.3")
    LLM_MODEL_FALLBACK = env("LLM_MODEL_FALLBACK", "glm-4.6")

    # 本地 Embedding / Rerank
    EMBED_MODEL = env("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
    RERANK_MODEL = env("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    HF_ENDPOINT = env("HF_ENDPOINT", "https://hf-mirror.com")
    HF_HOME = env("HF_HOME", str(ROOT / "hf_cache"))

    # 检索
    RRF_K = int(env("RRF_K", "60"))
    RERANK_TOP_N = int(env("RERANK_TOP_N", "5"))
    SUPPORT_THRESHOLD = float(env("SUPPORT_THRESHOLD", "0.35"))  # 拒答阈值

    # 数据
    DATA_SAMPLE = int(env("DATA_SAMPLE", "1500"))

    # 路径
    DATA_RAW = ROOT / "data" / "raw"
    DATA_PROCESSED = ROOT / "data" / "processed"
    ARTIFACTS = ROOT / "artifacts"
    INDEX_DIR = ARTIFACTS / "index"
    GRAPH_PATH = ARTIFACTS / "graph" / "kg.json"
    WIKI_DIR = ARTIFACTS / "wiki"
    EVAL_DIR = ARTIFACTS / "eval"

    PORT = int(env("PORT", "8010"))

    @classmethod
    def ensure_dirs(cls) -> None:
        for d in (cls.DATA_RAW, cls.DATA_PROCESSED, cls.INDEX_DIR, cls.GRAPH_PATH.parent,
                  cls.WIKI_DIR, cls.EVAL_DIR):
            d.mkdir(parents=True, exist_ok=True)


os.environ.setdefault("HF_ENDPOINT", Config.HF_ENDPOINT)
os.environ.setdefault("HF_HOME", Config.HF_HOME)
