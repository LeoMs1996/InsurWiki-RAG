"""§1.3 混合检索与 RRF 融合 + Rerank 精排。

  - BM25（rank_bm25，jieba 分词）+ 向量检索（本地 bge-small-zh-v1.5，numpy 余弦）
    双路召回，各取 top-K
  - RRF（Reciprocal Rank Fusion, k=60）融合两路排序
  - Rerank：本地 BGE-reranker-v2-m3（cross-encoder）精排，输出 top-N 及相关性分数
  - 索引持久化到 artifacts/index/（chunks.jsonl + 向量 npy）
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import Config


# ---------------- Embedding（本地 sentence-transformers） ----------------
class Embedder:
    def __init__(self, model_name: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name or Config.EMBED_MODEL)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True,
                                 show_progress_bar=False).astype("float32")


# ---------------- 向量库（numpy 余弦，嵌入式即可支撑数千 chunk） ----------------
class VectorStore:
    def __init__(self):
        self.matrix: Optional[np.ndarray] = None

    def build(self, embeddings: np.ndarray) -> None:
        self.matrix = embeddings

    def search(self, qvec: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        if self.matrix is None:
            return []
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = self.matrix @ qvec
        sims = np.nan_to_num(sims, nan=-1.0, posinf=1.0, neginf=-1.0)
        idx = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in idx]


# ---------------- BM25（jieba 分词） ----------------
class BM25Store:
    def __init__(self, corpus_tokens: List[List[str]]):
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(corpus_tokens)

    def search(self, tokens: List[str], top_k: int) -> List[Tuple[int, float]]:
        scores = self.bm25.get_scores(tokens)
        idx = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]


def tokenize(text: str) -> List[str]:
    import jieba
    return [t for t in jieba.cut(text) if t.strip()]


# ---------------- RRF 融合 ----------------
def rrf_fuse(rank_lists: List[List[int]], k: int = 60) -> Dict[int, float]:
    """Reciprocal Rank Fusion: score = Σ 1/(k + rank_i)。"""
    scores: Dict[int, float] = {}
    for ranks in rank_lists:
        for r, doc_id in enumerate(ranks, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + r)
    return scores


# ---------------- Reranker（本地 BGE-reranker-v2-m3） ----------------
class Reranker:
    def __init__(self, model_name: Optional[str] = None):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name or Config.RERANK_MODEL,
                                  max_length=512)

    def rerank(self, query: str, docs: List[str]) -> List[float]:
        pairs = [(query, d) for d in docs]
        return [float(s) for s in self.model.predict(pairs)]


# ---------------- 混合检索器（组合门面） ----------------
class HybridRetriever:
    def __init__(self, chunks: List[Dict[str, Any]],
                 embeddings: np.ndarray, corpus_tokens: List[List[str]]):
        self.chunks = chunks
        self.vector = VectorStore()
        self.vector.build(embeddings)
        self.bm25 = BM25Store(corpus_tokens)
        self._reranker: Optional[Reranker] = None

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    def recall(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """双路召回 + RRF 融合（无 LLM/重排开销，供 Agent 工具与图谱通道使用）。"""
        from sentence_transformers import SentenceTransformer  # noqa
        qvec = _embed_query(query)
        vec_hits = self.vector.search(qvec, top_k)
        bm_hits = self.bm25.search(tokenize(query), top_k)
        fused = rrf_fuse([[i for i, _ in vec_hits], [i for i, _ in bm_hits]],
                         k=Config.RRF_K)
        ranked = sorted(fused.items(), key=lambda x: -x[1])[:top_k]
        out = []
        for idx, score in ranked:
            c = dict(self.chunks[idx])
            c["rrf_score"] = round(score, 6)
            c["vec_score"] = next((s for i, s in vec_hits if i == idx), None)
            c["bm25_score"] = next((s for i, s in bm_hits if i == idx), None)
            out.append(c)
        return out

    def search(self, query: str, top_k: int = 20, top_n: int = 5,
               use_rerank: bool = True,
               extra_channel: Optional[List[Dict[str, Any]]] = None
               ) -> List[Dict[str, Any]]:
        """完整链路：双路召回 →（可选图谱通道并入）→ RRF → Rerank → top-N。"""
        n = top_n or Config.RERANK_TOP_N
        qvec = _embed_query(query)
        vec_hits = self.vector.search(qvec, top_k)
        bm_hits = self.bm25.search(tokenize(query), top_k)

        rank_lists = [[i for i, _ in vec_hits], [i for i, _ in bm_hits]]
        extra_scores: Dict[int, float] = {}
        if extra_channel:  # GraphRAG 证据通道：作为第三路排名列表并入 RRF
            extra_ids = [self._chunk_index(e["chunk_id"]) for e in extra_channel
                         if self._chunk_index(e["chunk_id"]) is not None]
            if extra_ids:
                rank_lists.append(extra_ids)
                for r, i in enumerate(extra_ids, start=1):
                    extra_scores[i] = 1.0 / (Config.RRF_K + r)

        fused = rrf_fuse(rank_lists, k=Config.RRF_K)
        ranked = sorted(fused.items(), key=lambda x: -x[1])

        if use_rerank and len(ranked) > n:
            cand_idx = [i for i, _ in ranked[: max(top_k, 3 * n)]]
            scores = self.reranker.rerank(query, [self.chunks[i]["text"] for i in cand_idx])
            scored = sorted(zip(cand_idx, scores), key=lambda x: -x[1])[:n]
            out = []
            for idx, rs in scored:
                c = dict(self.chunks[idx])
                c["rerank_score"] = round(rs, 4)
                c["rrf_score"] = round(fused.get(idx, 0.0), 6)
                c["from_graph"] = idx in extra_scores
                out.append(c)
            return out

        out = []
        for idx, s in ranked[:n]:
            c = dict(self.chunks[idx])
            c["rrf_score"] = round(s, 6)
            c["from_graph"] = idx in extra_scores
            out.append(c)
        return out

    def _chunk_index(self, chunk_id: str) -> Optional[int]:
        for i, c in enumerate(self.chunks):
            if c["chunk_id"] == chunk_id:
                return i
        return None


# ---------------- 查询向量缓存（Embedder 进程级单例） ----------------
_embedder: Optional[Embedder] = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def _embed_query(query: str) -> np.ndarray:
    return get_embedder().encode([query])[0]


# ---------------- 索引构建与持久化 ----------------
def build_index(chunks: List[Dict[str, Any]], save: bool = True) -> HybridRetriever:
    texts = [c["text"] for c in chunks]
    print(f"[index] embedding {len(texts)} chunks with {Config.EMBED_MODEL} ...")
    emb = get_embedder().encode(texts)
    tokens = [tokenize(t) for t in texts]
    retriever = HybridRetriever(chunks, emb, tokens)
    if save:
        Config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        np.save(Config.INDEX_DIR / "embeddings.npy", emb)
        with open(Config.INDEX_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"[index] saved to {Config.INDEX_DIR}")
    return retriever


def load_index() -> HybridRetriever:
    emb = np.load(Config.INDEX_DIR / "embeddings.npy")
    chunks = [json.loads(l) for l in
              (Config.INDEX_DIR / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
              if l.strip()]
    tokens = [tokenize(c["text"]) for c in chunks]
    return HybridRetriever(chunks, emb, tokens)
