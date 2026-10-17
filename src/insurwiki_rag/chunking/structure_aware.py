"""§1.2 结构感知切分（Structure-Aware Chunking）。

针对保单条款"条-款-项"层级：
  1. 条款边界识别：正则匹配 `第X条`（中文数字/阿拉伯数字），保证一个 chunk 不跨条
  2. 款/项层级：`（一）（二）` / `1. 2.` / `（1）（2）` 作为次级边界
  3. 语义完整性：条内按句子边界聚合，受 最大token窗口(MAX_CHARS)+overlap 约束
  4. 费率表通道：含表格特征（| 分隔 / 多列数字 / "费率表"关键词）的片段走独立的
     表格→Markdown/KV 序列化通道，不与正文混切
  5. 元数据：chunk_id / policy_id / 条款章节路径 / 原文偏移(offset) / chunk_type
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional

from ..config import Config

# 条-款-项 正则
RE_ARTICLE = re.compile(r"第[一二三四五六七八九十百千零〇\d]+条")
RE_SUB_ITEM = re.compile(r"[（(][一二三四五六七八九十\d]+[)）]")
RE_SENT_END = re.compile(r"(?<=[。！？；])")

MAX_CHARS = 420      # 约等于 300 token（中文 1 字≈1.5 token 的保守值）
OVERLAP_CHARS = 60
TABLE_HINT = re.compile(r"费率|费率表|\|.+\||\t")
MIN_TABLE_COLS = 2


def _cid(prefix: str, text: str) -> str:
    return f"{prefix}-{hashlib.md5(text.encode()).hexdigest()[:10]}"


def looks_like_table(text: str) -> bool:
    """启发式判断是否为表格/费率类数据。"""
    if "费率" in text and any(ch.isdigit() for ch in text):
        return True
    if text.count("|") >= MIN_TABLE_COLS:
        return True
    # 多行以数字/百分号结尾的列式数据
    lines = [l for l in text.splitlines() if l.strip()]
    num_lines = sum(1 for l in lines if re.search(r"\d+(\.\d+)?%?", l))
    return len(lines) >= 3 and num_lines / len(lines) > 0.6


def table_to_kv(text: str) -> str:
    """表格 → Markdown/KV 序列化。"""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    md, kv = [], []
    for l in lines:
        if "|" in l:
            cells = [c.strip() for c in l.strip("|").split("|") if c.strip()]
            if len(cells) >= 2:
                md.append("| " + " | ".join(cells) + " |")
                kv.append(f"{cells[0]}：{' '.join(cells[1:])}")
        else:
            md.append(l)
            kv.append(l)
    header = "（费率表/表格数据，Markdown 序列化）\n" if md and md[0].startswith("|") else ""
    return header + "\n".join(md) + ("\n\nKV 摘要：\n" + "\n".join(kv[:20]) if kv else "")


def split_sentences(text: str) -> List[str]:
    return [s for s in RE_SENT_END.split(text) if s.strip()]


def _window_sentences(sents: List[str]) -> List[str]:
    """句子边界 + 最大窗口 + overlap 聚合。"""
    out, buf, size = [], [], 0
    for s in sents:
        if buf and size + len(s) > MAX_CHARS:
            out.append("".join(buf))
            tail = "".join(buf)
            keep = tail[-OVERLAP_CHARS:]
            buf, size = [keep] if keep else [], len(keep)
        buf.append(s)
        size += len(s)
    if buf:
        out.append("".join(buf))
    return out


def chunk_fragment(frag: str, start_offset: int) -> List[Dict[str, Any]]:
    """切分单个证据片段，保留条款路径与偏移。"""
    if looks_like_table(frag):
        body = table_to_kv(frag)
        return [{
            "text": body, "section_path": "费率表/表格", "offset": start_offset,
            "chunk_type": "table",
        }]
    # 条款边界：按 第X条 切
    article_spans: List[tuple] = []
    for m in RE_ARTICLE.finditer(frag):
        article_spans.append((m.start(), m.group()))
    chunks = []
    if article_spans:
        bounds = [s for s, _ in article_spans] + [len(frag)]
        titles = [t for _, t in article_spans]
        for i in range(len(bounds) - 1):
            seg = frag[bounds[i]:bounds[i + 1]]
            path = titles[i]
            for j, part in enumerate(_window_sentences(split_sentences(seg))):
                body = part if part.startswith(path) else (f"{path} {part}" if j == 0 else f"{path}(续) {part}")
                chunks.append({
                    "text": body,
                    "section_path": path,
                    "offset": start_offset + bounds[i],
                    "chunk_type": "clause",
                })
    else:
        for part in _window_sentences(split_sentences(frag)):
            chunks.append({"text": part, "section_path": "未编号段落",
                           "offset": start_offset, "chunk_type": "clause"})
    return chunks


def chunk_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """对全部合同文档做结构感知切分，产出带元数据的 chunk 列表。"""
    all_chunks: List[Dict[str, Any]] = []
    for d in docs:
        offset = 0
        for frag in d["fragments"]:
            for c in chunk_fragment(frag, offset):
                c.update({
                    "chunk_id": _cid("c", d["doc_id"] + c["text"]),
                    "doc_id": d["doc_id"],
                    "policy_id": d["policy_id"],
                    "policy_name": d["policy_name"],
                })
                all_chunks.append(c)
            offset += len(frag) + 2  # 连接符 "\n\n"
    return all_chunks


def load_docs() -> List[Dict[str, Any]]:
    docs = []
    path = Config.DATA_PROCESSED / "corpus_docs.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            docs.append(__import__("json").loads(line))
    return docs


def load_triples() -> List[Dict[str, Any]]:
    triples = []
    path = Config.DATA_PROCESSED / "qa_triples.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            triples.append(__import__("json").loads(line))
    return triples
