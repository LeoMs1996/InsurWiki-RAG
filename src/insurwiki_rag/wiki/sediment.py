"""§1.7 LLM Wiki 知识沉淀。

  - 主题聚类：按实体类型 + 图谱共现关系对条款主题聚类（无需 sklearn）
  - 聚合摘要：GLM 对每簇条款做聚合摘要，生成结构化 Wiki 页（Markdown + front-matter）
  - 页结构：主题概述 / 核心条款要点 / 关联条款引用（互链）/ 常见问答 / 更新时间与数据来源
  - 闭环：Wiki 页正文重新向量化并入向量库（“沉淀→反哺检索”），标记 chunk_type=wiki
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ..config import Config
from ..llm import get_llm
from ..retrieval.hybrid import get_embedder

WIKI_TMPL = """---
title: {title}
topic_id: {topic_id}
entity_type: {etype}
keywords: {keywords}
source: InsQABench 合同问答子集（InsurWiki-RAG 知识沉淀流水线）
created_at: {created}
chunk_ids: {n_chunks}
---

# {title}

## 主题概述
{overview}

## 核心条款要点
{points}

## 关联条款引用
{links}

## 常见问答
{qas}

> 数据来源：InsQABench（华中科技大学 VLR Lab，Apache-2.0）· InsurWiki-RAG 自动沉淀 · {created}
"""


def cluster_by_entity(kg, chunks_by_id: Dict[str, Dict[str, Any]],
                      min_cluster: int = 3, max_topics: int = 24
                      ) -> List[Dict[str, Any]]:
    """按图谱实体聚类：每个实体（mentions≥min_cluster）关联的 chunk 构成一个主题簇。"""
    topics = []
    for name, data in kg.g.nodes(data=True):
        cids = [c for c in data.get("chunk_ids", set()) if c in chunks_by_id]
        if len(cids) >= min_cluster:
            topics.append({
                "topic_id": "T" + re.sub(r"\W", "", name)[:12],
                "entity": name,
                "etype": data.get("type", ""),
                "chunk_ids": cids[:20],
            })
    topics.sort(key=lambda t: -len(t["chunk_ids"]))
    return topics[:max_topics]


def generate_wiki_pages(kg, chunks: List[Dict[str, Any]],
                        min_cluster: int = 3, max_topics: int = 24) -> List[Dict[str, Any]]:
    """沉淀流水线：聚类 → GLM 聚合摘要 → 生成 Wiki 页 → 回灌向量索引。"""
    llm = get_llm()
    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    triples_by_entity: Dict[str, List[str]] = {}
    for h, t, d in kg.g.edges(data=True):
        triples_by_entity.setdefault(h, [])
        triples_by_entity[h].append(f"{h} --{d['relation']}--> {t}")
    topics = cluster_by_entity(kg, chunks_by_id, min_cluster, max_topics)
    print(f"[wiki] {len(topics)} topics clustered")
    pages = []
    for t in topics:
        ents = "\n\n".join(
            f"-（{chunks_by_id[c]['policy_name']}·{chunks_by_id[c].get('section_path','')}）"
            f"{chunks_by_id[c]['text'][:200]}"
            for c in t["chunk_ids"][:10])
        graph_ctx = "\n".join(triples_by_entity.get(t["entity"], [])[:8])
        try:
            body = llm.chat_json(
                "你是保险知识 Wiki 编辑。基于给定条款证据生成结构化摘要，字段："
                'overview（80字内主题概述）、points（数组，3-6条核心条款要点，'
                '每条含 [条款号/合同名]）、keywords（4-6个关键词数组）。只输出JSON。',
                f"主题实体：{t['entity']}（类型 {t['etype']}）\n"
                f"图谱关系：\n{graph_ctx}\n\n条款证据：\n{ents}", max_tokens=1500)
        except Exception:  # noqa: BLE001
            continue
        pages.append({**t, "body": body})
    return pages


def render_pages(pages: List[Dict[str, Any]],
                 qa_by_chunk: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """渲染 Markdown 页（front-matter + 固定小节），落盘 artifacts/wiki/。"""
    Config.WIKI_DIR.mkdir(parents=True, exist_ok=True)
    rendered = []
    for p in pages:
        b = p["body"]
        title = f"{p['entity']}（{p['etype']}）"
        created = datetime.now().strftime("%Y-%m-%d %H:%M")
        links = "\n".join(f"- `{cid}`"
                          for cid in p["chunk_ids"][:10]) or "（无）"
        qas = []
        for cid in p["chunk_ids"][:10]:
            for qa in qa_by_chunk.get(cid, [])[:2]:
                qas.append(f"**Q：{qa['question']}**\n\nA：{qa['answer'][:200]}")
        md = WIKI_TMPL.format(
            title=title, topic_id=p["topic_id"], etype=p["etype"],
            keywords="、".join(b.get("keywords", [])), created=created,
            n_chunks=len(p["chunk_ids"]),
            overview=b.get("overview", ""),
            points="\n".join(f"- {_fmt_point(x)}" for x in b.get("points", [])),
            links=links,
            qas=("\n\n".join(qas[:6]) if qas else "（暂无关联问答）"))
        fname = f"{p['topic_id']}.md"
        (Config.WIKI_DIR / fname).write_text(md, encoding="utf-8")
        rendered.append({
            "topic_id": p["topic_id"], "title": title, "file": fname,
            "etype": p["etype"], "entity": p["entity"],
            "keywords": b.get("keywords", []),
            "overview": b.get("overview", ""),
            "chunk_ids": p["chunk_ids"], "created_at": created,
        })
def _fmt_point(x: Any) -> str:
    """LLM 要点可能是 dict（如 {'条款': ...}），统一格式化为字符串。"""
    if isinstance(x, dict):
        return "　".join(str(v) for v in x.values())
    return str(x)


ZpG
        json.dumps(rendered, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[wiki] rendered {len(rendered)} pages -> {Config.WIKI_DIR}")
    return rendered


def _chunks_sec(p, cid):
    return cid


def wiki_pages_as_chunks(rendered: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Wiki 页 → 检索 chunk（沉淀→反哺检索闭环）。"""
    out = []
    for r in rendered:
        path = Config.WIKI_DIR / r["file"]
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"^---.*?---\s*", "", text, flags=re.S).strip()
        out.append({
            "chunk_id": f"wiki-{r['topic_id']}",
            "doc_id": f"wiki-{r['topic_id']}",
            "policy_id": "WIKI",
            "policy_name": f"Wiki·{r['title']}",
            "text": text[:1500],
            "section_path": "Wiki 沉淀页",
            "chunk_type": "wiki",
            "offset": 0,
        })
    return out


def augment_index_with_wiki(wiki_chunks: List[Dict[str, Any]]) -> None:
    """把 Wiki chunk 追加进持久化索引（向量矩阵 + chunks.jsonl）。"""
    if not wiki_chunks:
        return
    existing = [json.loads(l) for l in
                (Config.INDEX_DIR / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()]
    existing_ids = {c["chunk_id"] for c in existing}
    new = [c for c in wiki_chunks if c["chunk_id"] not in existing_ids]
    if not new:
        return
    emb = np.load(Config.INDEX_DIR / "embeddings.npy")
    new_emb = get_embedder().encode([c["text"] for c in new])
    emb = np.vstack([emb, new_emb])
    np.save(Config.INDEX_DIR / "embeddings.npy", emb)
    with open(Config.INDEX_DIR / "chunks.jsonl", "a", encoding="utf-8") as f:
        for c in new:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[wiki] index augmented with {len(new)} wiki chunks "
          f"(total {emb.shape[0]})")
