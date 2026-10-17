#!/usr/bin/env python3
"""InsurWiki-RAG 数据流水线（一条命令跑通：拉取→清洗→切分→索引→图谱→Wiki）。

用法：
  .venv/bin/python scripts/pipeline.py            # 全量
  .venv/bin/python scripts/pipeline.py --skip-data # 复用已有 processed 数据
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from insurwiki_rag.config import Config  # noqa: E402
from insurwiki_rag.data import pipeline as data_pipeline  # noqa: E402
from insurwiki_rag.chunking.structure_aware import (  # noqa: E402
    chunk_docs, load_docs, load_triples)
from insurwiki_rag.retrieval.hybrid import build_index  # noqa: E402


def build_qa_by_chunk(triples, chunks):
    """QA 对 → chunk 映射（证据包含 chunk 文本即关联），供 Wiki 常见问答小节使用。"""
    qa_by_chunk = {}
    for qa in triples:
        ev = qa["evidence"]
        for c in chunks:
            t = c["text"]
            key = c["chunk_id"]
            if (t[:60] in ev) or (ev[:60] in t):
                qa_by_chunk.setdefault(key, []).append(qa)
    return qa_by_chunk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-data", action="store_true", help="复用 data/processed 已有数据")
    ap.add_argument("--skip-wiki", action="store_true")
    args = ap.parse_args()
    Config.ensure_dirs()

    # 1) 数据
    if not args.skip_data:
        data_pipeline.run()
    else:
        print("[pipeline] skip data stage")

    # 2) 切分 + 3) 索引
    docs = load_docs()
    triples = load_triples()
    chunks = chunk_docs(docs)
    retriever = build_index(chunks)

    # 4) GraphRAG 抽取
    from insurwiki_rag.graphrag.kg import KnowledgeGraph, extract_graph
    if Config.GRAPH_PATH.exists():
        kg = KnowledgeGraph.load()
        print(f"[pipeline] graph loaded: {kg.stats()['nodes']} nodes")
    else:
        kg = extract_graph(retriever.chunks, batch_size=6, workers=6)

    # 5) Wiki 沉淀 + 回灌
    if not args.skip_wiki:
        from insurwiki_rag.wiki.sediment import (
            augment_index_with_wiki, generate_wiki_pages, render_pages,
            wiki_pages_as_chunks)
        pages = generate_wiki_pages(kg, retriever.chunks)
        qa_by_chunk = build_qa_by_chunk(triples, retriever.chunks)
        rendered = render_pages(pages, qa_by_chunk)
        augment_index_with_wiki(wiki_pages_as_chunks(rendered))

    print("[pipeline] ALL DONE")


if __name__ == "__main__":
    main()
