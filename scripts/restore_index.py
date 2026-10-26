#!/usr/bin/env python3
"""离线重建 artifacts/index（复用已有语料与 Wiki 产物，不调用 LLM）。

等价于 scripts/pipeline.py --skip-data --skip-wiki，但额外把已渲染的
Wiki 页回灌进索引（原 pipeline 第 5 步的 augment 部分，跳过 LLM 生成）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from insurwiki_rag.config import Config  # noqa: E402


def main() -> None:
    Config.ensure_dirs()

    # 2) 切分 + 3) 索引（BM25 + 向量）
    from insurwiki_rag.chunking.structure_aware import chunk_docs, load_docs
    from insurwiki_rag.retrieval.hybrid import build_index

    docs = load_docs()
    chunks = chunk_docs(docs)
    retriever = build_index(chunks)

    # 5-离线部分) 已渲染 Wiki 页 → chunk → 回灌索引
    from insurwiki_rag.wiki.sediment import augment_index_with_wiki, wiki_pages_as_chunks

    idx = Config.WIKI_DIR / "index.json"
    if idx.exists():
        rendered = json.loads(idx.read_text(encoding="utf-8"))
        augment_index_with_wiki(wiki_pages_as_chunks(rendered))

    print(f"[restore] DONE: {len(retriever.chunks)} corpus chunks (+ wiki chunks)")


if __name__ == "__main__":
    main()
