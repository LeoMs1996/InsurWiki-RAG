"""单元/集成测试：pytest tests/"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from insurwiki_rag.chunking.structure_aware import (  # noqa: E402
    chunk_fragment, looks_like_table, split_sentences)
from insurwiki_rag.retrieval.hybrid import rrf_fuse  # noqa: E402
from insurwiki_rag.selfrag.graph import sensitive_check  # noqa: E402


def test_pipeline_outputs_exist():
    from insurwiki_rag.config import Config
    assert (Config.DATA_PROCESSED / "qa_triples.jsonl").exists()
    assert (Config.DATA_PROCESSED / "corpus_docs.jsonl").exists()
    stats = json.loads((Config.DATA_PROCESSED / "stats.json").read_text())
    assert stats["sampled"] > 0


def test_structure_chunking_clause():
    frag = "第五条 下列原因造成的损失保险人不负责赔偿：（一）战争；（二）核辐射。第六条 保险金额由双方约定。"
    chunks = chunk_fragment(frag, 0)
    assert all(c["chunk_type"] == "clause" for c in chunks)
    paths = {c["section_path"] for c in chunks}
    assert "第五条" in paths and "第六条" in paths  # 条款边界不跨条
    assert all(c["text"] for c in chunks)


def test_table_channel():
    frag = "费率表：\n家庭财产 0.15%\n贵重物品 0.35%"
    assert looks_like_table(frag)
    chunks = chunk_fragment(frag, 0)
    assert chunks[0]["chunk_type"] == "table"
    assert "0.15%" in chunks[0]["text"]


def test_rrf_fusion():
    fused = rrf_fuse([[1, 2, 3], [2, 4, 1]], k=60)
    assert fused[2] > fused[4]  # 双路命中者得分更高
    assert abs(fused[1] - (1/61 + 1/63)) < 1e-9  # doc1: 两路分别 rank1/rank3


def test_sensitive_check():
    assert sensitive_check("本产品保证收益") == ["保证收益"]
    assert sensitive_check("依据条款约定") == []


def test_llm_client_protocol_switch():
    from insurwiki_rag.llm import GLMClient
    from insurwiki_rag.config import Config
    c = GLMClient()
    assert c.anthropic is ("/anthropic" in Config.LLM_BASE_URL)
    assert c.available


@pytest.mark.skipif(not Path("artifacts/index/chunks.jsonl").exists(),
                    reason="index not built")
def test_index_roundtrip():
    from insurwiki_rag.retrieval.hybrid import load_index
    r = load_index()
    assert len(r.chunks) > 1000
    res = r.search("保险 免责", top_n=3, use_rerank=False)
    assert len(res) >= 1
