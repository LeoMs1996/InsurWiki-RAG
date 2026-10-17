#!/usr/bin/env python3
"""评测脚本：Retrieval Hit@5 / 答案支撑率（citation faithfulness 代理）。

从 QA 三元组随机抽取 N 题（默认 60）：
  - Hit@5：混合检索 top-5 是否命中该题的证据 chunk（chunk 文本与 evidence 重叠）
  - 支撑率：走完整 Self-RAG 链路的 support_score 均值 + 拒答率
输出 artifacts/eval/report.json + 控制台表格。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from insurwiki_rag.config import Config  # noqa: E402
from insurwiki_rag.chunking.structure_aware import load_triples  # noqa: E402
from insurwiki_rag.retrieval.hybrid import load_index  # noqa: E402


def evidence_hit(top_chunks, evidence: str) -> bool:
    ev_key = evidence[:60]
    for c in top_chunks:
        if c["text"][:60] in evidence or ev_key in c["text"]:
            return True
    # 宽松判据：任一 top-5 chunk 与证据的字符二元组重叠 ≥0.6
    ev_bg = {evidence[i:i+2] for i in range(len(evidence)-1)}
    for c in top_chunks:
        t = c["text"]
        bg = [t[i:i+2] for i in range(len(t)-1)]
        if bg and sum(1 for b in bg if b in ev_bg) / len(bg) >= 0.6:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--skip-llm", action="store_true",
                    help="只评 Hit@5，不跑 LLM 支撑率（快速模式）")
    args = ap.parse_args()
    Config.EVAL_DIR.mkdir(parents=True, exist_ok=True)

    triples = load_triples()
    rng = random.Random(7)
    sample = rng.sample(triples, min(args.n, len(triples)))
    retriever = load_index()

    hits, rows = 0, []
    for i, qa in enumerate(sample):
        top = retriever.search(qa["question"], top_k=15, top_n=5)
        hit = evidence_hit(top, qa["evidence"])
        hits += int(hit)
        rows.append({"question": qa["question"], "hit": hit,
                     "top_chunk_ids": [c["chunk_id"] for c in top],
                     "gold_evidence": qa["evidence"][:100]})
        if (i + 1) % 20 == 0:
            print(f"[eval] {i+1}/{len(sample)} Hit@5={hits/(i+1):.3f}")

    report = {"n": len(sample), "hit_at_5": round(hits / len(sample), 4), "rows": rows}

    if not args.skip_llm:
        from insurwiki_rag.graphrag.kg import KnowledgeGraph
        from insurwiki_rag.selfrag.graph import SelfRAGGraph
        kg = KnowledgeGraph.load() if Config.GRAPH_PATH.exists() else None
        rag = SelfRAGGraph(retriever, kg)
        supports, refused = [], 0
        sub = sample[:20]  # 控制 LLM 调用量
        for j, qa in enumerate(sub):
            out = rag.ask(qa["question"])
            supports.append(out.get("support_score", 0.0))
            refused += int(bool(out.get("refused")))
            print(f"[eval-llm] {j+1}/{len(sub)} support={out.get('support_score')}")
        report.update({
            "support_rate_mean": round(sum(supports) / len(supports), 4),
            "refusal_rate": round(refused / len(sub), 4),
            "support_samples": len(sub),
        })

    out_path = Config.EVAL_DIR / "report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n===== 评测结果 =====")
    print(f"Hit@5: {report['hit_at_5']:.2%}  (n={report['n']})")
    if "support_rate_mean" in report:
        print(f"答案支撑率均值: {report['support_rate_mean']:.2%}")
        print(f"拒答率: {report['refusal_rate']:.2%}")
    print(f"报告已写入 {out_path}")


if __name__ == "__main__":
    main()
