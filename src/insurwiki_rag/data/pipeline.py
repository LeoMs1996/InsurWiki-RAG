"""§1.1 数据获取与处理：InsQABench 合同问答三元组的拉取、解析、清洗、采样。

流程（scripts/pipeline.py 调用）：
  1. fetch:   从 HF（hf-mirror 镜像）下载 clause_train.json 到 data/raw/
  2. parse:   从 instruction 中解析出 <Q, A, Evidence> 三元组 + 合同名（policy_name）
  3. clean:   去重（问题+证据指纹）、去噪（空/超短/超长）、格式规范化
  4. sample:  按 Config.DATA_SAMPLE 随机采样（固定随机种子，可复现）
  5. export:  data/processed/qa_triples.jsonl + corpus_docs.jsonl（按合同聚合的条款语料）
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from ..config import Config

RAW_URL = ("https://hf-mirror.com/datasets/JaneDing2025/InsQABench/"
           "resolve/main/clause_train.json")

_RE_CONTRACT = re.compile(r"这段文字来自的合同是[:：]\s*(?P<name>.+?)\n")
_RE_FRAGMENT = re.compile(r"片段内容[:：]\s*(?P<frag>.*?)\n用户提问[:：]\s*(?P<q>.+)$",
                          re.S)


def fetch(raw_path: Path | None = None) -> Path:
    """下载 InsQABench 合同问答训练集到 data/raw/。"""
    Config.ensure_dirs()
    path = raw_path or Config.DATA_RAW / "insqabench_clause_train.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[fetch] downloading {RAW_URL} ...")
        req = urllib.request.Request(RAW_URL, headers={"User-Agent": "InsurWiki-RAG/1.0"})
        with urllib.request.urlopen(req, timeout=600) as resp, open(path, "wb") as f:
            f.write(resp.read())
    print(f"[fetch] raw data ready: {path} ({path.stat().st_size/1e6:.1f} MB)")
    return path


def parse_instruction(item: Dict[str, str]) -> Dict[str, str] | None:
    """从 InsQABench 的 instruction 字段解析 <Q, Evidence, 合同名>。"""
    instr, output = item.get("instruction", ""), item.get("output", "")
    m_name = _RE_CONTRACT.search(instr)
    m_frag = _RE_FRAGMENT.search(instr)
    if not (m_name and m_frag and instr and output):
        return None
    frag = m_frag.group("frag").strip()
    q = m_frag.group("q").strip()
    if len(frag) < 30 or len(q) < 4 or len(output) < 4:
        return None  # 去噪：空/超短样本
    return {
        "policy_name": m_name.group("name").strip(),
        "evidence": frag,
        "question": q,
        "answer": output,
    }


def _fp(text: str) -> str:
    return hashlib.md5(re.sub(r"\s+", "", text).encode()).hexdigest()


def clean_and_sample(records: List[Dict[str, str]],
                     sample_n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """去重（问题+证据指纹）→ 均匀随机采样 → 规范化字段。"""
    seen, uniq = set(), []
    for r in records:
        fp = _fp(r["question"] + "|" + r["evidence"][:120])
        if fp in seen:
            continue
        seen.add(fp)
        uniq.append({
            "qa_id": f"qa-{len(seen):05d}",
            "policy_name": r["policy_name"],
            "policy_id": "P" + _fp(r["policy_name"])[:10],
            "question": r["question"],
            "answer": r["answer"],
            "evidence": r["evidence"],
        })
    rng = random.Random(seed)
    picked = uniq if len(uniq) <= sample_n else rng.sample(uniq, sample_n)
    picked.sort(key=lambda x: x["qa_id"])
    print(f"[clean] {len(records)} raw -> {len(uniq)} unique -> sampled {len(picked)}")
    return picked


def build_corpus(triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按合同聚合证据片段为"合同文档"，供结构感知切分使用。"""
    by_policy: Dict[str, Dict[str, Any]] = {}
    for t in triples:
        p = by_policy.setdefault(t["policy_id"], {
            "policy_id": t["policy_id"], "policy_name": t["policy_name"],
            "fragments": [], "qa_ids": [],
        })
        if t["evidence"] not in p["fragments"]:
            p["fragments"].append(t["evidence"])
        p["qa_ids"].append(t["qa_id"])
    docs = []
    for i, p in enumerate(sorted(by_policy.values(), key=lambda x: x["policy_id"])):
        docs.append({
            "doc_id": f"doc-{i:04d}",
            "policy_id": p["policy_id"],
            "policy_name": p["policy_name"],
            "text": "\n\n".join(p["fragments"]),
            "fragments": p["fragments"],
            "qa_ids": sorted(set(p["qa_ids"])),
        })
    return docs


def run(sample_n: int | None = None) -> Dict[str, Any]:
    """执行完整数据处理流水线，返回统计信息。"""
    n = sample_n if sample_n is not None else Config.DATA_SAMPLE
    raw_path = fetch()
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    parsed = [r for r in (parse_instruction(x) for x in raw) if r]
    triples = clean_and_sample(parsed, n)
    docs = build_corpus(triples)

    Config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    _dump_jsonl(Config.DATA_PROCESSED / "qa_triples.jsonl", triples)
    _dump_jsonl(Config.DATA_PROCESSED / "corpus_docs.jsonl", docs)
    stats = {
        "raw_records": len(raw),
        "parsed": len(parsed),
        "sampled": len(triples),
        "policies": len(docs),
        "fragments": sum(len(d["fragments"]) for d in docs),
    }
    (Config.DATA_PROCESSED / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[export] {stats}")
    return stats


def _dump_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
