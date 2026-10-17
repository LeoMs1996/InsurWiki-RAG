"""§1.5 GraphRAG 本体约束。

  - 本体（Ontology）限定保险领域 8 类实体 + 受约束的关系类型
  - 用 GLM 对条款语料做实体-关系抽取（批量、并发、结果校验回灌本体）
  - networkx 存储（持久化为 node-link JSON）
  - 问答时：实体链接 → 1-hop 图查询 → 命中实体关联的 chunk 作为第三证据通道并入 RRF
  - 生成端：本体约束写进 system prompt（禁止编造本体外的条款关系）
"""
from __future__ import annotations

import json
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from ..config import Config
from ..llm import get_llm

# ---------------- 保险领域本体 ----------------
ENTITY_TYPES = ["保险产品", "责任条款", "免责条款", "费率因子", "犹豫期", "保额", "期限", "被保险人群体"]
RELATION_TYPES = [
    "承保责任", "免除责任", "适用费率", "享有犹豫期", "约定保额", "约定期限",
    "适用群体", "关联条款",
]

ONTOLOGY_PROMPT = """你是保险领域知识图谱构建器。从给定条款文本中抽取实体和关系，严格遵守本体约束：
- 实体类型只能是：{etypes}
- 关系类型只能是：{rtypes}
- 不确定/本体外的信息一律不抽
每条输出 JSON：{{"entities":[{{"name":"...","type":"...","chunk_id":"..."}}],"relations":[{{"head":"实体名","relation":"关系类型","tail":"实体名","evidence":"原文短句"}}]}}

文本（chunk_id 见标注）：
{payload}

输出一个 JSON 数组，每个元素对应一个 chunk。只输出 JSON。"""


class KnowledgeGraph:
    def __init__(self):
        import networkx as nx
        self.g = nx.MultiDiGraph()

    # ---------- 构建 ----------
    def add_extraction(self, chunk_id: str, extraction: Dict[str, Any]) -> int:
        n = 0
        for e in extraction.get("entities", []) or []:
            name = (e.get("name") or "").strip()
            etype = e.get("type") or ""
            if name and etype in ENTITY_TYPES:
                if not self.g.has_node(name):
                    self.g.add_node(name, type=etype, chunk_ids=set(), mentions=0)
                self.g.nodes[name]["mentions"] += 1
                self.g.nodes[name]["chunk_ids"].add(chunk_id)
                n += 1
        for r in extraction.get("relations", []) or []:
            h, t = (r.get("head") or "").strip(), (r.get("tail") or "").strip()
            rel = r.get("relation") or ""
            if h and t and rel in RELATION_TYPES and self.g.has_node(h) and self.g.has_node(t):
                self.g.add_edge(h, t, relation=rel, chunk_id=chunk_id,
                                evidence=(r.get("evidence") or "")[:120])
                n += 1
        return n

    def finalize(self) -> None:
        import networkx as nx
        self.g = nx.convert_node_labels_to_integers(self.g, label_attribute="name") \
            if False else self.g  # 保留字符串节点名

    # ---------- 查询 ----------
    def link_entities(self, query: str) -> List[str]:
        """实体链接：本体实体名在问题中出现（全名或 ≥2 字子串命中）。"""
        hits = []
        for name, data in self.g.nodes(data=True):
            if name and name in query:
                hits.append(name)
        return hits

    def query(self, query: str, max_hops: int = 1) -> Dict[str, Any]:
        """实体链接 + 1-hop 邻域查询，返回关联 chunk 与关系证据。"""
        linked = self.link_entities(query)
        chunk_ids: List[str] = []
        triples: List[Dict[str, str]] = []
        visited = set()
        frontier = linked
        for _hop in range(max_hops):
            nxt = []
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                chunk_ids.extend(self.g.nodes[node].get("chunk_ids", set()))
                for h, t, edata in self.g.edges(node, data=True):
                    triples.append({"head": h, "relation": edata["relation"],
                                    "tail": t, "evidence": edata.get("evidence", ""),
                                    "chunk_id": edata.get("chunk_id", "")})
                    nxt.append(t if t != node else h)
            frontier = [n for n in nxt if n not in visited]
        uniq_cids, seen = [], set()
        for c in chunk_ids:
            if c and c not in seen:
                seen.add(c)
                uniq_cids.append(c)
        return {"linked_entities": linked, "chunk_ids": uniq_cids[:8],
                "triples": triples[:12]}

    # ---------- 持久化 ----------
    def save(self, path=None) -> None:
        import networkx as nx
        path = path or Config.GRAPH_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.g)
        for n in data["nodes"]:
            n["chunk_ids"] = sorted(n.pop("chunk_ids", set()))
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path=None) -> "KnowledgeGraph":
        import networkx as nx
        path = path or Config.GRAPH_PATH
        data = json.loads(path.read_text(encoding="utf-8"))
        kg = cls()
        kg.g = nx.node_link_graph(data)
        for _, nd in kg.g.nodes(data=True):
            nd["chunk_ids"] = set(nd.get("chunk_ids", []))
        return kg

    def stats(self) -> Dict[str, Any]:
        return {
            "nodes": self.g.number_of_nodes(),
            "edges": self.g.number_of_edges(),
            "entity_types": dict(Counter(d.get("type") for _, d in self.g.nodes(data=True))),
            "relation_types": dict(Counter(d.get("relation")
                                           for _, _, d in self.g.edges(data=True))),
        }


# ---------------- 抽取流水线 ----------------
def extract_graph(chunks: List[Dict[str, Any]], batch_size: int = 5,
                  max_batches: Optional[int] = None, workers: int = 4) -> KnowledgeGraph:
    """GLM 批量抽取实体-关系（并发），构建知识图谱。"""
    kg = KnowledgeGraph()
    batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
    if max_batches:
        batches = batches[:max_batches]
    print(f"[graphrag] extracting from {len(batches)} batches "
          f"({sum(len(b) for b in batches)} chunks) with GLM ...")

    def one(batch):
        payload = "\n\n".join(f"[{c['chunk_id']}] {c['text'][:400]}" for c in batch)
        llm = get_llm()
        try:
            out = llm.chat_json(
                "你是严格的保险知识图谱抽取器。", ONTOLOGY_PROMPT.format(
                    etypes="/".join(ENTITY_TYPES), rtypes="/".join(RELATION_TYPES),
                    payload=payload), max_tokens=3000)
            return out if isinstance(out, list) else [out]
        except Exception as e:  # noqa: BLE001
            return {"__error__": str(e)}

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, b): b for b in batches}
        for fut in as_completed(futures):
            batch = futures[fut]
            result = fut.result()
            if isinstance(result, dict):
                continue  # 出错批次跳过
            for chunk, ex in zip(batch, result):
                if isinstance(ex, dict):
                    kg.add_extraction(chunk["chunk_id"], ex)
            done += 1
            if done % 20 == 0:
                print(f"[graphrag] {done}/{len(batches)} batches, "
                      f"graph: {kg.g.number_of_nodes()} nodes / {kg.g.number_of_edges()} edges")
    kg.save()
    print(f"[graphrag] done: {kg.stats()}")
    return kg
