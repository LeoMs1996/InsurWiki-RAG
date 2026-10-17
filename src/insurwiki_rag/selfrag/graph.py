"""§1.4 Self-RAG + §1.6 金融合规问答：LangGraph 显式建模反思循环。

状态机（SelfRAGGraph）：
  retrieve → grade_relevant [IsRelevant?] →（不相关 → rewrite+re-retrieve 一次）
    → generate（带引用生成）→ grade_supported [IsSupported?]（逐句与证据比对）
    → grade_useful [IsUseful?] → 合规检查（拒答阈值/敏感词/免责声明）→ 装配引用输出

反思 token：[Retrieve?] [IsRelevant?] [IsSupported?] [IsUseful?] 全程记录在 trace，
可通过 /api/chat 响应与日志查看（验收标准 #3）。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, TypedDict

from ..config import Config
from ..llm import get_llm
from ..retrieval.hybrid import HybridRetriever

logger = logging.getLogger("insurwiki.selfrag")

# ---------------- 合规护栏 ----------------
SENSITIVE_WORDS = ["保证收益", "稳赚不赔", "绝对安全", "保本保息", "理赔全赔"]
DISCLAIMER = ("以上内容基于所引保险条款原文生成，仅供参考，不构成保险要约或承诺；"
              "具体保障责任、免责事项与理赔标准以保险合同及保险人官方解释为准。")


def sensitive_check(text: str) -> List[str]:
    return [w for w in SENSITIVE_WORDS if w in text]


class SelfRAGState(TypedDict, total=False):
    query: str
    rewrite: str
    retrieve_decision: str            # [Retrieve?] yes/no
    candidates: List[Dict[str, Any]]  # 检索候选（含图谱通道）
    graph_evidence: Dict[str, Any]
    relevant: List[Dict[str, Any]]    # [IsRelevant?] 通过的证据
    reflection: List[Dict[str, Any]]  # 反思 token 轨迹
    answer: str
    support_score: float              # [IsSupported?]
    useful: bool                      # [IsUseful?]
    refused: bool
    citations: List[Dict[str, Any]]
    retried: bool


class SelfRAGGraph:
    """基于 LangGraph 的 Self-RAG 反思循环编排。"""

    def __init__(self, retriever: HybridRetriever,
                 kg: Optional[Any] = None, llm: Optional[Any] = None):
        self.retriever = retriever
        self.kg = kg
        self.llm = llm or get_llm()
        self._graph = self._build()

    # ---------- 节点 ----------
    def _node_retrieve(self, state: SelfRAGState) -> SelfRAGState:
        q = state.get("rewrite") or state["query"]
        graph_evidence: Dict[str, Any] = {"linked_entities": [], "chunk_ids": [], "triples": []}
        extra_channel = None
        if self.kg is not None:
            graph_evidence = self.kg.query(q)
            if graph_evidence["chunk_ids"]:
                by_id = {c["chunk_id"]: c for c in self.retriever.chunks}
                extra_channel = [by_id[cid] for cid in graph_evidence["chunk_ids"]
                                 if cid in by_id]
        candidates = self.retriever.search(q, top_k=15, top_n=6,
                                           extra_channel=extra_channel)
        state["retrieve_decision"] = "[Retrieve?] yes"
        state["candidates"] = candidates
        state["graph_evidence"] = graph_evidence
        if not candidates:
            state["retrieve_decision"] = "[Retrieve?] no"
            state["refused"] = True
            state["answer"] = "依据不足：知识库中未检索到与该问题相关的条款内容。"
        state.setdefault("reflection", []).append({
            "token": "[Retrieve?]", "value": state["retrieve_decision"],
            "n_candidates": len(candidates),
            "graph_linked_entities": graph_evidence["linked_entities"]})
        return state

    def _node_grade_relevant(self, state: SelfRAGState) -> SelfRAGState:
        cands = state.get("candidates", [])
        relevant = [c for c in cands if (c.get("rerank_score") or 0) > 0.3
                    or (c.get("rrf_score") or 0) >= 0.03]
        state["relevant"] = relevant
        state["reflection"].append({
            "token": "[IsRelevant?]",
            "value": "yes" if relevant else "no",
            "detail": [c["chunk_id"] for c in relevant]})
        return state

    def _node_rewrite(self, state: SelfRAGState) -> SelfRAGState:
        """不相关 → LLM 改写查询，重检索一次（Self-RAG 的 Retrieve-again 循环）。"""
        if not state.get("relevant") and not state.get("retried") and not state.get("refused"):
            try:
                state["rewrite"] = self.llm.chat(
                    "你是保险领域检索查询改写器。把用户问题改写为更适合检索的"
                    "包含关键词的完整句子，只输出改写后的问题。",
                    state["query"], max_tokens=200).strip()
                state["retried"] = True
                state["reflection"].append({
                    "token": "[Retrieve?]", "value": "yes(rewrite)",
                    "rewrite": state["rewrite"]})
                return self._node_retrieve(state) | {"reflection": state["reflection"]}
            except Exception:  # noqa: BLE001
                pass
        state["refused"] = not bool(state.get("relevant"))
        return state

    def _node_generate(self, state: SelfRAGState) -> SelfRAGState:
        if state.get("refused"):
            state.setdefault("answer", "依据不足：未找到与问题相关的条款证据，无法作答。")
            return state
        ev = "\n\n".join(
            f"[{i+1}] chunk_id={c['chunk_id']}（{c.get('policy_name','')}·{c.get('section_path','')}）\n{c['text']}"
            for i, c in enumerate(state["relevant"][:5]))
        graph_ctx = ""
        ge = state.get("graph_evidence") or {}
        if ge.get("triples"):
            graph_ctx = "\n\n图谱证据（本体约束内，仅可引用，不得编造其外关系）：\n" + "\n".join(
                f"- {t['head']} --{t['relation']}--> {t['tail']}（{t['evidence'][:40]}）"
                for t in ge["triples"][:6])
        answer = self.llm.chat(
            "你是严谨的保险条款助手。只依据给出的证据回答；"
            "引用证据时标注编号[n]；不得编造；若证据不足以回答，明确说“依据不足”。"
            "回答以「结论：」开头，先给结论再给条款依据。",
            f"问题：{state['query']}\n\n证据：\n{ev}{graph_ctx}", max_tokens=1500)
        state["answer"] = answer
        return state

    def _node_grade_supported(self, state: SelfRAGState) -> SelfRAGState:
        """[IsSupported?] 逐句答案-证据词面支撑度（citation faithfulness 代理指标）。"""
        if state.get("refused"):
            state["support_score"] = 0.0
            state["reflection"].append({"token": "[IsSupported?]", "value": "skip(refused)"})
            return state
        sents = [s for s in re.split(r"(?<=[。；！？])", state.get("answer", "")) if s.strip()]
        ev_text = "".join(c["text"] for c in state.get("relevant", []))
        ev_bigrams = {ev_text[i:i+2] for i in range(len(ev_text) - 1)}
        supported = 0
        for s in sents:
            bg = [s[i:i+2] for i in range(len(s) - 1)]
            if not bg:
                continue
            cover = sum(1 for b in bg if b in ev_bigrams) / len(bg)
            if cover >= 0.5:
                supported += 1
        score = supported / len(sents) if sents else 0.0
        state["support_score"] = round(score, 4)
        state["reflection"].append({
            "token": "[IsSupported?]",
            "value": f"{score:.2f}",
            "detail": f"{supported}/{len(sents)} 句有证据支撑"})
        if score < Config.SUPPORT_THRESHOLD:
            state["refused"] = True
            state["answer"] = ("依据不足：现有条款证据对问题的支撑度较低"
                               f"（支撑率 {score:.0%}，低于阈值 {Config.SUPPORT_THRESHOLD:.0%}），"
                               "为避免误导不作答。请补充更具体的问题或查阅原合同。")
        return state

    def _node_grade_useful(self, state: SelfRAGState) -> SelfRAGState:
        if state.get("refused"):
            state["useful"] = False
            state["reflection"].append({"token": "[IsUseful?]", "value": "skip(refused)"})
            return state
        useful = ("依据不足" not in state["answer"]) and len(state["answer"]) > 20
        state["useful"] = useful
        state["reflection"].append({"token": "[IsUseful?]", "value": "yes" if useful else "no"})
        return state

    def _node_compliance(self, state: SelfRAGState) -> SelfRAGState:
        if not state.get("refused"):
            hit = sensitive_check(state["answer"])
            if hit:
                state["answer"] += f"\n\n（提示：检测到非规范表述 {hit}，请以条款原文为准。）"
            state["answer"] += f"\n\n{DISCLAIMER}"
        # 引用装配：chunk_id → 条款号 → 原文摘录
        cits = []
        for c in state.get("relevant", [])[:5]:
            excerpt = re.sub(r"\s+", " ", c["text"])[:80]
            cits.append({"chunk_id": c["chunk_id"], "n": c.get("section_path", ""),
                         "policy": c.get("policy_name", ""),
                         "excerpt": excerpt,
                         "scores": {"rrf": c.get("rrf_score"),
                                    "rerank": c.get("rerank_score"),
                                    "from_graph": c.get("from_graph", False)}})
        state["citations"] = cits
        logger.info("selfrag %s refused=%s support=%s",
                    state["query"][:20], state.get("refused"), state.get("support_score"))
        return state

    # ---------- 图 ----------
    def _build(self):
        from langgraph.graph import StateGraph, END, START
        g: StateGraph = StateGraph(SelfRAGState)
        g.add_node("retrieve", self._node_retrieve)
        g.add_node("grade_relevant", self._node_grade_relevant)
        g.add_node("rewrite", self._node_rewrite)
        g.add_node("generate", self._node_generate)
        g.add_node("grade_supported", self._node_grade_supported)
        g.add_node("grade_useful", self._node_grade_useful)
        g.add_node("compliance", self._node_compliance)
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "grade_relevant")
        g.add_conditional_edges("grade_relevant", lambda s: "rewrite" if not s.get("relevant") else "generate",
                                {"rewrite": "rewrite", "generate": "generate"})
        g.add_conditional_edges("rewrite", lambda s: END if s.get("refused") else "generate",
                                {END: END, "generate": "generate"})
        g.add_edge("generate", "grade_supported")
        g.add_conditional_edges("grade_supported",
                                lambda s: "compliance",
                                {"compliance": "compliance"})
        g.add_edge("compliance", "grade_useful")
        g.add_edge("grade_useful", END)
        return g.compile()

    # ---------- 执行 ----------
    def ask(self, query: str) -> Dict[str, Any]:
        import time
        t0 = time.time()
        final = self._graph.invoke({"query": query})
        final.pop("candidates", None)
        final.pop("relevant", None)
        final.setdefault("refused", False)
        final.setdefault("support_score", 0.0)
        final.setdefault("citations", [])
        final["latency_ms"] = int((time.time() - t0) * 1000)
        final["model"] = self.llm.active_model
        return final
