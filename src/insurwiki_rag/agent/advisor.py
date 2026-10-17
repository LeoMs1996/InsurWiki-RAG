"""作业2：LangGraph 保险顾问 Agent —— 多轮对话 + 意图路由 + 工具调用 + 人工确认。

状态机：
  intent（意图识别）→ 路由到
    ├─ rag 分支（知识问答 → rag_search + graph_query）
    ├─ graph 分支（图谱查询 → graph_query）
    ├─ calc 分支（费率/保额计算 → calculator）
    └─ chat 分支（闲聊/澄清）
  → synthesize（综合生成，带引用）→ compliance（合规护栏）→ 引用装配
  human-in-the-loop：意图置信度低或工具结果冲突时挂起，等待用户确认（interrupt）。

工具集：rag_search / graph_query / wiki_lookup / calculator / compliance_check
记忆：会话级（SessionStore，多轮上下文）
"""
from __future__ import annotations

import ast
import datetime
import json
import logging
import operator
import re
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from ..config import Config
from ..llm import get_llm
from ..retrieval.hybrid import HybridRetriever
from ..selfrag.graph import DISCLAIMER, sensitive_check

logger = logging.getLogger("insurwiki.agent")

INTENT_SYSTEM = """你是保险顾问 Agent 的意图分类器。输出 JSON：
{"intent":"rag|graph|calc|chat","confidence":0~1,"reason":"简短理由"}
- rag：保险条款/责任/免责/犹豫期等知识问答
- graph：询问实体关系（如"X和Y什么关系""X承保哪些责任"）
- calc：费率、保额、保费等数值计算
- chat：问候、闲聊、澄清。只输出 JSON。"""


def _safe_eval(expr: str) -> float:
    node = ast.parse(expr, mode="eval")
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd, ast.Mod)
    for n in ast.walk(node):
        if not isinstance(n, allowed):
            raise ValueError(f"illegal token: {ast.dump(n)}")
    return float(eval(compile(node, "", "eval")))  # noqa: S307


# ---------------- 工具 ----------------
class ToolBox:
    """Agent 五工具：rag_search / graph_query / wiki_lookup / calculator / compliance_check。"""

    def __init__(self, retriever: HybridRetriever, kg=None):
        self.retriever = retriever
        self.kg = kg

    def schema(self) -> List[Dict[str, Any]]:
        return [
            {"name": "rag_search", "desc": "混合检索保单条款（BM25+向量+RRF+Rerank）",
             "args": {"query": "str", "top_n": "int=5"}},
            {"name": "graph_query", "desc": "知识图谱查询（保险本体实体关系）",
             "args": {"query": "str"}},
            {"name": "wiki_lookup", "desc": "查询已沉淀的 Wiki 知识页",
             "args": {"keyword": "str"}},
            {"name": "calculator", "desc": "费率/保额计算（安全算术表达式）",
             "args": {"expression": "str 如 '100000*0.0025'"}},
            {"name": "compliance_check", "desc": "合规护栏检查（敏感词/免责）",
             "args": {"text": "str"}},
        ]

    def rag_search(self, query: str, top_n: int = 5) -> List[Dict[str, Any]]:
        extra = None
        if self.kg is not None:
            gq = self.kg.query(query)
            if gq["chunk_ids"]:
                by_id = {c["chunk_id"]: c for c in self.retriever.chunks}
                extra = [by_id[c] for c in gq["chunk_ids"] if c in by_id]
        res = self.retriever.search(query, top_n=top_n, extra_channel=extra)
        return [{"chunk_id": c["chunk_id"], "policy": c.get("policy_name"),
                 "section": c.get("section_path"), "text": c["text"][:200],
                 "scores": {"rrf": c.get("rrf_score"), "rerank": c.get("rerank_score")}}
                for c in res]

    def graph_query(self, query: str) -> Dict[str, Any]:
        if self.kg is None:
            return {"error": "graph not loaded"}
        return self.kg.query(query)

    def wiki_lookup(self, keyword: str) -> List[Dict[str, Any]]:
        idx_path = Config.WIKI_DIR / "index.json"
        if not idx_path.exists():
            return []
        pages = json.loads(idx_path.read_text(encoding="utf-8"))
        hits = []
        for p in pages:
            hay = p["title"] + p.get("overview", "") + "".join(p.get("keywords", []))
            if keyword and keyword in hay:
                hits.append({"topic_id": p["topic_id"], "title": p["title"],
                             "overview": p.get("overview", ""), "file": p["file"]})
        return hits[:5]

    def calculator(self, expression: str) -> Dict[str, Any]:
        try:
            expr = expression.replace("×", "*").replace("÷", "/").replace("%", "/100")
            return {"expression": expression, "result": _safe_eval(expr)}
        except Exception as e:  # noqa: BLE001
            return {"error": f"无法计算：{e}"}

    def compliance_check(self, text: str) -> Dict[str, Any]:
        return {"sensitive_words": sensitive_check(text),
                "disclaimer": DISCLAIMER}

    def run(self, name: str, args: Dict[str, Any]) -> Any:
        fn = getattr(self, name, None)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**args)
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}


# ---------------- 会话记忆 ----------------
class SessionStore:
    """会话级对话记忆（演示用内存实现，接口可换 Redis）。"""

    def __init__(self, max_turns: int = 8):
        self.max_turns = max_turns
        self._data: Dict[str, List[Dict[str, str]]] = {}

    def history(self, sid: str) -> List[Dict[str, str]]:
        return self._data.setdefault(sid, [])

    def append(self, sid: str, role: str, content: str) -> None:
        h = self.history(sid)
        h.append({"role": role, "content": content})
        if len(h) > self.max_turns * 2:
            del h[: len(h) - self.max_turns * 2]


class AgentState(TypedDict, total=False):
    session_id: str
    user_input: str
    history: List[Dict[str, str]]
    intent: Dict[str, Any]
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Any]
    answer: str
    citations: List[Dict[str, Any]]
    pending_confirmation: Dict[str, Any]  # human-in-the-loop 挂起状态
    need_human: bool


class AdvisorAgent:
    """LangGraph 状态机：意图路由 → 工具调用 → 综合生成 → 合规检查 → 引用装配。"""

    LOW_CONFIDENCE = 0.55

    def __init__(self, retriever: HybridRetriever, kg=None, llm=None):
        self.llm = llm or get_llm()
        self.tools = ToolBox(retriever, kg)
        self.sessions = SessionStore()
        self._on_event = None  # 流式回调：(type, payload) -> None
        self._graph = self._build()

    def _emit(self, etype: str, payload: Any) -> None:
        if self._on_event is not None:
            try:
                self._on_event(etype, payload)
            except Exception:  # noqa: BLE001  事件回调失败不影响主流程
                logger.exception("on_event callback failed (%s)", etype)

    # ---------- 节点 ----------
    def _n_intent(self, s: AgentState) -> AgentState:
        hist = "\n".join(f"{m['role']}: {m['content'][:80]}" for m in s["history"][-4:])
        if s.get("intent", {}).get("reason") == "user-confirmed":
            return s  # 用户已确认意图，跳过 LLM 重判
        try:
            it = self.llm.chat_json(
                INTENT_SYSTEM,
                f"对话历史：\n{hist}\n\n当前输入：{s['user_input']}", max_tokens=300)
            s["intent"] = {"intent": it.get("intent", "chat"),
                           "confidence": float(it.get("confidence", 0.5)),
                           "reason": it.get("reason", "")}
        except Exception:  # noqa: BLE001
            s["intent"] = {"intent": "rag", "confidence": 0.5, "reason": "fallback"}
        # human-in-the-loop：低置信度 → 挂起确认
        self._emit("intent", s["intent"])
        confirmed = s.get("intent", {}).get("reason") == "user-confirmed"
        vague = (not confirmed) and (len(s["user_input"].strip()) < 8 or re.search(r"那个|这个|嗯|随便|帮我看看", s["user_input"]))
        if s["intent"]["confidence"] < self.LOW_CONFIDENCE or vague:
            s["need_human"] = True
            s["pending_confirmation"] = {
                "question": f"您是想咨询保险条款知识（rag）、实体关系（graph）、"
                            f"还是保费/保额计算（calc）？当前识别为「{s['intent']['intent']}」"
                            f"（置信度 {s['intent']['confidence']:.2f}），请确认。",
                "guessed_intent": s["intent"]["intent"]}
        return s

    def _n_tools(self, s: AgentState) -> AgentState:
        it = s["intent"]["intent"]
        calls, results = [], []
        if it == "rag":
            calls.append({"tool": "rag_search", "args": {"query": s["user_input"], "top_n": 5}})
            gq = self.tools.graph_query(s["user_input"])
            if gq.get("triples"):
                calls.append({"tool": "graph_query", "args": {"query": s["user_input"][:50]}})
                results.append(gq)
        elif it == "graph":
            calls.append({"tool": "graph_query", "args": {"query": s["user_input"]}})
        elif it == "calc":
            raw = s["user_input"]
            norm = (raw.replace("万", "*10000").replace("万保额", "*10000")
                       .replace("×", "*").replace("x", "*").replace("X", "*")
                       .replace("÷", "/").replace("％", "%"))
            ms = re.findall(r"[\d.]+(?:\s*[%*/+\-]\s*\d+(?:\.\d+)?%?|\d)*%?", norm)
            if len(ms) == 2 and ms[1].endswith("%"):
                expr = ms[0] + "*" + ms[1]  # 保额 × 费率%
            elif ms:
                expr = ms[0]
            else:
                expr = raw
            calls.append({"tool": "calculator", "args": {"expression": expr}})
        for c in calls:
            if c["tool"] != "graph_query":
                results.insert(0, self.tools.run(c["tool"], c["args"]))
        s["tool_calls"], s["tool_results"] = calls, results
        self._emit("tools", {
            "tool_calls": [{"tool": c["tool"], "args": c["args"]} for c in calls],
            "tool_results": json.dumps(results, ensure_ascii=False, default=str)[:1200],
        })
        return s

    def _n_synthesize(self, s: AgentState) -> AgentState:
        if s.get("need_human"):
            s["answer"] = s["pending_confirmation"]["question"]
            self._emit("delta", {"text": s["answer"]})
            return s
        it = s["intent"]["intent"]
        ctx = json.dumps(s.get("tool_results"), ensure_ascii=False)[:3000]
        hist = "\n".join(f"{m['role']}: {m['content'][:60]}" for m in s["history"][-4:])
        cits = []
        for r in s.get("tool_results") or []:
            if isinstance(r, list):
                for c in r:
                    if isinstance(c, dict) and "chunk_id" in c:
                        cits.append({"chunk_id": c["chunk_id"], "n": c.get("section"),
                                     "policy": c.get("policy"), "excerpt": (c.get("text") or "")[:80]})
        if it == "chat":
            sys_p = "你是友好的保险顾问助手，简短回应，并引导用户提出保险条款问题。"
        elif it == "calc":
            sys_p = "你是保险顾问。基于计算工具结果给出简要解读（费率、保额含义），提醒以合同为准。"
        else:
            sys_p = ("你是严谨的保险顾问。只依据工具返回的条款/图谱证据回答，"
                     "不得编造；以「结论：」开头；引用格式 [条款号/合同名]。"
                     "证据不足时明确说明。")
        prompt = (f"对话历史：\n{hist}\n\n用户：{s['user_input']}\n\n工具结果：\n{ctx}")
        if self._on_event is not None:  # 流式：逐增量回调
            parts: List[str] = []
            for delta in self.llm.chat_stream(sys_p, prompt, max_tokens=1200):
                parts.append(delta)
                self._emit("delta", {"text": delta})
            s["answer"] = "".join(parts)
        else:
            s["answer"] = self.llm.chat(sys_p, prompt, max_tokens=1200)
        s["citations"] = cits[:5]
        return s

    def _n_compliance(self, s: AgentState) -> AgentState:
        if s.get("need_human"):
            return s
        chk = self.tools.compliance_check(s["answer"])
        if chk["sensitive_words"]:
            s["answer"] += f"\n\n（合规提示：非规范表述 {chk['sensitive_words']}）"
        s["answer"] += f"\n\n{DISCLAIMER}"
        self.sessions.append(s["session_id"], "user", s["user_input"])
        self.sessions.append(s["session_id"], "assistant", s["answer"])
        return s

    # ---------- 图 ----------
    def _build(self):
        from langgraph.graph import END, START, StateGraph
        g: StateGraph = StateGraph(AgentState)
        g.add_node("intent", self._n_intent)
        g.add_node("tools", self._n_tools)
        g.add_node("synthesize", self._n_synthesize)
        g.add_node("compliance", self._n_compliance)
        g.add_edge(START, "intent")
        g.add_conditional_edges(
            "intent",
            lambda s: "synthesize" if s.get("need_human") else "tools",
            {"tools": "tools", "synthesize": "synthesize"})
        g.add_edge("tools", "synthesize")
        g.add_edge("synthesize", "compliance")
        g.add_edge("compliance", END)
        return g.compile()

    # ---------- 对外 ----------
    def chat(self, session_id: str, user_input: str,
             confirmed_intent: Optional[str] = None,
             on_event=None) -> Dict[str, Any]:
        """运行 Agent。on_event(type, payload) 存在时启用流式：
        intent → tools → delta(×N) → citations → done。"""
        import time
        t0 = time.time()
        self._on_event = on_event
        try:
            init: AgentState = {
                "session_id": session_id, "user_input": user_input,
                "history": list(self.sessions.history(session_id)),
            }
            if confirmed_intent:  # 用户确认后重入
                init["intent"] = {"intent": confirmed_intent, "confidence": 0.9,
                                  "reason": "user-confirmed"}
            out = self._graph.invoke(init)
        finally:
            self._on_event = None
        out.pop("history", None)
        out["session_id"] = session_id
        out["latency_ms"] = int((time.time() - t0) * 1000)
        out["model"] = self.llm.active_model
        if on_event is not None:
            self._on_event = on_event
            try:
                self._emit("citations", {"citations": out.get("citations", [])})
                self._emit("done", out)
            finally:
                self._on_event = None
        return out
