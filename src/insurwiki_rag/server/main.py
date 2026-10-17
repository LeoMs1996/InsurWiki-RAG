"""InsurWiki-RAG FastAPI 服务：/api/chat /api/wiki /api/agent/chat /health /docs站。"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import Config
from ..retrieval.hybrid import load_index
from ..selfrag.graph import SelfRAGGraph

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("insurwiki.server")

app = FastAPI(title="InsurWiki-RAG", version="1.0.0",
              description="基于 InsQABench 的金融级保单条款 RAG 与 LLM Wiki 知识沉淀平台")

_state: dict = {}


def _bootstrap():
    """惰性加载（首个请求时）检索索引、图谱、Self-RAG 图与 Agent。"""
    if _state:
        return
    log.info("bootstrapping: index + graph + selfrag + agent ...")
    from ..graphrag.kg import KnowledgeGraph
    from ..agent.advisor import AdvisorAgent
    retriever = load_index()
    kg = KnowledgeGraph.load() if Config.GRAPH_PATH.exists() else None
    _state.update(retriever=retriever, kg=kg)
    _state["selfrag"] = SelfRAGGraph(retriever, kg)
    _state["agent"] = AdvisorAgent(retriever, kg)
    log.info("bootstrap done: %d chunks, graph=%s",
             len(retriever.chunks),
             kg.stats() if kg else None)


class ChatIn(BaseModel):
    question: str


class AgentIn(BaseModel):
    session_id: str = "default"
    message: str
    confirmed_intent: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "model": Config.LLM_MODEL,
            "indexed": _ready() and len(_state["retriever"].chunks)}


def _ready() -> bool:
    _bootstrap()
    return True


@app.post("/api/chat")
def chat(body: ChatIn):
    """作业1·合规问答（Self-RAG 全链路 + 引用可追溯）。"""
    _bootstrap()
    result = _state["selfrag"].ask(body.question)
    return JSONResponse(result)


@app.get("/api/search")
def search(q: str, top_n: int = 5):
    """裸混合检索（调试/演示）。"""
    _bootstrap()
    return _state["retriever"].search(q, top_n=top_n)


@app.get("/api/graph")
def graph(entity: Optional[str] = None):
    _bootstrap()
    kg = _state.get("kg")
    if kg is None:
        raise HTTPException(503, "graph not built")
    if entity:
        return kg.query(entity)
    st = kg.stats()
    nodes = [{"name": n, **{k: v for k, v in d.items() if k != "chunk_ids"}}
             for n, d in list(kg.g.nodes(data=True))[:400]]
    edges = [{"head": h, "tail": t, **d} for h, t, d in list(kg.g.edges(data=True))[:600]]
    return {**st, "nodes": nodes, "edges": edges}


@app.get("/api/wiki")
def wiki_list():
    idx = Config.WIKI_DIR / "index.json"
    if not idx.exists():
        return {"pages": []}
    return {"pages": json.loads(idx.read_text(encoding="utf-8"))}


@app.get("/api/wiki/{topic_id}")
def wiki_detail(topic_id: str):
    path = Config.WIKI_DIR / f"{topic_id}.md"
    if not path.exists():
        raise HTTPException(404, "wiki page not found")
    return {"topic_id": topic_id, "markdown": path.read_text(encoding="utf-8")}


@app.get("/api/export/obsidian")
def export_obsidian():
    """导出 Obsidian 库 zip（Wiki 页 + 实体双链笔记 + Graph.canvas）。"""
    from .obsidian_export import export_zip
    if not Config.GRAPH_PATH.exists():
        raise HTTPException(503, "graph not built")
    zp = Config.ARTIFACTS / "obsidian_vault.zip"
    export_zip(zp)
    return FileResponse(zp, filename="InsurWiki-Obsidian.zip",
                        media_type="application/zip")


@app.post("/api/agent/chat")
def agent_chat(body: AgentIn):
    """作业2·保险顾问 Agent（多轮 + 工具调用 + human-in-the-loop）。"""
    _bootstrap()
    return _state["agent"].chat(body.session_id, body.message, body.confirmed_intent)


@app.post("/api/agent/chat/stream")
def agent_chat_stream(body: AgentIn):
    """流式版 Agent 对话：SSE 事件 intent → tools → delta(×N) → citations → done。"""
    import queue
    import threading

    _bootstrap()
    q: queue.Queue = queue.Queue()

    def worker():
        try:
            _state["agent"].chat(
                body.session_id, body.message, body.confirmed_intent,
                on_event=lambda etype, payload: q.put((etype, payload)))
        except Exception as e:  # noqa: BLE001
            log.exception("agent stream worker failed")
            q.put(("error", {"message": str(e)}))
        finally:
            q.put(None)  # 结束哨兵

    def sse():
        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = q.get()
            if item is None:
                break
            etype, payload = item
            yield (f"event: {etype}\n"
                   f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n")

    return StreamingResponse(
        sse(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ---------- Web UI ----------
_WEB = Path(__file__).resolve().parents[3] / "web"
_DOCS = Path(__file__).resolve().parents[3] / "docs" / "html"
app.mount("/static", StaticFiles(directory=str(_WEB)), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    return (_WEB / "index.html").read_text(encoding="utf-8")


@app.get("/docs/", response_class=HTMLResponse)
@app.get("/docs/{page}", response_class=HTMLResponse)
def docs(page: str = "index.html"):
    p = _DOCS / page
    if not p.exists():
        raise HTTPException(404)
    return p.read_text(encoding="utf-8")
