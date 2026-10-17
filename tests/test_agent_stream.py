"""Agent 流式输出测试：llm.chat_stream / AdvisorAgent(on_event) / SSE 端点。"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from insurwiki_rag.agent.advisor import AdvisorAgent  # noqa: E402


class _FakeLLM:
    """mock LLM：chat_json 返回 rag 意图；chat/chat_stream 返回固定文本。"""

    active_model = "fake-model"

    def chat_json(self, system, user, max_tokens=2048):
        return {"intent": "rag", "confidence": 0.9, "reason": "test"}

    def chat(self, system, user, max_tokens=2048, temperature=0.3):
        return "结论：测试回答。"

    def chat_stream(self, system, user, max_tokens=2048, temperature=0.3):
        for piece in ["结论：", "流式", "回答。"]:
            yield piece


def _make_agent() -> AdvisorAgent:
    retriever = MagicMock()
    retriever.search.return_value = [
        {"chunk_id": "c1", "policy_name": "测试保单", "section_path": "第一条",
         "text": "测试条款内容" * 10, "rrf_score": 0.5, "rerank_score": 0.8}]
    return AdvisorAgent(retriever=retriever, kg=None, llm=_FakeLLM())


def test_chat_stream_events_order_and_answer():
    agent = _make_agent()
    events = []
    out = agent.chat("s1", "充电桩保险的免赔额是怎么规定的？",
                     on_event=lambda t, p: events.append((t, p)))
    types = [t for t, _ in events]
    # 事件顺序：intent → tools → delta(×3) → citations → done
    assert types[0] == "intent"
    assert "tools" in types
    assert types.count("delta") == 3
    assert types[-2] == "citations" and types[-1] == "done"
    # done 事件的 answer 与增量拼接一致（叠加合规免责后）
    done_payload = events[-1][1]
    assert done_payload["answer"].startswith("结论：流式回答。")
    assert out["answer"] == done_payload["answer"]
    assert out["latency_ms"] >= 0 and out["model"] == "fake-model"


def test_chat_without_callback_unchanged():
    agent = _make_agent()
    out = agent.chat("s2", "充电桩保险的免赔额是怎么规定的？")
    assert out["answer"].startswith("结论：测试回答。")
    assert "结论：流式" not in out["answer"]  # 非流式走 chat()


def test_sse_endpoint():
    from fastapi.testclient import TestClient
    from insurwiki_rag.server import main as server

    server._state.update(retriever=MagicMock(), kg=None,
                         selfrag=MagicMock(), agent=_make_agent())
    client = TestClient(server.app)
    resp = client.post("/api/agent/chat/stream",
                       json={"session_id": "t", "message": "家财险免除哪些责任？"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert "event: intent" in body
    assert "event: tools" in body
    assert "event: delta" in body
    assert "event: citations" in body
    assert "event: done" in body
    # done 帧可解析且含完整 answer
    for frame in body.split("\n\n"):
        if frame.startswith("event: done"):
            payload = json.loads(frame.split("data: ", 1)[1])
            assert payload["answer"].startswith("结论：流式回答。")
            break
    else:
        pytest.fail("no done frame")


def test_llm_stream_sse_parsing(monkeypatch):
    """Anthropic SSE 响应能被正确解析为增量。"""
    import io
    from insurwiki_rag import llm as llm_mod
    from insurwiki_rag.config import Config

    sse = ("\r\n".join([
        'event: message_start',
        'data: {"type":"message_start"}',
        '',
        'event: content_block_delta',
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"你好"}}',
        '',
        'event: content_block_delta',
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"，世界"}}',
        '',
        'event: message_stop',
        'data: {"type":"message_stop"}',
        '',
    ]))

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        return _Resp(sse.encode())

    client = llm_mod.GLMClient()
    monkeypatch.setattr(client, "base_url", Config.LLM_BASE_URL.rstrip("/") or "x/anthropic")
    client.anthropic = True
    client.api_key = "test-key"
    client.fallback = None
    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_urlopen)
    parts = list(client.chat_stream("sys", "user"))
    assert parts == ["你好", "，世界"]
