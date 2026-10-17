"""GLM LLM 客户端：Anthropic Messages 协议（GLM Coding Plan 端点），urllib 零 SDK。

- base_url 含 /anthropic 时自动走 Anthropic 协议（x-api-key + anthropic-version）
- 显式禁用 thinking，避免思考块吃掉 max_tokens
- glm-5.3 调用被拒（如 1113）时自动降级 LLM_MODEL_FALLBACK
- 兼容 OpenAI /chat/completions 端点（如需切换供应商）
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from typing import Any

from .config import Config


class LLMError(RuntimeError):
    pass


class GLMClient:
    def __init__(self) -> None:
        self.base_url = Config.LLM_BASE_URL.rstrip("/")
        self.api_key = Config.LLM_API_KEY
        self.model = Config.LLM_MODEL
        self.fallback = Config.LLM_MODEL_FALLBACK
        self.anthropic = "/anthropic" in self.base_url
        self.active_model = self.model
        self.last_latency_ms: int = 0

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    # ---------- 对外主入口 ----------
    def chat(self, system: str, user: str, max_tokens: int = 2048,
             temperature: float = 0.3) -> str:
        """单轮 system+user 调用，返回纯文本。失败自动降级到 fallback 模型一次。"""
        try:
            return self._call(self.active_model, system, user, max_tokens, temperature)
        except LLMError:
            if self.fallback and self.fallback != self.active_model:
                self.active_model = self.fallback
                return self._call(self.fallback, system, user, max_tokens, temperature)
            raise

    def chat_stream(self, system: str, user: str, max_tokens: int = 2048,
                    temperature: float = 0.3):
        """流式版 chat：yield 文本增量。首个增量前失败自动降级 fallback 一次。"""
        try:
            yield from self._call_stream(self.active_model, system, user,
                                         max_tokens, temperature)
        except LLMError:
            if self.fallback and self.fallback != self.active_model:
                self.active_model = self.fallback
                yield from self._call_stream(self.fallback, system, user,
                                            max_tokens, temperature)
            else:
                raise

    # ---------- 协议实现 ----------
    def _call(self, model: str, system: str, user: str,
              max_tokens: int, temperature: float) -> str:
        t0 = time.time()
        if self.anthropic:
            payload = {
                "model": model, "max_tokens": max_tokens, "temperature": temperature,
                "system": system,
                "messages": [{"role": "user", "content": user}],
                "thinking": {"type": "disabled"},
            }
            req = urllib.request.Request(
                f"{self.base_url}/v1/messages",
                data=json.dumps(payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
            )
        else:
            payload = {
                "model": model, "max_tokens": max_tokens, "temperature": temperature,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
            }
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self.api_key}"},
            )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:300]
            raise LLMError(f"HTTP {e.code}: {body}") from e
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"LLM request failed: {e}") from e
        self.last_latency_ms = int((time.time() - t0) * 1000)
        text = self._extract(data)
        if not text.strip():
            raise LLMError("empty LLM response")
        return text

    def _call_stream(self, model: str, system: str, user: str,
                    max_tokens: int, temperature: float):
        """SSE 流式调用：逐行读取响应体，yield 文本增量（生成器）。"""
        t0 = time.time()
        if self.anthropic:
            payload = {
                "model": model, "max_tokens": max_tokens,
                "temperature": temperature, "stream": True,
                "system": system,
                "messages": [{"role": "user", "content": user}],
                "thinking": {"type": "disabled"},
            }
            req = urllib.request.Request(
                f"{self.base_url}/v1/messages",
                data=json.dumps(payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
            )
        else:
            payload = {
                "model": model, "max_tokens": max_tokens,
                "temperature": temperature, "stream": True,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
            }
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json",
                         "Accept": "text/event-stream",
                         "Authorization": f"Bearer {self.api_key}"},
            )
        got_any = False
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str or data_str == "[DONE]":
                        continue
                    try:
                        ev = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    delta = self._extract_delta(ev)
                    if delta:
                        got_any = True
                        yield delta
                    elif ev.get("type") in ("message_stop",) or (
                            ev.get("choices") and ev["choices"][0].get("finish_reason")):
                        break
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:300]
            raise LLMError(f"HTTP {e.code}: {body}") from e
        except LLMError:
            raise
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"LLM stream failed: {e}") from e
        self.last_latency_ms = int((time.time() - t0) * 1000)
        if not got_any:
            raise LLMError("empty LLM stream")

    def _extract_delta(self, ev: dict[str, Any]) -> str:
        """从 SSE 事件中取文本增量（兼容 Anthropic / OpenAI 协议）。"""
        if self.anthropic:
            if ev.get("type") == "content_block_delta":
                d = ev.get("delta", {}) or {}
                if d.get("type") == "text_delta":
                    return d.get("text", "")
            return ""
        choices = ev.get("choices") or []
        if choices:
            return (choices[0].get("delta") or {}).get("content", "") or ""
        return ""
    def _extract(self, data: dict[str, Any]) -> str:
        if self.anthropic:
            blocks = data.get("content", [])
            return "".join(b.get("text", "") for b in blocks
                           if isinstance(b, dict) and b.get("type") == "text")
        return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")

    # ---------- 结构化 JSON 输出辅助 ----------
    def chat_json(self, system: str, user: str, max_tokens: int = 2048) -> Any:
        raw = self.chat(system + "\n只输出合法 JSON，不要任何其他文字、不要代码围栏。", user, max_tokens)
        return json.loads(self._strip_fences(raw))

    @staticmethod
    def _strip_fences(text: str) -> str:
        t = text.strip()
        if t.startswith("```"):
            t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
            t = re.sub(r"\n?```$", "", t)
        return t.strip()


_client: GLMClient | None = None


def get_llm() -> GLMClient:
    global _client
    if _client is None:
        _client = GLMClient()
    return _client
