# InsurWiki-RAG

> InsurWiki-RAG —— 基于 InsQABench 的金融级保单条款 RAG 与 LLM Wiki 知识沉淀平台

第二期《非结构化保单条款 RAG 深度优化》课程作业交付：结构感知切分、BM25+向量混合检索与 RRF 融合、Rerank 精排、Self-RAG 反思循环、GraphRAG 本体约束、引用可追溯的金融合规问答、LLM Wiki 知识沉淀闭环、LangGraph 保险顾问 Agent 编排。

## 快速开始（一键启动）

```bash
# 1) 环境
python3 -m venv .venv
.venv/bin/pip install fastapi "uvicorn[standard]" langchain langgraph sentence-transformers rank_bm25 networkx datasets jieba numpy pytest

# 2) 配置：复制 .env.example 为 .env，填入 GLM API Key
cp .env.example .env   # 编辑 LLM_API_KEY

# 3) 一条命令跑通数据流水线（拉取→清洗→切分→索引→图谱→Wiki）
.venv/bin/python scripts/pipeline.py

# 4) 起服务
.venv/bin/python -m uvicorn insurwiki_rag.server.main:app --port 8010 --app-dir src
```

- Web 工作台：<http://127.0.0.1:8010/>（问答/Wiki/Agent/图谱 四个页签）
- 流程文档站：<http://127.0.0.1:8010/docs/>（10 类 Mermaid 高级图，离线可开 `docs/html/index.html`）
- 新人入门：`docs/deep-dive/00-beginner.html`（图解 Self-RAG 何时/在哪用、GraphRAG 与 Wiki 闭环，建议先读再看 01–06 号页面）

## 技术栈

| 组件 | 选择 |
|---|---|
| LLM | GLM **glm-5.3**（Anthropic 兼容端点，Coding Plan 套餐；glm-4.6 自动降级） |
| Embedding | BAAI/bge-small-zh-v1.5（本地，不用智谱 embedding-3 API） |
| Rerank | BAAI/bge-reranker-v2-m3（本地 cross-encoder，不用智谱 rerank API） |
| 检索 | rank_bm25 + numpy 向量余弦 双路 → RRF(k=60) → Rerank |
| 编排 | LangGraph（Self-RAG 反思循环 + 顾问 Agent 状态机） |
| 图谱 | networkx（8 类保险本体实体 / 8 类受约束关系） |
| 服务 | FastAPI |

## API

| 路由 | 说明 |
|---|---|
| `POST /api/chat` | 合规问答（Self-RAG 全链路，返回反思轨迹+可点引用） |
| `GET /api/search?q=` | 裸混合检索（调试） |
| `GET /api/graph` / `?entity=` | 图谱统计 / 实体查询 |
| `GET /api/wiki`、`/api/wiki/{id}` | Wiki 列表 / 详情（Markdown） |
| `POST /api/agent/chat` | 保险顾问 Agent（多轮、工具调用、人工确认） |
| `POST /api/agent/chat/stream` | Agent 流式对话（SSE：`intent` → `tools` → `delta`… → `citations` → `done`） |
| `GET /health` | 健康检查 |

## 评测

```bash
.venv/bin/python scripts/evaluate.py --n 60      # Hit@5 + 支撑率
.venv/bin/python -m pytest tests/ -q             # 单元/回归测试
```

当前实例实测：**Hit@5 75%（n=60）· 答案支撑率 68.6% · 拒答率 15%**（详见 `artifacts/eval/report.json`）。

## 目录结构

```
src/insurwiki_rag/
  data/pipeline.py          # §1.1 数据获取与处理
  chunking/structure_aware.py  # §1.2 结构感知切分
  retrieval/hybrid.py       # §1.3 混合检索+RRF+Rerank
  selfrag/graph.py          # §1.4 Self-RAG（LangGraph）
  graphrag/kg.py            # §1.5 GraphRAG 本体约束
  qa（护栏内嵌 selfrag）     # §1.6 合规问答+评测
  wiki/sediment.py          # §1.7 LLM Wiki 沉淀闭环
  agent/advisor.py          # 作业2 保险顾问 Agent
  server/main.py            # FastAPI
scripts/pipeline.py scripts/evaluate.py
docs/html/                  # 离线文档站（10 类图）
tests/  data/  artifacts/
```

## 数据来源

InsQABench（华中科技大学 VLR Lab，Apache-2.0）：合同问答子集 `<Q, A, Evidence>` 三元组采样 1500 条，覆盖 454 份保险合同。经 hf-mirror.com 下载。

## Obsidian 知识库导出

图谱与 Wiki 可一键导出为 Obsidian 仓库（`[[双链]]` + 原生 Canvas 图谱）：

```bash
python scripts/export_obsidian.py          # 导出到 artifacts/obsidian_vault/
python scripts/export_obsidian.py --zip    # 打包 artifacts/obsidian_vault.zip
```

或直接在 Web 工作台「知识图谱」页点 **⬇ 导出 Obsidian 库（zip）**（即 `GET /api/export/obsidian`）。

导出内容：`实体/` 8 类本体 1600+ 实体笔记（关系双链）、`Wiki/` 24 个知识沉淀页、
`Graph.canvas`（Obsidian 原生画布图谱，按本体类型分色分区）、`Index.md` MOC 总目录。
Obsidian → 打开文件夹作为仓库 → 点开 `Graph.canvas` 即为可视化图谱。
