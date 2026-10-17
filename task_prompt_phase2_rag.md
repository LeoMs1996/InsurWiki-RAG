# 任务 Prompt：第二期《非结构化保单条款 RAG 深度优化》企业级项目

**项目名：InsurWiki-RAG**

> InsurWiki-RAG —— 基于 InsQABench 的金融级保单条款 RAG 与 LLM Wiki 知识沉淀平台

（以下所有交付物——仓库名、README、文档站 header、报告封面——统一使用该名称与副标题）

## 角色设定
你是一名金融科技方向的资深 AI 工程师，需要独立完成一个**企业级保单条款 RAG + LLM Wiki + Agent 编排**项目（InsurWiki-RAG），作为第二期课程作业交付。

## 总体目标

## 技术决策（已确认）
| 决策点 | 选择 |
|---|---|
| 技术栈 | 纯 Python（FastAPI 服务 + LangChain/LangGraph 生态） |
| LLM | 智谱 GLM（GLM Coding Plan 套餐 API Key，模型 glm-4.x 系列） |
| Embedding | 优先智谱 embedding-3 API（与 GLM 同 Key）；文档中说明 BGE-M3 本地部署方案（FlagEmbedding + dense/sparse/colbert 三模态）作为企业级替代，并对比选型理由 |
| Rerank | BGE-reranker-v2-m3（本地）或智谱 rerank API，二选一并说明 |
| Agent 框架 | LangGraph 状态机编排 |
| 数据规模 | 合同问答三元组抽取 1~2k 条 + 对应条款原文（控制 LLM token 消耗） |
| 交付形态 | 独立项目仓库 InsurWiki-RAG + 内置 docs/ HTML 文档站（Mermaid 高级图） |

## 作业1：RAG 构建与 Wiki 知识沉淀

### 1.1 数据获取与处理（要有完整可复现流程）
- 从 HuggingFace `datasets` 拉取 InsQABench，筛选保险合同问答子集（`<Q, A, Evidence>` 三元组）
- 随机采样 1~2k 条；下载/提取对应条款原文段落
- 数据清洗：去重、去噪、格式规范化（统一 UTF-8 JSON Lines）
- 产出 `data/raw/`（原始）与 `data/processed/`（处理后）两级目录

### 1.2 结构感知切分（Structure-Aware Chunking）
- 针对保单条款的"条-款-项"层级结构实现切分器：按条款编号正则识别章节边界，结合语义完整性（句子边界 + 最大 token 窗口 + overlap）切分
- 费率表类表格数据单独走 表格→Markdown/KV 序列化 通道，不与正文混切
- 每个 chunk 保留元数据：`policy_id / 条款章节路径 / 原文偏移 / chunk_id`

### 1.3 混合检索与 RRF 融合
- BM25（rank_bm25 / Elasticsearch 可选）+ 向量检索（embedding-3）双路召回
- RRF（Reciprocal Rank Fusion, k=60）融合两路排序，输出 top-K
- Rerank 精排：BGE-reranker 或智谱 rerank，输出 top-N 及相关性分数

### 1.4 Self-RAG
- 生成节点输出 `reflection tokens`：`[Retrieve?] [IsRelevant?] [IsSupported?] [IsUseful?]`
- 检索相关性 / 答案支撑度（逐句与证据比对）/ 有用性三级反思，不合格触发重检索或降级回答
- 用 LangGraph 节点显式建模反思循环

### 1.5 GraphRAG 本体约束
- 用 GLM 对条款语料做实体-关系抽取，本体（Ontology）限定为保险领域：`保险产品 / 责任条款 / 免责条款 / 费率因子 / 犹豫期 / 保额 / 期限 / 被保险人群体`
- 构建轻量知识图谱（networkx 存储，节点=实体、边=带约束类型的关系）
- 问答时先做实体链接，图谱查询结果作为额外证据通道并入 RRF 融合，并对 LLM 生成加本体约束（禁止编造本体外的条款关系）

### 1.6 金融合规问答（引用可追溯、答案可复核）
- 每条答案必须携带引用：`[chunk_id → 条款号 → 原文摘录]`，前端/接口可点击回溯原文
- 合规护栏：免责声明模板、拒答阈值（证据支撑度低于阈值时明确说"依据不足"而非编造）、敏感词过滤
- 评测：从测试集抽 50~100 题做 Retrieval Hit@5 / 答案支撑率（citation faithfulness）统计

### 1.7 LLM Wiki 知识沉淀
- 定期/手动触发"知识沉淀"流水线：LLM 对聚类后的条款主题做聚合摘要，生成结构化 Wiki 页（Markdown + front-matter 元数据）
- Wiki 页结构：主题概述 / 核心条款要点 / 关联条款引用（互链）/ 常见问答（源自 QA 对）/ 更新时间与数据来源标注
- Wiki 页本身也进入向量库形成"沉淀→反哺检索"闭环
- 提供只读 Wiki 浏览页（Web 端）

## 作业2：Agent 编排与知识库结合（LangGraph）
- 编排一个"保险顾问 Agent"：多轮对话 + 意图识别 → 工具调用
- 工具集：`rag_search`（混合检索）/ `graph_query`（图谱查询）/ `wiki_lookup`（Wiki 页查询）/ `calculator`（费率/保额计算）/ `compliance_check`（合规护栏）
- LangGraph 状态机图：意图路由 → （RAG 分支 / 图谱分支 / 计算分支）→ 综合生成 → 合规检查 → 引用装配
- 支持 memory（会话级对话记忆）与 human-in-the-loop（低置信度转人工确认节点）

## 工程与交付要求
- 目录规范：`src/`（核心模块按上述环节分包）、`data/`、`docs/html/`、`scripts/`、`tests/`
- `.env.example` 完整列出所需 Key（GLM API Key 等），README 一键启动
- FastAPI 提供：`/api/chat`（合规问答）、`/api/wiki`（Wiki 列表/详情）、`/api/agent/chat`（Agent 对话）、`/health`
- 评测脚本 + 指标输出（JSON/表格）

## 流程文档站（docs/，HTML + Mermaid 高级图）
必须覆盖、且每步都要有"怎么做 + 用了什么技术 + 为什么"：
1. 总体架构图（flowchart：数据层→切分→索引→检索→反思→生成→Wiki→Agent）
2. 数据获取与处理流程图（含 InsQABench 来源、采样策略、清洗规则）
3. 结构感知切分示意图（条款层级切分前后对比）
4. 混合检索 + RRF + Rerank 流程图与时序图
5. Self-RAG 反思循环状态图（stateDiagram）
6. GraphRAG 本体 ER 图/类图（erDiagram / classDiagram）
7. 引用追溯与合规护栏机制图
8. Wiki 知识沉淀闭环图
9. LangGraph Agent 编排图（节点+边+条件路由）
10. 技术选型表：每个组件（GLM-4.x、embedding-3、BGE-M3、BGE-reranker、BM25、RRF、LangGraph、networkx、FastAPI…）的用途与替代方案

## 验收标准
- [ ] 一条命令跑通数据流水线（拉取→清洗→切分→索引）
- [ ] 问答接口返回带可点引用的合规答案，支撑率评测有数据
- [ ] Self-RAG 反思轨迹可在响应/日志中查看
- [ ] 图谱查询与本体约束生效（构造越界问题验证拒答）
- [ ] 至少生成 10+ 篇结构化 Wiki 页并可在 Web 浏览
- [ ] Agent 多轮对话可演示工具调用与人工确认分支
- [ ] HTML 文档站含全部 10 类图，可离线打开
