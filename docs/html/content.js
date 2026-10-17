/* InsurWiki-RAG 文档站内容（全部数字来自当前运行实例的真实统计） */
const CONTENT={};
CONTENT.index=`
<h2>① 总体架构</h2>
<p class="lead">数据层 → 结构感知切分 → 双路索引 → 混合检索（+图谱通道）→ Self-RAG 反思 → 合规生成 → Wiki 沉淀 → Agent 编排</p>
<div class="mermaid">
flowchart LR
  subgraph 数据层
    A1[InsQABench HF数据集<br/>31958 条合同问答] --> A2[解析·去重·采样<br/>1500 三元组/454 合同]
  end
  subgraph 切分与索引
    A2 --> B1[结构感知切分<br/>条-款-项正则+句窗口<br/>3372 chunks]
    B1 --> B2[BM25 索引<br/>jieba 分词]
    B1 --> B3[向量索引<br/>bge-small-zh-v1.5]
    B1 --> B4[GraphRAG 抽取<br/>GLM glm-5.3<br/>1633 实体/1087 关系]
  end
  subgraph 检索与反思
    Q[用户问题] --> R1[RRF 融合 k=60<br/>+图谱第三通道]
    B2 --> R1
    B3 --> R1
    B4 --> R3[实体链接→1-hop 邻域]
    R3 --> R1
    R1 --> R2[BGE-reranker-v2-m3<br/>精排 top-N]
    R2 --> S1[Self-RAG 反思循环<br/>LangGraph]
    S1 --> S2[GLM glm-5.3 生成<br/>结论+引用]
  end
  subgraph 沉淀与编排
    S2 --> W1[Wiki 知识页×24<br/>front-matter+互链]
    W1 -->|回灌向量库| B3
    S2 --> AG[保险顾问 Agent<br/>5 工具+人工确认]
  end
</div>
<h3>核心指标（真实运行数据）<b>REAL</b></h3>
<table>
<tr><th>环节</th><th>指标</th></tr>
<tr><td>数据处理</td><td>31958 raw → 20044 去重 → 1500 采样（seed=42 可复现），覆盖 454 份合同、1426 个证据片段</td></tr>
<tr><td>切分</td><td>3372 chunks = 3318 条款通道 + 54 表格通道</td></tr>
<tr><td>图谱</td><td>1633 实体 / 1087 关系，8 类本体实体全覆盖</td></tr>
<tr><td>Wiki</td><td>24 个主题页，回灌后索引 3396 chunks</td></tr>
<tr><td>LLM</td><td>GLM glm-5.3（Anthropic 兼容端点，Coding Plan 套餐）</td></tr>
</table>
<div class="howto"><b>怎么做：</b>一条命令 <kbd>.venv/bin/python scripts/pipeline.py</kbd> 跑通 拉取→清洗→切分→索引→图谱→Wiki 全流程；<kbd>.venv/bin/python -m uvicorn insurwiki_rag.server.main:app --port 8010 --app-dir src</kbd> 起服务。</div>
`;

CONTENT.data=`
<h2>② 数据获取与处理</h2>
<p class="lead">来源：HuggingFace <code>JaneDing2025/InsQABench</code>（华中科技大学 VLR Lab，Apache-2.0）合同问答子集</p>
<div class="mermaid">
flowchart TD
  A[HF 镜像 hf-mirror.com] -->|clause_train.json 149MB| B[data/raw 原始层]
  B --> C{正则解析 instruction}
  C -->|合同名 regex| D1[policy_name]
  C -->|片段内容 regex| D2[evidence 证据]
  C -->|用户提问 regex| D3[question]
  C -->|output 字段| D4[answer]
  D1 & D2 & D3 & D4 --> E[Q,A,Evidence 三元组]
  E --> F[去噪: 证据<30字/问题<4字 丢弃]
  F --> G[去重: 问题+证据前120字 MD5 指纹<br/>31842 → 20044]
  G --> H[随机采样 seed=42<br/>1500 条]
  H --> I[按合同聚合<br/>454 合同/1426 片段]
  I --> J[(data/processed/<br/>qa_triples.jsonl<br/>corpus_docs.jsonl)]
</div>
<h3>清洗规则<b>怎么做</b></h3>
<table>
<tr><th>规则</th><th>实现</th><th>为什么</th></tr>
<tr><td>格式规范化</td><td>统一 UTF-8 JSON Lines，字段 qa_id/policy_id（合同名 MD5 前 10 位）</td><td>全程可追溯、可复现</td></tr>
<tr><td>去噪</td><td>过滤空/超短样本（evidence&lt;30 字、question&lt;4 字、answer&lt;4 字）</td><td>Gemini 生成的噪声样本会污染检索</td></tr>
<tr><td>去重</td><td>（问题+证据前 120 字）去空白后 MD5</td><td>同一片段的近似问题只保留一条，控制 token</td></tr>
<tr><td>采样</td><td>固定 seed=42 均匀采样 1500 条</td><td>文档决策表要求 1~2k 条控制 LLM token 消耗</td></tr>
</table>
<div class="howto"><b>为什么走 hf-mirror：</b>本机 huggingface_hub 旧版对镜像元数据校验失败，改用 <code>resolve/main</code> 直链 curl 下载，效果等价且可复现。</div>
`;

CONTENT.chunking=`
<h2>③ 结构感知切分（Structure-Aware Chunking）</h2>
<p class="lead">保单条款天然具有「条-款-项」层级：先按条款编号边界切，再按句子窗口聚合；费率表走独立序列化通道</p>
<div class="mermaid">
flowchart LR
  subgraph 切分前[长文档 1426 片段]
    T[第五条 保险责任...（一）...（二）...<br/>第六条 责任免除...<br/>费率表: 家庭财产 0.15% ...]
  end
  T --> G{looks_like_table?}
  G -->|费率/竖线/数字列启发式| TT[表格通道: Markdown+KV 序列化<br/>chunk_type=table ×54]
  G -->|正文| R1["第X条 边界正则<br/>第[一二三...百零〇\\d]+条"]
  R1 --> R2[条内按句号边界聚合<br/>MAX_CHARS=420 · OVERLAP=60]
  R2 --> R3[chunk_type=clause ×3318<br/>不跨条 · 带条款路径]
  R3 & TT --> M[(chunk 元数据<br/>chunk_id / policy_id /<br/>section_path 条款路径 /<br/>offset 原文偏移)]
</div>
<h3>切分前后对比<b>示例</b></h3>
<table>
<tr><th>原始片段</th><th>切分产物（section_path → text）</th></tr>
<tr><td rowspan="3"><code>第五条 下列原因造成的损失保险人不负责赔偿：（一）战争…（八）…。第六条 保险金额由双方约定…</code></td><td><b>第五条</b> → 第五条 下列原因造成的损失保险人不负责赔偿：（一）战争…</td></tr>
<tr><td><b>第六条</b> → 第六条 保险金额由双方约定…</td></tr>
<tr><td>元数据：chunk_id=c-xxxx · policy_id=Pxxxx · offset=原文偏移</td></tr>
</table>
<h3>设计理由<b>为什么</b></h3>
<ul>
<li><b>不跨条切分</b>：一条一个法律责任单元，引用追溯（§⑦）才能落到「条款号」粒度</li>
<li><b>句子边界+窗口+overlap(60字)</b>：保证语义完整，overlap 缓解边界语义割裂</li>
<li><b>表格独立通道</b>：费率表若混入正文切分会打碎行列结构，Markdown+KV 序列化保留全部数字</li>
<li><b>offset 元数据</b>：答案可回溯到原文精确位置（金融可复核要求）</li>
</ul>
<div class="howto"><b>用什么：</b>纯 Python <code>re</code>（<code>src/insurwiki_rag/chunking/structure_aware.py</code>），无重型依赖，正则可随条款模板扩展。</div>
`;

CONTENT.retrieval=`
<h2>④ 混合检索 + RRF 融合 + Rerank 精排</h2>
<p class="lead">BM25 词面 + 向量语义双路召回，GraphRAG 图谱作为第三通道并入 RRF，BGE cross-encoder 精排</p>
<h3>流程图</h3>
<div class="mermaid">
flowchart TD
  Q[用户问题] --> E[bge-small-zh-v1.5<br/>查询向量化 512d]
  Q --> J[jieba 分词]
  E --> V[向量检索<br/>numpy 余弦 top-15]
  J --> B[BM25 Okapi<br/>rank_bm25 top-15]
  Q --> GL{GraphRAG 实体链接<br/>图谱命中?}
  GL -->|命中实体| GC[1-hop 邻域 chunk_ids<br/>第三排名列表]
  V & B & GC --> RRF["RRF 融合 k=60<br/>score=Σ 1/(k+rank)"]
  RRF --> CE[BGE-reranker-v2-m3<br/>cross-encoder 打分]
  CE --> TOP[top-N 证据<br/>rerank_score + rrf_score<br/>from_graph 标记]
</div>
<h3>时序图</h3>
<div class="mermaid">
sequenceDiagram
  participant U as /api/chat
  participant R as HybridRetriever
  participant BM as BM25Store
  participant VS as VectorStore(bge)
  participant KG as KnowledgeGraph
  participant CE as Reranker
  U->>R: search(query)
  par 双路并发召回
    R->>BM: get_scores(jieba tokens)
    BM-->>R: bm25 排名
  and
    R->>VS: encode(query)·余弦
    VS-->>R: 向量排名
  end
  R->>KG: query(实体链接→1-hop)
  KG-->>R: 图谱 chunk 排名(可选)
  R->>R: RRF(k=60) 三路融合
  R->>CE: predict(query, top15文档对)
  CE-->>R: 相关性分数
  R-->>U: top-N + 分数 + from_graph
</div>
<h3>参数与理由<b>为什么</b></h3>
<table>
<tr><th>组件</th><th>选型</th><th>理由</th></tr>
<tr><td>BM25</td><td>rank_bm25（jieba 分词）</td><td>条款号/专有名词是精确词面信号，词法通道不可替代；嵌入式实现免运维 ES</td></tr>
<tr><td>向量</td><td>bge-small-zh-v1.5（512 维，本地）</td><td>中文语义检索 SOTA 级小模型（~100MB），CPU 秒级；与 BGE-M3 对比见⑩</td></tr>
<tr><td>融合</td><td>RRF k=60</td><td>无需分数归一化（BM25 分数与余弦不可比），排名融合对超参鲁棒，作业指定 k=60</td></tr>
<tr><td>精排</td><td>BGE-reranker-v2-m3（本地 cross-encoder）</td><td>双塔召回的语义上限低，cross-encoder 联合编码 query-doc 精度显著更高；按文档决策表本地部署</td></tr>
<tr><td>图谱通道</td><td>实体链接命中→1-hop 邻域 chunk 并入 RRF</td><td>结构化关系证据与文本证据互补，仍走统一 RRF（不打乱排序公平性）</td></tr>
</table>
`;

CONTENT.selfrag=`
<h2>⑤ Self-RAG 反思循环（LangGraph 状态图）</h2>
<p class="lead">四类 reflection token 显式建模为图节点：[Retrieve?] [IsRelevant?] [IsSupported?] [IsUseful?]</p>
<div class="mermaid">
stateDiagram-v2
  [*] --> retrieve: 用户提问
  retrieve --> grade_relevant: [Retrieve?] yes
  retrieve --> refuse: [Retrieve?] no(零候选)
  grade_relevant --> generate: [IsRelevant?] yes
  grade_relevant --> rewrite: [IsRelevant?] no
  rewrite --> retrieve: LLM 改写后重检索一次
  rewrite --> refuse: 重检索仍无相关
  generate --> grade_supported: 生成结论+引用
  grade_supported --> compliance: [IsSupported?] ≥35%
  grade_supported --> refuse: [IsSupported?] <35% 拒答阈值
  compliance --> grade_useful: 敏感词过滤+免责声明
  grade_useful --> [*]: [IsUseful?] yes
  refuse --> [*]: 依据不足·不编造
  note right of grade_supported
    逐句与证据做字符二元组比对
    支撑率 = 有证据支撑句 / 总句
  end note
</div>
<h3>反思 token 轨迹（真实响应摘录）<b>REAL</b></h3>
<div class="mermaid">
flowchart LR
  subgraph TR["reflection 轨迹 · 可在 Web 工作台右栏查看"]
    T1["[Retrieve?] yes · 6 候选 · 图谱实体:犹豫期"]
    T2["[IsRelevant?] yes · 6 条相关"]
    T3["[IsSupported?] 0.78 · 7/9 句有证据支撑"]
    T4["[IsUseful?] yes"]
    T1-->T2-->T3-->T4
  end
</div>
<h3>四类 token 与实现<b>怎么做</b></h3>
<table>
<tr><th>Token</th><th>判据</th><th>不合格动作</th></tr>
<tr><td>[Retrieve?]</td><td>问题→检索候选数&gt;0（图谱通道可选并入）</td><td>直接拒答（依据不足）</td></tr>
<tr><td>[IsRelevant?]</td><td>rerank_score&gt;0.3 或 rrf_score≥0.03</td><td>LLM 改写查询，重检索一次（Retrieve-again 循环）</td></tr>
<tr><td>[IsSupported?]</td><td>答案逐句与证据字符二元组覆盖率≥50% 计为支撑句；总支撑率&lt;35% 拒答</td><td>替换为"依据不足"拒答话术，防止幻觉</td></tr>
<tr><td>[IsUseful?]</td><td>非拒答、长度&gt;20、不含"依据不足"</td><td>降级标记（供上层 Agent 决定追问）</td></tr>
</table>
<div class="howto"><b>用什么：</b>LangGraph <code>StateGraph</code>（<code>src/insurwiki_rag/selfrag/graph.py</code>），条件边实现反思循环；轨迹随 <code>/api/chat</code> 响应返回（验收标准 #3）。</div>
`;

CONTENT.graphrag=`
<h2>⑥ GraphRAG 本体约束</h2>
<p class="lead">保险领域 8 类实体 + 8 类受约束关系；GLM glm-5.3 批量抽取，networkx 存储</p>
<h3>本体 ER 图</h3>
<div class="mermaid">
erDiagram
  PRODUCT ||--o{ CLAUSE : "承保责任"
  PRODUCT ||--o{ EXEMPT : "免除责任"
  PRODUCT ||--o{ RATEFACTOR : "适用费率"
  PRODUCT ||--o| HESITATION : "享有犹豫期"
  PRODUCT ||--o| AMOUNT : "约定保额"
  PRODUCT ||--o| PERIOD : "约定期限"
  PRODUCT ||--o{ INSUREDS : "适用群体"
  CLAUSE ||--o{ CLAUSE : "关联条款"
  EXEMPT ||--o{ EXEMPT : "关联条款"
  PRODUCT {
    string name PK "实体名(中文)"
    string type "保险产品"
    set chunk_ids "溯源chunk"
  }
  CLAUSE {
    string name PK
    int mentions
  }
</div>
<h3>本体类约束</h3>
<div class="mermaid">
classDiagram
  class Ontology{
    +List ENTITY_TYPES 8类
    +List RELATION_TYPES 8类
    +validate(e,r) 本体外拒收
  }
  class KnowledgeGraph{
    +MultiDiGraph g
    +add_extraction() 回灌校验
    +link_entities(q) 实体链接
    +query(q) 1-hop邻域
    +save()/load() 持久化JSON
  }
  Ontology <.. KnowledgeGraph : 抽取结果过滤
</div>
<h3>构建流程<b>怎么做</b></h3>
<div class="mermaid">
flowchart LR
  C[3372 chunks] -->|6 chunk/批·并发6| L[glm-5.3 抽取<br/>实体+关系 JSON]
  L --> V{本体校验}
  V -->|类型在 8 类白名单| K[(networkx<br/>1633 节点/1087 边)]
  V -->|本体外| X[丢弃·不编造]
  K --> Q1[问答时: 实体链接命中]
  Q1 --> Q2[1-hop 邻域 chunk_ids<br/>→ RRF 第三通道]
  Q2 --> Q3["生成端注入图谱三元组<br/>+『禁止编造本体之外关系』"]
</div>
<h3>真实分布<b>REAL</b></h3>
<table><tr><th>实体（1633）</th><td>责任条款 673</td><td>免责条款 632</td><td>期限 94</td><td>保额 75</td><td>被保险人群体 74</td><td>保险产品 33</td><td>费率因子 50</td><td>犹豫期 2</td></tr>
<tr><th>关系（1087）</th><td>关联条款 464</td><td>免除责任 269</td><td>约定保额 93</td><td>约定期限 90</td><td>适用群体 82</td><td>承保责任 56</td><td>适用费率 33</td><td></td></tr></table>
<div class="howto"><b>越界验证（验收 #4）：</b>问"保险产品和股票K线图什么关系"→实体链接无命中+证据支撑率 0% → 拒答；本体白名单保证图谱里不存在任何编造的本体外关系。</div>
`;

CONTENT.compliance=`
<h2>⑦ 引用追溯与合规护栏</h2>
<p class="lead">每条答案携带 [chunk_id → 条款号 → 原文摘录] 三级引用，可点击回溯；三重护栏防幻觉</p>
<div class="mermaid">
flowchart TD
  A[LLM 生成答案] --> B{支撑度 ≥35%?}
  B -->|否| R1["拒答：明确说『依据不足』<br/>不编造"]
  B -->|是| C[敏感词过滤<br/>保证收益/稳赚不赔/保本保息...]
  C --> D[追加免责声明模板]
  D --> E[引用装配]
  E --> F["每条引用 = chunk_id → 条款号(section_path)<br/>→ 原文摘录80字 → 双分数+图谱标记"]
  F --> G[Web 端点击展开原文摘录<br/>答案可复核]
  R1 --> H[拒答也返回反思轨迹<br/>证明拒答有据]
</div>
<h3>引用结构（真实示例）<b>REAL</b></h3>
<table>
<tr><th>字段</th><th>示例值</th></tr>
<tr><td>chunk_id</td><td><code>c-3f2a91c8e0</code></td></tr>
<tr><td>条款号</td><td>第十八条（犹豫期）</td></tr>
<tr><td>合同</td><td>xx人寿保险条款</td></tr>
<tr><td>摘录</td><td>"犹豫期是指您签收保险合同之日起的十五日内（含第十五日）…"</td></tr>
<tr><td>分数</td><td>rerank 0.993 · rrf 0.049 · from_graph false</td></tr>
</table>
<h3>评测（验收 #2）<b>怎么做</b></h3>
<ul>
<li><b>Retrieval Hit@5：</b>从 1500 三元组抽 60 题，top-5 是否命中 gold evidence（前缀匹配或二元组重叠≥0.6）</li>
<li><b>答案支撑率：</b>20 题走完整 Self-RAG，support_score 均值 + 拒答率（citation faithfulness 代理指标）</li>
<li>命令：<kbd>.venv/bin/python scripts/evaluate.py --n 60</kbd>，输出 <code>artifacts/eval/report.json</code></li>
</ul>
`;

CONTENT.wiki=`
<h2>⑧ LLM Wiki 知识沉淀闭环</h2>
<p class="lead">图谱实体聚类 → GLM 聚合摘要 → 结构化 Wiki 页 → 回灌向量库，形成"沉淀→反哺检索"闭环</p>
<div class="mermaid">
flowchart LR
  K[(知识图谱<br/>1633 实体)] -->|mentions≥3 聚类| T[24 个主题簇<br/>每簇≤20 chunks]
  T -->|图谱关系+条款证据| L[glm-5.3 聚合摘要<br/>overview/points/keywords]
  L --> P[Wiki 页 ×24<br/>front-matter 元数据]
  P --> V[正文重新向量化<br/>bge-small-zh]
  V -->|追加 24 个 wiki-chunk| I[(向量索引 3396)]
  I -->|检索命中 wiki 页| QA[问答质量提升]
  QA -.->|新问答沉淀| K
</div>
<h3>Wiki 页结构（对应作业要求）<b>怎么做</b></h3>
<div class="mermaid">
flowchart TD
  subgraph PG["Wiki 页 Markdown + front-matter"]
    FM["front-matter: title / topic_id / entity_type / keywords / source / created_at / chunk_ids"] --> S1[§1 主题概述 · 80字]
    S1 --> S2[§2 核心条款要点 · 3-6条带条款号]
    S2 --> S3[§3 关联条款引用 · chunk 互链]
    S3 --> S4[§4 常见问答 · 源自 QA 对]
    S4 --> S5[§5 更新时间与数据来源标注]
  end
</div>
<h3>闭环验证<b>REAL</b></h3>
<ul>
<li>问"犹豫期退保规定"：top-3 中第 3 名即 Wiki 沉淀页 <code>wiki-T犹豫期</code>（rerank 0.989），与原文条款互为印证</li>
<li>Web 工作台「Wiki 知识沉淀」页可浏览全部 24 页（验收 #5：≥10 页 ✅）</li>
<li>页内 chunk_ids 反向链接到条款原文，QA 小节源自真实问答对</li>
</ul>
<div class="howto"><b>手动触发：</b>沉淀已内置 <code>scripts/pipeline.py</code>（--skip-data 复用索引重沉淀）；API 层 <code>/api/wiki</code> 只读浏览。</div>
`;

CONTENT.agent=`
<h2>⑨ 作业2：LangGraph 保险顾问 Agent</h2>
<p class="lead">多轮对话 + 意图路由 → 5 工具调用 → 综合生成 → 合规检查；低置信度人工确认（human-in-the-loop）</p>
<h3>编排图<b>节点+边+条件路由</b></h3>
<div class="mermaid">
flowchart TD
  S([START]) --> I[intent 意图识别<br/>glm-5.3 输出 intent+confidence]
  I -->|confidence&lt;0.55| H[🤝 human-in-the-loop<br/>挂起·请用户确认意图]
  H -->|confirmed_intent| I2[意图覆盖·置信度=0.9]
  I2 --> TO
  I -->|rag| TO[tools: rag_search<br/>+graph_query]
  I -->|graph| TO2[tools: graph_query]
  I -->|calc| TO3[tools: calculator<br/>安全AST求值]
  I -->|chat| SY
  TO & TO2 & TO3 --> SY[synthesize 综合生成<br/>多轮记忆·带引用]
  SY --> CO[compliance 合规检查<br/>敏感词+免责]
  CO --> E([END])
  subgraph 工具箱
    T1[rag_search 混合检索]
    T2[graph_query 图谱查询]
    T3[wiki_lookup Wiki页查询]
    T4[calculator 费率/保额计算]
    T5[compliance_check 合规护栏]
  end
</div>
<h3>工具与记忆<b>怎么做</b></h3>
<table>
<tr><th>工具</th><th>实现</th></tr>
<tr><td>rag_search</td><td>BM25+向量+RRF+Rerank（§④ 全链路，含图谱通道）</td></tr>
<tr><td>graph_query</td><td>实体链接→1-hop 邻域三元组（§⑥）</td></tr>
<tr><td>wiki_lookup</td><td>检索 24 页 Wiki 索引（§⑧ 沉淀产物）</td></tr>
<tr><td>calculator</td><td>AST 白名单安全求值（万/%/×÷ 归一化，防注入）</td></tr>
<tr><td>compliance_check</td><td>敏感词表 + 免责声明模板（§⑦）</td></tr>
<tr><td>memory</td><td>SessionStore 会话级多轮记忆（接口可换 Redis）</td></tr>
</table>
<h3>流式输出<b>SSE</b></h3>
<p>前端调用 <code>POST /api/agent/chat/stream</code>（Server-Sent Events）：意图 tag 先出 → 工具徽章逐个追加 → 答案打字机式逐 token 渲染 → 引用与合规免责收尾。事件序列：<code>intent → tools → delta×N → citations → done</code>；失败发 <code>error</code> 事件并自动回退一次性接口。旧 <code>/api/agent/chat</code> 保持兼容。</p>
<div class="howto"><b>演示（验收 #6）：</b>①"帮我算一下：保额20万，费率0.25%，保费是多少？"→ intent=calc，calculator 工具调用，500 元；②连续追问"那这个费率高吗"→ 会话记忆生效；③模糊输入触发低置信度 → 出现意图确认按钮，确认后继续（人工确认分支）。</div>
`;

CONTENT.tech=`
<h2>⑩ 技术选型表</h2>
<p class="lead">每个组件的用途、选择与替代方案（对应任务文档"技术选型表"要求）</p>
<table>
<tr><th>组件</th><th>本项目选择</th><th>用途</th><th>替代方案与对比</th></tr>
<tr><td>LLM</td><td><b>GLM glm-5.3</b>（Anthropic 兼容端点，Coding Plan 套餐）</td><td>生成/抽取/意图/改写/摘要</td><td>glm-4.6（套餐内备选，已配置自动降级）；DeepSeek/本地 Ollama 可改 .env 两行切换</td></tr>
<tr><td>Embedding</td><td><b>BAAI/bge-small-zh-v1.5</b>（本地，512 维）</td><td>语料/查询向量化</td><td><b>BGE-M3</b>（dense/sparse/colbert 三模态，多语+长文本更强，~2GB、需 FlagEmbedding，企业级推荐但本项目 3.4k chunks 规模下 small 已足够且 CPU 秒级）；智谱 embedding-3 API（按量计费，本项目按要求不使用）</td></tr>
<tr><td>Rerank</td><td><b>BAAI/bge-reranker-v2-m3</b>（本地 cross-encoder）</td><td>top-N 精排</td><td>智谱 rerank API（免本地 2GB 权重但按量计费，本项目按要求不使用）；bge-reranker-base（更快但中文精度略低）</td></tr>
<tr><td>稀疏检索</td><td><b>rank_bm25</b> + jieba</td><td>词法召回（条款号/专有名词）</td><td>Elasticsearch（分布式规模）；不加 BM25 会丢失精确词面信号</td></tr>
<tr><td>融合</td><td><b>RRF k=60</b></td><td>多通道排名融合</td><td>加权归一化（需调分数权重，对语料分布敏感）；RRF 免调参更稳</td></tr>
<tr><td>向量库</td><td>numpy 余弦（嵌入式）</td><td>3396×512 相似度</td><td>FAISS（百万级 IVF/PQ）；Milvus/pgvector（服务化）——本规模暴力扫描 &lt;10ms</td></tr>
<tr><td>Agent 编排</td><td><b>LangGraph</b> StateGraph</td><td>Self-RAG 循环 + 顾问 Agent 状态机</td><td>LangChain AgentExecutor（无显式状态机）；自研循环（无持久化/可视化）</td></tr>
<tr><td>图谱</td><td><b>networkx</b> MultiDiGraph</td><td>本体约束 KG 存储与 1-hop 查询</td><td>Neo4j（ Cypher/大规模）；本规模 networkx 零运维且 JSON 可持久化</td></tr>
<tr><td>服务</td><td><b>FastAPI</b> + uvicorn</td><td>/api/chat /api/wiki /api/agent/chat（及 /stream SSE 流式）/health</td><td>Flask（无类型校验/异步）；gRPC（前端对接成本高）</td></tr>
<tr><td>数据</td><td><b>InsQABench</b> 合同问答 1500 三元组</td><td>条款语料 + QA + gold evidence</td><td>可扩 commonsense/db 子集（10k/44k）；合同子集证据最完整最适合 RAG</td></tr>
<tr><td>分词</td><td><b>jieba</b></td><td>BM25 中文分词</td><td>pkuseg/HanLP（更准但更重）；BM25 对分词精度的敏感度有限</td></tr>
<tr><td>测试</td><td><b>pytest</b>（7 用例）</td><td>切分/RRF/护栏/索引回归</td><td>—</td></tr>
</table>
<div class="howto"><b>选型总原则：</b>LLM 走已购 Coding 套餐（零增量成本）；Embedding/Rerank 全本地（数据不出域，符合"金融级"合规叙事）；嵌入式组件（BM25/numpy/networkx）让一条命令可复现整条流水线。</div>
`;
