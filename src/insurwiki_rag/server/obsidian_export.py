# -*- coding: utf-8 -*-
"""导出 Obsidian 库：Wiki 知识页 + 图谱实体笔记（双链）+ Canvas 图谱。

- Wiki/         作业1 沉淀的 24 个知识页（追加图谱互链）
- 实体/         图谱 8 类本体实体笔记，含 [[双链]] 关系
- Graph.canvas  Obsidian Canvas 原生图谱视图
- Index.md      MOC 总目录
"""
from __future__ import annotations

import json
import math
import re
import zipfile
from pathlib import Path
from typing import Dict, List

from ..config import Config

ILLEGAL = r'[#^|\[\]\\/:*?"<>]'


def safe_name(name: str) -> str:
    s = re.sub(ILLEGAL, "－", str(name)).strip()
    return (s[:80] or "未命名")


TYPE_COLORS = {  # Obsidian canvas 十六进制（不带 #）
    "保险产品": "ff7eb6", "责任条款": "5b8cff", "免责条款": "e5646e",
    "费率因子": "e6a23c", "犹豫期": "3fb96f", "保额": "8f5bff",
    "期限": "4dd0e1", "被保险人群体": "aed581",
}


def build_vault(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    kg = json.loads(Config.GRAPH_PATH.read_text(encoding="utf-8"))
    nodes: Dict[str, dict] = {}
    for n in kg["nodes"]:
        nodes[n["id"]] = n
    edges: List[dict] = kg["links"] if "links" in kg else kg.get("edges", [])

    # ---- 实体笔记 ----
    ent_dir = out_dir / "实体"
    ent_dir.mkdir(exist_ok=True)
    deg: Dict[str, int] = {}
    for e in edges:
        deg[e["source"]] = deg.get(e["source"], 0) + 1
        deg[e["target"]] = deg.get(e["target"], 0) + 1
    by_type: Dict[str, List[str]] = {}
    for name, meta in nodes.items():
        et = meta.get("type", "其他")
        by_type.setdefault(et, []).append(name)
        out = [f"# {name}", "",
               f"- **本体类型**：{et}",
               f"- **连接度**：{deg.get(name, 0)}",
               f"- **关联条款块**：{len(meta.get('chunk_ids', []) or [])}", ""]
        rels = [(e["relation"], e["target"]) for e in edges if e["source"] == name]
        revs = [(e["relation"], e["source"]) for e in edges if e["target"] == name]
        if rels:
            out += ["## 关系（出边）"] + [f"- {r} → [[{safe_name(t)}|{t}]]" for r, t in rels[:30]] + [""]
        if revs:
            out += ["## 关系（入边）"] + [f"- [[{safe_name(s)}|{s}]] → {r} → 本实体" for r, s in revs[:30]] + [""]
        (ent_dir / f"{safe_name(name)}.md").write_text("\n".join(out), encoding="utf-8")

    # ---- Wiki 页（追加互链段）----
    wiki_dir = out_dir / "Wiki"
    wiki_dir.mkdir(exist_ok=True)
    idx_file = Config.WIKI_DIR / "index.json"
    pages = json.loads(idx_file.read_text(encoding="utf-8")) if idx_file.exists() else []
    name_map = {}
    for p in pages:
        tid = p.get("topic_id", "")
        fn = next((fn for fn in Config.WIKI_DIR.glob("*.md")
                   if (Config.WIKI_DIR / fn).read_text(encoding="utf-8", errors="ignore").find(f"topic_id: {tid}") >= 0), None)
        if not fn:
            continue
        text = (Config.WIKI_DIR / fn).read_text(encoding="utf-8")
        links = []
        for kw in p.get("keywords", [])[:6]:
            cand = [n for n in nodes if kw and kw in n and n != p.get("title")]
            links += [safe_name(c) for c in cand[:2]]
        if links:
            uniq = sorted(set(links))
            text += f"\n\n## 图谱互链\n" + " ".join(f"[[{l}]]" for l in uniq) + "\n"
        (wiki_dir / safe_name(p.get("title", tid))).with_suffix(".md").write_text(text, encoding="utf-8")
        name_map[p.get("title", tid)] = safe_name(p.get("title", tid))

    # ---- Canvas 图谱（按本体类型分扇区环形布局）----
    ids = list(nodes.keys())
    cnodes, cedges = [], []
    cx = cy = 2000
    etypes = [t for t in TYPE_COLORS if t in by_type] or list(by_type)
    sector = 2 * math.pi / max(len(etypes), 1)
    counters = {t: 0 for t in etypes}
    totals = {t: len(by_type.get(t, [])) for t in etypes}
    pos = {}
    for nid in ids:
        et = nodes[nid].get("type", "其他")
        if et not in counters:
            et = etypes[-1]
        k = counters[et]; counters[et] += 1
        base = etypes.index(et) * sector
        # 每类一圈，节点越多半径越大
        rr = 260 + 90 * math.sqrt(totals[et]) * (k % 12 + 1) / 12
        ang = base + sector * (k / max(totals[et], 1))
        x, y = cx + rr * math.cos(ang), cy + rr * math.sin(ang)
        pos[nid] = (x, y)
        cnodes.append({"id": nid, "type": "text", "text": f"[[实体/{safe_name(nid)}|{nid}]]",
                       "x": int(x), "y": int(y), "width": 200, "height": 60,
                       "color": TYPE_COLORS.get(et, "8b93a7")})
    for i, e in enumerate(edges):
        if e["source"] in pos and e["target"] in pos:
            cedges.append({"id": f"e{i}", "fromNode": e["source"], "fromSide": "right",
                           "toNode": e["target"], "toSide": "left",
                           "label": e.get("relation", ""), "color": "555555"})
    canvas = {"nodes": cnodes, "edges": cedges}
    (out_dir / "Graph.canvas").write_text(json.dumps(canvas, ensure_ascii=False), encoding="utf-8")

    # ---- MOC ----
    moc = ["# InsurWiki-RAG 知识库（Obsidian）", "",
           "来源：InsQABench 保单条款知识沉淀流水线（Self-RAG + GraphRAG + Wiki 沉淀）。", "",
           "## 使用说明",
           "- `Graph.canvas`：**用 Obsidian 打开本库后直接点开即是可视化图谱**（核心入口）",
           "- `实体/`：8 类本体实体笔记，关系用 [[双链]] 连接，可用反向链接与关系图谱（局部视图）",
           "- `Wiki/`：作业1 沉淀的知识页，含条款要点与图谱互链", "",
           "## 本体类型目录"]
    for t in etypes:
        moc.append(f"### {t}（{totals.get(t, 0)}）")
        for n in by_type.get(t, []):
            moc.append(f"- [[实体/{safe_name(n)}|{n}]]")
        moc.append("")
    (out_dir / "Index.md").write_text("\n".join(moc), encoding="utf-8")
    return out_dir


def export_zip(zip_path: Path) -> Path:
    tmp = zip_path.parent / "obsidian_vault"
    if tmp.exists():
        for p in sorted(tmp.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
    build_vault(tmp)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in tmp.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(tmp.parent))
    return zip_path
