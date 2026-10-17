#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出 Obsidian 库：python scripts/export_obsidian.py [输出目录]

生成可直接用 Obsidian「打开文件夹作为仓库」的知识库：
  实体/   8 类本体实体笔记（[[双链]] 关系）
  Wiki/   作业1 沉淀知识页 + 图谱互链
  Graph.canvas  原生 Canvas 图谱
  Index.md      MOC 总目录
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from insurwiki_rag.config import Config  # noqa: E402
from insurwiki_rag.server.obsidian_export import build_vault, export_zip  # noqa: E402

if __name__ == "__main__":
    if "--zip" in sys.argv:
        zp = export_zip(Config.ARTIFACTS / "obsidian_vault.zip")
        print(f"✅ Obsidian 库 zip：{zp}")
    else:
        out = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
            else Config.ARTIFACTS / "obsidian_vault"
        p = build_vault(out)
        n_ent = len(list((p / "实体").glob("*.md")))
        n_wiki = len(list((p / "Wiki").glob("*.md")))
        print(f"✅ Obsidian 库：{p}")
        print(f"   实体笔记 {n_ent} · Wiki 页 {n_wiki} · Graph.canvas + Index.md")
        print("   Obsidian → 打开文件夹作为仓库 → 点开 Graph.canvas")
