# -*- coding: utf-8 -*-
"""为案例（case）生成 anchor 字段，帮助其更容易被检索 query 命中。

思路：让每条 case 在知识树下逐层导航（默认按 dialog_trigger，可选 case_trigger），
积累沿途经过节点的 background；再依据「背景知识 + case 内容」提炼出一串**关键词**
作为 anchor。anchor 的关键词与 query 生成时使用的背景知识同源、同体系，因此
case 的 anchor 与对话生成的 query 用词一致，可提高检索召回。

anchor 形态：一句话，但内容是空格隔开的关键词（不是自然语句、不是模拟用户问法），
例如 "WeCon 历史消息同步 database.ini 本地数据库 多端同步"。

用法：
  python case_anchor/build_anchor.py \
      --case data/case/text.json \
      --tree build/output/knowledge_tree.json \
      --out case_anchor/text_with_anchor.json

  # 用 case_trigger 导航（默认 dialog_trigger）
  python case_anchor/build_anchor.py --nav-by case_trigger

  # 关闭思考模式 / 离线自测
  python case_anchor/build_anchor.py --no-thinking
  python case_anchor/build_anchor.py --provider mock
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# 复用 build/khtree 的 LLM 客户端、数据模型与工具
# ---------------------------------------------------------------------------
ANCHOR_DIR = Path(__file__).resolve().parent
ROOT_DIR = ANCHOR_DIR.parent
BUILD_DIR = ROOT_DIR / "build"
for p in (str(BUILD_DIR), str(ROOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from khtree.config_types import LLMSettings  # noqa: E402
from khtree.llm import LLMClient, _analysis_before_json, _detect_entity  # noqa: E402
from khtree.models import Case, Node, Tree  # noqa: E402
from khtree.utils import extract_json  # noqa: E402

try:
    from tqdm import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


def log(msg: str) -> None:
    print(f"[anchor] {msg}", file=sys.stderr, flush=True)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"已写出：{p}")


def init_dialog_trigger(node: Node) -> int:
    """dialog_trigger 为空的节点用 case_trigger 兜底填充（不覆盖已有值）。返回填充数。"""
    count = 0
    for c in node.children:
        if not (c.dialog_trigger or "").strip() and (c.case_trigger or "").strip():
            c.dialog_trigger = c.case_trigger
            count += 1
        count += init_dialog_trigger(c)
    return count


# ---------------------------------------------------------------------------
# 提示词：case 逐层导航 + anchor 关键词生成
# ---------------------------------------------------------------------------
def navigate_case_messages(case: Case, children: List[Node], nav_by: str) -> List[Dict[str, str]]:
    """让 case 在某一层选择子类别。nav_by 决定看哪个 trigger 字段。"""
    trig_field = "dialog_trigger" if nav_by == "dialog_trigger" else "case_trigger"
    items = [{"name": c.name, "trigger": getattr(c, trig_field)} for c in children]
    system = (
        "你是知识分类助手。给定一条案例和若干候选类别（每个类别有 name 和 trigger，"
        "trigger 描述什么样的内容应进入该类别），判断该案例应进入哪个类别。"
    )
    user = (
        f"案例：\n标题：{case.case_name}\n内容：{case.text}\n\n"
        f"候选类别：\n{json.dumps(items, ensure_ascii=False, indent=2)}\n\n"
        '请输出 JSON：{"name":"<最匹配类别name，没有则空字符串>","reason":"..."}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def anchor_messages(case: Case, backgrounds: List[Dict]) -> List[Dict[str, str]]:
    """依据 case 内容 + 导航积累的背景知识，提炼检索关键词作为 anchor。

    刻意贴近 generate_query_messages：同样吃「背景知识（节点名：背景）」，
    使 anchor 关键词与对话生成的 query 用词同体系，从而更易互相命中。
    """
    system = (
        "你是企业内部检索优化助手。下面给出一条案例，以及它在知识体系中归类时"
        "沿途积累的背景知识。请为该案例提炼一组**检索关键词**作为 anchor，"
        "目的是让用户咨询相关问题时生成的检索 query 更容易命中这条案例。"
    )
    bg = "\n".join(
        f"- {b.get('name', '')}：{b.get('background', '')}"
        for b in backgrounds if b and b.get("background")
    )
    user = (
        f"案例：\n标题：{case.case_name}\n内容：{case.text}\n\n"
        f"背景知识（按归类经过的节点列出）：\n{bg}\n\n"
        "请先分析：该案例的核心问题是什么、用户可能用哪些术语/关键词检索它，"
        "结合背景知识补充该领域的专有名词、同义说法、关键实体。背景知识仅作辅助，不必强凑。\n"
        "anchor 必须是**一行用空格隔开的关键词**（不是句子、不是模拟用户问法），"
        "覆盖案例的核心实体、术语与典型检索词。\n"
        '请先给出分析，最后用 ```json``` 代码块输出：{"anchor":"<空格隔开的关键词>"}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# 单条 case：导航 + 生成 anchor
# ---------------------------------------------------------------------------
async def build_one(
    llm: LLMClient, tree: Tree, case: Case, nav_by: str
) -> Dict[str, Any]:
    # 1. 逐层导航，积累背景
    visited: List[str] = []
    backgrounds: List[Dict[str, str]] = []
    node = tree.root
    while node.children:
        chosen = await navigate_case(llm, case, node.children, nav_by)
        if not chosen:
            break
        child = node.find_by_name(chosen)
        if child is None:
            break
        visited.append(child.name)
        if child.background:
            backgrounds.append({"name": child.name, "background": child.background})
        node = child

    # 2. 生成 anchor 关键词
    anchor, analysis = await gen_anchor(llm, case, backgrounds)
    return {
        "case_id": case.case_id,
        "case_name": case.case_name,
        "text": case.text,
        "visited": visited,
        "analysis": analysis,
        "anchor": anchor,
    }


async def navigate_case(llm: LLMClient, case: Case, children: List[Node], nav_by: str) -> str:
    if llm.provider == "mock":
        # 离线启发式：按实体/名称匹配
        ent = _detect_entity(case.to_text())
        for c in children:
            trig = c.dialog_trigger if nav_by == "dialog_trigger" else c.case_trigger
            hay = (c.name + " " + (trig or "") + " " + (c.background or "")).lower()
            if ent and ent.lower() in hay:
                return c.name
            if c.name and c.name.lower() in case.to_text().lower():
                return c.name
        return ""
    msgs = navigate_case_messages(case, children, nav_by)
    data = extract_json(await llm._chat(msgs))
    name = str(data.get("name", "")).strip()
    valid = {c.name for c in children}
    return name if name in valid else ""


async def gen_anchor(llm: LLMClient, case: Case, backgrounds: List[Dict]) -> tuple:
    if llm.provider == "mock":
        # 离线启发式：实体 + 标题分词，拼成空格关键词
        ent = _detect_entity(case.to_text()) or ""
        kws = [ent] + [w for w in case.case_name.replace("、", " ").split() if w]
        seen, out = set(), []
        for w in kws:
            if w and w not in seen:
                seen.add(w)
                out.append(w)
        return " ".join(out), ""
    msgs = anchor_messages(case, backgrounds)
    raw = await llm._chat(msgs)
    data = extract_json(raw)
    anchor = str(data.get("anchor", "")).strip()
    return anchor, _analysis_before_json(raw)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
async def main_async(args: argparse.Namespace) -> int:
    cases = Case.load_all(read_json(Path(args.case)))
    tree = Tree.from_dict(read_json(Path(args.tree)))
    n_filled = init_dialog_trigger(tree.root)
    log(f"加载案例 {len(cases)} 条；知识树：{args.tree}"
        + (f"（dialog_trigger 兜底填充 {n_filled} 个节点）" if n_filled else ""))
    log(f"导航依据：{args.nav_by}")

    enable_thinking = not args.no_thinking
    llm = LLMClient(LLMSettings(
        provider=args.provider, base_url=args.base_url, api_key=args.api_key,
        model=args.model, concurrency=args.concurrency, timeout_seconds=args.timeout,
        temperature=args.temperature, max_retries=args.max_retries,
        max_tokens=args.max_tokens, enable_thinking=enable_thinking,
    ))
    log(f"provider={args.provider} 思考模式={'开' if enable_thinking else '关'}")

    case_list = list(cases.values())
    sem = asyncio.Semaphore(max(1, args.concurrency))
    bar = _tqdm(total=len(case_list), desc="生成anchor", file=sys.stderr) if _tqdm else None

    async def _one(c: Case) -> Dict[str, Any]:
        async with sem:
            try:
                return await build_one(llm, tree, c, args.nav_by)
            except Exception as exc:  # noqa: BLE001
                log(f"生成 anchor 失败(已跳过) case_id={c.case_id}: {exc}")
                return {"case_id": c.case_id, "case_name": c.case_name, "text": c.text,
                        "visited": [], "analysis": "", "anchor": "", "error": str(exc)}
            finally:
                if bar is not None:
                    bar.update(1)

    records = await asyncio.gather(*[_one(c) for c in case_list])
    if bar is not None:
        bar.close()

    # 调试中间文件：含 visited / analysis
    write_json(Path(args.detail_file) if args.detail_file
               else ANCHOR_DIR / "anchor_detail.json", records)

    # 主产物：原 case 结构 + anchor 字段（{case_id: {case_name, text, anchor}}）
    out_cases: Dict[str, Any] = {}
    for r in records:
        out_cases[r["case_id"]] = {
            "case_name": r["case_name"],
            "text": r["text"],
            "anchor": r["anchor"],
        }
    write_json(Path(args.out), out_cases)

    n_empty = sum(1 for r in records if not r["anchor"])
    log(f"完成：{len(records)} 条，其中 anchor 为空 {n_empty} 条")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="为 case 生成 anchor 关键词字段")
    p.add_argument("--case", default=str(ROOT_DIR / "data" / "case" / "text.json"),
                   help="案例文件（{case_id: {case_name, text}}）")
    p.add_argument("--tree", default=str(BUILD_DIR / "output" / "knowledge_tree.json"),
                   help="知识树文件")
    p.add_argument("--out", default=str(ANCHOR_DIR / "text_with_anchor.json"),
                   help="输出：原 case + anchor 字段")
    p.add_argument("--detail-file", default=None,
                   help="调试中间文件（含 visited / analysis），默认 case_anchor/anchor_detail.json")
    p.add_argument("--nav-by", choices=["dialog_trigger", "case_trigger"],
                   default="dialog_trigger", help="导航依据的 trigger 字段")
    p.add_argument("--provider", default="openai", help="LLM provider：openai 或 mock")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="1234")
    p.add_argument("--model", default="gpt-4.1-mini")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--no-thinking", action="store_true", help="关闭思考模式")
    p.add_argument("--thinking", dest="no_thinking", action="store_false",
                   help="开启思考模式（默认）")
    p.set_defaults(no_thinking=False)
    return p.parse_args()


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
