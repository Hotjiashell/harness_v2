# -*- coding: utf-8 -*-
"""阶段二导航/检索评测脚本（独立于 build 框架，可单独运行）。

给定对话数据 + 已构建的知识树，跑完「逐层导航 → 收集背景 → 生成 query」全过程，
再调用根目录 retrieve.py::retrieve_case 计算 top-1/3/5/10/20/50 召回率。

三个阶段（--stage）：
  full    ：导航+生成 query+检索，跑完整流程（默认）
  query   ：只跑到生成 query，产出 query 中间文件，不检索
  retrieve：从已有 query 中间文件读取 query，只跑检索与召回统计

支持：
  - 自定义输入/输出路径（对话、知识树、query 中间文件、结果文件）
  - 开启/关闭思考模式（--thinking / --no-thinking）

用法示例：
  python eval/eval_recall.py \
      --dialog data/dialog/dialog.json \
      --tree build/output/knowledge_tree.json \
      --out-dir eval/output

  python eval/eval_recall.py --stage query --no-thinking
  python eval/eval_recall.py --stage retrieve --query-file eval/output/queries.json
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 路径与依赖：复用 build/khtree 的 LLM 客户端与数据模型
# ---------------------------------------------------------------------------
EVAL_DIR = Path(__file__).resolve().parent
ROOT_DIR = EVAL_DIR.parent
BUILD_DIR = ROOT_DIR / "build"
for p in (str(BUILD_DIR), str(ROOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from khtree.config_types import LLMSettings  # noqa: E402
from khtree.llm import LLMClient  # noqa: E402
from khtree.models import Dialog, Node, Tree  # noqa: E402

TOP_KS = [1, 3, 5, 10, 20, 50]


# ---------------------------------------------------------------------------
# 简单日志
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[eval] {msg}", file=sys.stderr, flush=True)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"已写出：{p}")


# ---------------------------------------------------------------------------
# 加载根目录 retrieve.py::retrieve_case（动态加载，避免与包名冲突）
# ---------------------------------------------------------------------------
def load_retrieve_case():
    path = ROOT_DIR / "retrieve.py"
    spec = importlib.util.spec_from_file_location("_eval_retrieve", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    fn = getattr(mod, "retrieve_case", None)
    if fn is None:
        raise RuntimeError("retrieve.py 中找不到 retrieve_case")
    return fn


# ---------------------------------------------------------------------------
# 导航 + 生成 query
# ---------------------------------------------------------------------------
async def navigate_and_make_query(
    llm: LLMClient, tree: Tree, dialog: Dialog
) -> Dict[str, Any]:
    """对单条对话逐层导航并生成 query，记录经过的节点与最终 query。"""
    visited: List[str] = []
    backgrounds: List[str] = []
    node = tree.root
    while node.children:
        chosen = await llm.navigate(dialog.chat_content, node.children)
        if not chosen:
            break
        child = node.find_by_name(chosen)
        if child is None:
            break
        visited.append(child.name)
        if child.background:
            backgrounds.append(child.background)
        node = child
    query = await llm.generate_query(dialog.chat_content, backgrounds)
    return {
        "call_sno": dialog.call_sno,
        "case_id": dialog.case_id,
        "chat_content": dialog.chat_content,
        "visited": visited,
        "query": query,
    }


async def run_navigation(
    llm: LLMClient, tree: Tree, dialogs: List[Dialog], concurrency: int
) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(max(1, concurrency))
    results: List[Optional[Dict[str, Any]]] = [None] * len(dialogs)

    async def _one(i: int, d: Dialog) -> None:
        async with sem:
            try:
                results[i] = await navigate_and_make_query(llm, tree, d)
            except Exception as exc:  # noqa: BLE001
                log(f"导航失败(已跳过) call_sno={d.call_sno}: {exc}")
                results[i] = {"call_sno": d.call_sno, "case_id": d.case_id,
                              "chat_content": d.chat_content, "visited": [],
                              "query": "", "error": str(exc)}

    await asyncio.gather(*[_one(i, d) for i, d in enumerate(dialogs)])
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# 检索 + 召回率
# ---------------------------------------------------------------------------
async def run_retrieval(
    retrieve_case, records: List[Dict[str, Any]], concurrency: int
) -> List[Dict[str, Any]]:
    """对每条记录的 query 调 retrieve_case 取 top-max(TOP_KS)，记录命中目标案例的最小排名。"""
    max_k = max(TOP_KS)
    sem = asyncio.Semaphore(max(1, concurrency))
    is_async = inspect.iscoroutinefunction(retrieve_case)

    async def _one(rec: Dict[str, Any]) -> Dict[str, Any]:
        query = rec.get("query", "")
        case_id = rec.get("case_id", "")
        hit_rank = None
        try:
            async with sem:
                if is_async:
                    res = await retrieve_case(query, top_k=max_k)
                else:
                    # 同步函数：丢线程池执行，避免阻塞事件循环、实现真并行
                    res = await asyncio.to_thread(retrieve_case, query, max_k)
            res = res or []
            for idx, item in enumerate(res):
                if str(item.get("caseID")) == str(case_id):
                    hit_rank = idx + 1  # 1-based 排名
                    break
        except Exception as exc:  # noqa: BLE001
            log(f"检索失败(已跳过) call_sno={rec.get('call_sno')}: {exc}")
            rec = {**rec, "retrieve_error": str(exc)}
        return {**rec, "hit_rank": hit_rank,
                "hit_at": {f"top{k}": (hit_rank is not None and hit_rank <= k) for k in TOP_KS}}

    return await asyncio.gather(*[_one(r) for r in records])


def summarize_recall(retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(retrieved)
    recall = {}
    for k in TOP_KS:
        hits = sum(1 for r in retrieved if r.get("hit_rank") is not None and r["hit_rank"] <= k)
        recall[f"top{k}"] = {"hits": hits, "total": total,
                             "recall": (hits / total) if total else 0.0}
    return {"total": total, "recall": recall}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
async def main_async(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    query_file = Path(args.query_file) if args.query_file else out_dir / "queries.json"
    nav_file = out_dir / "navigation.json"
    result_file = Path(args.result_file) if args.result_file else out_dir / "recall_result.json"

    enable_thinking = not args.no_thinking

    # 阶段：full / query 需要导航；retrieve 直接读 query 文件
    if args.stage in ("full", "query"):
        dialogs = Dialog.load_all(read_json(Path(args.dialog)))
        tree = Tree.from_dict(read_json(Path(args.tree)))
        log(f"加载对话 {len(dialogs)} 条；知识树：{args.tree}")
        llm = LLMClient(LLMSettings(
            provider=args.provider, base_url=args.base_url, api_key=args.api_key,
            model=args.model, concurrency=args.concurrency,
            timeout_seconds=args.timeout, temperature=args.temperature,
            max_retries=args.max_retries, max_tokens=args.max_tokens,
            enable_thinking=enable_thinking,
        ))
        log(f"provider={args.provider} 思考模式={'开' if enable_thinking else '关'}")
        records = await run_navigation(llm, tree, dialogs, args.concurrency)
        # 导航中间文件（经过的节点 + query）
        write_json(nav_file, records)
        write_json(query_file,
                   [{"call_sno": r["call_sno"], "case_id": r["case_id"], "query": r["query"]}
                    for r in records])
        if args.stage == "query":
            log("阶段 query：已生成 query，结束（未检索）。")
            return 0
    else:  # retrieve
        if not query_file.exists():
            log(f"找不到 query 文件：{query_file}。请先用 --stage query 生成，或用 --query-file 指定。")
            return 2
        records = read_json(query_file)
        log(f"从 query 文件读取 {len(records)} 条：{query_file}")

    # 检索 + 召回率
    retrieve_case = load_retrieve_case()
    retrieved = await run_retrieval(retrieve_case, records, args.concurrency)
    summary = summarize_recall(retrieved)

    write_json(out_dir / "retrieval_detail.json", retrieved)
    write_json(result_file, {
        "config": {
            "stage": args.stage, "provider": args.provider, "model": args.model,
            "enable_thinking": enable_thinking,
            "dialog": args.dialog, "tree": args.tree,
        },
        "summary": summary,
    })

    # 控制台输出召回率
    log("=" * 50)
    log(f"召回率（共 {summary['total']} 条对话）：")
    for k in TOP_KS:
        r = summary["recall"][f"top{k}"]
        log(f"  top{k:<2}: {r['recall']:.4f}  ({r['hits']}/{r['total']})")
    log("=" * 50)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="阶段二导航/检索召回率评测")
    p.add_argument("--stage", choices=["full", "query", "retrieve"], default="full",
                   help="full=全流程；query=只到生成query；retrieve=从已有query只跑检索")
    # 输入
    p.add_argument("--dialog", default=str(ROOT_DIR / "data" / "dialog" / "dialog.json"),
                   help="对话数据文件")
    p.add_argument("--tree", default=str(BUILD_DIR / "output" / "knowledge_tree.json"),
                   help="知识树文件")
    p.add_argument("--query-file", default=None,
                   help="query 中间文件路径（retrieve 阶段从此读取；其他阶段写入此处）")
    # 输出
    p.add_argument("--out-dir", default=str(EVAL_DIR / "output"), help="输出目录")
    p.add_argument("--result-file", default=None, help="召回率结果文件路径")
    # LLM
    p.add_argument("--provider", default="openai", help="LLM provider：openai 或 mock")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="1234")
    p.add_argument("--model", default="gpt-4.1-mini")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=2048)
    # 思考模式：默认开启，传 --no-thinking 关闭
    p.add_argument("--no-thinking", action="store_true", help="关闭思考模式")
    p.add_argument("--thinking", dest="no_thinking", action="store_false",
                   help="开启思考模式（默认）")
    p.set_defaults(no_thinking=False)
    return p.parse_args()


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
