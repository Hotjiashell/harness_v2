# -*- coding: utf-8 -*-
"""Baseline 召回率评测：不走知识树导航，直接由对话生成 query，再检索算召回率。

用作对照组——衡量「知识树导航 + 节点背景知识」相比「直接拿对话生成 query」
到底带来多少召回率增益。

复用 eval_recall.py 的检索/召回/进度条逻辑，只把「导航+收集背景」换成
「直接用对话内容、空背景生成 query」。

三个阶段（--stage）与 eval_recall 一致：
  full    ：生成 query + 检索 + 召回率（默认）
  query   ：只生成 query，写出 queries.json，不检索
  retrieve：从已有 query 文件只跑检索 + 召回统计

用法示例：
  python eval/eval_baseline.py --dialog data/dialog/dialog.json --out-dir eval/baseline_output
  python eval/eval_baseline.py --stage query --no-thinking
  python eval/eval_baseline.py --stage retrieve --query-file eval/baseline_output/queries.json
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List

EVAL_DIR = Path(__file__).resolve().parent
ROOT_DIR = EVAL_DIR.parent
BUILD_DIR = ROOT_DIR / "build"
for p in (str(EVAL_DIR), str(BUILD_DIR), str(ROOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from khtree.config_types import LLMSettings  # noqa: E402
from khtree.llm import LLMClient  # noqa: E402
from khtree.models import Dialog  # noqa: E402

# 复用 eval_recall 的工具：日志、IO、检索、召回汇总、进度条
import eval_recall as ER  # noqa: E402
from eval_recall import (  # noqa: E402
    TOP_KS, log, read_json, write_json, load_retrieve_case,
    run_retrieval, summarize_recall, _gather_with_progress,
)


# ---------------------------------------------------------------------------
# baseline：直接由对话生成 query（空背景，不导航）
# ---------------------------------------------------------------------------
async def run_baseline_query(
    llm: LLMClient, dialogs: List[Dialog], concurrency: int
) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(d: Dialog) -> Dict[str, Any]:
        async with sem:
            try:
                # 不导航、无节点背景：query 仅来自对话内容
                query, analysis = await llm.generate_query_ex(d.chat_content, [])
            except Exception as exc:  # noqa: BLE001
                log(f"生成 query 失败(已跳过) call_sno={d.call_sno}: {exc}")
                return {"call_sno": d.call_sno, "case_id": d.case_id,
                        "chat_content": d.chat_content, "query": "", "error": str(exc)}
            return {"call_sno": d.call_sno, "case_id": d.case_id,
                    "chat_content": d.chat_content, "analysis": analysis, "query": query}

    results = await _gather_with_progress([_one(d) for d in dialogs], "baseline生成query")
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
async def main_async(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    query_file = Path(args.query_file) if args.query_file else out_dir / "queries.json"
    result_file = Path(args.result_file) if args.result_file else out_dir / "recall_result.json"
    enable_thinking = not args.no_thinking

    if args.stage in ("full", "query"):
        dialogs = Dialog.load_all(read_json(Path(args.dialog)))
        log(f"加载对话 {len(dialogs)} 条（baseline：不走知识树导航）")
        llm = LLMClient(LLMSettings(
            provider=args.provider, base_url=args.base_url, api_key=args.api_key,
            model=args.model, concurrency=args.concurrency,
            timeout_seconds=args.timeout, temperature=args.temperature,
            max_retries=args.max_retries, max_tokens=args.max_tokens,
            enable_thinking=enable_thinking,
        ))
        log(f"provider={args.provider} 思考模式={'开' if enable_thinking else '关'}")
        records = await run_baseline_query(llm, dialogs, args.concurrency)
        write_json(query_file,
                   [{"call_sno": r["call_sno"], "case_id": r["case_id"], "query": r["query"]}
                    for r in records])
        if args.stage == "query":
            log("阶段 query：已生成 baseline query，结束（未检索）。")
            return 0
    else:  # retrieve
        if not query_file.exists():
            log(f"找不到 query 文件：{query_file}。请先用 --stage query 生成，或用 --query-file 指定。")
            return 2
        records = read_json(query_file)
        log(f"从 query 文件读取 {len(records)} 条：{query_file}")

    retrieve_case = load_retrieve_case()
    log(f"检索策略：{args.strategy}；index：{args.index}")
    retrieved = await run_retrieval(
        retrieve_case, records, args.concurrency, strategy=args.strategy, index=args.index
    )
    summary = summarize_recall(retrieved)

    write_json(out_dir / "retrieval_detail.json", retrieved)
    write_json(result_file, {
        "config": {
            "mode": "baseline（无导航，对话直生 query）",
            "stage": args.stage, "provider": args.provider, "model": args.model,
            "enable_thinking": enable_thinking, "dialog": args.dialog,
            "strategy": args.strategy, "index": args.index,
        },
        "summary": summary,
    })

    log("=" * 50)
    log(f"Baseline 召回率（共 {summary['total']} 条对话）：")
    for k in TOP_KS:
        r = summary["recall"][f"top{k}"]
        log(f"  top{k:<2}: {r['recall']:.4f}  ({r['hits']}/{r['total']})")
    log("=" * 50)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline（无导航）召回率评测")
    p.add_argument("--stage", choices=["full", "query", "retrieve"], default="full",
                   help="full=生成query+检索；query=只生成query；retrieve=从已有query只跑检索")
    p.add_argument("--dialog", default=str(ROOT_DIR / "data" / "dialog" / "dialog.json"),
                   help="对话数据文件")
    p.add_argument("--query-file", default=None,
                   help="query 中间文件路径（retrieve 阶段读取，其他阶段写入）")
    p.add_argument("--strategy", default="lexical&semantic",
                   help="传给 retrieve.py::retrieve_case 的检索策略")
    p.add_argument("--index", default="document_12",
                   help="传给 retrieve.py::retrieve_case 的检索索引名称")
    p.add_argument("--out-dir", default=str(EVAL_DIR / "baseline_output"), help="输出目录")
    p.add_argument("--result-file", default=None, help="召回率结果文件路径")
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
