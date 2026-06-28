# -*- coding: utf-8 -*-
"""query 融合召回评测的公共逻辑。

两种融合脚本（fuse_split_half.py / fuse_score_sort.py）共用这里的：
  - 读取两个 query 文件、按 call_sno 对齐
  - 用两个 query 各自检索 top-max(TOP_KS)
  - 调用各自的融合函数得到融合后的案例排序
  - 计算 top-1/3/5/10/20/50 召回率、写结果文件

融合函数本身在各脚本里实现，通过 fuse_fn(items_a, items_b, n) 注入。
items_* 为 retrieve_case 的返回列表，元素含 caseID 与 score（按相关度降序）。
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

EVAL_QM_DIR = Path(__file__).resolve().parent
ROOT_DIR = EVAL_QM_DIR.parent.parent          # query_mix -> idea_validation -> repo root
EVAL_DIR = ROOT_DIR / "eval"
BUILD_DIR = ROOT_DIR / "build"
for p in (str(EVAL_DIR), str(BUILD_DIR), str(ROOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# 复用 eval_recall 的工具
from eval_recall import (  # noqa: E402
    TOP_KS, log, read_json, write_json, load_retrieve_case,
    summarize_recall, summarize_top_cases, _gather_with_progress,
)


def case_id_of(item: Dict[str, Any]) -> str:
    return str(item.get("caseID") or item.get("case_id") or "")


def score_of(item: Dict[str, Any]) -> float:
    try:
        return float(item.get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def load_query_map(path: Path) -> Dict[str, Dict[str, Any]]:
    """读取 query 文件（[{call_sno, case_id, query}]），返回 {call_sno: record}。"""
    records = read_json(path)
    return {str(r.get("call_sno")): r for r in records}


async def retrieve_two(
    retrieve_case, query_a: str, query_b: str, max_k: int, strategy: str,
    sem: asyncio.Semaphore,
) -> tuple:
    """用两个 query 各检索 top-max_k，返回 (items_a, items_b)。"""
    is_async = inspect.iscoroutinefunction(retrieve_case)

    async def _run(q: str) -> List[Dict[str, Any]]:
        if not q:
            return []
        async with sem:
            if is_async:
                res = await retrieve_case(q, top_k=max_k, strategy=strategy)
            else:
                res = await asyncio.to_thread(retrieve_case, q, top_k=max_k, strategy=strategy)
        return res or []

    a, b = await asyncio.gather(_run(query_a), _run(query_b))
    return a, b


def hit_rank_in(fused_ids: List[str], case_id: str) -> Any:
    """目标案例在融合排序里的 1-based 名次；未命中返回 None。"""
    for idx, cid in enumerate(fused_ids):
        if cid == str(case_id):
            return idx + 1
    return None


async def run_fusion_eval(
    args: argparse.Namespace,
    fuse_fn: Callable[[List[Dict[str, Any]], List[Dict[str, Any]], int], List[str]],
    method_name: str,
) -> int:
    """通用主流程。fuse_fn(items_a, items_b, n) -> 融合后的 caseID 列表（长度≤n，已按融合规则排序）。"""
    qa_path, qb_path = Path(args.query_a), Path(args.query_b)
    qa = load_query_map(qa_path)
    qb = load_query_map(qb_path)
    log(f"融合方法：{method_name}")
    log(f"query A：{qa_path}（{len(qa)} 条）")
    log(f"query B：{qb_path}（{len(qb)} 条）")

    common = sorted(set(qa) & set(qb))
    only_a, only_b = set(qa) - set(qb), set(qb) - set(qa)
    if only_a or only_b:
        log(f"提示：仅 A 有 {len(only_a)} 条，仅 B 有 {len(only_b)} 条，未纳入对比")
    log(f"共同 call_sno：{len(common)} 条")

    retrieve_case = load_retrieve_case()
    max_k = max(TOP_KS)
    sem = asyncio.Semaphore(max(1, args.concurrency))

    async def _one(sno: str) -> Dict[str, Any]:
        ra, rb = qa[sno], qb[sno]
        case_id = str(ra.get("case_id") or rb.get("case_id") or "")
        query_a = ra.get("query", "")
        query_b = rb.get("query", "")
        rec: Dict[str, Any] = {
            "call_sno": sno, "case_id": case_id,
            "query_a": query_a, "query_b": query_b,
        }
        try:
            items_a, items_b = await retrieve_two(
                retrieve_case, query_a, query_b, max_k, args.strategy, sem
            )
            fused_ids = fuse_fn(items_a, items_b, max_k)
            hit_rank = hit_rank_in(fused_ids, case_id)
            rec.update({
                "hit_rank": hit_rank,
                "topA": summarize_top_cases(items_a, 5),
                "topB": summarize_top_cases(items_b, 5),
                "fused_top5": fused_ids[:5],
                "hit_at": {f"top{k}": (hit_rank is not None and hit_rank <= k) for k in TOP_KS},
            })
        except Exception as exc:  # noqa: BLE001
            log(f"融合检索失败(已跳过) call_sno={sno}: {exc}")
            rec.update({"hit_rank": None, "fuse_error": str(exc),
                        "hit_at": {f"top{k}": False for k in TOP_KS}})
        return rec

    retrieved = await _gather_with_progress([_one(s) for s in common], f"融合检索[{method_name}]")
    summary = summarize_recall(retrieved)

    out_dir = Path(args.out_dir)
    write_json(out_dir / "fusion_detail.json", retrieved)
    result_file = Path(args.result_file) if args.result_file else out_dir / "recall_result.json"
    write_json(result_file, {
        "config": {
            "method": method_name,
            "query_a": str(qa_path), "query_b": str(qb_path),
            "strategy": args.strategy, "compared": len(common),
        },
        "summary": summary,
    })

    log("=" * 50)
    log(f"融合召回率（方法：{method_name}，共 {summary['total']} 条）：")
    for k in TOP_KS:
        r = summary["recall"][f"top{k}"]
        log(f"  top{k:<2}: {r['recall']:.4f}  ({r['hits']}/{r['total']})")
    log("=" * 50)
    return 0


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--query-a", required=True, help="query 文件 A（[{call_sno, case_id, query}]）")
    p.add_argument("--query-b", required=True, help="query 文件 B")
    p.add_argument("--out-dir", default=str(EVAL_QM_DIR / "output"), help="输出目录")
    p.add_argument("--result-file", default=None, help="召回率结果文件路径")
    p.add_argument("--strategy", default="lexical&semantic", help="传给 retrieve_case 的检索策略")
    p.add_argument("--concurrency", type=int, default=8)
