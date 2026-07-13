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


def load_dialog_map(path: Path) -> Dict[str, Dict[str, Any]]:
    """读取原始对话文件，返回 {call_sno: 完整对话记录}。"""
    if not path or not Path(path).exists():
        if path:
            log(f"提示：找不到对话文件 {path}，无法提供调试信息或 use_chat 检索内容")
        return {}
    raw = read_json(Path(path))
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            sno = str(item.get("call_sno", "")).strip()
            if sno:
                out[sno] = item
        return out
    if isinstance(raw, dict):
        for key, value in raw.items():
            if not isinstance(value, dict):
                continue
            sno = str(value.get("call_sno") or key).strip()
            if sno:
                out[sno] = value
        return out
    log(f"提示：对话文件格式不支持：{path}")
    return out


def load_case_text_map(path: Path) -> Dict[str, Dict[str, Any]]:
    """读取案例标题/内容文件，返回 {case_id: 案例记录}。"""
    if not path or not Path(path).exists():
        if path:
            log(f"提示：找不到案例文件 {path}，无法补充 GT 案例调试信息")
        return {}
    raw = read_json(Path(path))
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            cid = str(key).strip()
            if not cid:
                continue
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("case_id", cid)
                item.setdefault("caseID", cid)
                out[cid] = item
            else:
                out[cid] = {"case_id": cid, "caseID": cid, "text": str(value)}
        return out
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            cid = case_id_of(item)
            if cid:
                out[cid] = item
        return out
    log(f"提示：案例文件格式不支持：{path}")
    return out


async def retrieve_two(
    retrieve_case, query_a: str, query_b: str, max_k: int, strategy: str,
    sem: asyncio.Semaphore, index: str = "document_12",
    use_similar_question: bool = False, use_chat: bool = False, chat_content: str = "",
) -> tuple:
    """用两个 query 各检索 top-max_k，返回 (items_a, items_b)。"""
    is_async = inspect.iscoroutinefunction(retrieve_case)

    async def _run(q: str) -> List[Dict[str, Any]]:
        if not q:
            return []
        kwargs = dict(top_k=max_k, strategy=strategy, index=index,
                      use_similar_question=use_similar_question,
                      use_chat=use_chat,
                      chat_content=chat_content if use_chat else "")
        async with sem:
            if is_async:
                res = await retrieve_case(q, **kwargs)
            else:
                res = await asyncio.to_thread(retrieve_case, q, **kwargs)
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
    log(f"检索策略：{args.strategy}；index：{args.index}"
        f"；use_similar_question={args.use_similar_question}；use_chat={args.use_chat}")

    common = sorted(set(qa) & set(qb))
    only_a, only_b = set(qa) - set(qb), set(qb) - set(qa)
    if only_a or only_b:
        log(f"提示：仅 A 有 {len(only_a)} 条，仅 B 有 {len(only_b)} 条，未纳入对比")
    log(f"共同 call_sno：{len(common)} 条")

    retrieve_case = load_retrieve_case()
    max_k = max(TOP_KS)
    sem = asyncio.Semaphore(max(1, args.concurrency))

    dialog_map = load_dialog_map(Path(args.dialog))
    case_text_map = load_case_text_map(Path(args.case_text))
    if dialog_map:
        log(f"已加载对话调试数据：{len(dialog_map)} 条")
    if case_text_map:
        log(f"已加载 GT 案例调试数据：{len(case_text_map)} 条")

    # 检索用对话只允许来自 --dialog；query A/B 中即使带 chat_content 也一律忽略。
    dialog_chat = {sno: str(item.get("chat_content", "")) for sno, item in dialog_map.items()}
    if args.use_chat:
        missing = [sno for sno in common if not dialog_chat.get(sno, "")]
        log(f"检索对话仅从 --dialog 加载：{args.dialog}（{len(dialog_chat)} 条）")
        if missing:
            preview = ", ".join(missing[:5])
            suffix = "..." if len(missing) > 5 else ""
            log(f"错误：--dialog 中缺少 {len(missing)} 条共同 query 的非空 chat_content：{preview}{suffix}")
            return 2

    async def _one(sno: str) -> Dict[str, Any]:
        ra, rb = qa[sno], qb[sno]
        dialog_item = dialog_map.get(sno, {})
        case_id = str(ra.get("case_id") or rb.get("case_id") or dialog_item.get("caseID") or "")
        gt_case_item = case_text_map.get(case_id, {})
        query_a = ra.get("query", "")
        query_b = rb.get("query", "")
        # A/B 共用 --dialog 中同一 call_sno 的对话，绝不读取 query 文件内的对话字段。
        chat_content = dialog_chat.get(sno, "")
        rec: Dict[str, Any] = {
            "call_sno": sno, "case_id": case_id,
            "query_a": query_a, "query_b": query_b,
            "dialog": dialog_item,
            "gt_case": {
                "case_id": case_id,
                "case_name": str(gt_case_item.get("case_name", "")),
                "text": str(gt_case_item.get("text", "")),
            },
        }
        try:
            items_a, items_b = await retrieve_two(
                retrieve_case, query_a, query_b, max_k, args.strategy, sem,
                index=args.index, use_similar_question=args.use_similar_question,
                use_chat=args.use_chat, chat_content=chat_content,
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
            "dialog": str(Path(args.dialog)),
            "strategy": args.strategy, "index": args.index,
            "use_similar_question": args.use_similar_question, "use_chat": args.use_chat,
            "compared": len(common),
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
    p.add_argument("--index", default="document_12", help="传给 retrieve_case 的检索索引名称")
    p.add_argument("--use-similar-question", action="store_true",
                   help="检索时启用相似问题（retrieve_case 的 use_similar_question）")
    p.add_argument("--use-chat", action="store_true",
                   help="检索时启用对话内容；chat_content 只从 --dialog 按 call_sno 读取")
    p.add_argument("--dialog", default=str(ROOT_DIR / "data" / "dialog" / "dialog.json"),
                   help="原始对话文件：用于调试信息，也是 use_chat 时检索对话的唯一来源")
    p.add_argument("--case-text", default=str(ROOT_DIR / "data" / "case" / "text.json"),
                   help="案例标题/内容文件：用于 fusion_detail.json 补充 GT case_name 和 text")
    p.add_argument("--concurrency", type=int, default=8)
