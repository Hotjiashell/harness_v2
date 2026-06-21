# -*- coding: utf-8 -*-
"""对比两个评测输出目录，找出「目录1 检索到、但目录2 没检索到」的对话。

按 top-1/3/5/10/20/50 分别整理：在该 top-k 下，目录1 命中而目录2 未命中的对话清单。
用于定位「方案1 相比 方案2 多召回了哪些样本」（如导航版 vs baseline）。

输入：两个目录，各自包含 retrieval_detail.json（由 eval_recall.py / eval_baseline.py 产出）。
输出：一个 JSON 文件。

用法：
  python eval/compare_recall.py --dir1 eval/output --dir2 eval/baseline_output \
      --out eval/compare_dir1_minus_dir2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

TOP_KS = [1, 3, 5, 10, 20, 50]


def log(msg: str) -> None:
    print(f"[compare] {msg}", file=sys.stderr, flush=True)


def load_detail(d: Path) -> Dict[str, Dict[str, Any]]:
    """读取目录下 retrieval_detail.json，返回 {call_sno: record}。"""
    path = d / "retrieval_detail.json"
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    return {str(r.get("call_sno")): r for r in records}


def hit_at(rec: Dict[str, Any], k: int) -> bool:
    """该记录在 top-k 下是否命中（优先用 hit_at，回退用 hit_rank）。"""
    if not rec:
        return False
    ha = rec.get("hit_at") or {}
    if f"top{k}" in ha:
        return bool(ha[f"top{k}"])
    rank = rec.get("hit_rank")
    return rank is not None and rank <= k


def main() -> int:
    ap = argparse.ArgumentParser(description="对比两个评测目录：目录1命中但目录2未命中的对话")
    ap.add_argument("--dir1", required=True, help="评测输出目录1（命中方）")
    ap.add_argument("--dir2", required=True, help="评测输出目录2（未命中方）")
    ap.add_argument("--out", required=True, help="输出 JSON 文件路径")
    args = ap.parse_args()

    d1 = load_detail(Path(args.dir1))
    d2 = load_detail(Path(args.dir2))
    log(f"目录1 {len(d1)} 条；目录2 {len(d2)} 条")

    # 以两边都有的 call_sno 为比较基准
    common = sorted(set(d1) & set(d2))
    only1 = sorted(set(d1) - set(d2))
    only2 = sorted(set(d2) - set(d1))
    if only1 or only2:
        log(f"提示：仅目录1有 {len(only1)} 条，仅目录2有 {len(only2)} 条，未纳入对比")

    result: Dict[str, Any] = {
        "dir1": str(args.dir1),
        "dir2": str(args.dir2),
        "compared": len(common),
        "diff": {},
    }
    for k in TOP_KS:
        items: List[Dict[str, Any]] = []
        for sno in common:
            r1, r2 = d1[sno], d2[sno]
            if hit_at(r1, k) and not hit_at(r2, k):
                items.append({
                    "call_sno": sno,
                    "case_id": r1.get("case_id"),
                    "dir1_rank": r1.get("hit_rank"),
                    "dir2_rank": r2.get("hit_rank"),
                    "dir1_query": r1.get("query"),
                    "dir2_query": r2.get("query"),
                })
        result["diff"][f"top{k}"] = {"count": len(items), "items": items}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"已写出：{out}")

    log("=" * 50)
    log("目录1 命中、目录2 未命中 的对话数：")
    for k in TOP_KS:
        log(f"  top{k:<2}: {result['diff'][f'top{k}']['count']} 条")
    log("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
