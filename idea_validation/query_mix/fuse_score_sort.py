# -*- coding: utf-8 -*-
"""融合方式 2：去重后按相关度得分排序，取前 top-N。

做法：两个 query 各自检索，合并两边结果并去重（同一案例若两边都出现，
取较高的相关度得分），再按得分从高到低排序，取前 N 个。

用法：
  python idea_validation/query_mix/fuse_score_sort.py \
      --query-a eval/output/queries.json \
      --query-b eval/baseline_output/queries.json \
      --out-dir idea_validation/query_mix/output_score
"""
from __future__ import annotations

import argparse
import asyncio
from typing import Any, Dict, List

import _fusion_common as FC


def fuse_score_sort(
    items_a: List[Dict[str, Any]], items_b: List[Dict[str, Any]], n: int
) -> List[str]:
    """合并去重 -> 按得分降序 -> 取前 n。重复案例取两边较高得分。"""
    best: Dict[str, float] = {}
    order: List[str] = []  # 记录首次出现顺序，作为同分时的稳定次序
    for it in list(items_a) + list(items_b):
        cid = FC.case_id_of(it)
        if not cid:
            continue
        s = FC.score_of(it)
        if cid not in best:
            best[cid] = s
            order.append(cid)
        else:
            best[cid] = max(best[cid], s)
    # 按得分降序，稳定排序保证同分时维持首次出现顺序
    ranked = sorted(order, key=lambda cid: best[cid], reverse=True)
    return ranked[:n]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="query 融合召回评测（方式2：去重后按得分排序取topN）")
    FC.add_common_args(p)
    p.set_defaults(out_dir=str(FC.EVAL_QM_DIR / "output_score"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(FC.run_fusion_eval(args, fuse_score_sort, "score_sort(去重按得分取topN)"))


if __name__ == "__main__":
    raise SystemExit(main())
