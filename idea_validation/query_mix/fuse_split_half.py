# -*- coding: utf-8 -*-
"""融合方式 1：两个 query 各分一半，去重后尽量平分名额。

做法（对每个 top-N）：两个 query 各自检索，去除重复案例（每个案例只保留一次），
然后把 N 个名额尽量在 A、B 之间平分；当 N 为奇数、最后一个名额二选一时，
取相关度得分更高的那个候选。

实现为「交替取数」：A、B 轮流贡献各自下一个**未被选过**的案例。由得分更高的一方
先手，于是奇数名额的多出的那 1 个自然落给得分更高的一方；偶数则恰好平分。
这样得到一条统一的融合排序，对任意 top-N 取前 N 个即满足上述规则。

用法：
  python idea_validation/query_mix/fuse_split_half.py \
      --query-a eval/output/queries.json \
      --query-b eval/baseline_output/queries.json \
      --out-dir idea_validation/query_mix/output_split
"""
from __future__ import annotations

import argparse
import asyncio
from typing import Any, Dict, List

import _fusion_common as FC


def fuse_split_half(
    items_a: List[Dict[str, Any]], items_b: List[Dict[str, Any]], n: int
) -> List[str]:
    """交替取数融合：得分高的一方先手，轮流取各自下一个未选案例，去重，长度≤n。"""
    ids_a = [FC.case_id_of(it) for it in items_a]
    ids_b = [FC.case_id_of(it) for it in items_b]
    score_a = {FC.case_id_of(it): FC.score_of(it) for it in items_a}
    score_b = {FC.case_id_of(it): FC.score_of(it) for it in items_b}

    # 先手方：两列表头部得分更高者先取（决定奇数名额的归属）
    head_a = score_a.get(ids_a[0]) if ids_a else float("-inf")
    head_b = score_b.get(ids_b[0]) if ids_b else float("-inf")
    a_first = (head_a if head_a is not None else float("-inf")) >= \
              (head_b if head_b is not None else float("-inf"))

    i, j = 0, 0
    chosen: List[str] = []
    seen = set()
    take_a = a_first
    # 当两边都还有候选时严格交替；一方耗尽后另一方补齐
    while len(chosen) < n and (i < len(ids_a) or j < len(ids_b)):
        progressed = False
        if take_a:
            while i < len(ids_a):
                cid = ids_a[i]; i += 1
                if cid and cid not in seen:
                    seen.add(cid); chosen.append(cid); progressed = True
                    break
        else:
            while j < len(ids_b):
                cid = ids_b[j]; j += 1
                if cid and cid not in seen:
                    seen.add(cid); chosen.append(cid); progressed = True
                    break
        # 轮转到另一方；若当前方已无新案例，强制切换避免空转
        if not progressed:
            if take_a and i >= len(ids_a) and j >= len(ids_b):
                break
            if not take_a and j >= len(ids_b) and i >= len(ids_a):
                break
        take_a = not take_a
    return chosen[:n]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="query 融合召回评测（方式1：各分一半，去重平分）")
    FC.add_common_args(p)
    p.set_defaults(out_dir=str(FC.EVAL_QM_DIR / "output_split"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(FC.run_fusion_eval(args, fuse_split_half, "split_half(各分一半去重平分)"))


if __name__ == "__main__":
    raise SystemExit(main())
