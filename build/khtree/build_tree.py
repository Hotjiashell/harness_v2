# -*- coding: utf-8 -*-
"""阶段一：基于案例库逐层构建知识树。

流程（每一层）：
  指定/聚类得到初始类别
    -> 案例分类 (Classification, 并行)
    -> Batch Reflection (并行) -> Proposals
    -> Proposal Aggregation -> Update Plan
    -> Complexity Check -> Coverage Validation
    -> Accept / Feedback
  对每个类别下的案例递归构建下一层，直到 max_depth。

所有阶段都会写出中间结果，便于调试与续跑。
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config_types import HarnessConfig
from .llm import LLMClient
from .models import (
    ADD,
    MODIFY,
    Case,
    ClassificationResult,
    Node,
    Operation,
    Tree,
)
from .utils import (
    ErrorRecorder,
    chunk,
    dump_intermediate,
    gather_limited,
    log,
    stage_banner,
    write_json,
)


class CaseTreeBuilder:
    def __init__(
        self,
        config: HarnessConfig,
        llm: LLMClient,
        cases: Dict[str, Case],
        recorder: ErrorRecorder,
    ):
        self.config = config
        self.llm = llm
        self.cases = cases
        self.recorder = recorder
        self.rng = random.Random(config.runtime.random_seed)
        self._step = 0

    # -- 中间结果编号 --------------------------------------------------------
    def _dump(self, tag: str, data) -> None:
        self._step += 1
        name = f"{self._step:03d}_{tag}.json"
        dump_intermediate(self.config.paths.intermediate_dir, name, data)

    # =======================================================================
    # 入口
    # =======================================================================
    async def build(self, seed_l1: List[Dict], resume_tree: Optional[Tree] = None,
                    resume_from_level: int = 1) -> Tree:
        if resume_tree is not None:
            tree = resume_tree
            log(f"从已有树续跑，起始层 L{resume_from_level}", stage="BUILD")
        else:
            tree = Tree.new()
            # 初始 L1 节点
            for item in seed_l1:
                tree.root.children.append(
                    Node(
                        name=str(item.get("name", "")),
                        case_trigger=str(item.get("case_trigger") or item.get("trigger", "")),
                        dialog_trigger=str(item.get("dialog_trigger", "")),
                        background=str(item.get("background", "")),
                    )
                )
            resume_from_level = 1

        # 逐层处理
        await self._process_levels(tree, resume_from_level)
        return tree

    async def _process_levels(self, tree: Tree, start_level: int) -> None:
        max_depth = self.config.build.max_depth

        # 准备每层要处理的 (parent_node, cases_subset)
        # L1 的 parent 是 Root，案例是全部案例
        # 第一层特殊：初始类别来自 seed，已放入 tree
        for level in range(start_level, max_depth + 1):
            stage_banner(f"开始构建第 L{level} 层")
            parents = self._collect_layer_parents(tree, level)
            if not parents:
                log(f"L{level} 没有可处理的父节点，停止。", stage="BUILD")
                break

            for parent, case_ids in parents:
                subset = [self.cases[c] for c in case_ids if c in self.cases]
                if level == 1:
                    # L1 初始类别已存在于 tree（来自 seed）
                    await self._optimize_layer(parent, subset, level, seeded=True)
                else:
                    if len(subset) < self.config.build.min_cases_to_split:
                        log(f"父类别 [{parent.name}] 案例数 {len(subset)} < "
                            f"{self.config.build.min_cases_to_split}，不再分裂。", stage="BUILD")
                        continue
                    # 聚类得到初始子类别
                    # 直接由模型从案例总结中归纳初始子类别
                    init_cats = await self._init_categories(parent, subset, level)
                    if not init_cats:
                        log(f"父类别 [{parent.name}] 聚类未得到初始类别，跳过。", stage="BUILD")
                        continue
                    parent.children = init_cats
                    await self._optimize_layer(parent, subset, level, seeded=True)

            # 每层结束后落盘一次（便于续跑）
            self._dump(f"L{level}_tree_after", tree.to_dict(include_debug=True))

    def _collect_layer_parents(self, tree: Tree, level: int) -> List[Tuple[Node, List[str]]]:
        """返回该层每个父节点及其应处理的案例 id 集合。

        L1: parent = Root, cases = 全部
        L>=2: parents = 上一层的所有叶子类别节点，cases = 各自 case_ids
        """
        if level == 1:
            return [(tree.root, list(self.cases.keys()))]

        result: List[Tuple[Node, List[str]]] = []

        def walk(node: Node, depth: int) -> None:
            # depth 为 node 所在层级；Root=0
            if depth == level - 1:
                # node 是 level 层的父节点
                if node is not tree.root:
                    result.append((node, list(node.case_ids)))
                return
            for child in node.children:
                walk(child, depth + 1)

        for child in tree.root.children:
            walk(child, 1)
        return result

    # =======================================================================
    # L2+ 初始类别：直接由模型从案例总结中归纳
    # =======================================================================
    async def _init_categories(
        self, parent: Node, cases: List[Case], level: int
    ) -> List[Node]:
        stage_banner(f"L{level} 父类别[{parent.name}] 归纳初始子类别")

        # 1. 对每个案例做总结（并行）
        summaries = await gather_limited(
            cases,
            lambda c: self.llm.summarize_case(c, parent),
            concurrency=self.config.llm.concurrency,
            desc=f"summarize[{parent.name}]",
            use_tqdm=self.config.runtime.use_tqdm,
            recorder=self.recorder,
            where="summarize_case",
        )
        pairs = [(c, s) for c, s in zip(cases, summaries) if s]
        self._dump(
            f"L{level}_{parent.name}_summaries",
            [{"case_id": c.case_id, "summary": s} for c, s in pairs],
        )
        if not pairs:
            return []

        # 2. 把父类别下所有案例总结一次性交给模型，让其直接归纳出初始子类别
        all_summaries = [s for _, s in pairs]
        final_cats = await self.llm.discover_categories(all_summaries, parent)
        self._dump(f"L{level}_{parent.name}_init_categories", final_cats)

        nodes = []
        for c in final_cats:
            nm = str(c.get("name", "")).strip()
            if not nm:
                continue
            nodes.append(
                Node(
                    name=nm,
                    case_trigger=str(c.get("case_trigger", "")),
                    dialog_trigger=str(c.get("dialog_trigger", "")),
                    background=str(c.get("background", "")),
                )
            )
        return nodes

    # =======================================================================
    # 单层优化：分类 -> 反思 -> 聚合 -> 复杂度 -> 覆盖率
    # =======================================================================
    async def _optimize_layer(
        self, parent: Node, cases: List[Case], level: int, seeded: bool
    ) -> None:
        scope = f"L{level}/{parent.name}"
        stage_banner(f"优化 {scope}：{len(cases)} 个案例，初始 {len(parent.children)} 个类别")

        if not parent.children:
            log(f"{scope} 无初始类别，跳过优化。", stage="BUILD")
            return

        # ---- 1. 分类 ----
        classifications = await self._classify_all(parent.children, cases, scope)
        self._dump(
            f"{_safe(scope)}_classification",
            [{"case_id": r.case_id, "category": r.category, "reason": r.reason}
             for r in classifications],
        )

        # ---- 2~6. 反思 + 聚合 + 复杂度 + 覆盖率（带重试） ----
        plan_feedback = ""
        accepted_plan: Optional[List[Operation]] = None
        for plan_attempt in range(self.config.build.max_plan_retries + 1):
            log(f"{scope} 生成 Update Plan（第 {plan_attempt+1} 次尝试）", stage="BUILD")

            # Batch Reflection -> Proposals
            proposals = await self._batch_reflection(
                parent, cases, classifications, scope, plan_feedback
            )
            self._dump(f"{_safe(scope)}_proposals_try{plan_attempt+1}", proposals)

            # Aggregation -> Update Plan，含 Complexity Check
            plan = await self._aggregate_with_complexity_check(
                parent, proposals, scope, plan_attempt
            )
            self._dump(
                f"{_safe(scope)}_update_plan_try{plan_attempt+1}",
                [op.to_dict() for op in plan],
            )

            # Coverage Validation
            ok, feedback, new_classifications = await self._coverage_validation(
                parent, cases, plan, classifications, scope
            )
            if ok:
                accepted_plan = plan
                classifications = new_classifications
                log(f"{scope} Update Plan 通过覆盖率验证，接受。", stage="BUILD")
                break
            else:
                plan_feedback = feedback
                log(f"{scope} 覆盖率验证未通过，反馈后重试。", stage="BUILD")

        if accepted_plan is None:
            log(f"{scope} 达到最大重试次数仍未通过，使用最后一版 plan 强制执行并兜底。",
                stage="WARN")
            accepted_plan = plan
            # 强制执行并对仍 UNKNOWN 的案例兜底
            self._apply_plan(parent, accepted_plan)
            classifications = await self._classify_all(parent.children, cases, scope)
            self._fallback_unknown(parent, cases, classifications)
        else:
            self._apply_plan(parent, accepted_plan)
            classifications = await self._classify_all(parent.children, cases, scope)
            self._fallback_unknown(parent, cases, classifications)

        # ---- 7. 把案例落到各子类别 ----
        self._assign_cases(parent, classifications)
        self._dump(
            f"{_safe(scope)}_final_assignment",
            {child.name: list(child.case_ids) for child in parent.children},
        )

    # -- 分类 ----------------------------------------------------------------
    async def _classify_all(
        self, categories: List[Node], cases: List[Case], scope: str
    ) -> List[ClassificationResult]:
        results = await gather_limited(
            cases,
            lambda c: self.llm.classify(c, categories),
            concurrency=self.config.llm.concurrency,
            desc=f"classify[{scope}]",
            use_tqdm=self.config.runtime.use_tqdm,
            recorder=self.recorder,
            where="classify",
        )
        out: List[ClassificationResult] = []
        for case, r in zip(cases, results):
            if r is None:
                out.append(ClassificationResult(case.case_id, ClassificationResult.UNKNOWN,
                                                "分类调用失败"))
            else:
                out.append(r)
        return out

    # -- Batch Reflection ----------------------------------------------------
    async def _batch_reflection(
        self, parent: Node, cases: List[Case],
        classifications: List[ClassificationResult], scope: str, feedback: str,
    ) -> List[Dict]:
        case_by_id = {c.case_id: c for c in cases}
        unknown_ids = [r.case_id for r in classifications if r.is_unknown()]
        known_ids = [r.case_id for r in classifications if not r.is_unknown()]
        log(f"{scope} 待反思：{len(unknown_ids)} 个 UNKNOWN，{len(known_ids)} 个已分类",
            stage="BUILD")

        if not unknown_ids:
            log(f"{scope} 无 UNKNOWN 案例，无需反思。", stage="BUILD")
            return []

        # 构建 batch：每个 batch 含 unknown_per_batch 个 unknown + 其余补已分类案例
        batches = self._make_batches(unknown_ids, known_ids)
        self._dump(f"{_safe(scope)}_batches",
                   [{"unknown": b[0], "known": b[1]} for b in batches])

        async def _reflect_one(batch: Tuple[List[str], List[str]]) -> List[Dict]:
            b_unknown, b_known = batch
            batch_cases = [case_by_id[c] for c in (b_unknown + b_known) if c in case_by_id]
            return await self.llm.reflect_batch(
                batch_cases, parent.children, b_unknown, feedback
            )

        results = await gather_limited(
            batches, _reflect_one,
            concurrency=self.config.llm.concurrency,
            desc=f"reflect[{scope}]",
            use_tqdm=self.config.runtime.use_tqdm,
            recorder=self.recorder,
            where="reflect_batch",
        )
        proposals: List[Dict] = []
        for r in results:
            if r:
                proposals.extend(r)
        return proposals

    def _make_batches(
        self, unknown_ids: List[str], known_ids: List[str]
    ) -> List[Tuple[List[str], List[str]]]:
        cfg = self.config.build
        unknown_per = max(1, cfg.unknown_per_batch)
        known_per = max(0, cfg.batch_size - unknown_per)

        unknown_chunks = chunk(unknown_ids, unknown_per)
        known_pool = list(known_ids)
        self.rng.shuffle(known_pool)

        batches: List[Tuple[List[str], List[str]]] = []
        ki = 0
        for uc in unknown_chunks:
            kk = known_pool[ki: ki + known_per]
            ki += known_per
            if ki >= len(known_pool):
                ki = 0  # 循环复用已分类案例
            batches.append((uc, kk))
        return batches

    # -- Aggregation + Complexity Check -------------------------------------
    async def _aggregate_with_complexity_check(
        self, parent: Node, proposals: List[Dict], scope: str, plan_attempt: int
    ) -> List[Operation]:
        if not proposals:
            return []

        feedback = ""
        for c_try in range(self.config.build.max_complexity_retries + 1):
            agg = await self.llm.aggregate(proposals, parent.children, feedback)
            plan = [Operation.from_dict(op) for op in agg if isinstance(op, dict)]
            plan = [op for op in plan if op.op_type in (ADD, MODIFY)]

            # Complexity Check
            current = len(parent.children)
            add_count = sum(1 for op in plan if op.op_type == ADD)
            new_count = current + add_count
            log(f"{scope} Complexity Check: 现有 {current} + 新增 {add_count} = "
                f"{new_count}（上限 {self.config.build.max_node_count}）", stage="BUILD")
            if new_count <= self.config.build.max_node_count:
                return plan

            feedback = (
                "当前修改方案新增类别过多。\n\n"
                "请重新审视新增类别，\n"
                "总结能够覆盖多个Add的更高级别抽象的Add，\n"
                "减少 Add 操作数量。"
            )
            log(f"{scope} 复杂度超限，反馈后重新生成 Update Plan（第 {c_try+1} 次）。",
                stage="WARN")

        # 仍超限：截断 add（兜底，保证不超限）
        log(f"{scope} 复杂度仍超限，截断多余的 Add 操作。", stage="WARN")
        return self._truncate_adds(parent, plan)

    def _truncate_adds(self, parent: Node, plan: List[Operation]) -> List[Operation]:
        budget = self.config.build.max_node_count - len(parent.children)
        out, added = [], 0
        for op in plan:
            if op.op_type == ADD:
                if added < budget:
                    out.append(op)
                    added += 1
            else:
                out.append(op)
        return out

    # -- Coverage Validation -------------------------------------------------
    async def _coverage_validation(
        self, parent: Node, cases: List[Case], plan: List[Operation],
        classifications: List[ClassificationResult], scope: str,
    ) -> Tuple[bool, str, List[ClassificationResult]]:
        """试执行 plan，重新分类原 UNKNOWN，验证是否全部可分类。"""
        unknown_ids = {r.case_id for r in classifications if r.is_unknown()}
        if not unknown_ids:
            return True, "", classifications

        # 在副本上试执行
        trial = self._clone_children(parent.children)
        trial_parent = Node(name=parent.name, children=trial)
        self._apply_plan(trial_parent, plan)

        unknown_cases = [c for c in cases if c.case_id in unknown_ids]
        recls = await self._classify_all(trial_parent.children, unknown_cases, scope + "/verify")

        still_unknown = [r.case_id for r in recls if r.is_unknown()]
        if not still_unknown:
            # 合并：原已分类 + 新分类的结果
            merged = {r.case_id: r for r in classifications if not r.is_unknown()}
            for r in recls:
                merged[r.case_id] = r
            full = [merged.get(c.case_id,
                    ClassificationResult(c.case_id, ClassificationResult.UNKNOWN))
                    for c in cases]
            return True, "", full

        names = [self.cases[c].case_name for c in still_unknown if c in self.cases]
        feedback = (
            f"执行该 Update Plan 后，仍有 {len(still_unknown)} 个案例无法分类：\n"
            + "\n".join(f"- {n}" for n in names[:10])
            + "\n请调整 Update Plan（新增或修改类别），使这些案例也能被覆盖。"
        )
        return False, feedback, classifications

    # -- 应用 plan -----------------------------------------------------------
    def _apply_plan(self, parent: Node, plan: List[Operation]) -> None:
        for op in plan:
            try:
                if op.op_type == ADD:
                    if parent.find_by_name(op.name):
                        continue
                    parent.children.append(
                        Node(
                            name=op.name,
                            case_trigger=op.case_trigger,
                            dialog_trigger=op.dialog_trigger,
                            background=op.background,
                        )
                    )
                elif op.op_type == MODIFY:
                    target = parent.find_by_name(op.target) or parent.find_by_name(op.name)
                    if target is None:
                        continue
                    if op.name and op.name != target.name:
                        target.name = op.name
                    if op.case_trigger:
                        target.case_trigger = op.case_trigger
                    if op.background:
                        target.background = op.background
            except Exception as exc:  # noqa: BLE001
                self.recorder.record("apply_plan", exc, context=str(op.to_dict())[:200])

    # -- 兜底：仍 UNKNOWN 的案例 ------------------------------------------
    def _fallback_unknown(
        self, parent: Node, cases: List[Case], classifications: List[ClassificationResult]
    ) -> None:
        """对仍无法分类的案例，建立/归入“其他”类别，保证全覆盖。"""
        unknown = [r for r in classifications if r.is_unknown()]
        if not unknown:
            return
        other = parent.find_by_name("其他")
        if other is None:
            other = Node(
                name="其他",
                case_trigger="无法归入其他类别的案例",
                background="兜底类别，收纳暂未细分的案例。",
            )
            parent.children.append(other)
        for r in unknown:
            r.category = "其他"
        log(f"{parent.name}: {len(unknown)} 个案例兜底归入「其他」", stage="WARN")

    # -- 分配案例到子节点 ----------------------------------------------------
    def _assign_cases(
        self, parent: Node, classifications: List[ClassificationResult]
    ) -> None:
        by_name: Dict[str, Node] = {c.name: c for c in parent.children}
        for child in parent.children:
            child.case_ids = []
        for r in classifications:
            node = by_name.get(r.category)
            if node is not None:
                node.case_ids.append(r.case_id)

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _clone_children(children: List[Node]) -> List[Node]:
        return [Node.from_dict(c.to_dict(include_debug=True)) for c in children]


def _safe(text: str) -> str:
    return text.replace("/", "_").replace(" ", "")
