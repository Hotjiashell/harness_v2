# -*- coding: utf-8 -*-
"""阶段二：基于对话数据优化节点内容。

不新增/删除节点，只优化已有节点的 dialog_trigger 或 background。

流程：
  对话导航(navigate) -> 收集背景 -> 生成 query -> 调用 retrieve 检索
    -> 失败则错误归因(哪个节点的 trigger/background 出问题) [并行]
    -> 按节点聚合错误样本成 Batch
    -> 对每个 Batch 反思，形成对节点的修改
    -> 用训练/验证对话验证召回率是否提高 -> 接受/反馈重试
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .config_types import HarnessConfig
from .llm import LLMClient
from .models import Case, Dialog, Node, Tree
from .retrieval import Retriever
from .utils import (
    ErrorRecorder,
    dump_intermediate,
    gather_limited,
    log,
    stage_banner,
)


class NavResult:
    """单条对话的导航+检索结果。"""

    def __init__(self, dialog: Dialog):
        self.dialog = dialog
        self.visited: List[Dict] = []   # 经过的节点 [{name, dialog_trigger, background, dead_end}]
        # 每个导航决策层的候选信息：
        #   [{candidates: [{name, dialog_trigger}], chosen_name: str|None}]
        # candidates 是该层「可能被选的」全部节点（即当前节点的所有子节点）。
        self.levels: List[Dict] = []
        self.query: str = ""
        self.success: bool = False
        self.attribution: Optional[Dict] = None  # {node_name, problem, reason}


class DialogOptimizer:
    def __init__(
        self,
        config: HarnessConfig,
        llm: LLMClient,
        tree: Tree,
        cases: Dict[str, Case],
        dialogs: List[Dialog],
        recorder: ErrorRecorder,
    ):
        self.config = config
        self.llm = llm
        self.tree = tree
        self.cases = cases
        self.dialogs = dialogs
        self.recorder = recorder
        self.retriever = Retriever(cases)
        self._step = 0

    def _dump(self, tag: str, data) -> None:
        self._step += 1
        name = f"opt_{self._step:03d}_{tag}.json"
        dump_intermediate(self.config.paths.intermediate_dir / "optimize", name, data)

    # =======================================================================
    # 入口
    # =======================================================================
    async def optimize(self) -> Tree:
        stage_banner("阶段二：基于对话数据优化节点内容")

        # 基线召回率（仅作整体观测，不参与接受判定）
        base_recall = await self._eval_recall(self.dialogs, "baseline")
        log(f"基线召回率：{base_recall:.3f}", stage="OPTIMIZE")
        self._dump("baseline_recall", {"recall": base_recall})

        # 导航+检索+归因
        results = await self._navigate_and_retrieve(self.dialogs, "all")
        failures = [r for r in results if r is not None and not r.success]
        log(f"失败样本：{len(failures)}/{len(self.dialogs)}", stage="OPTIMIZE")

        if not failures:
            log("无失败样本，无需优化。", stage="OPTIMIZE")
            return self.tree

        # 按 (节点, 操作类型) 分 batch：每个节点最多一个 trigger batch + 一个 background batch
        batches = self._group_batches(failures)
        self._dump(
            "error_batches",
            {f"{name}::{problem}": [s["call_sno"] for s in samples]
             for (name, problem), samples in batches.items()},
        )

        # 先处理所有 trigger batch（修好导航），再处理所有 background batch（导航到位后优化 query）
        trigger_batches = {k: v for k, v in batches.items() if k[1] == "trigger"}
        background_batches = {k: v for k, v in batches.items() if k[1] == "background"}

        for (node_name, _), samples in trigger_batches.items():
            node = self._find_node(node_name)
            if node is None:
                log(f"归因节点 [{node_name}] 不存在，跳过 trigger batch。", stage="WARN")
                continue
            await self._refine_trigger_batch(node, samples)

        for (node_name, _), samples in background_batches.items():
            node = self._find_node(node_name)
            if node is None:
                log(f"归因节点 [{node_name}] 不存在，跳过 background batch。", stage="WARN")
                continue
            await self._refine_background_batch(node, samples)

        final_recall = await self._eval_recall(self.dialogs, "final")
        log(f"优化后召回率：{base_recall:.3f} -> {final_recall:.3f}", stage="OPTIMIZE")
        self._dump("final_recall", {"baseline": base_recall, "final": final_recall})
        return self.tree

    # =======================================================================
    # 导航 + 检索 + 归因（并行）
    # =======================================================================
    async def _navigate_and_retrieve(
        self, dialogs: List[Dialog], tag: str
    ) -> List[Optional[NavResult]]:
        async def _one(d: Dialog) -> NavResult:
            res = NavResult(d)
            # 1. 导航（同时记录每层候选）
            backgrounds: List[str] = []
            node = self.tree.root
            while node.children:
                candidates = [
                    {"name": c.name, "dialog_trigger": c.dialog_trigger}
                    for c in node.children
                ]
                chosen = await self.llm.navigate(d.chat_content, node.children)
                res.levels.append({"candidates": candidates, "chosen_name": chosen or None})
                if not chosen:
                    if res.visited:
                        res.visited[-1]["dead_end"] = True
                    else:
                        res.visited.append({"name": node.name, "dialog_trigger": "",
                                            "background": node.background, "dead_end": True})
                    break
                child = node.find_by_name(chosen)
                if child is None:
                    break
                res.visited.append({
                    "name": child.name,
                    "dialog_trigger": child.dialog_trigger,
                    "background": child.background,
                    "dead_end": False,
                })
                if child.background:
                    backgrounds.append(child.background)
                node = child

            # 2. 生成 query + 检索
            res.query = await self.llm.generate_query(d.chat_content, backgrounds)
            res.success = await self.retriever.retrieve(res.query, d.case_id)

            # 3. 失败则归因
            if not res.success:
                res.attribution = await self._attribute(res)
            return res

        results = await gather_limited(
            dialogs, _one,
            concurrency=self.config.llm.concurrency,
            desc=f"navigate[{tag}]",
            use_tqdm=self.config.runtime.use_tqdm,
            recorder=self.recorder,
            where="navigate_retrieve",
        )
        self._dump(
            f"navigate_{tag}",
            [{
                "call_sno": r.dialog.call_sno,
                "case_id": r.dialog.case_id,
                "query": r.query,
                "success": r.success,
                "visited": [v["name"] for v in r.visited],
                "levels": r.levels,
                "attribution": r.attribution,
            } for r in results if r is not None],
        )
        return results

    # =======================================================================
    # 错误归因（oneshot / multistage）
    # =======================================================================
    async def _attribute(self, res: NavResult) -> Dict:
        """对一条失败样本做错误归因，返回 {node_name, problem, reason}。

        problem 取值：
          - "background"：路径上某节点的 background 不足，导致 query 不好；
          - "trigger"：某候选节点的 dialog_trigger 有问题，导致没导到正确节点。
        """
        gt = self.cases.get(res.dialog.case_id)
        gt_name = gt.case_name if gt else ""
        gt_text = gt.text if gt else ""
        # 路径上实际走过的节点（含 background），供 background 归因用
        path_nodes = [
            {"name": v["name"], "dialog_trigger": v.get("dialog_trigger", ""),
             "background": v.get("background", "")}
            for v in res.visited
        ]

        if self.config.optimize.attribution_mode == "multistage":
            return await self._attribute_multistage(
                res, path_nodes, gt_name, gt_text
            )
        return await self._attribute_oneshot(res, path_nodes, gt_name, gt_text)

    async def _attribute_oneshot(
        self, res: NavResult, path_nodes: List[Dict], gt_name: str, gt_text: str
    ) -> Dict:
        """一次性归因：把所有层的候选一并交给模型判断。"""
        return await self.llm.attribute_error_oneshot(
            chat_content=res.dialog.chat_content,
            path_nodes=path_nodes,
            levels=res.levels,
            query=res.query,
            gt_case_name=gt_name,
            gt_case_text=gt_text,
        )

    async def _attribute_multistage(
        self, res: NavResult, path_nodes: List[Dict], gt_name: str, gt_text: str
    ) -> Dict:
        """多阶段归因：从最低层开始，逐层向上判断。

        每一阶段模型可判定：
          - background：路径某节点背景不足（仅最低层允许，因为 query 由整条路径生成）；
          - trigger：本层某候选节点 trigger 有问题；
          - upper_level：本层之上的分类就错了 -> 进入上一层继续（root 层不可再上）。
        """
        levels = res.levels
        if not levels:
            return {"node_name": res.visited[-1]["name"] if res.visited else "Root",
                    "problem": "trigger", "reason": "无候选层信息"}

        for idx in range(len(levels) - 1, -1, -1):
            is_deepest = (idx == len(levels) - 1)
            can_escalate = (idx > 0)  # 还能再往上（idx==0 是 root 的下一层，不能再上）
            attr = await self.llm.attribute_error_stage(
                chat_content=res.dialog.chat_content,
                path_nodes=path_nodes,
                level=levels[idx],
                query=res.query,
                gt_case_name=gt_name,
                gt_case_text=gt_text,
                allow_background=is_deepest,
                allow_escalate=can_escalate,
                stage_depth=idx + 1,
            )
            problem = attr.get("problem", "")
            if problem == "upper_level" and can_escalate:
                continue  # 进入上一层
            return attr
        # 兜底：到 root 层仍判 upper_level，归到该层首个候选的 trigger
        first = levels[0]["candidates"][0]["name"] if levels[0]["candidates"] else "Root"
        return {"node_name": first, "problem": "trigger",
                "reason": "多阶段归因到达顶层，归因为首层候选 trigger"}

    def _group_batches(self, failures: List[NavResult]) -> Dict[Tuple[str, str], List[Dict]]:
        """按 (节点名, 操作类型) 分组。每个节点最多一个 trigger batch + 一个 background batch。"""
        batches: Dict[Tuple[str, str], List[Dict]] = {}
        for r in failures:
            attr = r.attribution or {}
            node_name = attr.get("node_name", "")
            if not node_name:
                node_name = r.visited[-1]["name"] if r.visited else "Root"
            problem = attr.get("problem", "trigger")
            if problem not in ("trigger", "background"):
                problem = "trigger"
            gt = self.cases.get(r.dialog.case_id)
            batches.setdefault((node_name, problem), []).append({
                "reason": attr.get("reason", ""),
                "chat_content": r.dialog.chat_content,
                "case_id": r.dialog.case_id,
                "query": r.query,
                # GT：该对话本应检索到的目标案例（当前 query 没检索到它）
                "gt_case_name": gt.case_name if gt else "",
                "gt_case_text": gt.text if gt else "",
                "call_sno": r.dialog.call_sno,
                # 该对话实际走过的路径节点名（用于 background 验证时重建路径背景）
                "path_names": [v["name"] for v in r.visited],
                "dialog_obj": r.dialog,
            })
        return batches

    # =======================================================================
    # trigger batch：修改节点 trigger -> 在该节点所在层重导航验证
    # =======================================================================
    async def _refine_trigger_batch(self, node: Node, samples: List[Dict]) -> None:
        scope = f"node[{node.name}].trigger"
        stage_banner(f"优化 {scope}：{len(samples)} 个失败样本")

        parent = self._find_parent(node.name)
        if parent is None or not parent.children:
            log(f"{scope} 找不到父层，跳过。", stage="WARN")
            return

        original = node.dialog_trigger
        best_trigger, best_rate = original, -1.0
        feedback = ""
        for attempt in range(self.config.optimize.max_reflection_retries + 1):
            res = await self.llm.refine_trigger(node, samples, feedback)
            new_trigger = res.get("dialog_trigger", node.dialog_trigger) or node.dialog_trigger
            node.dialog_trigger = new_trigger  # 试应用（影响同层导航）

            # 验证：batch 内每条对话在父层重导航，应选中本节点
            misroute: List[Dict] = []
            for s in samples:
                chosen = await self.llm.navigate(s["chat_content"], parent.children)
                if chosen != node.name:
                    misroute.append({"call_sno": s["call_sno"],
                                     "chat_content": s["chat_content"],
                                     "chosen": chosen or "（未选中任何节点）"})
            rate = (len(samples) - len(misroute)) / len(samples)
            self._dump(f"{_safe(node.name)}_trigger_try{attempt+1}",
                       {"dialog_trigger": new_trigger, "reason": res.get("reason", ""),
                        "rate": rate, "misroute": misroute})
            log(f"{scope} 试改后命中率：{rate:.3f}（{len(samples)-len(misroute)}/{len(samples)}）",
                stage="OPTIMIZE")

            if rate > best_rate:
                best_trigger, best_rate = new_trigger, rate
            if not misroute:
                log(f"{scope} 全部正确分类，接受。", stage="OPTIMIZE")
                return

            node.dialog_trigger = original  # 回滚后带反馈重试
            feedback = _trigger_feedback(new_trigger, misroute, node.name)
            log(f"{scope} 仍有 {len(misroute)} 条未分类正确，重试。", stage="OPTIMIZE")

        node.dialog_trigger = best_trigger
        log(f"{scope} 达到最大重试，采用历史最优版本（命中率 {best_rate:.3f}）。", stage="WARN")

    # =======================================================================
    # background batch：修改节点 background -> 重建路径背景生成 query 验证检索
    # =======================================================================
    async def _refine_background_batch(self, node: Node, samples: List[Dict]) -> None:
        scope = f"node[{node.name}].background"
        stage_banner(f"优化 {scope}：{len(samples)} 个失败样本")

        original = node.background
        best_bg, best_rate = original, -1.0
        feedback = ""
        for attempt in range(self.config.optimize.max_reflection_retries + 1):
            res = await self.llm.refine_background(node, samples, feedback)
            new_bg = res.get("background", node.background) or node.background
            node.background = new_bg  # 试应用

            # 验证：沿对话原路径重建背景（含改后的新 BG）-> 生成 query -> 检索是否命中 GT
            still_fail: List[Dict] = []
            for s in samples:
                backgrounds = self._collect_path_backgrounds(s["path_names"])
                query = await self.llm.generate_query(s["chat_content"], backgrounds)
                ok = await self.retriever.retrieve(query, s["case_id"])
                if not ok:
                    still_fail.append({"call_sno": s["call_sno"], "query": query,
                                       "gt_case_name": s["gt_case_name"],
                                       "gt_case_text": s["gt_case_text"]})
            rate = (len(samples) - len(still_fail)) / len(samples)
            self._dump(f"{_safe(node.name)}_background_try{attempt+1}",
                       {"background": new_bg, "reason": res.get("reason", ""),
                        "rate": rate, "still_fail": still_fail})
            log(f"{scope} 试改后召回率：{rate:.3f}（{len(samples)-len(still_fail)}/{len(samples)}）",
                stage="OPTIMIZE")

            if rate > best_rate:
                best_bg, best_rate = new_bg, rate
            if not still_fail:
                log(f"{scope} 全部成功检索，接受。", stage="OPTIMIZE")
                return

            node.background = original  # 回滚后带反馈重试
            feedback = _background_feedback(new_bg, still_fail)
            log(f"{scope} 仍有 {len(still_fail)} 条未检索到，重试。", stage="OPTIMIZE")

        node.background = best_bg
        log(f"{scope} 达到最大重试，采用历史最优版本（召回率 {best_rate:.3f}）。", stage="WARN")

    def _collect_path_backgrounds(self, path_names: List[str]) -> List[str]:
        """按路径节点名，从当前树取各节点 background（自动反映已修改的新 BG）。"""
        backgrounds: List[str] = []
        for name in path_names:
            n = self._find_node(name)
            if n is not None and n.background:
                backgrounds.append(n.background)
        return backgrounds

    # =======================================================================
    # 召回率评估
    # =======================================================================
    async def _eval_recall(self, dialogs: List[Dialog], tag: str) -> float:
        if not dialogs:
            return 0.0
        results = await self._navigate_and_retrieve_light(dialogs, tag)
        ok = sum(1 for r in results if r)
        return ok / len(dialogs)

    async def _navigate_and_retrieve_light(
        self, dialogs: List[Dialog], tag: str
    ) -> List[bool]:
        """只算成功与否，不做归因，用于召回率评估。"""
        async def _one(d: Dialog) -> bool:
            backgrounds: List[str] = []
            node = self.tree.root
            while node.children:
                chosen = await self.llm.navigate(d.chat_content, node.children)
                if not chosen:
                    break
                child = node.find_by_name(chosen)
                if child is None:
                    break
                if child.background:
                    backgrounds.append(child.background)
                node = child
            query = await self.llm.generate_query(d.chat_content, backgrounds)
            return await self.retriever.retrieve(query, d.case_id)

        results = await gather_limited(
            dialogs, _one,
            concurrency=self.config.llm.concurrency,
            desc=f"recall[{tag}]",
            use_tqdm=self.config.runtime.use_tqdm,
            recorder=self.recorder,
            where="eval_recall",
        )
        return [bool(r) for r in results]

    # -- helpers -------------------------------------------------------------
    def _find_node(self, name: str) -> Optional[Node]:
        found: List[Node] = []

        def walk(n: Node) -> None:
            if n.name == name:
                found.append(n)
            for c in n.children:
                walk(c)

        walk(self.tree.root)
        return found[0] if found else None

    def _find_parent(self, name: str) -> Optional[Node]:
        """返回 name 节点的父节点（root 的子节点的父节点即 root）。"""
        found: List[Node] = []

        def walk(n: Node) -> None:
            for c in n.children:
                if c.name == name:
                    found.append(n)
                walk(c)

        walk(self.tree.root)
        return found[0] if found else None


def _trigger_feedback(prev_trigger: str, misroute: List[Dict], node_name: str) -> str:
    lines = [
        f"上一次你把 dialog_trigger 改成了：{prev_trigger}",
        f"但以下对话仍未被正确分类到节点「{node_name}」（导航选到了别处或未选中）：",
    ]
    for m in misroute:
        lines.append(f"  - 实际选到「{m['chosen']}」｜对话：{m['chat_content']}")
    lines.append("请进一步调整 dialog_trigger，使这些对话能正确进入本节点，"
                 "同时不要把明显不相关的对话也包含进来。")
    return "\n".join(lines)


def _background_feedback(prev_bg: str, still_fail: List[Dict]) -> str:
    lines = [
        f"上一次你把 background 改成了：{prev_bg}",
        "但以下对话据此生成的 query 仍未能检索到目标案例：",
    ]
    for f in still_fail:
        lines.append(
            f"  - 生成的 query：{f['query']}｜目标案例：{f['gt_case_name']}：{f['gt_case_text']}"
        )
    lines.append("请进一步补充/调整 background，使其能指导生成更贴近目标案例、"
                 "更容易检索到目标案例的 query。")
    return "\n".join(lines)


def _safe(text: str) -> str:
    return text.replace("/", "_").replace(" ", "")
