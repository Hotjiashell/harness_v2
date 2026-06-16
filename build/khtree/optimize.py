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
        train: List[Dialog],
        val: List[Dialog],
        recorder: ErrorRecorder,
    ):
        self.config = config
        self.llm = llm
        self.tree = tree
        self.cases = cases
        self.train = train
        self.val = val
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

        # 基线召回率
        base_train = await self._eval_recall(self.train, "baseline-train")
        base_val = await self._eval_recall(self.val, "baseline-val")
        log(f"基线召回率：train={base_train:.3f}, val={base_val:.3f}", stage="OPTIMIZE")
        self._dump("baseline_recall", {"train": base_train, "val": base_val})

        # 用训练集做错误归因
        results = await self._navigate_and_retrieve(self.train, "train")
        failures = [r for r in results if r is not None and not r.success]
        log(f"训练集失败样本：{len(failures)}/{len(self.train)}", stage="OPTIMIZE")

        if not failures:
            log("无失败样本，无需优化。", stage="OPTIMIZE")
            return self.tree

        # 错误归因（并行已在 navigate 阶段完成），按节点聚合成 batch
        node_batches = self._group_by_node(failures)
        self._dump(
            "error_batches",
            {name: [s["dialog"]["call_sno"] for s in samples]
             for name, samples in node_batches.items()},
        )

        # 逐节点反思 + 验证
        cur_train, cur_val = base_train, base_val
        for node_name, samples in node_batches.items():
            node = self._find_node(node_name)
            if node is None:
                log(f"归因节点 [{node_name}] 不存在，跳过。", stage="WARN")
                continue
            cur_train, cur_val = await self._reflect_and_validate(
                node, samples, cur_train, cur_val
            )

        log(f"优化后召回率：train={cur_train:.3f}, val={cur_val:.3f}", stage="OPTIMIZE")
        self._dump("final_recall", {"train": cur_train, "val": cur_val})
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

    def _group_by_node(self, failures: List[NavResult]) -> Dict[str, List[Dict]]:
        batches: Dict[str, List[Dict]] = {}
        for r in failures:
            attr = r.attribution or {}
            node_name = attr.get("node_name", "")
            if not node_name:
                # 没有明确归因则归到最后访问节点
                node_name = r.visited[-1]["name"] if r.visited else "Root"
            gt = self.cases.get(r.dialog.case_id)
            batches.setdefault(node_name, []).append({
                "problem": attr.get("problem", "trigger"),
                "reason": attr.get("reason", ""),
                "chat_content": r.dialog.chat_content,
                "case_id": r.dialog.case_id,
                "query": r.query,
                # GT：该对话本应检索到的目标案例（当前 query 没检索到它）
                "gt_case_name": gt.case_name if gt else "",
                "gt_case_text": gt.text if gt else "",
                "dialog": {"call_sno": r.dialog.call_sno},
            })
        return batches

    # =======================================================================
    # 反思 + 验证
    # =======================================================================
    async def _reflect_and_validate(
        self, node: Node, samples: List[Dict], cur_train: float, cur_val: float
    ) -> Tuple[float, float]:
        scope = f"node[{node.name}]"
        stage_banner(f"优化 {scope}：{len(samples)} 个失败样本")

        original = (node.dialog_trigger, node.background)
        feedback = ""
        for attempt in range(self.config.optimize.max_reflection_retries + 1):
            reflection = await self.llm.reflect_errors(node, samples, feedback)
            new_trigger = reflection.get("dialog_trigger", node.dialog_trigger) or node.dialog_trigger
            new_bg = reflection.get("background", node.background) or node.background
            self._dump(
                f"{_safe(node.name)}_reflection_try{attempt+1}",
                {"dialog_trigger": new_trigger, "background": new_bg,
                 "reason": reflection.get("reason", "")},
            )

            # 试应用
            node.dialog_trigger, node.background = new_trigger, new_bg

            new_train = await self._eval_recall(self.train, f"{node.name}-train-try{attempt+1}")
            new_val = await self._eval_recall(self.val, f"{node.name}-val-try{attempt+1}")
            log(f"{scope} 试改后：train {cur_train:.3f}->{new_train:.3f}, "
                f"val {cur_val:.3f}->{new_val:.3f}", stage="OPTIMIZE")

            # 接受条件：训练集与验证集召回率均未下降，且至少一个提高
            improved = (new_train + new_val) > (cur_train + cur_val)
            not_worse = new_train >= cur_train and new_val >= cur_val
            if improved and not_worse:
                log(f"{scope} 修改有效，接受。", stage="OPTIMIZE")
                return new_train, new_val

            # 回滚，生成反馈重试
            node.dialog_trigger, node.background = original
            feedback = (
                f"上一次修改未提升召回率（train {cur_train:.3f}->{new_train:.3f}, "
                f"val {cur_val:.3f}->{new_val:.3f}）。请换一种思路调整 "
                f"dialog_trigger 或 background。"
            )
            log(f"{scope} 修改无效，回滚并重试。", stage="OPTIMIZE")

        log(f"{scope} 达到最大重试，保留原内容。", stage="WARN")
        return cur_train, cur_val

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


def _safe(text: str) -> str:
    return text.replace("/", "_").replace(" ", "")
