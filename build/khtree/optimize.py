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

import asyncio
import re
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
    read_json,
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
        val: Optional[List[Dialog]] = None,
    ):
        self.config = config
        self.llm = llm
        self.tree = tree
        self.cases = cases
        self.dialogs = dialogs
        self.val = val or []
        self.recorder = recorder
        self.retriever = Retriever(cases)
        self._step = 0
        # 优化前快照：所有 batch 的验证都基于这份"未改动"的树，使各 batch 互相解耦、可并行。
        #   _orig_trigger / _orig_background：各节点原始 dialog_trigger / background
        #   _layer_siblings：各节点所在层（同父）的全部兄弟节点名，用于 trigger 验证重建候选
        self._orig_trigger: Dict[str, str] = {}
        self._orig_background: Dict[str, str] = {}
        self._layer_siblings: Dict[str, List[str]] = {}
        # 入口兜底：dialog_trigger 为空的节点用 case_trigger 初始化（载入的树可能由旧版本
        # 构建、或未经初始化）。建树末尾已做一次，这里对加载进来的树再保证一遍。
        self._init_dialog_trigger(self.tree.root)
        self._build_snapshot()
        # 全局共享并发闸：无论多少 batch 并行，真正在途的 LLM/检索调用数 ≤ concurrency。
        # 延迟到 optimize() 内（事件循环已就绪）创建，避免 Semaphore 绑定到错误的 loop。
        self._sem: Optional[asyncio.Semaphore] = None
        # 断点续跑：检测到 optimize 中间产物时置 True，启用 navigate_all 复用 + 逐 batch 续跑。
        self._resume = False

    def _build_snapshot(self) -> None:
        def walk(n: Node) -> None:
            self._orig_trigger[n.name] = n.dialog_trigger
            self._orig_background[n.name] = n.background
            sibling_names = [c.name for c in n.children]
            for c in n.children:
                self._layer_siblings[c.name] = sibling_names
                walk(c)

        walk(self.tree.root)

    def _init_dialog_trigger(self, node: Node) -> None:
        """dialog_trigger 为空的节点用 case_trigger 兜底初始化（不覆盖已有值）。"""
        for c in node.children:
            if not (c.dialog_trigger or "").strip() and (c.case_trigger or "").strip():
                c.dialog_trigger = c.case_trigger
            self._init_dialog_trigger(c)

    def _dump(self, tag: str, data) -> None:
        self._step += 1
        name = f"opt_{self._step:03d}_{tag}.json"
        dump_intermediate(self.config.paths.intermediate_dir / "optimize", name, data)

    # =======================================================================
    # 断点续跑：复用 optimize 中间产物（不改既有文件格式）
    # =======================================================================
    def _optimize_dir(self):
        return self.config.paths.intermediate_dir / "optimize"

    def _load_optimize_resume(self) -> Optional[List[Dict]]:
        """若存在 navigate_all 产物则返回其记录（启用续跑），否则 None（全新跑）。

        同时把 _step 推进到已有文件的最大序号，使后续 _dump 接着编号、不覆盖旧文件。
        """
        d = self._optimize_dir()
        if not d.exists():
            return None

        def prefix_of(p) -> int:
            m = re.match(r"^opt_(\d+)_", p.name)
            return int(m.group(1)) if m else 0

        nav_files = sorted(d.glob("opt_*_navigate_all.json"), key=prefix_of)
        if not nav_files:
            return None
        # 续跑时接着已有最大序号编号
        self._step = max((prefix_of(p) for p in d.glob("opt_*.json")), default=0)
        try:
            records = read_json(nav_files[-1])  # 取序号最大（最新）的一份
        except Exception as exc:  # noqa: BLE001
            log(f"navigate_all 产物损坏，放弃续跑改为全新跑：{exc}", stage="WARN")
            return None
        return records if isinstance(records, list) else None

    def _load_baseline_recall(self) -> Optional[Dict]:
        """续跑时读取已落盘的 baseline_recall 产物（取序号最大的一份），无则 None。"""
        d = self._optimize_dir()
        if not d.exists():
            return None

        def prefix_of(p) -> int:
            m = re.match(r"^opt_(\d+)_", p.name)
            return int(m.group(1)) if m else 0

        files = sorted(d.glob("opt_*_baseline_recall.json"), key=prefix_of)
        if not files:
            return None
        try:
            data = read_json(files[-1])
        except Exception:  # noqa: BLE001
            return None
        return data if isinstance(data, dict) else None

    def _reconstruct_results(self, records: List[Dict]) -> List["NavResult"]:
        """用 navigate_all 记录 + 当前对话集重建 NavResult（按 call_sno 关联取回 chat_content）。"""
        by_sno = {d.call_sno: d for d in self.dialogs}
        out: List[NavResult] = []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            d = by_sno.get(str(rec.get("call_sno", "")))
            if d is None:
                continue
            res = NavResult(d)
            res.query = rec.get("query", "")
            res.success = bool(rec.get("success"))
            res.attribution = rec.get("attribution")
            res.levels = rec.get("levels", []) or []
            res.visited = [{"name": n} for n in (rec.get("visited", []) or [])]
            out.append(res)
        return out

    def _scan_tries(self, node_name: str, problem: str,
                    value_key: str, fail_key: str) -> Dict[int, Dict]:
        """扫描某 batch 的所有 try 文件，返回 {try序号: {value, rate, fails}}。

        健壮性：损坏/缺字段的文件按"不存在"处理；同一 try 序号有多份时取序号最大的有效版本。
        """
        d = self._optimize_dir()
        if not d.exists():
            return {}
        pat = re.compile(rf"^opt_(\d+)_{re.escape(_safe(node_name))}_{problem}_try(\d+)$")
        best_per_k: Dict[int, Tuple[int, Dict]] = {}
        for p in d.glob("opt_*_try*.json"):
            m = pat.match(p.stem)
            if not m:
                continue
            prefix, k = int(m.group(1)), int(m.group(2))
            try:
                data = read_json(p)
            except Exception:  # noqa: BLE001
                continue  # 损坏 -> 当作不存在
            if not isinstance(data, dict) or "rate" not in data:
                continue
            if k not in best_per_k or prefix > best_per_k[k][0]:
                best_per_k[k] = (prefix, data)
        out: Dict[int, Dict] = {}
        for k, (_prefix, data) in best_per_k.items():
            out[k] = {"value": data.get(value_key, ""),
                      "rate": float(data.get("rate", -1.0)),
                      "fails": data.get(fail_key, []) or []}
        return out

    def _resume_batch_state(self, node_name: str, problem: str,
                            value_key: str, fail_key: str) -> Optional[Dict]:
        """根据已有 try 文件判定 batch 续跑状态。非续跑模式或无 try 文件返回 None。"""
        if not self._resume:
            return None
        tries = self._scan_tries(node_name, problem, value_key, fail_key)
        if not tries:
            return None
        max_attempts = self.config.optimize.max_reflection_retries + 1
        # 最优版本：rate 最高，并列取 try 序号最小（与在线逻辑的 strict > 一致）
        best_k = max(tries, key=lambda k: (tries[k]["rate"], -k))
        accepted = any(t["rate"] >= 1.0 for t in tries.values())
        exhausted = max_attempts in tries
        last_k = max(tries)
        return {
            "completed": accepted or exhausted,
            "best_value": tries[best_k]["value"],
            "best_rate": tries[best_k]["rate"],
            "start_attempt": last_k,           # 已完成 last_k 轮，从 try{last_k+1} 续
            "last_value": tries[last_k]["value"],
            "last_fails": tries[last_k]["fails"],
        }

    # =======================================================================
    # 入口
    # =======================================================================
    async def optimize(self) -> Tree:
        stage_banner("阶段二：基于对话数据优化节点内容")
        self._sem = asyncio.Semaphore(max(1, self.config.llm.concurrency))

        # 断点续跑检测：若 optimize 中间产物里已有 navigate_all，则复用它跳过导航+归因
        cached_base = None
        records = self._load_optimize_resume()
        if records is not None:
            self._resume = True
            log(f"检测到 optimize 中间产物，启用断点续跑（复用导航结果 {len(records)} 条）",
                stage="OPTIMIZE")
            results = self._reconstruct_results(records)
            cached_base = self._load_baseline_recall()
        else:
            # 导航+检索+归因：优化集只跑这一遍，基线召回率直接从结果统计（避免重复导航）
            results = await self._navigate_and_retrieve(self.dialogs, "all")
        valid = [r for r in results if r is not None]
        failures = [r for r in valid if not r.success]
        base_recall = (len(valid) - len(failures)) / len(valid) if valid else 0.0
        log(f"基线召回率（优化集）：{base_recall:.3f}", stage="OPTIMIZE")
        log(f"失败样本：{len(failures)}/{len(self.dialogs)}", stage="OPTIMIZE")

        # 验证集：独立对话集，仅在优化前后各跑一次召回率供人工观测，不参与优化/反馈。
        # 续跑且已有 baseline_recall 产物时直接复用，验证集基线不再重跑。
        if cached_base is not None:
            base_val = cached_base.get("val")
            log(f"基线召回率（验证集，复用产物）：{base_val if base_val is not None else '—'}",
                stage="OPTIMIZE")
        else:
            base_val = None
            if self.val:
                base_val = await self._eval_recall(self.val, "baseline-val")
                log(f"基线召回率（验证集，仅观测）：{base_val:.3f}", stage="OPTIMIZE")
            self._dump("baseline_recall", {"train": base_recall, "val": base_val})

        if not failures:
            log("无失败样本，无需优化。", stage="OPTIMIZE")
            return self.tree

        # 按 (节点, 操作类型) 分 batch：每个节点最多一个 trigger batch + 一个 background batch
        batches = self._group_batches(failures)
        self._dump(
            "error_batches",
            {f"{name}::{problem}": [
                {"call_sno": s["call_sno"], "reason": s.get("reason", "")}
                for s in samples
            ]
             for (name, problem), samples in batches.items()},
        )

        # 所有 batch（trigger / background）完全并行：每个 batch 的验证都基于优化前快照，
        # 互不依赖；总并发由共享 semaphore 统一限制在 config.llm.concurrency 内。
        async def _run_batch(key: Tuple[str, str], samples: List[Dict]) -> None:
            node_name, problem = key
            try:
                node = self._find_node(node_name)
                if node is None:
                    log(f"归因节点 [{node_name}] 不存在，跳过 {problem} batch。", stage="WARN")
                    return
                if problem == "trigger":
                    await self._refine_trigger_batch(node, samples)
                else:
                    await self._refine_background_batch(node, samples)
            except Exception as exc:  # noqa: BLE001
                # 单个 batch 失败不影响其他 batch：记录并跳过（该节点保持原内容）
                self.recorder.record(f"refine_batch[{node_name}::{problem}]", exc,
                                     context=f"{len(samples)} samples")

        await asyncio.gather(*[_run_batch(k, v) for k, v in batches.items()])

        final_recall = await self._eval_recall(self.dialogs, "final")
        log(f"优化后召回率（优化集）：{base_recall:.3f} -> {final_recall:.3f}", stage="OPTIMIZE")
        final_val = None
        if self.val:
            final_val = await self._eval_recall(self.val, "final-val")
            log(f"优化后召回率（验证集，仅观测）：{base_val:.3f} -> {final_val:.3f}", stage="OPTIMIZE")
        self._dump("final_recall", {"train": {"baseline": base_recall, "final": final_recall},
                                    "val": {"baseline": base_val, "final": final_val}})
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
    # trigger batch：改节点 trigger -> 在该节点所在层（快照兄弟）重导航验证
    # =======================================================================
    async def _refine_trigger_batch(self, node: Node, samples: List[Dict]) -> None:
        scope = f"node[{node.name}].trigger"
        log(f"优化 {scope}：{len(samples)} 个失败样本", stage="OPTIMIZE")

        sibling_names = self._layer_siblings.get(node.name)
        if not sibling_names:
            log(f"{scope} 找不到所在层，跳过。", stage="WARN")
            return

        original = self._orig_trigger.get(node.name, node.dialog_trigger)
        best_trigger, best_rate = original, -1.0
        feedback = ""
        start_attempt = 0
        # 断点续跑：扫描已有 try 文件，已完成则直接采用最优值，未完成则从断点续跑
        state = self._resume_batch_state(node.name, "trigger", "dialog_trigger", "misroute")
        if state is not None:
            if state["completed"]:
                node.dialog_trigger = state["best_value"]
                log(f"{scope} 续跑：已完成（命中率 {state['best_rate']:.3f}），跳过。", stage="OPTIMIZE")
                return
            best_trigger, best_rate = state["best_value"], state["best_rate"]
            start_attempt = state["start_attempt"]
            feedback = _trigger_feedback(state["last_value"], state["last_fails"], node.name)
            log(f"{scope} 续跑：已有 {start_attempt} 轮，从 try{start_attempt+1} 继续。", stage="OPTIMIZE")
        for attempt in range(start_attempt, self.config.optimize.max_reflection_retries + 1):
            try:
                async with self._sem:
                    res = await self.llm.refine_trigger(node, samples, feedback)
            except Exception as exc:  # noqa: BLE001
                # 本轮无法产出新 trigger：记录并停止重试，保留历史最优
                self.recorder.record(f"refine_trigger[{node.name}]", exc, context=feedback[:200])
                break
            new_trigger = res.get("dialog_trigger") or original

            # 验证：用快照同层兄弟构造候选（本节点用 new_trigger，其余用原始 trigger），
            # batch 内每条对话重导航应选中本节点。并行，受共享 semaphore 限流。
            candidates = self._snapshot_siblings(sibling_names, node.name, new_trigger)

            async def _check(s: Dict) -> Optional[Dict]:
                try:
                    async with self._sem:
                        chosen = await self.llm.navigate(s["chat_content"], candidates)
                except Exception as exc:  # noqa: BLE001
                    # 单条验证失败：记录并保守计为"未选中本节点"
                    self.recorder.record(f"trigger_check[{node.name}]", exc,
                                         context=s.get("call_sno", ""))
                    chosen = ""
                if chosen == node.name:
                    return None
                return {"call_sno": s["call_sno"], "chat_content": s["chat_content"],
                        "chosen": chosen or "（未选中任何节点）"}

            checked = await asyncio.gather(*[_check(s) for s in samples])
            misroute = [m for m in checked if m is not None]
            rate = (len(samples) - len(misroute)) / len(samples)
            self._dump(f"{_safe(node.name)}_trigger_try{attempt+1}",
                       {"dialog_trigger": new_trigger, "reason": res.get("reason", ""),
                        "rate": rate, "misroute": misroute})
            log(f"{scope} 试改后命中率：{rate:.3f}（{len(samples)-len(misroute)}/{len(samples)}）",
                stage="OPTIMIZE")

            if rate > best_rate:
                best_trigger, best_rate = new_trigger, rate
            if not misroute:
                node.dialog_trigger = new_trigger
                log(f"{scope} 全部正确分类，接受。", stage="OPTIMIZE")
                return
            feedback = _trigger_feedback(new_trigger, misroute, node.name)
            log(f"{scope} 仍有 {len(misroute)} 条未分类正确，重试。", stage="OPTIMIZE")

        node.dialog_trigger = best_trigger
        log(f"{scope} 达到最大重试，采用历史最优版本（命中率 {best_rate:.3f}）。", stage="WARN")

    # =======================================================================
    # background batch：改节点 background -> 用快照重建路径背景生成 query 验证检索
    # =======================================================================
    async def _refine_background_batch(self, node: Node, samples: List[Dict]) -> None:
        scope = f"node[{node.name}].background"
        log(f"优化 {scope}：{len(samples)} 个失败样本", stage="OPTIMIZE")

        original = self._orig_background.get(node.name, node.background)
        best_bg, best_rate = original, -1.0
        feedback = ""
        start_attempt = 0
        # 断点续跑：扫描已有 try 文件，已完成则直接采用最优值，未完成则从断点续跑
        state = self._resume_batch_state(node.name, "background", "background", "still_fail")
        if state is not None:
            if state["completed"]:
                node.background = state["best_value"]
                log(f"{scope} 续跑：已完成（召回率 {state['best_rate']:.3f}），跳过。", stage="OPTIMIZE")
                return
            best_bg, best_rate = state["best_value"], state["best_rate"]
            start_attempt = state["start_attempt"]
            feedback = _background_feedback(state["last_value"], state["last_fails"])
            log(f"{scope} 续跑：已有 {start_attempt} 轮，从 try{start_attempt+1} 继续。", stage="OPTIMIZE")
        for attempt in range(start_attempt, self.config.optimize.max_reflection_retries + 1):
            try:
                async with self._sem:
                    res = await self.llm.refine_background(node, samples, feedback)
            except Exception as exc:  # noqa: BLE001
                self.recorder.record(f"refine_background[{node.name}]", exc, context=feedback[:200])
                break
            new_bg = res.get("background") or original

            # 验证：沿对话原路径重建背景（本节点用 new_bg，其余用原始 BG）-> 生成 query -> 检索。
            # 并行，受共享 semaphore 限流。
            async def _check(s: Dict) -> Optional[Dict]:
                try:
                    backgrounds = self._snapshot_path_backgrounds(s["path_names"], node.name, new_bg)
                    async with self._sem:
                        query = await self.llm.generate_query(s["chat_content"], backgrounds)
                        ok = await self.retriever.retrieve(query, s["case_id"])
                except Exception as exc:  # noqa: BLE001
                    # 单条验证失败：记录并保守计为"未检索到"
                    self.recorder.record(f"background_check[{node.name}]", exc,
                                         context=s.get("call_sno", ""))
                    query, ok = "", False
                if ok:
                    return None
                return {"call_sno": s["call_sno"], "query": query,
                        "gt_case_name": s["gt_case_name"], "gt_case_text": s["gt_case_text"]}

            checked = await asyncio.gather(*[_check(s) for s in samples])
            still_fail = [m for m in checked if m is not None]
            rate = (len(samples) - len(still_fail)) / len(samples)
            self._dump(f"{_safe(node.name)}_background_try{attempt+1}",
                       {"background": new_bg, "reason": res.get("reason", ""),
                        "rate": rate, "still_fail": still_fail})
            log(f"{scope} 试改后召回率：{rate:.3f}（{len(samples)-len(still_fail)}/{len(samples)}）",
                stage="OPTIMIZE")

            if rate > best_rate:
                best_bg, best_rate = new_bg, rate
            if not still_fail:
                node.background = new_bg
                log(f"{scope} 全部成功检索，接受。", stage="OPTIMIZE")
                return
            feedback = _background_feedback(new_bg, still_fail)
            log(f"{scope} 仍有 {len(still_fail)} 条未检索到，重试。", stage="OPTIMIZE")

        node.background = best_bg
        log(f"{scope} 达到最大重试，采用历史最优版本（召回率 {best_rate:.3f}）。", stage="WARN")

    def _snapshot_siblings(
        self, sibling_names: List[str], target: str, target_trigger: str
    ) -> List[Node]:
        """用优化前快照构造同层候选节点：目标节点用 target_trigger，其余用原始 trigger。"""
        out: List[Node] = []
        for name in sibling_names:
            trig = target_trigger if name == target else self._orig_trigger.get(name, "")
            out.append(Node(name=name, dialog_trigger=trig,
                            background=self._orig_background.get(name, "")))
        return out

    def _snapshot_path_backgrounds(
        self, path_names: List[str], target: str, target_bg: str
    ) -> List[str]:
        """用优化前快照重建路径背景：目标节点用 target_bg，其余用原始 background。"""
        backgrounds: List[str] = []
        for name in path_names:
            bg = target_bg if name == target else self._orig_background.get(name, "")
            if bg:
                backgrounds.append(bg)
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
