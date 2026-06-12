# -*- coding: utf-8 -*-
"""异步 LLM 客户端。

提供两个 provider：
  - openai: 调用 openai 兼容 /chat/completions 接口；
  - mock:   纯启发式，不联网，用于在无模型环境下跑通并调试整个流程。

对外暴露的是“高层语义方法”（classify / reflect_batch / ...），
build_tree.py 与 optimize.py 只依赖这些方法，与底层 provider 解耦。
"""
from __future__ import annotations

import asyncio
import re
from typing import Dict, List, Optional

from . import prompts
from .config_types import LLMSettings
from .models import Case, ClassificationResult, Node
from .utils import extract_json, log


# ---------------------------------------------------------------------------
# 领域关键词（mock provider 使用）
# ---------------------------------------------------------------------------
# 每个软件/主题实体 -> 触发关键词
_ENTITY_KEYWORDS: Dict[str, List[str]] = {
    "WeCon": ["wecon", "聊天记录", "历史消息", "im", "单聊", "群聊", "推送"],
    "MeetWe": ["meetwe", "会议", "摄像头", "黑屏", "麦克风", "屏幕共享", "入会", "rtc"],
    "Outlook": ["outlook", "邮件", "插件", "日历", "发件箱", "凭据", "exchange"],
    "Git": ["git", "推送代码", "merge", "histories", "publickey", "仓库", "ssh"],
    "OA": ["oa", "报表", "excel", "导出"],
    "网盘": ["网盘", "同步冲突", "冲突文件"],
    "VPN": ["vpn", "证书过期", "证书", "远程连接"],
}


def _detect_entity(text: str) -> Optional[str]:
    low = text.lower()
    best, best_hits = None, 0
    for entity, kws in _ENTITY_KEYWORDS.items():
        hits = sum(1 for kw in kws if kw.lower() in low)
        if hits > best_hits:
            best, best_hits = entity, hits
    return best if best_hits > 0 else None


def _entity_background(entity: str) -> str:
    return f"{entity} 相关问题的领域背景与常见处理经验。"


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------
class LLMClient:
    def __init__(self, settings: LLMSettings):
        self.settings = settings
        self.provider = settings.provider
        self._client = None
        if self.provider == "openai":
            self._init_openai()

    def _init_openai(self) -> None:
        try:
            from openai import AsyncOpenAI
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("使用 openai provider 需要安装 openai 包") from exc
        self._client = AsyncOpenAI(
            base_url=self.settings.base_url,
            api_key=self.settings.api_key,
            timeout=self.settings.timeout_seconds,
        )

    # -- 底层 chat（带重试） ------------------------------------------------
    async def _chat(self, messages: List[Dict[str, str]]) -> str:
        assert self._client is not None
        last_exc: Optional[BaseException] = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                resp = await self._client.chat.completions.create(
                    model=self.settings.model,
                    messages=messages,
                    temperature=self.settings.temperature,
                    max_tokens=self.settings.max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                wait = min(2 ** attempt, 8)
                log(f"chat 调用失败(第{attempt+1}次)，{wait}s 后重试: {exc}", stage="LLM")
                await asyncio.sleep(wait)
        raise RuntimeError(f"chat 调用在重试后仍失败: {last_exc}")

    # =======================================================================
    # 阶段一：基于案例
    # =======================================================================
    async def classify(self, case: Case, categories: List[Node]) -> ClassificationResult:
        if self.provider == "mock":
            return self._mock_classify(case, categories)
        msgs = prompts.classify_messages(case, categories)
        data = extract_json(await self._chat(msgs))
        cat = str(data.get("category", "")).strip()
        names = {c.name for c in categories}
        if cat not in names:
            cat = ClassificationResult.UNKNOWN
        return ClassificationResult(case.case_id, cat or ClassificationResult.UNKNOWN,
                                    str(data.get("reason", "")))

    def _mock_classify(self, case: Case, categories: List[Node]) -> ClassificationResult:
        text = case.to_text()
        entity = _detect_entity(text)
        # 先按 category name / case_trigger 关键词匹配
        for c in categories:
            hay = (c.name + " " + c.case_trigger).lower()
            if entity and entity.lower() in hay:
                return ClassificationResult(case.case_id, c.name, f"命中实体 {entity}")
            # 名称直接出现在案例里
            if c.name and c.name.lower() in text.lower():
                return ClassificationResult(case.case_id, c.name, f"名称匹配 {c.name}")
        # 通过实体关键词命中类别背景
        if entity:
            for c in categories:
                if entity.lower() in (c.background or "").lower():
                    return ClassificationResult(case.case_id, c.name, f"背景命中实体 {entity}")
        return ClassificationResult(case.case_id, ClassificationResult.UNKNOWN, "无法归入现有类别")

    async def reflect_batch(
        self, batch_cases: List[Case], categories: List[Node],
        unknown_ids: List[str], feedback: str = "",
    ) -> List[Dict]:
        if self.provider == "mock":
            return self._mock_reflect_batch(batch_cases, categories, unknown_ids)
        msgs = prompts.reflect_batch_messages(batch_cases, categories, unknown_ids, feedback)
        data = extract_json(await self._chat(msgs))
        return data if isinstance(data, list) else []

    def _mock_reflect_batch(
        self, batch_cases: List[Case], categories: List[Node], unknown_ids: List[str]
    ) -> List[Dict]:
        unknown_set = set(unknown_ids)
        existing = {c.name.lower() for c in categories}
        proposals: List[Dict] = []
        seen_entities = set()
        for c in batch_cases:
            if c.case_id not in unknown_set:
                continue
            entity = _detect_entity(c.to_text())
            if not entity:
                continue
            if entity.lower() in existing or entity in seen_entities:
                continue
            seen_entities.add(entity)
            proposals.append({
                "op_type": "add",
                "name": entity,
                "case_trigger": f"当用户问题涉及 {entity} 时考虑该类别",
                "background": _entity_background(entity),
                "reason": f"案例 {c.case_id} 无法归入现有类别，需要新增 {entity}",
            })
        return proposals

    async def aggregate(
        self, proposals: List[Dict], categories: List[Node], feedback: str = ""
    ) -> List[Dict]:
        if self.provider == "mock":
            return self._mock_aggregate(proposals, feedback)
        msgs = prompts.aggregate_messages(proposals, categories, feedback)
        data = extract_json(await self._chat(msgs))
        return data if isinstance(data, list) else []

    def _mock_aggregate(self, proposals: List[Dict], feedback: str = "") -> List[Dict]:
        # 按 (op_type, name/target) 去重合并
        merged: Dict[str, Dict] = {}
        for p in proposals:
            op = str(p.get("op_type", "")).lower()
            key = op + "::" + str(p.get("name") or p.get("target") or "")
            if key not in merged:
                merged[key] = dict(p)
        result = list(merged.values())
        # 若有反馈要求减少 add，则按 name 前缀做一次粗合并（mock 简单处理：保留全部）
        return result

    # -- L2+ 聚类初始化 -----------------------------------------------------
    async def summarize_case(self, case: Case, parent: Node) -> str:
        if self.provider == "mock":
            entity = _detect_entity(case.to_text()) or parent.name
            return f"{entity}：{case.case_name}"
        msgs = prompts.summarize_case_messages(case, parent)
        data = extract_json(await self._chat(msgs))
        return str(data.get("summary", case.case_name))

    async def label_cluster(self, summaries: List[str], parent: Node) -> List[Dict]:
        if self.provider == "mock":
            return self._mock_label_cluster(summaries, parent)
        msgs = prompts.label_cluster_messages(summaries, parent)
        data = extract_json(await self._chat(msgs))
        return data if isinstance(data, list) else []

    def _mock_label_cluster(self, summaries: List[str], parent: Node) -> List[Dict]:
        # 以簇内出现最多的关键词作为子类别名
        from collections import Counter
        counter: Counter = Counter()
        for s in summaries:
            for token in re.split(r"[：:，,。\s]+", s):
                token = token.strip()
                if len(token) >= 2:
                    counter[token] += 1
        if not counter:
            name = f"{parent.name}-子类"
        else:
            name = counter.most_common(1)[0][0]
        return [{
            "name": f"{parent.name}-{name}",
            "case_trigger": f"涉及 {name} 的 {parent.name} 子问题",
            "background": f"{parent.name} 下关于 {name} 的处理经验",
        }]

    async def synthesize_categories(
        self, candidate_categories: List[Dict], parent: Node
    ) -> List[Dict]:
        if self.provider == "mock":
            # 去重（按 name）
            seen, out = set(), []
            for c in candidate_categories:
                nm = str(c.get("name", "")).strip()
                if nm and nm not in seen:
                    seen.add(nm)
                    out.append(c)
            return out
        msgs = prompts.synthesize_categories_messages(candidate_categories, parent)
        data = extract_json(await self._chat(msgs))
        return data if isinstance(data, list) else []

    # =======================================================================
    # 阶段二：基于对话
    # =======================================================================
    async def navigate(self, chat_content: str, children: List[Node]) -> str:
        """返回选中的子节点 name（空串表示无匹配）。"""
        if self.provider == "mock":
            return self._mock_navigate(chat_content, children)
        msgs = prompts.navigate_messages(chat_content, children)
        data = extract_json(await self._chat(msgs))
        name = str(data.get("name", "")).strip()
        valid = {c.name for c in children}
        return name if name in valid else ""

    def _mock_navigate(self, chat_content: str, children: List[Node]) -> str:
        entity = _detect_entity(chat_content)
        for c in children:
            hay = (c.name + " " + c.dialog_trigger + " " + c.background).lower()
            if entity and entity.lower() in hay:
                return c.name
            if c.name and c.name.lower() in chat_content.lower():
                return c.name
        return ""

    async def generate_query(self, chat_content: str, backgrounds: List[str]) -> str:
        if self.provider == "mock":
            entity = _detect_entity(chat_content) or ""
            return f"{entity} {chat_content[:40]}".strip()
        msgs = prompts.generate_query_messages(chat_content, backgrounds)
        data = extract_json(await self._chat(msgs))
        return str(data.get("query", chat_content[:60]))

    async def attribute_error(
        self, chat_content: str, visited: List[Dict], query: str
    ) -> Dict:
        if self.provider == "mock":
            return self._mock_attribute_error(chat_content, visited)
        msgs = prompts.attribute_error_messages(chat_content, visited, query)
        data = extract_json(await self._chat(msgs))
        return data if isinstance(data, dict) else {}

    def _mock_attribute_error(self, chat_content: str, visited: List[Dict]) -> Dict:
        # 若导航没走到底（visited 短），归因为最后一个节点的 trigger 问题；
        # 否则归因为最后一个节点的 background 问题。
        if not visited:
            return {"node_name": "Root", "problem": "trigger", "reason": "未能进入任何类别"}
        last = visited[-1]
        problem = "trigger" if last.get("dead_end") else "background"
        return {
            "node_name": last.get("name", ""),
            "problem": problem,
            "reason": "mock 归因：导航中断" if problem == "trigger" else "mock 归因：背景知识不足",
        }

    async def reflect_errors(
        self, node: Node, error_samples: List[Dict], feedback: str = ""
    ) -> Dict:
        if self.provider == "mock":
            return self._mock_reflect_errors(node, error_samples)
        msgs = prompts.reflect_errors_messages(node, error_samples, feedback)
        data = extract_json(await self._chat(msgs))
        return data if isinstance(data, dict) else {}

    def _mock_reflect_errors(self, node: Node, error_samples: List[Dict]) -> Dict:
        # 收集失败样本对话里的关键词，补进 trigger/background
        kws = set()
        problems = set()
        for s in error_samples:
            problems.add(s.get("problem", ""))
            chat = s.get("chat_content", "")
            ent = _detect_entity(chat)
            if ent:
                kws.add(ent)
        extra = "、".join(sorted(kws))
        dialog_trigger = node.dialog_trigger
        background = node.background
        if "trigger" in problems and extra:
            dialog_trigger = (dialog_trigger + f" 当用户提到 {extra} 等相关问题时也应进入该类别。").strip()
        if "background" in problems and extra:
            background = (background + f" 补充：{extra} 相关常见问题的处理要点与检索线索。").strip()
        return {
            "dialog_trigger": dialog_trigger,
            "background": background,
            "reason": "mock 反思：根据失败样本补充触发条件/背景知识",
        }
