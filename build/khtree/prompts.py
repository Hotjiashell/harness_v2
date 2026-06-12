# -*- coding: utf-8 -*-
"""提示词模板。

每个函数返回一个 messages 列表（OpenAI chat 格式）。模型被要求输出 JSON，
便于解析。mock provider 不使用这些模板，仅 openai provider 使用。
"""
from __future__ import annotations

import json
from typing import Dict, List

from .models import Case, Node


def _categories_block(categories: List[Node]) -> str:
    items = []
    for i, c in enumerate(categories):
        items.append(
            {
                "name": c.name,
                "case_trigger": c.case_trigger,
                "background": c.background,
            }
        )
    return json.dumps(items, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 案例分类
# ---------------------------------------------------------------------------
def classify_messages(case: Case, categories: List[Node]) -> List[Dict[str, str]]:
    system = (
        "你是一个知识分类助手。给定若干候选类别和一条案例，"
        "判断该案例应归入哪个类别。若都不合适，返回 UNKNOWN。"
        "只能从给定类别的 name 中选择，或返回 UNKNOWN。"
    )
    user = (
        f"候选类别：\n{_categories_block(categories)}\n\n"
        f"案例：\n标题：{case.case_name}\n内容：{case.text}\n\n"
        '请输出 JSON：{"category": "<类别name或UNKNOWN>", "reason": "<判断理由>"}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Batch Reflection -> Proposals
# ---------------------------------------------------------------------------
def reflect_batch_messages(
    batch_cases: List[Case],
    categories: List[Node],
    unknown_ids: List[str],
    feedback: str = "",
) -> List[Dict[str, str]]:
    system = (
        "你是知识结构优化专家。下面给出当前类别体系，以及一批案例"
        "（其中部分无法归入现有类别，已标注 UNKNOWN）。请分析这批案例，"
        "提出修改操作，使得这些案例都能被合理归类。\n"
        "仅允许两种操作：\n"
        '  1) add：新增一个类别，需给出 name/case_trigger/background；\n'
        '  2) modify：修改现有类别的 name/case_trigger/background（target 指向现有类别 name）。\n'
        "尽量复用已有类别，只有确实无法归入时才新增。"
    )
    case_lines = []
    unknown_set = set(unknown_ids)
    for c in batch_cases:
        tag = "UNKNOWN(待归类)" if c.case_id in unknown_set else "已分类"
        case_lines.append(f"- [{tag}] {c.case_id} 标题：{c.case_name}；内容：{c.text}")
    fb = f"\n\n上一轮的反馈，请据此改进：\n{feedback}" if feedback else ""
    user = (
        f"当前类别：\n{_categories_block(categories)}\n\n"
        f"本批案例：\n" + "\n".join(case_lines) + fb + "\n\n"
        '请输出 JSON 数组，每个元素形如：\n'
        '{"op_type":"add","name":"...","case_trigger":"...","background":"...","reason":"..."}\n'
        '或 {"op_type":"modify","target":"<现有类别name>","name":"<新name可选>",'
        '"case_trigger":"<可选>","background":"<可选>","reason":"..."}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Proposal Aggregation -> Update Plan
# ---------------------------------------------------------------------------
def aggregate_messages(
    proposals: List[Dict],
    categories: List[Node],
    feedback: str = "",
) -> List[Dict[str, str]]:
    system = (
        "你是知识结构优化专家。多个 batch 产生了多条修改提议（proposal），"
        "其中可能有重复或可合并的操作。请汇总成一份精简、无冗余的最终修改方案"
        "（Update Plan）。语义相同的新增类别应合并为一个；针对同一类别的多次修改应合并。"
    )
    fb = f"\n\n反馈（请据此调整方案）：\n{feedback}" if feedback else ""
    user = (
        f"当前类别：\n{_categories_block(categories)}\n\n"
        f"全部提议：\n{json.dumps(proposals, ensure_ascii=False, indent=2)}{fb}\n\n"
        "请输出汇总后的 JSON 数组，元素格式与提议相同（op_type=add/modify）。"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# L2+ 聚类初始化
# ---------------------------------------------------------------------------
def summarize_case_messages(case: Case, parent: Node) -> List[Dict[str, str]]:
    system = (
        "你是领域专家。请在给定父类别的背景下，用一句话总结这条案例的核心子问题类型，"
        "便于后续聚类。"
    )
    user = (
        f"父类别：{parent.name}\n父类别背景：{parent.background}\n\n"
        f"案例标题：{case.case_name}\n案例内容：{case.text}\n\n"
        '请输出 JSON：{"summary":"<一句话总结>"}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def label_cluster_messages(summaries: List[str], parent: Node) -> List[Dict[str, str]]:
    system = (
        "你是领域专家。下面是同一聚类簇内若干案例的总结。请为该簇总结出 1-3 个"
        "候选子类别（属于父类别之下）。"
    )
    user = (
        f"父类别：{parent.name}\n父类别背景：{parent.background}\n\n"
        f"簇内案例总结：\n" + "\n".join(f"- {s}" for s in summaries) + "\n\n"
        '请输出 JSON 数组：[{"name":"...","case_trigger":"...","background":"..."}]（1-3个）'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def synthesize_categories_messages(
    candidate_categories: List[Dict], parent: Node
) -> List[Dict[str, str]]:
    system = (
        "你是知识结构设计专家。下面是各聚类簇分别提出的候选子类别集合。"
        "请综观全部候选，去重、归并、提炼，给出父类别之下一组互斥、覆盖完整的初始子类别。"
    )
    user = (
        f"父类别：{parent.name}\n父类别背景：{parent.background}\n\n"
        f"全部候选子类别：\n{json.dumps(candidate_categories, ensure_ascii=False, indent=2)}\n\n"
        '请输出最终初始子类别 JSON 数组：[{"name":"...","case_trigger":"...","background":"..."}]'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# 阶段二：对话导航 / query 生成 / 错误归因 / 错误反思
# ---------------------------------------------------------------------------
def navigate_messages(chat_content: str, children: List[Node]) -> List[Dict[str, str]]:
    system = (
        "你是对话路由助手。只看对话内容，结合候选类别的 dialog_trigger，"
        "判断当前对话应进入哪个类别。"
    )
    items = [{"name": c.name, "dialog_trigger": c.dialog_trigger} for c in children]
    user = (
        f"对话内容：\n{chat_content}\n\n"
        f"候选类别：\n{json.dumps(items, ensure_ascii=False, indent=2)}\n\n"
        '请输出 JSON：{"name":"<最匹配类别name，没有则空字符串>","reason":"..."}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_query_messages(chat_content: str, backgrounds: List[str]) -> List[Dict[str, str]]:
    system = (
        "你是检索 query 生成助手。结合对话内容和导航过程中收集到的背景知识，"
        "生成一个用于检索相关案例的检索 query。"
    )
    bg = "\n".join(f"- {b}" for b in backgrounds if b)
    user = (
        f"对话内容：\n{chat_content}\n\n"
        f"导航收集到的背景知识：\n{bg}\n\n"
        '请输出 JSON：{"query":"<检索query>"}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def attribute_error_messages(
    chat_content: str, visited: List[Dict], query: str
) -> List[Dict[str, str]]:
    system = (
        "你是错误归因专家。一次对话导航+检索失败了。请分析是哪个节点出了问题，"
        "以及问题类型：\n"
        "  - trigger：节点的 dialog_trigger 有问题，导致选不对正确类别；\n"
        "  - background：节点的 background 提供的知识/经验不足，导致生成的 query 不好。\n"
        "请指出最该负责的那个节点。"
    )
    user = (
        f"对话内容：\n{chat_content}\n\n"
        f"导航路径（依次经过的节点及其 trigger/background）：\n"
        f"{json.dumps(visited, ensure_ascii=False, indent=2)}\n\n"
        f"生成的检索 query：{query}\n\n"
        '请输出 JSON：{"node_name":"<问题节点name>","problem":"trigger或background","reason":"..."}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def reflect_errors_messages(
    node: Node, error_samples: List[Dict], feedback: str = ""
) -> List[Dict[str, str]]:
    system = (
        "你是知识节点优化专家。下面是归因到同一个节点的一批失败样本。"
        "请分析应如何改进该节点的 dialog_trigger 或 background（不新增/删除节点），"
        "给出修改后的内容。dialog_trigger 决定对话能否走到该类别；"
        "background 决定能否生成好的检索 query。"
    )
    fb = f"\n\n上一轮反馈（修改未提升召回，请重新调整）：\n{feedback}" if feedback else ""
    user = (
        f"待优化节点：\nname：{node.name}\n"
        f"当前 dialog_trigger：{node.dialog_trigger}\n"
        f"当前 background：{node.background}\n\n"
        f"失败样本（问题类型 + 对话）：\n"
        f"{json.dumps(error_samples, ensure_ascii=False, indent=2)}{fb}\n\n"
        '请输出 JSON：{"dialog_trigger":"<改进后，不变则原样返回>",'
        '"background":"<改进后，不变则原样返回>","reason":"..."}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
