# -*- coding: utf-8 -*-
"""提示词模板。

每个函数返回一个 messages 列表（OpenAI chat 格式）。模型被要求输出 JSON，
便于解析。mock provider 不使用这些模板，仅 openai provider 使用。
"""
from __future__ import annotations

import json
from typing import Dict, List

from .models import Case, Node


def _categories_block(categories: List[Node], include_background: bool = False) -> str:
    """渲染候选类别列表。

    分类与生成修改操作时只暴露 name 与 case_trigger（不含 background）。
    """
    items = []
    for c in categories:
        item = {"name": c.name, "case_trigger": c.case_trigger}
        if include_background:
            item["background"] = c.background
        items.append(item)
    return json.dumps(items, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 案例分类
# ---------------------------------------------------------------------------
def classify_messages(case: Case, categories: List[Node]) -> List[Dict[str, str]]:
    system = (
        "你是一个知识分类助手。给定若干候选类别（每个类别提供 name 和 case_trigger，"
        "case_trigger 描述什么样的案例应归入该类别）和一条案例，"
        "依据 case_trigger 判断该案例应归入哪个类别。若都不合适，返回 UNKNOWN。"
        "只能从给定类别的 name 中选择，或返回 UNKNOWN。"
    )
    user = (
        f"候选类别（仅 name 与 case_trigger）：\n{_categories_block(categories)}\n\n"
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
) -> List[Dict[str, str]]:
    system = (
        "你是知识结构优化专家。下面给出当前类别体系（每个类别仅提供 name 和 case_trigger），"
        "以及一批案例（其中部分无法归入现有类别，已标注 UNKNOWN）。请分析这批案例，"
        "提出修改操作，使得这些案例都能被合理归类。\n\n"
        "类别的三个字段含义：\n"
        "  - name：类别名称。\n"
        "  - case_trigger：针对案例的触发条件，即什么样的案例应归入该类别。\n"
        "  - background：该领域的相关背景知识（如术语定义、原理、常见处理经验）。\n\n"
        "仅允许两种操作：\n"
        '  1) add：新增一个类别，必须同时给出 name、case_trigger、background 三个字段；\n'
        '  2) modify：修改现有类别，只能改 name 或 case_trigger（不能改 background），'
        "target 指向现有类别 name。\n\n"
        "尽量复用已有类别，只有确实无法归入时才新增。"
    )
    case_lines = []
    unknown_set = set(unknown_ids)
    for c in batch_cases:
        tag = "UNKNOWN(待归类)" if c.case_id in unknown_set else "已分类"
        case_lines.append(f"- [{tag}] {c.case_id} 标题：{c.case_name}；内容：{c.text}")
    user = (
        f"当前类别（仅 name 与 case_trigger）：\n{_categories_block(categories)}\n\n"
        f"本批案例：\n" + "\n".join(case_lines) + "\n\n"
        '请输出 JSON 数组，每个元素形如：\n'
        '新增：{"op_type":"add","name":"...","case_trigger":"...","background":"...","reason":"..."}\n'
        '修改：{"op_type":"modify","target":"<现有类别name>","name":"<新name，可选>",'
        '"case_trigger":"<新case_trigger，可选>","reason":"..."}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Proposal Aggregation -> Update Plan
# ---------------------------------------------------------------------------
def aggregate_messages(
    proposals: List[Dict],
    categories: List[Node],
    feedback: str = "",
    max_add: int = 0,
) -> List[Dict[str, str]]:
    add_limit = (
        f"本层当前已有 {len(categories)} 个类别，最多还能新增 {max_add} 个类别"
        f"（即本次 add 操作数量不得超过 {max_add} 个）。"
        if max_add and max_add > 0 else
        "本层已无新增类别的余量，请不要再输出 add 操作，只能通过 modify 调整现有类别。"
    )
    system = (
        "你是知识结构优化专家。多个 batch 基于案例提出了多条修改提议（proposal），"
        "请据此产出一份精简、无冗余、且能覆盖所有案例的最终修改方案（Update Plan）。\n\n"
        "你的职责不限于对提议去重合并——当现有提议不足以覆盖某些案例时，"
        "你可以主动扩充：新增 add 操作，或调整 add/modify 的 name、case_trigger 使其"
        "覆盖面更广，从而让更多案例能被归类。若反馈中指出某些新增类别实际未接住任何案例，"
        "应删除这些无用的 add。\n\n"
        "操作约束：\n"
        '  - add：新增类别，必须给出 name、case_trigger、background 三个字段；\n'
        '  - modify：只能改现有类别的 name 或 case_trigger，不能改 background。\n'
        f"  - 数量上限：{add_limit}\n\n"
        "原则：语义相同的新增类别应合并为一个；针对同一类别的多次修改应合并；"
        "在保证覆盖的前提下尽量精简，不要产出冗余或过细的类别。\n\n"
        "请先简要分析：哪些提议可合并、是否需要扩充、有无无用 add，"
        "然后在最后用一个 ```json ``` 代码块输出最终的修改方案 JSON 数组。"
    )
    fb = f"\n\n反馈（请据此调整方案）：\n{feedback}" if feedback else ""
    user = (
        f"当前类别（仅 name 与 case_trigger）：\n{_categories_block(categories)}\n\n"
        f"全部提议：\n{json.dumps(proposals, ensure_ascii=False, indent=2)}{fb}\n\n"
        "请先给出分析，最后用 ```json``` 代码块输出汇总后的 JSON 数组，"
        "元素格式与提议相同（op_type=add/modify）。"
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


def discover_categories_messages(
    summaries: List[str], parent: Node, max_count: int = 0
) -> List[Dict[str, str]]:
    limit = (
        f"最多归纳出 {max_count} 个子类别（不要超过这个数量）。"
        if max_count and max_count > 0 else ""
    )
    system = (
        "你是知识结构设计专家。下面给出某个父类别之下全部案例的总结。"
        "请综观所有总结，直接归纳出父类别之下一组互斥、覆盖完整、粒度适中的初始子类别。"
        "类别数量应与案例的内在主题数相匹配，不要过细也不要过粗。" + limit
    )
    body = "\n".join(f"- {s}" for s in summaries)
    user = (
        f"父类别：{parent.name}\n父类别背景：{parent.background}\n\n"
        f"该父类别下全部案例总结：\n{body}\n\n"
        '请输出初始子类别 JSON 数组：[{"name":"...","case_trigger":"...","background":"..."}]'
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
