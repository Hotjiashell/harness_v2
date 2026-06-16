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


def _path_block(path_nodes: List[Dict]) -> str:
    """实际导航路径（含每个节点的 background），供 background 归因参考。"""
    if not path_nodes:
        return "（未进入任何类别）"
    lines = []
    for i, n in enumerate(path_nodes, 1):
        lines.append(
            f"  {i}. {n.get('name','')}\n"
            f"     dialog_trigger：{n.get('dialog_trigger','')}\n"
            f"     background：{n.get('background','')}"
        )
    return "\n".join(lines)


def _level_block(level: Dict) -> str:
    """单层候选节点（可能被选的全部节点）及实际选中者。"""
    cands = level.get("candidates", [])
    chosen = level.get("chosen_name")
    lines = [f"  实际选中：{chosen if chosen else '（未选中任何节点）'}"]
    for c in cands:
        mark = " ← 实际选中" if c.get("name") == chosen else ""
        lines.append(f"  - {c.get('name','')}：{c.get('dialog_trigger','')}{mark}")
    return "\n".join(lines)


_ATTR_PROBLEM_DOC = (
    "  - background：路径上某个【已走过节点】的 background 知识不足，"
    "导致据此生成的检索 query 与目标案例对不上（路径走对了但 query 不好）。"
    "background 的作用是提供额外知识，指导生成更容易检索到目标案例的 query；\n"
    "  - trigger：某个【候选节点】的 dialog_trigger 有问题，导致没能选到通往目标案例的正确节点。"
    "dialog_trigger 的作用是说明什么样的对话应进入该类别。\n"
)


def attribute_oneshot_messages(
    chat_content: str, path_nodes: List[Dict], levels: List[Dict],
    query: str, gt_case_name: str = "", gt_case_text: str = "",
) -> List[Dict[str, str]]:
    """一次性归因：把所有层候选一并给模型，让其一次判断 background / trigger 问题。"""
    system = (
        "你是错误归因专家。一次对话导航+检索失败了：模型依据对话逐层选择类别、"
        "再结合路径上各节点的 background 生成检索 query，但该 query 没能检索到"
        "这条对话本应命中的目标案例（GT 案例）。\n"
        "请结合目标案例内容，判断失败的根因属于以下哪一种：\n"
        + _ATTR_PROBLEM_DOC +
        "\n下面会给出：实际走过的路径（含各节点 background）、以及每一层"
        "「可能被选的」全部候选节点及其 dialog_trigger。请先逐步分析"
        "（query 为何检索不到目标案例：是某节点 background 不足，还是某层选错了类别、"
        "即某候选 trigger 不当），然后在最后用一个 ```json``` 代码块输出归因结论。"
    )
    level_blocks = []
    for i, lv in enumerate(levels, 1):
        level_blocks.append(f"第 {i} 层候选：\n{_level_block(lv)}")
    user = (
        f"对话内容：\n{chat_content}\n\n"
        f"实际导航路径（含 background）：\n{_path_block(path_nodes)}\n\n"
        f"各层「可能被选的」候选节点：\n" + "\n\n".join(level_blocks) + "\n\n"
        f"生成的检索 query（未能检索到目标案例）：{query}\n\n"
        f"本应检索到的目标案例（GT）：\n标题：{gt_case_name}\n内容：{gt_case_text}\n\n"
        "请先给出分析，最后用 ```json``` 代码块输出："
        '{"node_name":"<问题节点name>","problem":"trigger或background","reason":"..."}。'
        "node_name 必须是上面出现过的某个节点名；"
        "reason 说明应对该节点做怎样的修改，以及为什么这样修改。"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def attribute_stage_messages(
    chat_content: str, path_nodes: List[Dict], level: Dict,
    query: str, gt_case_name: str = "", gt_case_text: str = "",
    allow_background: bool = True, allow_escalate: bool = True,
    stage_depth: int = 1,
) -> List[Dict[str, str]]:
    """多阶段归因的单阶段：只给【本层】候选，让模型判断本层问题或上抛。"""
    options = []
    if allow_background:
        options.append(
            "  - background：路径上某个【已走过节点】的 background 知识不足，"
            "导致据此生成的检索 query 与目标案例对不上（本层及以上类别都选对了，只是 query 不好）。"
        )
    options.append(
        "  - trigger：本层某个【候选节点】的 dialog_trigger 有问题，"
        "导致没能选到通往目标案例的正确节点。"
    )
    if allow_escalate:
        options.append(
            "  - upper_level：本层的候选里压根没有通往目标案例的正确类别，"
            "说明问题在更上一层（上一层就选错了），需要向上追溯。"
        )
    system = (
        "你是错误归因专家。一次对话导航+检索失败了：模型依据对话逐层选择类别、"
        "再结合路径各节点 background 生成检索 query，但该 query 没能检索到目标案例（GT 案例）。\n"
        f"当前正在分析第 {stage_depth} 层（从根往下数）的候选。"
        "请结合目标案例内容，判断本层失败根因属于以下哪一种：\n"
        + "\n".join(options) + "\n\n"
        "请先逐步分析，然后在最后用一个 ```json``` 代码块输出结论。"
    )
    problem_enum = "/".join(
        (["background"] if allow_background else []) + ["trigger"]
        + (["upper_level"] if allow_escalate else [])
    )
    user = (
        f"对话内容：\n{chat_content}\n\n"
        f"实际导航路径（含 background）：\n{_path_block(path_nodes)}\n\n"
        f"本层「可能被选的」候选节点：\n{_level_block(level)}\n\n"
        f"生成的检索 query（未能检索到目标案例）：{query}\n\n"
        f"本应检索到的目标案例（GT）：\n标题：{gt_case_name}\n内容：{gt_case_text}\n\n"
        "请先给出分析，最后用 ```json``` 代码块输出："
        f'{{"node_name":"<问题节点name，problem=upper_level 时可留空>",'
        f'"problem":"{problem_enum}","reason":"..."}}。'
        "reason 说明判断依据；若为 trigger/background，说明应对该节点做怎样的修改。"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def refine_trigger_messages(
    node: Node, samples: List[Dict], feedback: str = ""
) -> List[Dict[str, str]]:
    """修改某节点的 dialog_trigger：让归因到该节点的对话能正确进入本类别。"""
    system = (
        "你是知识节点优化专家。下面这批对话本应被分类到指定节点，但当前没能正确进入。"
        "原因被归因为该节点的 dialog_trigger（触发条件）不当。\n"
        "dialog_trigger 的作用是说明「什么样的对话应进入该类别」。"
        "请根据这批对话的共性，改进该节点的 dialog_trigger，使这些对话能正确进入本节点，"
        "同时避免写得过宽而把明显不相关的对话也吸进来。只改 dialog_trigger，不动 background。\n"
        "请先简要分析，最后用 ```json``` 代码块输出。"
    )
    sample_lines = []
    for i, s in enumerate(samples, 1):
        sample_lines.append(
            f"样本{i}：\n"
            f"  归因原因：{s.get('reason', '')}\n"
            f"  对话内容：{s.get('chat_content', '')}"
        )
    fb = f"\n\n{feedback}" if feedback else ""
    user = (
        f"待优化节点：\nname：{node.name}\n"
        f"当前 dialog_trigger：{node.dialog_trigger}\n\n"
        f"本应进入该节点、却分类失败的对话：\n"
        + "\n\n".join(sample_lines) + fb + "\n\n"
        "请先给出分析，最后用 ```json``` 代码块输出："
        '{"dialog_trigger":"<改进后的触发条件>","reason":"说明为何这样改"}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def refine_background_messages(
    node: Node, samples: List[Dict], feedback: str = ""
) -> List[Dict[str, str]]:
    """修改某节点的 background：让据此生成的 query 能检索到目标案例。"""
    system = (
        "你是知识节点优化专家。下面这批对话已正确导航到指定节点，"
        "但结合该节点 background 生成的检索 query 没能检索到本应命中的目标案例（GT 案例）。"
        "原因被归因为该节点的 background（背景知识）不足。\n"
        "background 的作用是提供额外知识，指导生成更容易检索到目标案例的 query。"
        "请对照「生成的 query」与「目标案例标题/内容」，补充/调整该节点的 background，"
        "使其贴近目标案例的用语与要点。只改 background，不动 dialog_trigger。\n"
        "请先简要分析，最后用 ```json``` 代码块输出。"
    )
    sample_lines = []
    for i, s in enumerate(samples, 1):
        sample_lines.append(
            f"样本{i}：\n"
            f"  归因原因：{s.get('reason', '')}\n"
            f"  对话内容：{s.get('chat_content', '')}\n"
            f"  生成的检索 query（未能检索到目标案例）：{s.get('query', '')}\n"
            f"  目标案例标题（GT）：{s.get('gt_case_name', '')}\n"
            f"  目标案例内容（GT）：{s.get('gt_case_text', '')}"
        )
    fb = f"\n\n{feedback}" if feedback else ""
    user = (
        f"待优化节点：\nname：{node.name}\n"
        f"当前 background：{node.background}\n\n"
        f"导航已到位、但 query 检索失败的对话：\n"
        + "\n\n".join(sample_lines) + fb + "\n\n"
        "请先给出分析，最后用 ```json``` 代码块输出："
        '{"background":"<改进后的背景知识>","reason":"说明为何这样改"}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
