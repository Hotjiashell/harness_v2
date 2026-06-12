# -*- coding: utf-8 -*-
"""核心数据结构。"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# 输入数据
# ---------------------------------------------------------------------------
@dataclass
class Case:
    """一条案例：一个真实问题及其处理方式。"""

    case_id: str
    case_name: str
    text: str

    def to_text(self) -> str:
        return f"【{self.case_name}】{self.text}"

    @staticmethod
    def load_all(raw: Dict[str, Dict[str, Any]]) -> Dict[str, "Case"]:
        cases: Dict[str, Case] = {}
        for cid, payload in raw.items():
            cases[cid] = Case(
                case_id=cid,
                case_name=str(payload.get("case_name", "")),
                text=str(payload.get("text", "")),
            )
        return cases


@dataclass
class Dialog:
    """一条对话，对应一个 ground truth 案例。"""

    call_sno: str
    chat_content: str
    case_id: str

    @staticmethod
    def load_all(raw: List[Dict[str, Any]]) -> List["Dialog"]:
        dialogs: List[Dialog] = []
        for item in raw:
            dialogs.append(
                Dialog(
                    call_sno=str(item.get("call_sno", "")),
                    chat_content=str(item.get("chat_content", "")),
                    case_id=str(item.get("caseID", "")),
                )
            )
        return dialogs


# ---------------------------------------------------------------------------
# 知识树
# ---------------------------------------------------------------------------
@dataclass
class Node:
    """知识树节点。

    四个语义字段对应需求：name / case_trigger / dialog_trigger / background。
    case_ids 记录归属于该节点（含其字面层，不含子节点）的案例 id，用于调试与逐层处理。
    """

    name: str
    case_trigger: str = ""
    dialog_trigger: str = ""
    background: str = ""
    children: List["Node"] = field(default_factory=list)
    case_ids: List[str] = field(default_factory=list)
    # 内部稳定 id，便于 proposal/操作引用，不对外序列化进最终树
    node_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    # -- 序列化 --------------------------------------------------------------
    def to_dict(self, include_debug: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "name": self.name,
            "case_trigger": self.case_trigger,
            "dialog_trigger": self.dialog_trigger,
            "background": self.background,
        }
        if include_debug:
            data["node_id"] = self.node_id
            data["case_ids"] = list(self.case_ids)
        if self.children:
            data["children"] = [c.to_dict(include_debug) for c in self.children]
        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Node":
        node = Node(
            name=str(data.get("name", "")),
            case_trigger=str(data.get("case_trigger", "")),
            dialog_trigger=str(data.get("dialog_trigger", "")),
            background=str(data.get("background", "")),
            case_ids=[str(c) for c in (data.get("case_ids") or [])],
        )
        if data.get("node_id"):
            node.node_id = str(data["node_id"])
        node.children = [Node.from_dict(c) for c in (data.get("children") or [])]
        return node

    # -- 便捷方法 ------------------------------------------------------------
    def all_case_ids(self) -> List[str]:
        """该子树覆盖的全部案例 id（去重）。"""
        seen: List[str] = []
        s = set()
        for cid in self.case_ids:
            if cid not in s:
                s.add(cid)
                seen.append(cid)
        for child in self.children:
            for cid in child.all_case_ids():
                if cid not in s:
                    s.add(cid)
                    seen.append(cid)
        return seen

    def find_by_name(self, name: str) -> Optional["Node"]:
        for child in self.children:
            if child.name == name:
                return child
        return None


@dataclass
class Tree:
    """整棵知识树，Root 节点之下为 L1。"""

    root: Node

    @staticmethod
    def new() -> "Tree":
        return Tree(root=Node(name="Root"))

    def to_dict(self, include_debug: bool = False) -> Dict[str, Any]:
        return self.root.to_dict(include_debug)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Tree":
        return Tree(root=Node.from_dict(data))


# ---------------------------------------------------------------------------
# 修改操作 / Update Plan
# ---------------------------------------------------------------------------
# 仅允许两种操作：
#   ADD    新增一个类别
#   MODIFY 修改现有类别的 name / case_trigger / background
ADD = "add"
MODIFY = "modify"


@dataclass
class Operation:
    """一个修改操作（Proposal / Update Plan 的基本单元）。"""

    op_type: str  # ADD or MODIFY
    # ADD: 新类别的字段；MODIFY: target 指向现有类别名，fields 为要修改的字段
    name: str = ""
    case_trigger: str = ""
    background: str = ""
    dialog_trigger: str = ""
    target: str = ""  # MODIFY 时指向被修改类别的 name
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_type": self.op_type,
            "name": self.name,
            "case_trigger": self.case_trigger,
            "background": self.background,
            "dialog_trigger": self.dialog_trigger,
            "target": self.target,
            "reason": self.reason,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Operation":
        return Operation(
            op_type=str(data.get("op_type", "")).lower().strip(),
            name=str(data.get("name", "")),
            case_trigger=str(data.get("case_trigger", "")),
            background=str(data.get("background", "")),
            dialog_trigger=str(data.get("dialog_trigger", "")),
            target=str(data.get("target", "")),
            reason=str(data.get("reason", "")),
        )


@dataclass
class ClassificationResult:
    """单条案例的分类结果。"""

    case_id: str
    category: str  # 命中的类别 name；UNKNOWN 表示无法归类
    reason: str = ""

    UNKNOWN = "UNKNOWN"

    def is_unknown(self) -> bool:
        return self.category == ClassificationResult.UNKNOWN or not self.category
