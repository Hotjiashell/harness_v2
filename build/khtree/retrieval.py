# -*- coding: utf-8 -*-
"""检索适配器。

按需求，retrieve.py 当前不实现，这里只预留调用方式：优先调用根目录
retrieve.py::retrive(query, caseID) -> bool。

当其未实现（返回 None / 抛错）时，回退到一个本地的、确定性的“伪检索”，
依据 query 与目标案例文本的关键词重叠判定召回成功与否，使阶段二的
“修改验证（召回成功率是否提高）”闭环在离线环境下也能跑通。
"""
from __future__ import annotations

import importlib.util
import inspect
import re
from pathlib import Path
from typing import Dict, Optional

from .models import Case
from .utils import log


_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_root_retrieve():
    path = _ROOT / "retrieve.py"
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("_root_retrieve", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return getattr(mod, "retrive", None) or getattr(mod, "retrieve", None)
    except Exception as exc:  # noqa: BLE001
        log(f"加载根目录 retrieve.py 失败，使用本地回退检索: {exc}", stage="RETRIEVE")
        return None


_ROOT_FN = None
_ROOT_FN_LOADED = False


def _tokens(text: str) -> set:
    parts = re.split(r"[：:，,。、\s/\\()（）“”\"']+", text.lower())
    return {p for p in parts if len(p) >= 2}


class Retriever:
    """对检索接口的封装。带离线回退。"""

    def __init__(self, cases: Dict[str, Case]):
        self.cases = cases
        global _ROOT_FN, _ROOT_FN_LOADED
        if not _ROOT_FN_LOADED:
            _ROOT_FN = _load_root_retrieve()
            _ROOT_FN_LOADED = True
        self._root_fn = _ROOT_FN
        self._root_fn_ok = _ROOT_FN is not None

    async def retrieve(self, query: str, case_id: str) -> bool:
        """根据 query 检索，命中目标 case_id 返回 True。"""
        if self._root_fn_ok:
            try:
                result = self._root_fn(query, case_id)
                if inspect.isawaitable(result):
                    result = await result
                if result is None:
                    # 未实现，永久回退
                    self._root_fn_ok = False
                else:
                    return bool(result)
            except Exception as exc:  # noqa: BLE001
                log(f"调用 retrieve.py 失败，永久回退本地检索: {exc}", stage="RETRIEVE")
                self._root_fn_ok = False
        return self._fallback(query, case_id)

    def _fallback(self, query: str, case_id: str) -> bool:
        """本地伪检索：query 与目标案例的关键词重叠度需为全库最高（top-1）。"""
        target = self.cases.get(case_id)
        if target is None:
            return False
        q = _tokens(query)
        if not q:
            return False

        def score(case: Case) -> float:
            ct = _tokens(case.to_text())
            inter = len(q & ct)
            return inter / (len(q) or 1)

        target_score = score(target)
        if target_score <= 0:
            return False
        # top-1：没有其它案例得分严格更高
        for cid, case in self.cases.items():
            if cid == case_id:
                continue
            if score(case) > target_score:
                return False
        return True
