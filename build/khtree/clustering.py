# -*- coding: utf-8 -*-
"""聚类适配器。

优先调用仓库根目录的 cluster.py::cluster（其参数/返回格式已约定，本框架不实现它）。
当 cluster.py 未实现（返回 None）或调用失败时，回退到一个本地的、不依赖网络的
朴素聚类，保证 L2+ 初始化流程在离线环境下也能跑通并产出中间结果。

cluster.cluster 的返回格式：
    [{"cluster_id": int, "text": str}, ...]
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Dict, List, Optional

from .config_types import ClusterSettings
from .utils import log


# 仓库根目录（build 的上一级）
_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_root_cluster():
    """动态加载根目录的 cluster.py（避免与包内模块名冲突）。"""
    path = _ROOT / "cluster.py"
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("_root_cluster", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return getattr(mod, "cluster", None)
    except Exception as exc:  # noqa: BLE001
        log(f"加载根目录 cluster.py 失败，使用本地回退聚类: {exc}", stage="CLUSTER")
        return None


async def cluster_texts(
    texts: List[str], settings: ClusterSettings
) -> List[Dict]:
    """对文本聚类，返回 [{"cluster_id", "text"}]。"""
    fn = _load_root_cluster()
    if fn is not None:
        try:
            result = fn(
                texts,
                cluster_method=settings.method,
                n_clusters=settings.n_clusters,
                min_cluster_size=settings.min_cluster_size,
                embedding_concurrency=settings.embedding_concurrency,
                embedding_model=settings.embedding_model,
                embedding_url=settings.embedding_url,
                embedding_api_key=settings.embedding_api_key,
            )
            if hasattr(result, "__await__"):
                result = await result
            if result:  # 非空才采用
                log(f"使用 cluster.py 聚类，得到 {len(result)} 条标注", stage="CLUSTER")
                return result
            log("cluster.py 未实现/返回空，使用本地回退聚类", stage="CLUSTER")
        except Exception as exc:  # noqa: BLE001
            log(f"调用 cluster.py 失败，使用本地回退聚类: {exc}", stage="CLUSTER")

    return _fallback_cluster(texts, settings)


# ---------------------------------------------------------------------------
# 本地回退：基于关键词重叠的朴素聚类（无需 embedding 服务）
# ---------------------------------------------------------------------------
def _tokens(text: str) -> set:
    parts = re.split(r"[：:，,。、\s/\\()（）“”\"']+", text.lower())
    return {p for p in parts if len(p) >= 2}


def _fallback_cluster(texts: List[str], settings: ClusterSettings) -> List[Dict]:
    n = len(texts)
    if n == 0:
        return []
    token_sets = [_tokens(t) for t in texts]

    if settings.method == "hdbscan":
        target_k = None  # 由密度决定，用阈值贪心
    else:
        target_k = max(1, min(settings.n_clusters, n))

    # 贪心：以 jaccard 相似度归并
    assignments = [-1] * n
    centers: List[set] = []
    for i in range(n):
        best_c, best_sim = -1, 0.0
        for ci, center in enumerate(centers):
            inter = len(token_sets[i] & center)
            union = len(token_sets[i] | center) or 1
            sim = inter / union
            if sim > best_sim:
                best_c, best_sim = ci, sim
        if best_sim >= 0.15 and (target_k is None or best_c >= 0):
            assignments[i] = best_c
            centers[best_c] |= token_sets[i]
        else:
            if target_k is not None and len(centers) >= target_k and centers:
                # 已达上限，并入最相似的中心
                best_c = max(
                    range(len(centers)),
                    key=lambda ci: len(token_sets[i] & centers[ci]),
                )
                assignments[i] = best_c
                centers[best_c] |= token_sets[i]
            else:
                centers.append(set(token_sets[i]))
                assignments[i] = len(centers) - 1

    out = [{"cluster_id": assignments[i], "text": texts[i]} for i in range(n)]
    log(f"本地回退聚类得到 {len(set(assignments))} 个簇", stage="CLUSTER")
    return out
