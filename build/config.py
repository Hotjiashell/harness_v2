# -*- coding: utf-8 -*-
"""层级化知识树构建框架的全局配置。

所有可调参数集中在此处，便于复现与调试。需求文档中要求“可在 config.py
中配置”的项目都暴露在这里。类型定义见 khtree/config_types.py。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 保证可以 import khtree 包
BUILD_DIR = Path(__file__).resolve().parent
ROOT_DIR = BUILD_DIR.parent
if str(BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DIR))

from khtree.config_types import (  # noqa: E402
    BuildSettings,
    ClusterSettings,
    HarnessConfig,
    LLMSettings,
    OptimizeSettings,
    PathSettings,
    RuntimeSettings,
)


# ---------------------------------------------------------------------------
# 路径配置：输入数据与输出目录
# ---------------------------------------------------------------------------
PATHS = PathSettings(
    # 案例库：{case_id: {case_name, text}}
    case_path=ROOT_DIR / "data" / "case" / "text.json",
    # 对话训练集 / 验证集：[{call_sno, chat_content, caseID}]
    dialog_train_path=ROOT_DIR / "data" / "dialog" / "dialog.json",
    dialog_val_path=ROOT_DIR / "data" / "dialog" / "dialog.json",
    # L1 人工初始类别
    seed_l1_path=ROOT_DIR / "output" / "seed_L1.json",
    # 所有产物输出目录
    output_dir=BUILD_DIR / "output",
)


# ---------------------------------------------------------------------------
# LLM 配置
# provider="mock" 时不联网，使用内置启发式逻辑跑通全流程（便于调试）。
# provider="openai" 时调用 openai 兼容接口。
# ---------------------------------------------------------------------------
LLM = LLMSettings(
    provider="mock",
    base_url="http://localhost:8000/v1",
    api_key="1234",
    model="gpt-4.1-mini",
    concurrency=8,
    timeout_seconds=60,
    temperature=0.1,
    max_retries=3,
    max_tokens=2048,
)


# ---------------------------------------------------------------------------
# 聚类配置（L2 及以后初始类别）。默认 K-Means，兼容 hdbscan。
# ---------------------------------------------------------------------------
CLUSTER = ClusterSettings(
    method="kmeans",
    n_clusters=4,
    min_cluster_size=2,
    embedding_concurrency=5,
    embedding_model="text-embedding-3-small",
    embedding_url="http://localhost:8000/embeddings",
    embedding_api_key="1234",
)


# ---------------------------------------------------------------------------
# 阶段一：基于案例数据的知识树构建
# ---------------------------------------------------------------------------
BUILD = BuildSettings(
    max_depth=2,            # 最大树深（L1=1, L2=2 ...）
    batch_size=8,           # 每个 batch 总大小
    unknown_per_batch=4,    # 每个 batch 中无法归类案例的目标数量
    max_node_count=12,      # Complexity Check：单层最大节点数
    max_plan_retries=3,     # Coverage Validation 最大重试次数
    max_complexity_retries=3,  # Complexity Check 失败后重生成 Update Plan 的最大次数
    min_cases_to_split=3,   # 类别下案例数少于该值则不再向下分裂
)


# ---------------------------------------------------------------------------
# 阶段二：基于对话数据的节点内容优化
# ---------------------------------------------------------------------------
OPTIMIZE = OptimizeSettings(
    max_reflection_retries=3,
    nav_beam_width=1,
)


# ---------------------------------------------------------------------------
# 运行/调试
# ---------------------------------------------------------------------------
RUNTIME = RuntimeSettings(
    console_output=True,
    log_timestamps=True,
    use_tqdm=True,
    random_seed=42,
)


CONFIG = HarnessConfig(
    paths=PATHS,
    llm=LLM,
    cluster=CLUSTER,
    build=BUILD,
    optimize=OPTIMIZE,
    runtime=RUNTIME,
)
