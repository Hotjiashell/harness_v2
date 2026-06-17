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
    # 阶段二对话集：
    #   dialog_train_path —— 用于错误归因与节点优化（训练集）
    #   dialog_val_path   —— 仅在优化前后各跑一次召回率供人工观测泛化效果，
    #                        不参与优化/反馈；为发挥作用应指向与 train 不同的对话集。
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
# 阶段一：基于案例数据的知识树构建
# ---------------------------------------------------------------------------
BUILD = BuildSettings(
    max_depth=2,            # 最大树深（L1=1, L2=2 ...）
    batch_size=8,           # 每个 batch 总大小
    unknown_per_batch=4,    # 每个 batch 中无法归类案例的目标数量
    max_node_count=12,      # Complexity Check：单层最大节点数
    max_initial_node_count=8,  # L2+ 归纳初始子类别时最多产出的类别数
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
    attribution_mode="oneshot",   # 错误归因方式："oneshot" 或 "multistage"
)


# ---------------------------------------------------------------------------
# 运行/调试
# ---------------------------------------------------------------------------
RUNTIME = RuntimeSettings(
    console_output=True,
    log_timestamps=True,
    use_tqdm=True,
    random_seed=42,
    # "build" 只跑完基于案例库的知识树构建；"all" 构建后继续基于对话优化节点内容。
    run_stage="all",
)


CONFIG = HarnessConfig(
    paths=PATHS,
    llm=LLM,
    build=BUILD,
    optimize=OPTIMIZE,
    runtime=RUNTIME,
)
