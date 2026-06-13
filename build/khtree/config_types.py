# -*- coding: utf-8 -*-
"""配置数据结构定义。

实际的取值集中在仓库根的 build/config.py 中（CONFIG 实例）。
拆出类型定义是为了让 khtree 包内部可以直接引用类型而不产生循环依赖。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PathSettings:
    case_path: Path
    dialog_train_path: Path
    dialog_val_path: Path
    seed_l1_path: Path
    output_dir: Path

    @property
    def intermediate_dir(self) -> Path:
        return self.output_dir / "intermediate"

    @property
    def case_tree_path(self) -> Path:
        return self.output_dir / "knowledge_tree_case.json"

    @property
    def final_tree_path(self) -> Path:
        return self.output_dir / "knowledge_tree.json"

    @property
    def error_log_path(self) -> Path:
        return self.output_dir / "errors.log"


@dataclass
class LLMSettings:
    provider: str = "mock"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "1234"
    model: str = "gpt-4.1-mini"
    concurrency: int = 8
    timeout_seconds: int = 60
    temperature: float = 0.1
    max_retries: int = 3
    max_tokens: int = 2048


@dataclass
class BuildSettings:
    max_depth: int = 2
    batch_size: int = 8
    unknown_per_batch: int = 4
    max_node_count: int = 12
    # L2 及以后，归纳初始子类别时最多产出的类别数
    max_initial_node_count: int = 8
    max_plan_retries: int = 3
    max_complexity_retries: int = 3
    min_cases_to_split: int = 3


@dataclass
class OptimizeSettings:
    max_reflection_retries: int = 3
    nav_beam_width: int = 1


@dataclass
class RuntimeSettings:
    console_output: bool = True
    log_timestamps: bool = True
    use_tqdm: bool = True
    random_seed: int = 42
    # 运行阶段控制：
    #   "build"    只跑阶段一（基于案例库构建知识树）
    #   "all"      构建后接着跑阶段二（基于对话优化节点内容）
    # 命令行子命令（build/optimize/all）会覆盖该默认值。
    run_stage: str = "all"


@dataclass
class HarnessConfig:
    paths: PathSettings
    llm: LLMSettings = field(default_factory=LLMSettings)
    build: BuildSettings = field(default_factory=BuildSettings)
    build: BuildSettings = field(default_factory=BuildSettings)
    optimize: OptimizeSettings = field(default_factory=OptimizeSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
