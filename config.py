from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent


@dataclass
class PathSettings:
    case_path: Path = ROOT_DIR / "data" / "case" / "text.json"
    seed_l1_path: Path = ROOT_DIR / "output" / "seed_L1.json"
    output_dir: Path = ROOT_DIR / "output" / "harness"
    skills_dir: Path = ROOT_DIR / "output" / "_skills" / "harness"


@dataclass
class LLMSettings:
    provider: str = "heuristic"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "1234"
    model: str = "gpt-4.1-mini"
    concurrency: int = 8
    timeout_seconds: int = 60
    temperature: float = 0.1
    max_retries: int = 2


@dataclass
class ClusterSettings:
    method: str = "hdbscan"
    n_clusters: int = 5
    min_cluster_size: int = 2
    embedding_concurrency: int = 5
    embedding_model: str = "text-embedding-3-small"
    embedding_url: str = "http://localhost:8000/embeddings"
    embedding_api_key: str = "1234"
    local_similarity_threshold: float = 0.18


@dataclass
class PipelineSettings:
    max_depth: int = 3
    new_l1_min_cases: int = 2
    min_cases_to_split: int = 2
    software_alias_min_match: int = 3
    stop_after_l1: bool = False
    resume_tree_path: Path | None = None
    console_output: bool = True
    log_timestamps: bool = True
    progress_bar_width: int = 28
    stage_dir_name: str = "intermediate"
    initial_root_filename: str = "05_initial_root.json"
    final_tree_filename: str = "knowledge_tree.json"
    debug_tree_filename: str = "knowledge_tree_debug.json"
    config_snapshot_filename: str = "run_config.json"


@dataclass
class HarnessConfig:
    paths: PathSettings = field(default_factory=PathSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    cluster: ClusterSettings = field(default_factory=ClusterSettings)
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)


CONFIG = HarnessConfig()
