# -*- coding: utf-8 -*-
"""命令行入口。

用法：
  # 阶段一：基于案例库构建知识树
  python build/run.py build

  # 从某个中间树文件续跑（指定起始层）
  python build/run.py build --resume build/output/intermediate/001_L1_tree_after.json --from-level 2

  # 阶段二：基于对话数据优化节点内容（默认读取阶段一产物）
  python build/run.py optimize

  # 用指定的案例树作为优化输入
  python build/run.py optimize --tree build/output/knowledge_tree_case.json

  # 一键全流程
  python build/run.py all
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

BUILD_DIR = Path(__file__).resolve().parent
if str(BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DIR))

from config import CONFIG  # noqa: E402
from khtree import utils  # noqa: E402
from khtree.build_tree import CaseTreeBuilder  # noqa: E402
from khtree.llm import LLMClient  # noqa: E402
from khtree.models import Case, Dialog, Tree  # noqa: E402
from khtree.optimize import DialogOptimizer  # noqa: E402
from khtree.utils import (  # noqa: E402
    ErrorRecorder,
    configure_logging,
    log,
    read_json,
    stage_banner,
    write_json,
)


def _setup():
    rt = CONFIG.runtime
    configure_logging(console=rt.console_output, timestamps=rt.log_timestamps)
    CONFIG.paths.output_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.paths.intermediate_dir.mkdir(parents=True, exist_ok=True)
    recorder = ErrorRecorder(CONFIG.paths.error_log_path)
    llm = LLMClient(CONFIG.llm)
    log(f"LLM provider = {CONFIG.llm.provider}", stage="INIT")
    return recorder, llm


def _load_cases():
    raw = read_json(CONFIG.paths.case_path)
    cases = Case.load_all(raw)
    log(f"加载案例 {len(cases)} 条：{CONFIG.paths.case_path}", stage="INIT")
    return cases


def _load_dialogs(path: Path):
    raw = read_json(path)
    dialogs = Dialog.load_all(raw)
    log(f"加载对话 {len(dialogs)} 条：{path}", stage="INIT")
    return dialogs


def _save_tree(tree: Tree, path: Path) -> None:
    write_json(path, tree.to_dict(include_debug=False))
    write_json(path.with_name(path.stem + "_debug" + path.suffix),
               tree.to_dict(include_debug=True))
    log(f"知识树已保存：{path}", stage="SAVE")
    log(f"调试树（含 case_ids）已保存：{path.with_name(path.stem + '_debug' + path.suffix)}",
        stage="SAVE")


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
async def run_build(args) -> Tree:
    recorder, llm = _setup()
    cases = _load_cases()
    seed_l1 = read_json(CONFIG.paths.seed_l1_path)
    # 快照本次运行配置
    write_json(CONFIG.paths.output_dir / "run_config.json", {
        "provider": CONFIG.llm.provider, "model": CONFIG.llm.model,
        "max_depth": CONFIG.build.max_depth, "batch_size": CONFIG.build.batch_size,
        "unknown_per_batch": CONFIG.build.unknown_per_batch,
        "max_node_count": CONFIG.build.max_node_count,
        "run_stage": CONFIG.runtime.run_stage,
    })

    resume_tree = None
    from_level = 1
    if args.resume:
        resume_tree = Tree.from_dict(read_json(Path(args.resume)))
        from_level = args.from_level
        log(f"续跑：{args.resume}，起始层 L{from_level}", stage="BUILD")

    builder = CaseTreeBuilder(CONFIG, llm, cases, recorder)
    tree = await builder.build(seed_l1, resume_tree=resume_tree, resume_from_level=from_level)

    _save_tree(tree, CONFIG.paths.case_tree_path)
    stage_banner("阶段一完成：基于案例库的知识树")
    _print_tree(tree)
    log(f"错误总数：{recorder.count}（详见 {CONFIG.paths.error_log_path}）", stage="DONE")
    return tree


# ---------------------------------------------------------------------------
# optimize
# ---------------------------------------------------------------------------
async def run_optimize(args) -> Tree:
    recorder, llm = _setup()
    cases = _load_cases()
    dialogs = _load_dialogs(CONFIG.paths.dialog_train_path)
    # 验证集：仅用于优化前后各跑一次召回率供人工观测，不参与优化/反馈
    val = _load_dialogs(CONFIG.paths.dialog_val_path) if CONFIG.paths.dialog_val_path else []

    tree_path = Path(args.tree) if args.tree else CONFIG.paths.case_tree_path
    if not tree_path.exists():
        log(f"找不到输入知识树：{tree_path}。请先运行 build。", stage="ERROR")
        raise SystemExit(2)
    # 优先载入带 case_ids 的 debug 版本，保留案例归属信息
    debug_path = tree_path.with_name(tree_path.stem + "_debug" + tree_path.suffix)
    load_path = debug_path if (not args.tree and debug_path.exists()) else tree_path
    tree = Tree.from_dict(read_json(load_path))
    log(f"载入待优化知识树：{load_path}", stage="OPTIMIZE")

    optimizer = DialogOptimizer(CONFIG, llm, tree, cases, dialogs, recorder, val=val)
    tree = await optimizer.optimize()

    _save_tree(tree, CONFIG.paths.final_tree_path)
    stage_banner("阶段二完成：最终知识树")
    _print_tree(tree)
    log(f"错误总数：{recorder.count}（详见 {CONFIG.paths.error_log_path}）", stage="DONE")
    return tree


async def run_all(args) -> None:
    await run_build(args)
    # optimize 读取阶段一产物
    class _A:
        tree = None
    await run_optimize(_A())


def _print_tree(tree: Tree, node=None, prefix: str = "") -> None:
    node = node or tree.root
    if prefix == "":
        log(f"知识树结构：", stage="TREE")
        print(node.name, file=sys.stderr)
    children = node.children
    for i, c in enumerate(children):
        last = i == len(children) - 1
        conn = "`-- " if last else "|-- "
        print(f"{prefix}{conn}{c.name} (cases={len(c.all_case_ids())})", file=sys.stderr)
        _print_tree(tree, c, prefix + ("    " if last else "|   "))


def main() -> int:
    parser = argparse.ArgumentParser(description="层级化知识树构建与优化框架")
    sub = parser.add_subparsers(dest="cmd", required=False)

    p_build = sub.add_parser("build", help="阶段一：基于案例库构建知识树")
    p_build.add_argument("--resume", type=str, default=None,
                         help="从中间树 json 续跑")
    p_build.add_argument("--from-level", type=int, default=1,
                         help="续跑的起始层级（与 --resume 配合）")

    p_opt = sub.add_parser("optimize", help="阶段二：基于对话优化节点内容")
    p_opt.add_argument("--tree", type=str, default=None,
                       help="待优化的知识树 json（默认用阶段一产物）")

    p_all = sub.add_parser("all", help="依次执行 build 与 optimize")
    p_all.add_argument("--resume", type=str, default=None)
    p_all.add_argument("--from-level", type=int, default=1)

    args = parser.parse_args()

    # 未显式指定子命令时，按 config.py 的 runtime.run_stage 决定跑到哪一步
    cmd = args.cmd
    if cmd is None:
        cmd = CONFIG.runtime.run_stage
        # 补齐缺省参数
        for name, default in (("resume", None), ("from_level", 1), ("tree", None)):
            if not hasattr(args, name):
                setattr(args, name, default)
        log(f"未指定子命令，按 config.runtime.run_stage = '{cmd}' 运行", stage="INIT")

    if cmd == "build":
        asyncio.run(run_build(args))
    elif cmd == "optimize":
        asyncio.run(run_optimize(args))
    elif cmd == "all":
        asyncio.run(run_all(args))
    else:
        parser.error(f"未知运行阶段: {cmd}（应为 build/optimize/all）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
