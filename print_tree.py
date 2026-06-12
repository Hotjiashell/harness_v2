#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print a knowledge tree in terminal with total case counts for each node. "
            "When possible, this script automatically loads a sibling "
            "knowledge_tree_debug.json for exact counts."
        )
    )
    parser.add_argument(
        "tree_json",
        type=Path,
        help="Path to knowledge_tree.json or knowledge_tree_debug.json",
    )
    parser.add_argument(
        "--debug-tree",
        type=Path,
        default=None,
        help="Optional path to knowledge_tree_debug.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(payload).__name__}")
    return payload


def has_case_ids(node: dict[str, Any]) -> bool:
    return isinstance(node.get("case_ids"), list)


def guess_debug_tree_path(tree_path: Path) -> Path | None:
    if tree_path.name.endswith("_debug.json"):
        return tree_path

    candidates = []
    if tree_path.name == "knowledge_tree.json":
        candidates.append(tree_path.with_name("knowledge_tree_debug.json"))

    candidates.append(tree_path.with_name(f"{tree_path.stem}_debug{tree_path.suffix}"))

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def build_exact_count_map(
    node: dict[str, Any],
    path: tuple[str, ...] | None = None,
) -> dict[tuple[str, ...], int]:
    count_map, _ = _build_exact_count_map_with_ids(node, path=path)
    return count_map


def _build_exact_count_map_with_ids(
    node: dict[str, Any],
    path: tuple[str, ...] | None = None,
) -> tuple[dict[tuple[str, ...], int], set[str]]:
    name = str(node.get("name") or "<unnamed>")
    current_path = (*path, name) if path else (name,)

    case_ids = node.get("case_ids")
    if not isinstance(case_ids, list):
        raise ValueError("Debug tree node is missing list field 'case_ids'")

    subtree_case_ids = {str(case_id) for case_id in case_ids}
    count_map: dict[tuple[str, ...], int] = {}
    for child in node.get("children") or []:
        if isinstance(child, dict):
            child_count_map, child_case_ids = _build_exact_count_map_with_ids(child, current_path)
            count_map.update(child_count_map)
            subtree_case_ids.update(child_case_ids)

    count_map[current_path] = len(subtree_case_ids)
    return count_map, subtree_case_ids


def infer_count(
    node: dict[str, Any],
    exact_count_map: dict[tuple[str, ...], int],
    path: tuple[str, ...] | None = None,
) -> int | None:
    name = str(node.get("name") or "<unnamed>")
    current_path = (*path, name) if path else (name,)

    if current_path in exact_count_map:
        return exact_count_map[current_path]

    case_ids = node.get("case_ids")
    if isinstance(case_ids, list):
        return len(set(str(case_id) for case_id in case_ids))

    child_counts = []
    for child in node.get("children") or []:
        if not isinstance(child, dict):
            continue
        child_count = infer_count(child, exact_count_map, current_path)
        if child_count is None:
            return None
        child_counts.append(child_count)
    if child_counts:
        return sum(child_counts)
    return None


def format_label(name: str, count: int | None) -> str:
    if count is None:
        return f"{name} (cases=?)"
    return f"{name} (cases={count})"


def print_tree(
    node: dict[str, Any],
    exact_count_map: dict[tuple[str, ...], int],
    path: tuple[str, ...] | None = None,
    prefix: str = "",
    is_last: bool = True,
) -> None:
    name = str(node.get("name") or "<unnamed>")
    current_path = (*path, name) if path else (name,)
    count = infer_count(node, exact_count_map, path)
    label = format_label(name, count)

    if path is None:
        print(label)
    else:
        connector = "`-- " if is_last else "|-- "
        print(f"{prefix}{connector}{label}")

    children = [child for child in (node.get("children") or []) if isinstance(child, dict)]
    if path is None:
        next_prefix = ""
    else:
        next_prefix = prefix + ("    " if is_last else "|   ")
    for index, child in enumerate(children):
        child_is_last = index == len(children) - 1
        print_tree(
            child,
            exact_count_map=exact_count_map,
            path=current_path,
            prefix=next_prefix,
            is_last=child_is_last,
        )


def main() -> int:
    args = parse_args()
    tree_path = args.tree_json.expanduser().resolve()

    if not tree_path.exists():
        print(f"[print_tree] File not found: {tree_path}", file=sys.stderr)
        return 1

    tree = load_json(tree_path)

    exact_count_map: dict[tuple[str, ...], int] = {}
    debug_path = args.debug_tree.expanduser().resolve() if args.debug_tree else None

    if has_case_ids(tree):
        debug_tree = tree
        debug_source = tree_path
    else:
        if debug_path is None:
            debug_path = guess_debug_tree_path(tree_path)

        debug_tree = None
        debug_source = None
        if debug_path is not None and debug_path.exists():
            candidate = load_json(debug_path)
            if has_case_ids(candidate):
                debug_tree = candidate
                debug_source = debug_path

    print(f"[print_tree] Loading tree: {tree_path}", file=sys.stderr)
    if debug_source is not None and debug_tree is not None:
        exact_count_map = build_exact_count_map(debug_tree)
        print(f"[print_tree] Using exact counts from: {debug_source}", file=sys.stderr)
    else:
        print(
            "[print_tree] No debug tree with case_ids found. Counts may be unknown.",
            file=sys.stderr,
        )

    print_tree(tree, exact_count_map=exact_count_map)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
