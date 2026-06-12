# -*- coding: utf-8 -*-
"""通用工具：日志、IO、错误记录、并发与 JSON 解析。"""
from __future__ import annotations

import asyncio
import datetime
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, TypeVar

try:  # tqdm 可选
    from tqdm import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


T = TypeVar("T")
R = TypeVar("R")


# ---------------------------------------------------------------------------
# 控制台日志
# ---------------------------------------------------------------------------
_CONSOLE = True
_TIMESTAMPS = True


def configure_logging(console: bool = True, timestamps: bool = True) -> None:
    global _CONSOLE, _TIMESTAMPS
    _CONSOLE = console
    _TIMESTAMPS = timestamps


def _ts() -> str:
    if not _TIMESTAMPS:
        return ""
    return datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")


def log(message: str, stage: Optional[str] = None) -> None:
    """打印调试信息到 stderr，方便追踪进度。"""
    if not _CONSOLE:
        return
    prefix = _ts()
    if stage:
        prefix += f"[{stage}] "
    print(f"{prefix}{message}", file=sys.stderr, flush=True)


def stage_banner(title: str) -> None:
    if not _CONSOLE:
        return
    line = "=" * 70
    print(f"\n{line}\n{_ts()}>>> {title}\n{line}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# 错误记录：不能因为单个样例错误而让整个流程崩溃
# ---------------------------------------------------------------------------
class ErrorRecorder:
    """收集错误并写入日志文件，同时在控制台提示。"""

    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.count = 0
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, where: str, exc: BaseException, context: str = "") -> None:
        self.count += 1
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        entry = (
            f"{_ts()}ERROR #{self.count} at [{where}]\n"
            f"  context: {context}\n"
            f"  {type(exc).__name__}: {exc}\n"
            f"{tb}\n"
        )
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(entry)
        log(f"已记录错误 #{self.count} @ {where}: {type(exc).__name__}: {exc}", stage="ERROR")


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_intermediate(out_dir: Path, name: str, data: Any) -> Path:
    """写出一份中间结果，便于调试。返回写入路径。"""
    path = Path(out_dir) / name
    write_json(path, data)
    log(f"中间结果已写入: {path}", stage="DUMP")
    return path


# ---------------------------------------------------------------------------
# JSON 解析（容错：从模型输出中提取 JSON）
# ---------------------------------------------------------------------------
def extract_json(text: str) -> Any:
    """尽力从模型输出文本中提取 JSON 对象/数组。"""
    if text is None:
        raise ValueError("empty text")
    text = text.strip()
    # 去掉 ```json ... ``` 代码块围栏
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # 退而求其次：寻找第一个 { 或 [ 与最后一个 } 或 ]
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except Exception:
                continue
    raise ValueError(f"无法从输出中解析 JSON: {text[:200]!r}")


# ---------------------------------------------------------------------------
# 并发：受限并发地 map 一个 async 函数，带进度条
# ---------------------------------------------------------------------------
async def gather_limited(
    items: Sequence[T],
    func: Callable[[T], Awaitable[R]],
    concurrency: int,
    desc: str = "",
    use_tqdm: bool = True,
    recorder: Optional[ErrorRecorder] = None,
    where: str = "",
) -> List[Optional[R]]:
    """并发执行 func(item)，限制并发数。

    单项失败不会中断整体：失败项返回 None 并记录错误。
    结果顺序与输入顺序一致。
    """
    sem = asyncio.Semaphore(max(1, concurrency))
    results: List[Optional[R]] = [None] * len(items)

    bar = None
    if use_tqdm and _tqdm is not None and len(items) > 0:
        bar = _tqdm(total=len(items), desc=desc or "processing", file=sys.stderr)

    async def _run(idx: int, item: T) -> None:
        async with sem:
            try:
                results[idx] = await func(item)
            except Exception as exc:  # noqa: BLE001
                if recorder is not None:
                    recorder.record(where or desc or "gather_limited", exc, context=str(item)[:200])
                else:
                    log(f"任务失败(已忽略): {exc}", stage="WARN")
                results[idx] = None
            finally:
                if bar is not None:
                    bar.update(1)

    await asyncio.gather(*[_run(i, it) for i, it in enumerate(items)])
    if bar is not None:
        bar.close()
    return results


def chunk(items: Sequence[T], size: int) -> List[List[T]]:
    size = max(1, size)
    return [list(items[i : i + size]) for i in range(0, len(items), size)]
