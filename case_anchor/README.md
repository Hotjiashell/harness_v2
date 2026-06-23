# Case Anchor 构建

为每条案例（case）生成一个 `anchor` 字段——一串**空格隔开的检索关键词**，
帮助案例更容易被用户咨询时生成的检索 query 命中，从而提高召回。

## 原理

1. 每条 case 在**知识树**下逐层导航（默认按 `dialog_trigger`，可选 `case_trigger`），
   积累沿途经过节点的 `background`。
2. 依据「case 内容 + 积累的背景知识」提炼一组检索关键词作为 `anchor`。
   提示词刻意贴近阶段二 `generate_query_messages`（同样吃「节点名：背景知识」），
   使 anchor 关键词与对话生成的 query **同源、同体系**——两者用同一套术语，
   case 的 anchor 就更容易被 query 命中。

`anchor` 形态：一行用空格隔开的关键词（不是自然句子、不是模拟用户问法），
覆盖案例的核心实体、术语与典型检索词。例如：
`WeCon 历史消息同步 database.ini 本地数据库 多端同步`。

## 用法

```bash
# 默认：按 dialog_trigger 导航，openai provider
python case_anchor/build_anchor.py \
    --case data/case/text.json \
    --tree build/output/knowledge_tree.json \
    --out case_anchor/text_with_anchor.json

# 按 case_trigger 导航
python case_anchor/build_anchor.py --nav-by case_trigger

# 关闭思考模式 / 离线自测（内置启发式，不联网）
python case_anchor/build_anchor.py --no-thinking
python case_anchor/build_anchor.py --provider mock
```

## 参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `--case` | 案例文件 `{case_id: {case_name, text}}` | `data/case/text.json` |
| `--tree` | 知识树文件 | `build/output/knowledge_tree.json` |
| `--out` | 输出：原 case + `anchor` 字段 | `case_anchor/text_with_anchor.json` |
| `--detail-file` | 调试中间文件（含 `visited` / `analysis`） | `case_anchor/anchor_detail.json` |
| `--nav-by` | 导航依据：`dialog_trigger` / `case_trigger` | `dialog_trigger` |
| `--no-thinking` / `--thinking` | 关闭 / 开启思考模式 | 开启 |
| `--provider` | `openai` 或 `mock` | `openai` |
| `--base-url` / `--api-key` / `--model` | LLM 接口配置 | localhost / 1234 / gpt-4.1-mini |
| `--concurrency` | 并发数 | 8 |

## 输出

| 文件 | 内容 |
|------|------|
| `text_with_anchor.json` | 主产物：`{case_id: {case_name, text, anchor}}`，结构与输入一致、额外带 `anchor` |
| `anchor_detail.json` | 调试：每条 case 的 `visited`（导航路径）、`analysis`（提炼分析过程）、`anchor` |

导航与生成全程并行，受 `--concurrency` 限流，带进度条。单条 case 失败只跳过该条、
不中断整体（`anchor` 置空并在 detail 中记 `error`）。
