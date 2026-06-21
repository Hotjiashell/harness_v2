# 导航 / 检索召回率评测

独立评测脚本（不依赖 `build/` 的运行流程，但复用其 LLM 客户端与数据模型）。

给定**对话数据**和**已构建的知识树**，跑「逐层导航 → 收集背景 → 生成 query」全过程，
再调用根目录 `retrieve.py::retrieve_case` 计算 **top-1/3/5/10/20/50** 召回率。

## 用法

```bash
# 全流程：导航 + 生成 query + 检索 + 召回率（默认）
python eval/eval_recall.py \
    --dialog data/dialog/dialog.json \
    --tree build/output/knowledge_tree.json \
    --out-dir eval/output

# 只跑到生成 query（产出 queries.json，不检索）
python eval/eval_recall.py --stage query

# 从已有 query 文件只跑检索 + 召回率
python eval/eval_recall.py --stage retrieve --query-file eval/output/queries.json

# 关闭思考模式（默认开启）
python eval/eval_recall.py --no-thinking

# 离线自测（用内置启发式，不联网）
python eval/eval_recall.py --provider mock
```

## 阶段（`--stage`）

| 值 | 含义 |
|----|------|
| `full`（默认） | 导航 → 生成 query → 检索 → 召回率 |
| `query` | 只跑到生成 query，写出 `queries.json`，不检索 |
| `retrieve` | 从 `--query-file` 读取 query，只跑检索与召回统计 |

## 主要参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `--dialog` | 对话数据文件 | `data/dialog/dialog.json` |
| `--tree` | 知识树文件 | `build/output/knowledge_tree.json` |
| `--query-file` | query 中间文件（retrieve 阶段读取，其他阶段写入） | `<out-dir>/queries.json` |
| `--out-dir` | 输出目录 | `eval/output` |
| `--result-file` | 召回率结果文件 | `<out-dir>/recall_result.json` |
| `--stage` | 运行阶段 | `full` |
| `--no-thinking` / `--thinking` | 关闭 / 开启思考模式 | 开启 |
| `--provider` | `openai` 或 `mock` | `openai` |
| `--base-url` / `--api-key` / `--model` | LLM 接口配置 | localhost / 1234 / gpt-4.1-mini |
| `--concurrency` | 并发数 | 8 |

## 输出文件（写入 `--out-dir`）

| 文件 | 内容 |
|------|------|
| `navigation.json` | 每条对话：经过的节点 `visited` + 生成的 `query` + 对话原文 |
| `queries.json` | 精简的 `{call_sno, case_id, query}`，供 `retrieve` 阶段复用 |
| `retrieval_detail.json` | 每条对话的命中排名 `hit_rank` 与各 top-k 是否命中 |
| `recall_result.json` | 运行配置 + 各 top-k 召回率汇总 |

召回率同时打印到终端。命中排名 `hit_rank` 为目标案例在检索结果中的 1-based 名次，
`null` 表示 top-50 内未命中。
