# Query 融合召回评测（idea validation）

把**两个 query 文件**的检索结果融合后再算召回率，验证「融合两个 query」能否提升召回。
query 文件就是 `eval/eval_recall.py` / `eval/eval_baseline.py` 产出的 `queries.json`
（格式 `[{call_sno, case_id, query}]`）。

两个脚本对应两种融合方式，按 `call_sno` 对齐两个文件、用两个 query 各自检索 top-50，
再融合、算 top-1/3/5/10/20/50 召回率。

## 方式 1：各分一半，去重平分（`fuse_split_half.py`）

两个 query 各自检索，去除重复案例（每案例只留一次），把 top-N 的名额尽量在 A/B 间平分；
N 为奇数时多出的那 1 个名额给**相关度得分更高**的候选。实现为「得分高者先手的交替取数」。

```bash
python idea_validation/query_mix/fuse_split_half.py \
    --query-a eval/output/queries.json \
    --query-b eval/baseline_output/queries.json
```

## 方式 2：去重后按得分排序取 topN（`fuse_score_sort.py`）

合并两边结果并去重（同一案例两边都出现时取较高得分），按相关度得分降序取前 N。

```bash
python idea_validation/query_mix/fuse_score_sort.py \
    --query-a eval/output/queries.json \
    --query-b eval/baseline_output/queries.json
```

## 参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `--query-a` | query 文件 A（必填） | — |
| `--query-b` | query 文件 B（必填） | — |
| `--out-dir` | 输出目录 | `output_split` / `output_score` |
| `--result-file` | 召回率结果文件 | `<out-dir>/recall_result.json` |
| `--strategy` | 传给 `retrieve.py::retrieve_case` 的检索策略 | `lexical&semantic` |
| `--dialog` | 原始对话文件：用于 `fusion_detail.json` 补充完整对话；`--use-chat` 时也用于 `chat_content` 兜底 | `data/dialog/dialog.json` |
| `--case-text` | 案例标题/内容文件：用于 `fusion_detail.json` 补充 GT `case_name` 和 `text` | `data/case/text.json` |
| `--concurrency` | 并发数 | 8 |

## 输出

| 文件 | 内容 |
|------|------|
| `fusion_detail.json` | 每条：两个 `query_a`/`query_b`、完整对话 `dialog`、GT 案例标题/内容 `gt_case`、命中排名 `hit_rank`、各自 top5、融合后 `fused_top5`、各 top-k 是否命中 |
| `recall_result.json` | 配置（含两个 query 文件路径与融合方法）+ 各 top-k 召回率 |

日志中会打印两个 query 文件路径、各自条数、共同 call_sno 数；`fusion_detail.json`
逐条记录两个 query 原文、完整对话和 GT 案例内容，便于排查。检索按 `--concurrency` 并行（同步 `retrieve_case`
丢线程池）。两文件只取**共同的 call_sno** 做对比，仅单边存在的会提示并跳过。
