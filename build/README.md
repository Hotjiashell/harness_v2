# 层级化知识树构建与优化框架

基于案例库构建层级化知识树，再基于对话数据优化节点内容。实现自 `build/requirements.md`。

## 目录结构

```
build/
├── config.py            # 全局配置（所有可调参数）
├── run.py               # 命令行入口（build / optimize / all）
├── README.md
├── khtree/              # 框架包
│   ├── config_types.py  # 配置数据类定义
│   ├── models.py        # Node / Tree / Case / Dialog / Operation 等数据结构
│   ├── utils.py         # 日志、IO、错误记录、并发、JSON 解析
│   ├── llm.py           # 异步 LLM 客户端（openai / mock 两种 provider）
│   ├── prompts.py       # 提示词模板
│   ├── retrieval.py     # 检索适配器（调用根 retrieve.py，带离线回退）
│   ├── build_tree.py    # 阶段一：基于案例库构建知识树
│   └── optimize.py      # 阶段二：基于对话数据优化节点内容
└── output/              # 所有产物（运行后生成）
    ├── knowledge_tree_case.json        # 阶段一产物（最终用户字段）
    ├── knowledge_tree_case_debug.json  # 阶段一产物（含 case_ids）
    ├── knowledge_tree.json             # 阶段二最终产物
    ├── knowledge_tree_debug.json       # 含 case_ids 的最终产物
    ├── run_config.json                 # 本次运行配置快照
    ├── errors.log                      # 错误记录
    └── intermediate/                   # 每个阶段的中间结果（便于调试）
```

## 安装依赖

```bash
pip3 install openai tqdm httpx
```

（`mock` provider 仅 `tqdm` 为可选项；`openai` provider 需要 `openai` 包。）

## 运行

```bash
# 阶段一：基于案例库构建知识树
python3 build/run.py build

# 阶段二：基于对话数据优化节点内容（默认读取阶段一产物）
python3 build/run.py optimize

# 一键全流程
python3 build/run.py all

# 打印知识树（含案例数）
python3 print_tree.py build/output/knowledge_tree.json
```

### 按配置决定跑到哪一步

不带子命令运行时，按 `config.py` 的 `RUNTIME.run_stage` 决定：

- `run_stage="build"`：只跑完阶段一（基于案例库构建知识树）。
- `run_stage="all"`：阶段一构建完成后，接着跑阶段二（基于对话优化节点内容）。

```bash
# 等价于按 config.runtime.run_stage 运行（build 或 all）
python3 build/run.py
```

显式子命令（`build` / `optimize` / `all`）会覆盖该配置默认值。

### 续跑

```bash
# 从某层中间树文件继续往后构建
python3 build/run.py build \
    --resume build/output/intermediate/006_L1_tree_after.json \
    --from-level 2

# 知识树构建完成后，单独跑对话优化（指定输入树）
python3 build/run.py optimize --tree build/output/knowledge_tree_case.json
```

## 两个阶段

### 阶段一：基于案例库构建（`build_tree.py`）

逐层构建，每一层先做一次分类，再进入「聚合 → 复杂度 → 覆盖率」重试循环：

```
指定/归纳初始类别
  → 案例分类 (Classification, 并行)            ← known 案例只分这一次
  → Batch Reflection (并行) → Proposals        ← 只跑一次，基于案例原文产出"粗料"
  → 重试循环（最多 max_plan_retries+1 次）：
      Aggregation → Update Plan                ← 吃覆盖反馈；可去重合并，也可主动扩充 add
        └ Complexity Check（内层）：新增后节点数 > MaxNodeCount 则反馈重生成 / 兜底截断
      Coverage Validation（试执行 plan + 只重分类 Unknown）
        ├ 全覆盖 → 接受，break
        └ 有遗漏 → 生成覆盖反馈，回到 Aggregation 重试
  → 应用 plan → 改名重映射 → 只对仍 Unknown 的案例重分类 → 落案例
```

关于这套循环的几个设计点：

- **分类只做一次**：已分类（known）案例不再被反复重分类。覆盖率验证与最终落地时，只对仍为 Unknown 的案例重新分类，已分类结果直接沿用（modify 改名时做一次零成本的标签重映射）。
- **Batch Reflection 只跑一次**：它基于案例原文产出 add/modify 提议（粗料）。它不接收覆盖反馈——因为反馈针对的是「plan 覆盖不全」，而 plan 是聚合阶段的产物。
- **覆盖反馈只回聚合**：聚合阶段职责已不限于去重合并，**可以主动扩充**（新增 add、调整 name/case_trigger 扩大覆盖面），所以由它来消化覆盖反馈最合适。聚合 prompt 要求模型先分析、再用 ```json``` 代码块输出最终方案，并写明本层最多还能新增几个类别（`max_node_count − 现有节点数`）。
- **覆盖反馈包含三部分**：① 上一版 Update Plan；② 仍无法分类的案例（含归不进的原因）；③ 新增却一个案例都没接住的「无用 add」（提示删除）。让聚合在上一版基础上修正，而不是从头重猜。

- **L1** 初始类别人工定义（`output/seed_L1.json`）。
- **L2 及以后** 初始类别由模型直接归纳：对父类别下每个案例做总结（并行）→ 把全部总结一次性交给模型 → 模型直接归纳出该父类别下的初始子类别（不再做 embedding 聚类），数量上限由 `max_initial_node_count` 控制（prompt 提示 + 兜底截断）。初始类别只是起点，后续 Batch Reflection 仍会按需新增类别。
- 逐层递归直到 `max_depth`；类别下案例数小于 `min_cases_to_split` 不再分裂。

> 字段使用约定（阶段一）：**案例分类**与**生成/汇总修改操作**时，只把每个类别的 `name` 与 `case_trigger` 交给模型（不含 `background`）。修改操作（modify）也只允许改 `name` 或 `case_trigger`；新增操作（add）则要求同时写 `name`、`case_trigger`、`background` 三个字段（提示词中已说明三字段含义）。`background` 与 `dialog_trigger` 主要留待阶段二基于对话数据优化。

### 阶段二：基于对话优化节点内容（`optimize.py`）

不新增/删除节点，只优化已有节点的 `dialog_trigger` 或 `background`：

```
对话导航(只看对话) → 收集背景 → 生成 query → 调用 retrieve 检索 (并行)
  → 失败则错误归因（哪个节点的 trigger / background 出问题）
  → 按节点聚合错误样本成 Batch
  → 对每个 Batch 反思 → 修改 dialog_trigger / background
  → 用训练/验证对话验证召回率是否提高 → 接受 / 反馈重试
```

检索调用根目录 `retrieve.py::retrive`（当前为未实现的桩，框架已预留调用方式）。

## 配置说明（`config.py`）

所有需求中要求“可配置”的项都在 `config.py`：

| 配置 | 字段 | 说明 |
|------|------|------|
| 运行阶段 | `RUNTIME.run_stage` | `build` 只构建 / `all` 构建后接着优化（不带子命令时生效） |
| 案例库路径 | `PATHS.case_path` | |
| 对话训练/验证集路径 | `PATHS.dialog_train_path` / `dialog_val_path` | |
| L1 种子类别路径 | `PATHS.seed_l1_path` | |
| LLM provider | `LLM.provider` | `mock`（离线启发式）/ `openai` |
| LLM 接口 | `LLM.base_url` / `api_key` / `model` | |
| 并发数 | `LLM.concurrency` | 分类、反思、导航等并行度 |
| 最大树深 | `BUILD.max_depth` | |
| Batch 大小 | `BUILD.batch_size` | |
| 每 Batch 无法归类案例数 | `BUILD.unknown_per_batch` | |
| MaxNodeCount | `BUILD.max_node_count` | Complexity Check 上限 |
| 初始类别数上限 | `BUILD.max_initial_node_count` | L2+ 归纳初始子类别时最多产出的类别数 |
| 覆盖率验证重试次数 | `BUILD.max_plan_retries` | |
| 复杂度检查重试次数 | `BUILD.max_complexity_retries` | |
| 不再分裂阈值 | `BUILD.min_cases_to_split` | 类别下案例数低于此值不向下分层 |
| 优化反思重试次数 | `OPTIMIZE.max_reflection_retries` | |

## 关于 provider

- `mock`（默认）：不联网，用内置领域关键词启发式模拟所有 LLM 决策，可在无模型环境下完整跑通两个阶段并产出全部中间结果，便于调试与回归。
- `openai`：把 `config.py` 的 `LLM.provider` 改为 `"openai"`，并填好 `base_url` / `api_key` / `model` 即可切换到真实大模型；提示词见 `khtree/prompts.py`。

## 健壮性

- **错误隔离**：单条案例/对话处理出错不会中断整体流程，失败项被跳过并记录到 `output/errors.log`。
- **中间产物**：每个阶段都按序号写出 JSON 到 `output/intermediate/`，便于排查“跑到哪一步、结果是什么”。
- **进度可视**：命令行用 `tqdm` 进度条 + 带时间戳的阶段日志输出。
- **续跑**：阶段一可从任意层中间树继续；阶段二可独立基于已构建的树运行。
```
