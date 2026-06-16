# 层级化知识树构建与优化框架

# 1. 目标

构建一个层级化、可导航的知识树（Knowledge Tree），用于知识路由与案例检索。

知识树中的每个节点包含四个字段：

```json
{
  "name": "类别名称",
  "case_trigger": "针对案例，什么时候考虑该类别",
  "dialog_trigger": "针对对话，什么时候考虑该类别，如当用户提到什么时",
  "background": "领域内相关背景知识，如术语定义等",
}
```
可参考output/seed_L1.json

最终形成如下层级结构：

```text
Root
 ├── L1
 │    ├── L2
 │    └── ...
 └── ...
```

# 2. 输入数据

系统使用两类数据：
1. 案例库，形如：data/case/text.json。需要能在config.py中配置路径，每条案例代表一个真实问题及对应处理方式。
2. 对话数据，形如：data/dialog/dialog.json。分为训练集和验证集，都需要能在config.py中配置路径。每条对话都对应一个ground truth案例，用于后续优化知识节点内容。

# 3. 整体流程
整体分为两个阶段：
```text
先基于案例数据，逐层构建完整知识树
  ↓
再基于对话数据，优化知识节点内容
```

其中，知识树构建阶段对于每一层：
```text
人工定义初始类别（L1）或聚类得到初始类别（L2及以后）
  ↓
利用案例数据优化知识树结构
  ↓
按同样方式优化下一层
```

# 4. 根据案例数据优化知识结构
## 指定初始类别
L1构建时的初始类别是人工定义的，如output/seed_L1.json，需要能在config.py中配置。
L2及以后的初始类别是聚类得到的，后面详细介绍。


## 案例分类

使用种子初始类别对当前层待处理案例进行分类，调用LLM判断案例是否属于已有当前层类别。需要输出判断结果和理由，需要并行，需要输出中间处理结果。


## Batch Reflection

将案例划分为多个 Batch。

每个 Batch 主要包含一些无法归类到初始类别的案例，外加部分已经成功分类的案例。每个batch的大小，无法归类的到初始类别案例的数目要能够在config.py里指定。
模型分析 Batch 后生成一系列修改操作Proposal。分析Batch应该并行。

仅允许两种修改操作：1. 新增一个类别；2. 修改现有类别的name, case_trigger或background


## Proposal Aggregation

多个 Batch 会产生多个 Proposal。

由模型汇总这些proposal，有些修改操作可以归并为一种操作，得到最终的一系列修改操作Update Plan

## Complexity Check

在执行 Update Plan 之前进行复杂度检查。

计算：

```python
new_node_count =
current_node_count +
number_of_add_operations
```

如果：

```python
new_node_count > MaxNodeCount # MaxNodeCount要能在config.py里指定
```

则拒绝当前 Update Plan。并生成反馈：

```text
当前修改方案新增类别过多。

请重新审视新增类别，
总结能够覆盖多个Add的更高级别抽象的Add，
减少 Add 操作数量。
```

然后重新生成 Update Plan。


## Coverage Validation

通过 Complexity Check 后：

1. 试执行 Update Plan
2. 重新分类 Unknown Case
3. 验证全部Unknown Case都能成功分类

如果能则接受plan，不能就不接受plan，并生成反馈让模型重新调整update plan，直到达到最大重试次数（要能配置）。


完整流程如下：

```text
Case Data
    ↓
Classification
    ↓
Batch Reflection
    ↓
Proposal Generation
    ↓
Proposal Aggregation
    ↓
Update Plan
    ↓
Complexity Check
    ↓
Coverage Validation
    ↓
Accept / Feedback
```

# 5. L2及以后子类别初始化

L2 及以下层级不采用人工定义，而是通过聚类自动生成初始类别。首先根据父节点的类别名称和background，对每个案例进行总结。再对总结进行embedding聚类。然后让模型对每个聚类总结出1-3个类别，最后让模型同时看所有类别，总结出最终初始类别。聚类函数见cluster.py的参数和返回格式，不需要进行实现。要兼容k-means和HDBSCAN聚类，默认采用K-Means聚类。聚类参数可在config.py中指定。

# 6. 逐层处理

处理完L1类别后，就对L1的某个类别下的案例，采用同样的处理方式构建下一层，如此逐层，直到达到最大树深，最大树深可在config.py中指定。

# 7. 利用对话数据优化节点内容

在基于案例数据把整个知识结构建立起来后，再使用对话数据优化节点内容。这里不再新增或删除节点，而是优化已有节点的dialog_trigger或background。

## 对话导航

让模型只看对话内容，基于当前知识树进行导航。导航完成后，模型会看到一系列节点的背景知识。

## Query生成与案例检索

模型结合导航过程中获得的所有节点知识，生成一个query，调用retrieve.py检索案例。retrieve.py这部分当前不需要实现，只需要在流程中预留调用方式。

## 错误归因

如果检索失败，则让模型分析是哪个节点的trigger还是background出了问题，trigger导致选不对正确的类别，background导致提供的用于生产query地知识或经验不够。以上生成query、检索、反思问题的过程也要并行。

## Batch构建

按照节点聚合错误样本，同一个节点对应的错误归到同一个batch。

```text
Node A Batch
Node B Batch
...
```

## 错误反思

反思同一个batch里的所有问题，分析究竟应该如何改进对应节点的dialog_trigger或background，形成修改操作。

## 修改验证

使用对话训练数据和验证数据验证修改效果，看案例检索召回成功率是否提高。

如果提高，则接受修改操作；如果没有提高，则不接受，并生成反馈让模型继续调整。

# 8. 最终产物

输出一个可序列化的知识树：

```json
{
  "name": "Root",
  "children": [
    {
      "name": ...,
      "case_trigger": ...,
      "dialog_trigger": ...,
      "background": ...,
      "children": [
        {
          "name": ...,
          "case_trigger": ...,
          "dialog_trigger": ...,
          "background": ...,
          ...
        }
      ]
    }
  ]
}
```

# 9. 其他说明
要有详细的中间输出结果便于调试，比如跑完基于案例库知识构建是什么样的。

把基于案例库构建知识结构和基于对话优化节点内容分开实现。先做前面的在做后面的。

要能续跑，比如指定某一层基于案例库知识构建的中间文件，可以接着往后跑；也可以在知识树构建完成后，接着跑基于对话数据的节点内容优化。

要有错误处理机制，不能因为一个样例出现错误导致整个处理流程崩掉。要有错误记录。

要在命令行输出调试信息，方便追踪进度，比如当前跑到什么阶段了，进度怎么样（可以用tqdm），什么时间跑的什么。

你生成的代码放在build文件夹里
