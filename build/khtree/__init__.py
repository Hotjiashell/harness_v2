# -*- coding: utf-8 -*-
"""层级化知识树构建与优化框架。

包结构：
  models      数据结构（Node / Tree / Case / Dialog / Operation / UpdatePlan）
  utils       日志、IO、错误记录、并发等工具
  llm         异步 LLM 客户端（openai / mock 两种 provider）
  prompts     所有提示词模板
  build_tree  阶段一：基于案例库构建知识树
  optimize    阶段二：基于对话数据优化节点内容
"""
