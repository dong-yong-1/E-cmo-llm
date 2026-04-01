# 电商客服大模型评估设计文档

## 1. 评估目标

本项目的评估目标不是只看训练 loss，而是回答一个更接近真实业务的问题：

“模型生成的客服回复，是否遵守平台规则、能否解决用户问题、是否优于基座模型或其他 LoRA 配置？”

因此，评估体系需要同时覆盖：

- 规则正确性
- 回复可用性
- 相对胜率
- badcase 分析

## 2. 评估设计原则

本项目评估体系遵循以下原则：

- 业务优先：优先评估规则遵守和客服可用性
- 对比优先：强调候选模型相对 baseline 的提升
- 分桶分析：支持按场景、难度和错误类型统计
- 结果可追溯：保留逐样本明细，支持 badcase 回看

## 3. 当前评估脚本

当前评估脚本位于 [scripts/evaluate.py](/Users/dongyong/Project/Trea_code/ecom-llm-cs/scripts/evaluate.py)。

该脚本支持：

- 加载评测集
- 加载候选模型和 baseline 模型
- 对每条样本生成回复
- 使用 judge 模型做规则评估和 pairwise 对比
- 输出汇总报告和逐样本明细

输出文件默认包括：

- [reports/eval\_report.json](/Users/dongyong/Project/Trea_code/ecom-llm-cs/reports/eval_report.json)
- [reports/eval\_details.json](/Users/dongyong/Project/Trea_code/ecom-llm-cs/reports/eval_details.json)

## 4. 评估流程

完整评估流程如下：

1. 读取评测集样本
2. 将样本统一升级为 `schema v1`
3. 将 `instruction + input` 构造成模型输入 prompt
4. 用候选模型生成回复
5. 用 baseline 模型生成回复
6. 调用 judge 模型评估候选模型回复的规则正确性与可用性
7. 调用 judge 模型比较候选模型与 baseline 的 pairwise 胜负
8. 汇总规则准确率、胜率、错误分布和 badcase

## 5. 评估指标设计

### 5.1 Rule Accuracy

定义：

- 候选模型回复是否符合平台规则
- 是否满足样本中的 `policy_tags`
- 是否存在乱承诺、错判规则、忽略业务约束等问题

这是本项目最核心的指标，因为电商客服任务首先是业务规则受限任务，而不是纯开放式生成任务。

### 5.2 Reference Match Rate

定义：

- 模型回复是否与参考答案在处理方向上基本一致

该指标用于辅助判断模型是否偏离了预期解决路径，但优先级低于 `rule_accuracy`。因为参考答案只是一个标准写法，而不是唯一正确答案。

### 5.3 Helpfulness Score

定义：

- 模型回复对用户是否有帮助，评分范围为 1 到 5

主要看：

- 是否真正回答了问题
- 是否给出明确处理路径
- 是否避免空泛表述

### 5.4 Politeness Score

定义：

- 回复是否礼貌、自然，评分范围为 1 到 5

尤其适用于投诉、安抚类样本。

### 5.5 Overall Score

定义：

- 综合评分，评分范围为 1 到 5

综合考虑：

- 规则是否正确
- 是否解决问题
- 是否自然礼貌
- 是否简洁清晰

### 5.6 Win Rate vs Baseline

定义：

- 使用 pairwise 对比方式，判断候选模型回复是否优于 baseline

计算方式：

```text
win_rate = (wins + 0.5 * ties) / total
```

其中：

- `wins`
  - 候选模型优于 baseline 的样本数
- `ties`
  - 候选模型与 baseline 持平的样本数

该指标适合用于：

- 微调模型 vs 基座模型
- 不同 LoRA rank 对比
- 不同 target modules 对比

## 6. Judge 评估逻辑

当前评估中使用 judge 模型完成两类判断：

### 6.1 单回复评估

judge 根据以下信息做判断：

- 用户问题
- 结构化输入上下文
- 平台规则
- `policy_tags`
- 参考答案
- 候选模型回复

输出字段包括：

- `rule_correct`
- `reference_match`
- `helpfulness_score`
- `politeness_score`
- `overall_score`
- `error_type`
- `reason`

### 6.2 Pairwise 对比评估

judge 比较 A 和 B 两个回复，输出：

- `winner`
  - 取值为 `A`、`B` 或 `Tie`
- `reason`

优先比较维度为：

1. 是否遵守平台规则
2. 是否真正解决用户问题
3. 是否礼貌自然
4. 是否表达简洁

## 7. Error Type 设计

为了支撑 badcase 分析，当前评估将错误类型归类为：

- `none`
- `规则错判`
- `过度承诺`
- `信息缺失乱答`
- `安抚不足`
- `回复冗长`
- `其他`

这套错误分类直接服务于项目后续优化。例如：

- `规则错判` 多，说明规则数据或规则约束不够强
- `过度承诺` 多，说明模型过于迎合用户
- `信息缺失乱答` 多，说明模型需要更强的追问或保守回复能力
- `安抚不足` 多，说明投诉场景数据和训练信号不足

## 8. 分桶统计设计

为了避免只看单一总体分数，评估脚本支持按以下维度分桶：

- `category`
- `subcategory`
- `difficulty`
- `error_type`

这样的价值在于：

- 可以定位哪一类业务场景最弱
- 可以定位哪种难度最容易失败
- 可以识别 LoRA 配置在不同子任务上的收益差异

## 9. Badcase 输出设计

评估报告会保留所有规则错误样本，输出信息包括：

- 样本 `id`
- `category`
- `subcategory`
- `difficulty`
- `instruction`
- `candidate_response`
- `reference_output`
- `error_type`
- `reason`

这样做的意义是：

- 支持人工复查
- 支持归因分析
- 支持从 badcase 反推数据增强方向

## 10. 当前评估方式的价值

相较于只看训练日志或 loss，这套评估方式的价值主要体现在：

- 可以衡量业务规则是否真的学到了
- 可以比较不同 LoRA 配置的实际效果差异
- 可以量化模型是否优于基座模型
- 可以系统输出 badcase，而不是只凭主观感受

这使项目从“跑通训练”升级为“具备实验评估闭环”。

## 11. 当前评估方式的局限性

当前评估设计仍存在一些局限：

- judge 模型本身可能存在打分偏差
- 若评测集仍来自合成数据，可能和训练分布过于接近
- 单轮客服回复评估尚未覆盖多轮追问场景
- 当前规则准确率仍依赖 judge，而不是完全自动化规则引擎

因此，后续仍建议补充：

- 一批人工构造且人工审核的高质量 eval set
- 少量人工抽检结果
- 规则关键词或规则引擎辅助校验

## 12. 推荐实验用法

该评估体系特别适合以下几类实验：

### 12.1 微调模型 vs 基座模型

目标：

- 验证 SFT + LoRA 是否真正提升客服可用性

关注指标：

- `rule_accuracy`
- `win_rate_vs_baseline`

### 12.2 LoRA Rank 消融实验

目标：

- 比较 `r=8`、`r=16`、`r=32` 的差异

关注指标：

- `rule_accuracy`
- `avg_overall_score`
- `win_rate_vs_baseline`
- `by_difficulty`

### 12.3 Target Modules 消融实验

目标：

- 比较 `q_proj,v_proj` 与 `q_proj,k_proj,v_proj,o_proj`

关注指标：

- `rule_accuracy`
- `error_type_distribution`
- `badcases`

### 12.4 数据规模对比实验

目标：

- 比较 `120`、`1000`、`3000+` 样本下模型表现变化

关注指标：

- `rule_accuracy`
- `reference_match_rate`
- `win_rate_vs_baseline`
- `badcases`

## 13. 小结

本项目的评估设计已经从单纯观察训练 loss，升级为围绕业务规则、客服可用性和相对胜率展开的综合评估体系。该体系能够为后续 LoRA 消融实验、模型对比实验和 badcase 分析提供稳定支撑，也更符合真实电商客服大模型项目的评估方式。
