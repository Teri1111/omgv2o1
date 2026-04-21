# T12-T15 审查问题修复计划

## 问题清单

### 问题1：T13的AND/CMP生成坏表达式（高严重性）

**现状**：
- `_do_and(relation)` 生成 `AND(expression, "relation")` - 缺少第二个表达式
- `_do_cmp(relation)` 生成 `CMP(expression, "relation")` - 缺少operator和正确格式

**正确格式**（来自lf_construction.py和extend_expression_tool.py）：
- AND：`AND(expression1, expression2)` - 需要两个表达式
- CMP：`CMP("operator", "relation", expression)` - 需要operator、relation、expression

**修复方案**：
1. 修改`_do_and()`：需要两个表达式ID，生成`AND(expr1, expr2)`
2. 修改`_do_cmp()`：需要operator参数，生成`CMP("operator", "relation", expr)`
3. 修改`_dispatch_action()`中对and/cmp的调用，传递正确参数

### 问题2：T15的统计聚合逻辑错误（高严重性）

**现状**：
- `tool_usage_stats["extend_expression"]`永远为0，因为只从`observation_tool`聚合
- `kb_usage_stats`没有从`kb_stats`同步

**修复方案**：
1. 在`finalize()`中，根据`action`字段统计`extend_expression`调用次数
2. 将`kb_stats`同步到`kb_usage_stats`

### 问题3：T12没有真正实现"consult后重试"（中严重性）

**现状**：
- consult后只是清除了`_last_failure`，没有真正重试

**修复方案**：
1. 在consult成功后，增加一个重试机制
2. 可以在当前step内重新调用`_decide_action()`并执行

## 修复顺序

1. 先修问题1（AND/CMP）- 这是功能错误
2. 再修问题2（统计）- 这影响T15验收
3. 最后修问题3（重试）- 这是功能增强

## 验证标准

### 问题1验证
- `_do_and("rel1")`生成合法S-expression（需要两个表达式）
- `_do_cmp("GE", "rel1")`生成合法S-expression
- 新增单测覆盖AND/CMP的S-expression生成

### 问题2验证
- 只含join_forward的trace，`tool_usage_stats["extend_expression"]`>0
- 有kb_stats的trace，`kb_usage_stats`正确同步

### 问题3验证
- 失败后consult，然后有重试动作
- 新增单测覆盖重试逻辑