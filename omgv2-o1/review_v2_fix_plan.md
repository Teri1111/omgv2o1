# T12-T15 审查报告 v2 修复计划

## 问题清单

### 问题1：T12的consult后重试没有回滚失败动作（高严重性）

**现状**：
- `_dispatch_action()`已执行并修改了`function_list`和`selected_relations`
- 检测到空结果后触发consult
- consult成功后直接`continue`进入下一次循环，没有回滚失败动作

**修复方案**：
在consult重试前，回滚失败动作：
1. 保存当前`function_list`和`selected_relations`的快照
2. 如果需要重试，回滚到快照状态
3. 然后重新`_decide_action()`并执行

### 问题2：T13的CMP操作符推断顺序错误（中严重性）

**现状**：
```python
def _infer_cmp_operator(self):
    q = self.question.lower()
    if any(w in q for w in ["greater than", "more than", "older than"]):
        return "gt"
    if any(w in q for w in ["less than", "younger than"]):
        return "lt"
    if any(w in q for w in ["greater than or equal", "at least"]):
        return "ge"
    if any(w in q for w in ["less than or equal", "at most"]):
        return "le"
    return "gt"
```

问题："greater than or equal"包含"greater than"子串，会被第一个条件匹配。

**修复方案**：
调整判断顺序，先判断更长的短语：
```python
def _infer_cmp_operator(self):
    q = self.question.lower()
    # 先判断更长的短语
    if any(w in q for w in ["greater than or equal", "at least"]):
        return "ge"
    if any(w in q for w in ["less than or equal", "at most"]):
        return "le"
    # 再判断更短的短语
    if any(w in q for w in ["greater than", "more than", "older than"]):
        return "gt"
    if any(w in q for w in ["less than", "younger than"]):
        return "lt"
    return "gt"
```

### 问题3：T11回归测试失败3项（中严重性）

**现状**：
- test_t11_observation.py中的3个测试失败
- 原因：新的重试逻辑改变了控制流，导致mock iterator不够用

**修复方案**：
1. 查看失败测试的具体内容
2. 更新测试以适应新的控制流
3. 或者修复代码以保持向后兼容

## 修复顺序

1. 先修问题2（CMP操作符推断）- 简单且独立
2. 再修问题1（重试回滚）- 核心功能
3. 最后修问题3（T11测试）- 需要理解测试逻辑

## 验证标准

### 问题2验证
- "greater than or equal to 1000" → "ge"
- "less than or equal to 1000" → "le"
- "greater than 1000" → "gt"
- "less than 1000" → "lt"

### 问题1验证
- 失败动作后consult，function_list回滚到失败前状态
- 重试动作基于回滚后的状态执行
- 最终function_list不包含失败动作

### 问题3验证
- test_t11_observation.py全部通过