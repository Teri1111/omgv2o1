# T13: 非JOIN动作覆盖恢复 实施计划

## 目标
让系统除了 join_forward / join_reverse 之外，真正能走到 count / argmax / argmin / tc / cmp / and。

## 当前状态分析
1. `_dispatch_action()` 已支持 argmax、argmax、time_filter、count
2. `_decide_action()` 中有门控逻辑：`if self.step_count >= 2:` 才允许 count/argmax/argmin
3. 缺少 and、cmp、tc 的完整支持
4. LLM function calling 中缺少这些动作的解析

## 实施步骤

### 1. 修改 `reasoning/agent.py`

#### 1.1 修改 `_decide_action()` 门控逻辑
```python
def _decide_action(self, out_rels, in_rels):
    """Decide next action based on question and available relations."""
    # 关键词检测
    argmax_words = ["largest", "biggest", "highest", "greatest", "longest",
                    "last", "newest", "latest", "first", "oldest", "most recent",
                    "most famous", "most popular"]
    argmin_words = ["least", "smallest", "fewest", "lowest", "shortest", "earliest"]
    is_count = "how many" in self.question or "number of" in self.question
    is_argmax = any(w in self.question for w in argmax_words)
    is_argmin = any(w in self.question for w in argmin_words)
    is_tc = "when" in self.question or "time" in self.question or "date" in self.question
    
    # 放开门控：不再要求 step_count >= 2
    if out_rels:
        # 优先检测明确意图
        if is_count:
            return "count"
        if is_argmax and self._has_literal_relation(out_rels):
            return "argmax"
        if is_argmin and self._has_literal_relation(out_rels):
            return "argmin"
        if is_tc and self._has_literal_relation(out_rels):
            return "time_filter"
        # 默认 join
        return "join_forward"
    
    if in_rels:
        return "join_reverse"
    
    return "finish"
```

#### 1.2 修改 `_dispatch_action()` 增加 and/cmp 支持
```python
def _dispatch_action(self, action, filtered_out, filtered_in):
    """Dispatch an action name to the appropriate skill-backed method."""
    # 现有逻辑...
    
    # 新增 and 支持
    if action == "and" and filtered_out:
        # AND 需要两个表达式，这里简化处理
        # 实际实现需要维护 expression1 和 expression2
        rel = self._choose_join_relation(filtered_out, reverse=False)
        if rel:
            self._do_and(rel["relation"])
            return {"relation": "AND:" + rel["relation"]}
    
    # 新增 cmp 支持
    if action == "cmp" and filtered_out:
        rel = self._find_literal_relation(filtered_out)
        if rel:
            self._do_cmp(rel)
            return {"relation": "CMP:" + rel}
    
    return None
```

#### 1.3 增加 `_do_and()` 和 `_do_cmp()` 方法
```python
def _do_and(self, relation):
    """Execute AND operation."""
    eid = self.expression_id
    # AND 需要两个表达式，这里简化为 JOIN + AND
    fl = 'expression' + eid + ' = AND(expression' + eid + ', "' + relation + '")'
    self.function_list.append(fl)
    self.selected_relations.append("AND:" + relation)

def _do_cmp(self, relation):
    """Execute CMP (comparison) operation."""
    eid = self.expression_id
    fl = 'expression' + eid + ' = CMP(expression' + eid + ', "' + relation + '")'
    self.function_list.append(fl)
    self.selected_relations.append("CMP:" + relation)
```

### 2. 修改 `reasoning/llm_agent.py`

#### 2.1 修改 `choose_next_step_function_call()` 增加非JOIN动作支持
在 prompt 中增加示例：
```python
# 在 prompt_lines 中增加
'For counting: {"tool": "extend_expression", "args": {"action": "count"}, "thought": "..."}',
'For argmax: {"tool": "extend_expression", "args": {"action": "argmax", "relation": "some.literal.relation"}, "thought": "..."}',
'For argmin: {"tool": "extend_expression", "args": {"action": "argmin", "relation": "some.literal.relation"}, "thought": "..."}',
'For time filter: {"tool": "extend_expression", "args": {"action": "tc", "relation": "some.date.relation"}, "thought": "..."}',
'For and: {"tool": "extend_expression", "args": {"action": "and", "relation": "some.relation"}, "thought": "..."}',
'For comparison: {"tool": "extend_expression", "args": {"action": "cmp", "relation": "some.literal.relation"}, "thought": "..."}',
```

#### 2.2 修改解析逻辑
在 `choose_next_step_function_call()` 中增加解析：
```python
elif action_arg in ("argmax", "argmin"):
    parsed = {"action": action_arg, "relation": tool_args.get("relation"), "thought": tool_thought}
elif action_arg == "tc":
    parsed = {"action": "time_filter", "relation": tool_args.get("relation"), "thought": tool_thought}
elif action_arg == "and":
    parsed = {"action": "and", "relation": tool_args.get("relation"), "thought": tool_thought}
elif action_arg == "cmp":
    parsed = {"action": "cmp", "relation": tool_args.get("relation"), "thought": tool_thought}
```

### 3. 修改 `skills/tools/extend_expression_tool.py`

#### 3.1 增加 and/cmp 支持
```python
def extend_expression(action, relation=None, direction="forward", target_expr=None, **kwargs):
    """Extend the current logical form expression."""
    # 现有逻辑...
    
    # 增加 and 支持
    elif action == "and" and relation:
        result = _do_and_extension(relation, target_expr)
    
    # 增加 cmp 支持
    elif action == "cmp" and relation:
        result = _do_cmp_extension(relation, target_expr)
```

### 4. 修改 `skills/tools/__init__.py`

#### 4.1 更新 Tool Schema
在 `extend_expression` Tool Schema 中增加 action 选项：
```json
{
  "name": "extend_expression",
  "parameters": {
    "properties": {
      "action": {
        "type": "string",
        "enum": ["join", "count", "finish", "argmax", "argmin", "tc", "and", "cmp"],
        "description": "Action type to perform"
      }
    }
  }
}
```

## 验收标准

### 单元测试
1. 测试 count、argmax、argmin、tc、and、cmp 动作解析
2. 测试 `_decide_action()` 门控逻辑修改
3. 测试 `_dispatch_action()` 新增动作支持

### 集成测试
1. 小样本（10）运行中，至少一种非 join action 出现非 0 调用
2. 定向样本测试：针对 count/argmax/argmin 问题的样本
3. 原有测试全部通过

### 回归测试
```bash
cd /data/gt/omgv2-o1
/data/gt/envs/lf_gjq/bin/python3 -m py_compile reasoning/agent.py
/data/gt/envs/lf_gjq/bin/python3 -m py_compile reasoning/llm_agent.py
/data/gt/envs/lf_gjq/bin/python3 -m py_compile skills/tools/extend_expression_tool.py
/data/gt/envs/lf_gjq/bin/python3 test_smoke.py
/data/gt/envs/lf_gjq/bin/python3 test_closed_loop.py 3
```

## 风险控制

### 高风险点
1. **非JOIN动作产生脏表达式** → 保留 execute_partial/execute_final 约束
2. **门控逻辑修改导致误判** → 保留关键词检测，增加置信度阈值
3. **and/cmp 实现复杂** → 先实现简化版本，后续优化

### 回滚策略
- 每个步骤独立提交，可单独回滚
- 保留 `phase6-t12-experience-kb` 分支作为基线

## 时间估算
- 修改 agent.py 门控逻辑和 _dispatch_action(): 2 小时
- 修改 llm_agent.py 解析逻辑: 1 小时
- 修改 extend_expression_tool.py: 1 小时
- 修改 Tool Schema: 0.5 小时
- 编写测试: 1.5 小时
- 联调和验证: 2 小时
- **总计: 8 小时**

## 下一步行动
1. 修改 `reasoning/agent.py` 门控逻辑和 _dispatch_action()
2. 修改 `reasoning/llm_agent.py` 解析逻辑
3. 修改 `skills/tools/extend_expression_tool.py` 增加 and/cmp 支持
4. 修改 `skills/tools/__init__.py` 更新 Tool Schema
5. 编写测试用例
6. 跑回归测试
7. 完成后通知 PM 审查