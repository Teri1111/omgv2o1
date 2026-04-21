# T14: 多工具组合与自验证闭环 实施计划

## 目标
让系统出现真正的多工具链路，而不是"每轮直接join"。

## 当前状态分析
1. T11 已完成 observation tool 执行通路
2. T12 已完成 Experience KB 主动查询闭环
3. T13 已完成非JOIN动作覆盖恢复
4. 当前 observation tool 可以执行，但缺少与主推理流程的深度集成

## 实施步骤

### 1. 修改 `reasoning/agent.py`

#### 1.1 增加多工具链路支持
在 `LLMGuidedAgent.run()` 中，需要支持以下模式：

**模式1: explore_neighbors -> extend_expression**
```python
# 在 step 循环中，如果上一步是 explore_neighbors observation
if self._last_observation_tool == "explore_neighbors":
    # 从 observation 结果中提取发现的关系
    discovered_relations = self._extract_relations_from_observation(self._last_observation_result)
    if discovered_relations:
        # 将发现的关系加入可用关系列表
        # 在下一轮 step 中，LLM 可以选择 extend_expression 使用这些关系
```

**模式2: verify_expression -> revise next action**
```python
# 在 step 循环中，如果上一步是 verify_expression observation
if self._last_observation_tool == "verify_expression":
    # 从 observation 结果中获取验证结果
    verification_result = self._last_observation_result
    if verification_result.get("valid") == False:
        # 验证失败，设置 last_failure 并允许重试
        self._last_failure = "Expression verification failed: " + verification_result.get("error", "")
        # 清除当前表达式中的错误部分
        self._rollback_last_action()
```

**模式3: inspect_path -> decide whether to adopt path-guided draft**
```python
# 在 step 循环中，如果上一步是 inspect_path observation
if self._last_observation_tool == "inspect_path":
    # 从 observation 结果中获取路径信息
    path_info = self._last_observation_result
    if path_info and path_info.get("confidence", 0) > 0.7:
        # 路径置信度高，考虑采用 path-guided draft
        self._consider_path_guided_draft(path_info)
```

#### 1.2 增加 observation 结果跟踪
在 `LLMGuidedAgent` 中增加：
```python
self._last_observation_tool = None
self._last_observation_result = None
self._observation_history = []  # 记录所有 observation 结果
```

#### 1.3 修改 scratchpad 记录
确保 observation 与 step action 交替时，scratchpad 显式记录 Thought / Action / Observation：
```python
# 在 observation 执行后
obs_n = self.scratchpad.count("Observation") + 1
self.scratchpad += "Observation" + str(obs_n) + ": " + str(observation_text) + "\n"

# 在 step action 执行前
step_n = self.scratchpad.count("Thought") + 1
self.scratchpad += "Thought" + str(step_n) + ": " + thought + "\n"
self.scratchpad += "Action" + str(step_n) + ": " + action_description + "\n"
```

### 2. 修改 `reasoning/llm_agent.py`

#### 2.1 修改 prompt 构建
在 `choose_next_step_function_call()` 中，增加多工具链路的指导：
```python
# 在 prompt 中增加多工具组合的示例
"Example multi-tool sequence:",
"1. explore_neighbors to discover relations: {\"tool\": \"explore_neighbors\", \"args\": {\"entity\": \"some_entity\", \"direction\": \"forward\"}, \"thought\": \"...\"}",
"2. extend_expression using discovered relation: {\"tool\": \"extend_expression\", \"args\": {\"action\": \"join\", \"relation\": \"discovered_relation\", \"direction\": \"forward\"}, \"thought\": \"...\"}",
"",
"3. verify_expression to check current work: {\"tool\": \"verify_expression\", \"args\": {\"expression\": \"current_expression\", \"mode\": \"partial\"}, \"thought\": \"...\"}",
"4. extend_expression to revise if needed: {\"tool\": \"extend_expression\", \"args\": {\"action\": \"join\", \"relation\": \"corrected_relation\", \"direction\": \"forward\"}, \"thought\": \"...\"}",
```

#### 2.2 增加上下文感知
在 prompt 中注入上一步的 observation 结果：
```python
if last_observation_tool and last_observation_result:
    state_lines.append("Last observation tool: " + last_observation_tool)
    state_lines.append("Last observation result: " + str(last_observation_result)[:200])
```

### 3. 修改 `skills/tools/adapters.py`

#### 3.1 增强 explore_neighbors_adapter 返回值
```python
def explore_neighbors_adapter(entity, direction="forward", filter_pattern=None, subgraph=None, **kwargs):
    """增强返回值，包含发现的关系列表。"""
    # 现有逻辑...
    
    # 增加关系发现功能
    discovered_relations = []
    for rel, targets in neighbors.items():
        discovered_relations.append({
            "relation": rel,
            "targets": targets,
            "direction": direction
        })
    
    return {
        "neighbors": neighbors,
        "discovered_relations": discovered_relations,
        "count": len(neighbors)
    }
```

#### 3.2 增强 verify_expression_adapter 返回值
```python
def verify_expression_adapter(expression, mode="partial", **kwargs):
    """增强返回值，包含验证细节。"""
    # 现有逻辑...
    
    # 增加验证细节
    return {
        "valid": result.get("valid", False),
        "error": result.get("error", ""),
        "warnings": result.get("warnings", []),
        "suggestions": result.get("suggestions", []),  # 修复建议
        "partial_result": result.get("partial_result", None)
    }
```

### 4. 修改 `skills/tools/__init__.py`

#### 4.1 更新 Tool Schema
确保所有 Tool 的返回值 schema 包含多工具链路所需的信息。

## 验收标准

### 单元测试
1. 测试 explore_neighbors -> extend_expression 链路
2. 测试 verify_expression -> revise next action 链路
3. 测试 inspect_path -> decide whether to adopt path-guided draft 链路
4. 测试 scratchpad 记录完整性

### 集成测试
1. 小样本（10）运行中，多工具调用次数 > 0
2. 不引入 observation 死循环；budget 超限时能安全回退
3. 原有回归测试全部通过

### 回归测试
```bash
cd /data/gt/omgv2-o1
/data/gt/envs/lf_gjq/bin/python3 -m py_compile reasoning/agent.py
/data/gt/envs/lf_gjq/bin/python3 -m py_compile reasoning/llm_agent.py
/data/gt/envs/lf_gjq/bin/python3 -m py_compile skills/tools/adapters.py
/data/gt/envs/lf_gjq/bin/python3 test_smoke.py
/data/gt/envs/lf_gjq/bin/python3 test_closed_loop.py 3
```

## 风险控制

### 高风险点
1. **observation 死循环** → 限制 observation 预算，增加死循环检测
2. **多工具链路复杂度** → 保持简单，先实现基本链路
3. **性能影响** → 监控 LLM 调用次数，必要时优化 prompt

### 回滚策略
- 每个步骤独立提交，可单独回滚
- 保留 `phase6-t12-experience-kb` 分支作为基线

## 时间估算
- 修改 agent.py 增加多工具链路支持: 3 小时
- 修改 llm_agent.py 增加上下文感知: 1.5 小时
- 修改 adapters.py 增强返回值: 1.5 小时
- 编写测试: 2 小时
- 联调和验证: 2 小时
- **总计: 10 小时**

## 下一步行动
1. 修改 `reasoning/agent.py` 增加多工具链路支持
2. 修改 `reasoning/llm_agent.py` 增加上下文感知
3. 修改 `skills/tools/adapters.py` 增强返回值
4. 编写测试用例
5. 跑回归测试
6. 完成后通知 PM 审查