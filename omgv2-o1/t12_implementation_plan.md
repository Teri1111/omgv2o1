# T12: Experience KB 主动查询闭环 实施计划

## 目标
把 Experience KB 从"每步被动注入"改成"失败时显式 consult，并允许带 KB guidance 重试 1 次"。

## 当前状态分析
- T11 已完成 observation tool 执行通路，`consult_experience` 工具已可执行
- 当前 `search_experience_rules()` 在 `_decide_action()` 中被调用，用于被动注入
- 失败检测已有 `_last_failure` 机制，但未与 KB 查询联动
- 缺少"失败 → consult_experience → 重试"的闭环逻辑

## 实施步骤

### 1. 修改 `skills/experience_kb_skill.py`

#### 1.1 增加主动查询接口
```python
def consult_experience_tool(
    state_description: str,
    last_error: str = "",
    current_expr: str = "",
    available_relations: list = None,
    top_k: int = 3,
) -> dict:
    """主动查询 Experience KB，返回结构化结果供 observation tool 使用。"""
    # 复用 search_experience_rules 的逻辑，但返回结构化结果
    # 包含：matched_rules, guidance_text, rule_ids, confidence
```

#### 1.2 修改现有函数
- `search_experience_rules()` 增加 `trigger_mode` 参数：`"passive"`（默认）或 `"active"`（失败时调用）
- `active` 模式下降低阈值，增加 top_k，优先返回错误修复规则

### 2. 修改 `reasoning/agent.py`

#### 2.1 增加失败检测逻辑
在 `LLMGuidedAgent.run()` 中：
- 在 step 循环中增加失败检测
- 失败来源：
  1. `_last_failure` 非空（已有）
  2. `step_result.get("error")` 非空
  3. `step_answers` 为空且 `action` 不是 `finish`
  4. validation rejected（从 `_choose_join_relation` 中检测）

#### 2.2 增加 consult 触发逻辑
```python
# 在 step 循环中，action 执行后
if self._should_consult_experience(action, step_result):
    # 触发 consult_experience observation
    consult_result = self._execute_observation("consult_experience", {
        "state_description": f"Current entity: {self.current_entity}",
        "last_error": self._last_failure or "",
        "current_expr": function_list_to_sexpr(self.function_list),
        "available_relations": list(out_rels.keys()) + list(in_rels.keys())
    }, "Consulting Experience KB after failure")
    
    # 将 consultation 结果写入 scratchpad
    obs_n = self.scratchpad.count("Observation") + 1
    self.scratchpad += "Observation" + str(obs_n) + ": " + str(consult_result["observation"]) + "\n"
    
    # 允许重试一次（不增加 step_count）
    # 重试逻辑：根据 KB guidance 重新选择 action
    if consult_result.get("success"):
        # 从 consultation 结果中提取 guidance
        guidance = consult_result["observation"]
        # 设置 last_failure 为 None，允许重试
        self._last_failure = None
        # 在下一轮 step 中，LLM 会看到 guidance 并做出调整
```

#### 2.3 修改 trace 记录
在 `TraceCollector` 中增加：
- `passive_kb_injections`: 被动注入次数
- `active_kb_consultations`: 主动 consult 次数
- `kb_consultation_results`: 每次 consultation 的结果

### 3. 修改 `reasoning/llm_agent.py`

#### 3.1 修改 prompt 构建
在 `choose_next_step_function_call()` 中：
- 当有 KB consultation 结果时，在 prompt 中注入 guidance
- 区分"被动注入"和"主动 consult 后的 guidance"

#### 3.2 增加重试逻辑
- 当有 KB guidance 时，LLM 应优先考虑 guidance 建议
- 在 prompt 中明确提示："Based on Experience KB guidance: ..."

### 4. 修改 `skills/tools/adapters.py`

#### 4.1 修改 `consult_experience_adapter`
- 确保返回结构化结果，包含 `matched_rules`, `guidance_text`, `rule_ids`
- 增加 `confidence` 字段

## 验收标准

### 单元测试
1. 测试 `consult_experience_tool()` 返回结构化结果
2. 测试失败检测逻辑正确触发 consult
3. 测试重试逻辑（失败 → consult → 重试）
4. 测试 trace 区分被动注入和主动 consult

### 集成测试
1. 小样本（10）运行中，`consult_experience` 调用次数 > 0
2. trace 中 `active_kb_consultations` > 0
3. prompt 中不再出现 191/200 这种高频无差别被动注入
4. 原有测试全部通过

### 回归测试
```bash
cd /data/gt/omgv2-o1
/data/gt/envs/lf_gjq/bin/python3 -m py_compile reasoning/agent.py
/data/gt/envs/lf_gjq/bin/python3 -m py_compile reasoning/llm_agent.py
/data/gt/envs/lf_gjq/bin/python3 -m py_compile skills/experience_kb_skill.py
/data/gt/envs/lf_gjq/bin/python3 test_smoke.py
/data/gt/envs/lf_gjq/bin/python3 test_closed_loop.py 3
```

## 风险控制

### 高风险点
1. **无限 consult 循环** → 限制 consult 次数（每个 sample 最多 2 次）
2. **KB 查询性能** → 增加缓存，避免重复查询
3. **重试逻辑复杂** → 保持简单：失败 → consult → 重试一次 → 继续

### 回滚策略
- 每个步骤独立提交，可单独回滚
- 保留 `phase6-t11-observation` 分支作为基线

## 时间估算
- 修改 experience_kb_skill.py: 1 小时
- 修改 agent.py 失败检测和 consult 触发: 2 小时
- 修改 llm_agent.py prompt 逻辑: 1 小时
- 修改 adapters.py: 0.5 小时
- 编写测试: 1 小时
- 联调和验证: 1.5 小时
- **总计: 7 小时**

## 下一步行动
1. 创建分支 `phase6-t12-experience-kb` ✅
2. 修改 `skills/experience_kb_skill.py`，增加主动查询接口
3. 修改 `reasoning/agent.py`，增加失败检测和 consult 触发
4. 修改 `reasoning/llm_agent.py`，增加 KB guidance 注入
5. 修改 `skills/tools/adapters.py`，确保 consult_experience_adapter 返回结构化结果
6. 编写测试用例
7. 跑回归测试
8. 完成后通知 PM 审查