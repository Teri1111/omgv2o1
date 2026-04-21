# T12 详细实施步骤

## 第一阶段：修改 experience_kb_skill.py

### 1.1 增加主动查询函数
```python
def consult_experience_active(
    state_description: str,
    last_error: str = "",
    current_expr: str = "",
    available_relations: list = None,
    top_k: int = 5,  # 主动查询时增加 top_k
    threshold: float = 0.3,  # 主动查询时降低阈值
) -> dict:
    """主动查询 Experience KB，返回结构化结果。"""
    kb = _get_kb()
    if kb is None:
        return {
            "matched_rules": [],
            "guidance_text": "",
            "rule_ids": [],
            "confidence": 0.0,
            "query_type": "active_consult"
        }
    
    # 构建查询 - 优先使用错误信息
    queries = []
    if last_error:
        queries.append((f"error: {last_error}", 0.3))
    if available_relations:
        queries.append((f"relations: {', '.join(available_relations[:5])}", 0.35))
    if state_description:
        queries.append((state_description, 0.4))
    
    all_results = []
    seen_ids = set()
    for query_text, q_threshold in queries:
        try:
            results = kb.search(query_text, top_k=top_k, threshold=q_threshold)
            for r in results:
                rid = r.get("rule_id", r.get("title", ""))
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    all_results.append(r)
        except Exception:
            continue
    
    if not all_results:
        return {
            "matched_rules": [],
            "guidance_text": "",
            "rule_ids": [],
            "confidence": 0.0,
            "query_type": "active_consult"
        }
    
    # 按分数排序
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    top_results = all_results[:top_k]
    
    # 计算置信度
    avg_score = sum(r.get("score", 0) for r in top_results) / len(top_results)
    
    return {
        "matched_rules": top_results,
        "guidance_text": format_rules_for_prompt(top_results),
        "rule_ids": [r.get("rule_id", r.get("title", "")) for r in top_results],
        "confidence": avg_score,
        "query_type": "active_consult"
    }
```

### 1.2 修改现有函数
- `search_experience_rules()` 增加 `trigger_mode` 参数
- `passive` 模式：保持现有逻辑（阈值 0.4，top_k 3）
- `active` 模式：降低阈值到 0.3，增加 top_k 到 5

## 第二阶段：修改 agent.py

### 2.1 增加失败检测函数
```python
def _should_consult_experience(self, action, step_result):
    """判断是否需要主动查询 Experience KB。"""
    # 检查失败条件
    if self._last_failure:
        return True
    
    if step_result and step_result.get("error"):
        return True
    
    # 检查空结果（非 finish 动作）
    if action != "finish":
        step_answers = step_result.get("answers", []) if step_result else []
        if not step_answers and step_result.get("error") is None:
            return True
    
    # 检查 validation rejected（从 _choose_join_relation 中检测）
    # 这个逻辑需要在 _choose_join_relation 中增加返回值
    
    return False
```

### 2.2 增加 consult 触发逻辑
在 `LLMGuidedAgent.run()` 的 step 循环中：
```python
# 在 action 执行后，observation 写入前
if self._should_consult_experience(action, step_result):
    # 检查 consult 预算（每个 sample 最多 2 次）
    if self._observation_counts["consult_experience"] < self._observation_budget["consult_experience"]:
        # 触发 consult_experience observation
        consult_args = {
            "state_description": f"Current entity: {self.current_entity}",
            "last_error": self._last_failure or "",
            "current_expr": function_list_to_sexpr(self.function_list),
            "available_relations": list(out_rels.keys()) + list(in_rels.keys())
        }
        consult_result = self._execute_observation("consult_experience", consult_args, "Consulting KB after failure")
        
        if consult_result.get("success"):
            # 将 consultation 结果写入 scratchpad
            obs_n = self.scratchpad.count("Observation") + 1
            self.scratchpad += "Observation" + str(obs_n) + ": " + str(consult_result["observation"]) + "\n"
            
            # 更新 trace
            if step_trace is not None:
                step_trace["kb_consultation"] = {
                    "triggered": True,
                    "args": consult_args,
                    "result": consult_result["observation"],
                    "success": True
                }
            
            # 增加 consult 计数
            self._observation_counts["consult_experience"] += 1
            
            # 清除 last_failure，允许重试
            self._last_failure = None
```

### 2.3 修改 trace 记录
在 `TraceCollector.__init__()` 中增加：
```python
self.data["kb_stats"] = {
    "passive_injections": 0,
    "active_consultations": 0,
    "consultation_results": []
}
```

## 第三阶段：修改 llm_agent.py

### 3.1 修改 prompt 构建
在 `choose_next_step_function_call()` 中：
```python
# 检查是否有 KB consultation 结果
kb_guidance = ""
if scratchpad and "Consulting KB after failure" in scratchpad:
    # 从 scratchpad 中提取最近的 consultation 结果
    # 这里需要解析 scratchpad 获取 consultation 结果
    # 简化实现：在 prompt 中提示有 KB guidance
    kb_guidance = "Note: Experience KB consultation was performed after previous failure. Check scratchpad for guidance."

# 在 prompt 中注入 guidance
if kb_guidance:
    prompt_lines.insert(-1, kb_guidance)
```

### 3.2 增加重试逻辑
- 当有 KB guidance 时，LLM 应优先考虑 guidance 建议
- 在 prompt 中明确提示："Based on Experience KB guidance: ..."

## 第四阶段：修改 adapters.py

### 4.1 修改 consult_experience_adapter
```python
def consult_experience_adapter(
    state_description: str = "",
    last_error: str = "",
    current_expression: str = "",
    available_relations: List[str] = None,
    top_k: int = 3,
    query_type: str = "passive",  # 新增参数
    **kwargs
) -> Union[str, dict]:
    """Adapter for consult_experience Tool."""
    from skills.experience_kb_skill import search_experience_rules, consult_experience_active
    
    if query_type == "active":
        # 使用主动查询接口
        return consult_experience_active(
            state_description=state_description,
            last_error=last_error,
            current_expr=current_expression,
            available_relations=available_relations or [],
            top_k=top_k,
            threshold=0.3
        )
    else:
        # 使用被动查询接口（保持向后兼容）
        return search_experience_rules(
            question=state_description,
            current_entity=kwargs.get("current_entity", ""),
            current_expression=current_expression,
            available_relations=available_relations or [],
            last_failure=last_error,
            top_k=top_k,
            threshold=kwargs.get("threshold", 0.4),
        )
```

## 第五阶段：测试

### 5.1 单元测试
创建 `test_t12_experience_kb.py`：
1. 测试 `consult_experience_active()` 返回结构化结果
2. 测试失败检测逻辑正确触发 consult
3. 测试重试逻辑（失败 → consult → 重试）
4. 测试 trace 区分被动注入和主动 consult

### 5.2 集成测试
1. 小样本（10）运行中，`consult_experience` 调用次数 > 0
2. trace 中 `active_kb_consultations` > 0
3. prompt 中不再出现 191/200 这种高频无差别被动注入

## 实施顺序
1. 修改 `skills/experience_kb_skill.py`（第一阶段）
2. 修改 `skills/tools/adapters.py`（第四阶段）
3. 修改 `reasoning/agent.py`（第二阶段）
4. 修改 `reasoning/llm_agent.py`（第三阶段）
5. 编写测试（第五阶段）
6. 跑回归测试

## 注意事项
1. 保持向后兼容性
2. 避免无限 consult 循环
3. 控制 consult 预算（每个 sample 最多 2 次）
4. 确保原有测试全部通过