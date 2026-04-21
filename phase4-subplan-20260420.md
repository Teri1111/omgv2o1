# Phase 3 子计划 v2 - 2026-04-20 本轮执行

> 基于最新审查报告（2026-04-20 17:00+）
> 目标：真正关闭 v3 审查的两条关键阻断项
> 审查结论：当前状态仍为"部分完成"，不适合按"可关单"处理

---

## 审查发现（4 项，2 高 2 中）

| 编号 | 发现 | 严重度 | 当前状态 |
|------|------|--------|---------|
| 发现 1 | 复杂动作参数级闭环未完成 — parse_action 原样塞参，dispatch 用启发式覆盖 | **高** | ❌ 未关闭 |
| 发现 2 | 评测入口依赖仓外 — sys.path.insert 外部目录 | **中** | ❌ 未关闭 |
| 发现 3 | 长跑评测入口未就绪 — 无 --full-test/resume/workers | **中** | ❌ 未关闭 |
| 发现 4 | 回归护栏偏向 wiring — 无参数执行级断言 | **中** | ❌ 未关闭 |

---

## 本轮任务范围

| ID | 任务 | 优先级 | 负责 | 目标文件 |
|----|------|--------|------|---------|
| P4F-1 | parse_action() 参数结构化 | **P0** | SubAgent-A | `/data/gt/omgv2-o1/reasoning/llm_agent.py` |
| P4F-2 | _dispatch_action() 参数消费 | **P0** | SubAgent-A | `/data/gt/omgv2-o1/reasoning/agent.py` |
| P4F-3 | compute_f1() 收回主仓 | **P0** | SubAgent-A | `/data/gt/omgv2-o1/evaluate.py` + `test_closed_loop.py` |
| P4F-4 | 长跑入口 (--full-test/resume) | **P0** | SubAgent-A | `/data/gt/omgv2-o1/test_closed_loop.py` |
| P4F-5 | 参数级回归测试 | **P0** | SubAgent-B | `/data/gt/omgv2-o1/test_t7_review.py` |

---

## P4F-1: parse_action() 参数结构化

**问题**：llm_agent.py:136-144 对 Argmax/Argmin/Compare/Time_constraint 只是把参数原样塞进 relation 字段。

**当前代码**：
```python
elif action_type == "argmax" and allow_argmax:
    return {"action": "argmax", "relation": argument, "thought": thought}
elif action_type == "compare" and allow_cmp:
    return {"action": "cmp", "relation": argument, "thought": thought}  # "GT | film.film.revenue" 整体塞入
elif action_type == "time_constraint" and allow_time_filter:
    return {"action": "time_filter", "relation": argument, "thought": thought}  # "film.film.date | 2000" 整体塞入
```

**目标代码**：
```python
elif action_type == "argmax" and allow_argmax:
    return {"action": "argmax", "relation": argument.strip(), "thought": thought}
elif action_type == "argmin" and allow_argmin:
    return {"action": "argmin", "relation": argument.strip(), "thought": thought}
elif action_type == "compare" and allow_cmp:
    # Parse: "GT | film.film.revenue" -> operator + relation
    result = {"action": "cmp", "thought": thought}
    if "|" in argument:
        parts = [p.strip() for p in argument.split("|", 1)]
        result["operator"] = parts[0].strip().lower()  # "gt", "lt", "ge", "le"
        result["relation"] = parts[1].strip()
    else:
        token = argument.strip()
        if token.lower() in ("gt", "lt", "ge", "le"):
            result["operator"] = token.lower()
            result["relation"] = ""
        else:
            result["operator"] = None
            result["relation"] = token
    return result
elif action_type == "time_constraint" and allow_time_filter:
    # Parse: "film.film.initial_release_date | 2000" -> relation + time_value
    result = {"action": "time_filter", "thought": thought}
    if "|" in argument:
        parts = [p.strip() for p in argument.split("|", 1)]
        result["relation"] = parts[0].strip()
        result["time_value"] = parts[1].strip()
    else:
        result["relation"] = argument.strip()
        result["time_value"] = None
    return result
```

**验收**：`python3 -m py_compile reasoning/llm_agent.py` 通过

---

## P4F-2: _dispatch_action() 参数消费

**问题**：agent.py:849-902 对 argmax/argmin/cmp 直接调用 `_find_literal_relation()`，完全忽略 LLM 参数。

**当前代码**：
```python
if action == "argmax":
    rel = self._find_literal_relation(filtered_out)  # 忽略 pending
    if rel:
        self._do_arg("ARGMAX", rel)
        return {"relation": "ARGMAX:" + rel}

if action == "cmp":
    rel = self._find_literal_relation(filtered_out)  # 忽略 pending + 不传 operator
    if rel:
        self._do_cmp(rel)  # 缺少 operator 参数
        return {"relation": "CMP:" + rel}

if action == "time_filter" and filtered_out:
    pending = getattr(self, "_pending_llm_choice", None)
    if pending is not None:
        cand = pending.get("relation")
        if cand and cand in filtered_out:  # 要求完全相等
            time_rel = cand
```

**目标代码**：
```python
if action == "argmax":
    rel = None
    pending = getattr(self, "_pending_llm_choice", None)
    if pending is not None:
        rel = pending.get("relation")
    # 验证 relation 在 filtered_out 中（支持模糊匹配）
    if rel and rel not in filtered_out:
        for key in filtered_out:
            if rel in key or key in rel:
                rel = key
                break
    if rel is None or rel not in filtered_out:
        rel = self._find_literal_relation(filtered_out)
    if rel:
        self._do_arg("ARGMAX", rel)
        if hasattr(self, "_pending_llm_choice"):
            self._pending_llm_choice = None
        return {"relation": "ARGMAX:" + rel}

if action == "argmin":
    rel = None
    pending = getattr(self, "_pending_llm_choice", None)
    if pending is not None:
        rel = pending.get("relation")
    if rel and rel not in filtered_out:
        for key in filtered_out:
            if rel in key or key in rel:
                rel = key
                break
    if rel is None or rel not in filtered_out:
        rel = self._find_literal_relation(filtered_out)
    if rel:
        self._do_arg("ARGMIN", rel)
        if hasattr(self, "_pending_llm_choice"):
            self._pending_llm_choice = None
        return {"relation": "ARGMIN:" + rel}

if action == "cmp":
    rel = None
    operator = None
    pending = getattr(self, "_pending_llm_choice", None)
    if pending is not None:
        rel = pending.get("relation")
        operator = pending.get("operator")
    # 验证 relation 在 filtered_out 中
    if rel and rel not in filtered_out:
        for key in filtered_out:
            if rel in key or key in rel:
                rel = key
                break
    if rel is None or rel not in filtered_out:
        rel = self._find_literal_relation(filtered_out)
    if rel:
        self._do_cmp(rel, operator=operator)  # 传递 operator
        if hasattr(self, "_pending_llm_choice"):
            self._pending_llm_choice = None
        return {"relation": "CMP:" + rel}

if action == "time_filter" and filtered_out:
    time_rel = None
    time_value = None
    pending = getattr(self, "_pending_llm_choice", None)
    if pending is not None:
        cand = pending.get("relation") or pending.get("time_relation")
        time_value = pending.get("time_value")
        if cand:
            # Exact match first
            if cand in filtered_out:
                time_rel = cand
            else:
                # Fuzzy match
                for key in filtered_out:
                    if cand in key or key in cand:
                        time_rel = key
                        break
    if time_rel is None:
        time_rel = self._find_literal_relation(filtered_out)
    if time_rel and time_rel in filtered_out:
        tc_targets = filtered_out[time_rel]
        if time_value:
            tc_entity = time_value  # 使用 LLM 指定的 time_value
        else:
            tc_entity = tc_targets[0] if tc_targets else self.current_entity
        self._do_tc(time_rel, tc_entity)
        if hasattr(self, "_pending_llm_choice"):
            self._pending_llm_choice = None
        return {"relation": "TC:" + time_rel}
```

**验收**：`python3 -m py_compile reasoning/agent.py` 通过

---

## P4F-3: compute_f1() 收回主仓

**问题**：test_closed_loop.py:24-27 通过绝对路径插入外部目录。

**当前代码**：
```python
import sys as _sys
_sys.path.insert(0, "/data/gt/data_gen")
from evaluate_v4 import compute_f1
_sys.path.pop(0)
```

**改动**：
1. 新建 `/data/gt/omgv2-o1/evaluate.py`，包含 compute_f1()
2. 修改 test_closed_loop.py 导入：
```python
from evaluate import compute_f1
```

**验收**：`python3 test_closed_loop.py --eval-mode full 3` 正常输出 F1/P/R/EM

---

## P4F-4: 长跑入口

**问题**：CLI 没有 --full-test、--resume、--workers 参数。

**改动**：在 test_closed_loop.py 新增：
```python
parser.add_argument('--full-test', action='store_true', help='Run full CWQ test set')
parser.add_argument('--resume', action='store_true', help='Resume from existing results')
parser.add_argument('--output', type=str, default=None, help='Output JSONL path')
parser.add_argument('--workers', type=int, default=1, help='Parallel workers')
```

**验收**：`python3 test_closed_loop.py --help` 显示新参数

---

## P4F-5: 参数级回归测试

**问题**：现有测试只验证 wiring，不验证参数执行。

**改动**：在 test_t7_review.py 新增 TestT7ParameterExecution 类：
- test_argmax_uses_pending_relation
- test_argmin_uses_pending_relation
- test_cmp_passes_operator
- test_time_filter_uses_time_value

**验收**：新增测试能检测到 bug（先失败），修复后通过

---

## 执行策略

1. 所有代码修改直接在主仓 `/data/gt/omgv2-o1` 进行
2. SubAgent-A 执行 P4F-1 → P4F-2 → P4F-3 → P4F-4
3. SubAgent-B 执行 P4F-5（先写测试，验证 bug 存在，再修复）
4. 每个任务完成后验证语法 + 运行测试
5. 完成后创建分支 `phase10-p4-param-closure`

---

*创建时间: 2026-04-20 17:15*
*基于: 最新审查报告*