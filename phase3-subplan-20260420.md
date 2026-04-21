# Phase 3 子计划 - 2026-04-20 本轮执行

> 基于新版审查报告-v3
> 目标：修复 v3 审查发现的 3 个关键问题，使系统进入"可稳定长跑、可独立复现"状态
> 对照：v3 报告建议顺序 → 先补复杂动作参数闭环，再补评测工程化

---

## 本轮任务范围

| ID | 任务 | 优先级 | 负责 | 状态 |
|----|------|--------|------|------|
| P3F-1 | 复杂动作参数级闭环（解析+执行） | **P0** | SubAgent-A (Dev) | 待启动 |
| P3F-2 | compute_f1() 收回到主仓 | **P0** | SubAgent-A (Dev) | 待启动 |
| P3F-3 | P2-T2 长跑入口（--full-test + 断点续跑） | **P0** | SubAgent-A (Dev) | 待启动 |
| P3F-4 | 自动化回归升级（参数级测试） | **P0** | SubAgent-B (Test) | 待启动 |

---

## P3F-1: 复杂动作参数级闭环

**问题根因（v3 发现 1）**：
- `parse_action()` 对 `Compare [ GT | relation ]` 和 `Time_constraint [ relation | time ]` 只是把整段参数原样塞进 `relation` 字段，没有拆出 `operator / relation / time_value`
- `_dispatch_action()` 在 `argmax / argmin / cmp` 路径上并不读取 LLM 返回的 relation 参数，而是重新调用 `_find_literal_relation(filtered_out)` 选择"第一个字面量关系"
- `_dispatch_action()` 在 `time_filter` 路径上虽然会看 `_pending_llm_choice`，但它要求 `pending["relation"]` 与 `filtered_out` 的 relation key 完全相等；像 `film.film.initial_release_date | 2000` 这样的参数串不会命中，所以 time value 会被直接丢掉

**改动文件**：
1. `reasoning/llm_agent.py` — `parse_action()` 参数解析重构
2. `reasoning/agent.py` — `_dispatch_action()` 参数消费逻辑修复

**具体改动**：

### 1a. parse_action() 参数解析结构化
对于 `Compare` 动作：
```python
# 当前：param_str = "GT | film.film.initial_release_date"
# 期望：拆分为 {"operator": "GT", "relation": "film.film.initial_release_date"}
```

对于 `Time_constraint` 动作：
```python
# 当前：param_str = "film.film.initial_release_date | 2000"
# 期望：拆分为 {"relation": "film.film.initial_release_date", "time_value": "2000"}
```

对于 `Argmax/Argmin` 动作：
```python
# 当前：param_str = "film.film.initial_release_date"
# 期望：保持为 {"relation": "film.film.initial_release_date"}
```

### 1b. _dispatch_action() 参数消费修复
- `argmax/argmin` 路径：读取 `pending.get("relation")` 而不是重新调用 `_find_literal_relation`
- `cmp` 路径：读取 `pending.get("operator")` 和 `pending.get("relation")`
- `time_filter` 路径：读取 `pending.get("relation")` 和 `pending.get("time_value")`，使用模糊匹配而非完全相等

### 1c. 验收标准
- `python3 -m py_compile reasoning/llm_agent.py` 通过
- `python3 -m py_compile reasoning/agent.py` 通过
- 手动测试：Compare/Time_constraint/Argmax/Argmin 的参数能正确解析并执行

---

## P3F-2: compute_f1() 收回到主仓

**问题根因（v3 发现 2）**：
- `test_closed_loop.py` 通过 `sys.path.insert(0, "/data/gt/data_gen")` 从仓外导入 `compute_f1()`
- 仓库本身不能算"评测能力自包含"

**改动文件**：
1. `omgv2-o1/evaluate.py` — 新建文件，包含 `compute_f1()` 函数
2. `test_closed_loop.py` — 修改导入路径，使用本地 `evaluate.py`

**具体改动**：

### 2a. 新建 evaluate.py
将 `/data/gt/data_gen/evaluate_v4.py` 中的 `compute_f1()` 函数复制到主仓：
```python
# omgv2-o1/evaluate.py
def compute_f1(predicted_answers: set, golden_answers: set) -> dict:
    """Standard set-level KBQA F1 computation."""
    if not predicted_answers and not golden_answers:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not predicted_answers or not golden_answers:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = len(predicted_answers & golden_answers)
    precision = tp / len(predicted_answers)
    recall = tp / len(golden_answers)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
```

### 2b. test_closed_loop.py 导入修复
```python
# 删除或注释掉
# sys.path.insert(0, "/data/gt/data_gen")
# from evaluate_v4 import compute_f1

# 改为
from evaluate import compute_f1
```

### 2c. 验收标准
- `python3 -m py_compile evaluate.py` 通过
- `python3 test_closed_loop.py --eval-mode full 3` 能正常输出 F1/P/R/EM
- 不再依赖仓外路径

---

## P3F-3: P2-T2 长跑入口

**问题根因（v3 发现 2）**：
- CLI 仍只有样本数 + `--eval-mode full` 这一级入口，未见 `--full-test`、断点续跑、并行评测主链路

**改动文件**：
1. `test_closed_loop.py` — 新增 `--full-test` 模式、断点续跑、结果落盘

**具体改动**：

### 3a. 新增 --full-test 模式
```python
parser.add_argument('--full-test', action='store_true',
                    help='Run full CWQ test set (1639 samples)')
```

### 3b. 断点续跑实现
- 结果文件落盘（JSONL，每行一题）
- 重启时检测已有结果文件，跳过已完成样本
- 支持 `--resume` 参数

### 3c. 结果落盘格式
```json
{
  "question_id": "WebQTest-1234",
  "question": "...",
  "predicted_answers": ["m.0123", "m.0456"],
  "golden_answers": ["m.0123"],
  "lf_sexpr": "(JOIN ...)",
  "merged_sexpr": "(JOIN ...)",
  "f1": 0.5,
  "precision": 1.0,
  "recall": 0.5,
  "em": 0,
  "hit": 1,
  "trace": {...}
}
```

### 3d. 并行评测（可选）
- 支持 `--workers N` 参数
- 使用多进程并行处理样本
- 注意 LLM 调用的并发限制

### 3e. 验收标准
- `python3 test_closed_loop.py --full-test --eval-mode full` 能启动全集评测
- 断点续跑功能正常
- 结果文件正确生成

---

## P3F-4: 自动化回归升级

**问题根因（v3 发现 3）**：
- 现有测试覆盖仍偏向动作名是否能解析、allow 标志是否透传、full 模式是否能打印指标
- 没有覆盖：`Compare` 的 operator 是否被执行、`Time_constraint` 的 time value 是否被执行、`Argmax / Argmin` 的 relation 参数是否真的被执行、评测入口是否脱离仓外路径后仍然可运行

**改动文件**：
1. `test_t7_review.py` — 新增参数级测试

**具体改动**：

### 4a. 新增 TestT7ParameterExecution 测试类
```python
class TestT7ParameterExecution(unittest.TestCase):
    """Verify complex action parameters are actually executed."""
    
    def test_compare_operator_executed(self):
        """Verify Compare [ GT | relation ] uses GT operator."""
        
    def test_time_constraint_value_executed(self):
        """Verify Time_constraint [ relation | 2000 ] uses 2000."""
        
    def test_argmax_relation_executed(self):
        """Verify Argmax [ relation ] uses specified relation."""
        
    def test_argmin_relation_executed(self):
        """Verify Argmin [ relation ] uses specified relation."""
```

### 4b. 新增 TestT7EvaluateSelfContained 测试类
```python
class TestT7EvaluateSelfContained(unittest.TestCase):
    """Verify evaluation works without external dependencies."""
    
    def test_compute_f1_local(self):
        """Test local compute_f1 function."""
        
    def test_eval_mode_full_local(self):
        """Test --eval-mode full without external imports."""
```

### 4c. 验收标准
- `python3 test_t7_review.py` 全部通过（包括新增测试）
- 新增测试能检测到参数执行问题

---

## 执行策略

1. **SubAgent-A (Dev)** 按 P3F-1 → P3F-2 → P3F-3 顺序修改代码
2. 每个子任务完成后由架构师（我）审查 diff
3. **SubAgent-B (Test)** 在 P3F-1+P3F-2+P3F-3 全部完成后执行 P3F-4
4. 所有任务完成后，同步代码到主仓，创建分支 `phase10-p3-param-closure`
5. 每个阶段完成后发 Telegram 通知

---

## 风险与缓解

| 风险 | 概率 | 缓解措施 |
|------|------|---------|
| 参数解析正则过于复杂导致新bug | 中 | 保持简单，只处理标准格式，并提供fallback |
| compute_f1() 复制后格式不一致 | 低 | 直接从evaluate_v4.py复制，不做修改 |
| 全集评测时间过长 | 高 | 先用100样本测试，确认无误后再跑全集 |
| 参数级测试难以mock | 中 | 使用已知的测试数据，不依赖真实LLM |

---

## 验收标准

### 整体验收
- [ ] P3F-1: 复杂动作参数级闭环完成
- [ ] P3F-2: compute_f1() 收回到主仓
- [ ] P3F-3: P2-T2 长跑入口就绪
- [ ] P3F-4: 自动化回归升级完成

### 测试验收
- [ ] 所有语法检查通过
- [ ] test_t7_review.py 全部通过（包括新增测试）
- [ ] test_closed_loop.py --eval-mode full 正常工作
- [ ] 100样本验证无退化

### 文档验收
- [ ] 代码注释清晰
- [ ] 新增功能有使用说明

---

*创建时间: 2026-04-20 16:50*
*基于: 新版审查报告-v3*
*预计完成时间: 2026-04-20 20:00*