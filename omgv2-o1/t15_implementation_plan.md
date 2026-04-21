# T15: 指标监控与200样本回归 实施计划

## 目标
把本轮整改是否生效变成可量化指标，防止再次出现"接口接好了但运行态没用"的假完成。

## 当前状态分析
1. TraceCollector 已有 kb_stats 和 multi_tool_chain_stats
2. 缺少 tool usage、action distribution 等详细统计
3. 缺少 200 样本规模测试脚本
4. 缺少整改前后对比输出

## 实施步骤

### 1. 修改 `reasoning/agent.py`

#### 1.1 扩展 TraceCollector 统计
在 `TraceCollector.__init__()` 中增加：
```python
self.data["tool_usage_stats"] = {
    "extend_expression": 0,
    "explore_neighbors": 0,
    "verify_expression": 0,
    "consult_experience": 0,
    "inspect_path": 0,
    "multi_tool_calls": 0,  # 多工具调用次数
}

self.data["action_distribution"] = {
    "join_forward": 0,
    "join_reverse": 0,
    "count": 0,
    "argmax": 0,
    "argmin": 0,
    "time_filter": 0,
    "and": 0,
    "cmp": 0,
    "finish": 0,
    "observation": 0,
}

self.data["kb_usage_stats"] = {
    "passive_injections": 0,
    "active_consultations": 0,
    "consultation_success_rate": 0.0,
}

self.data["performance_stats"] = {
    "total_steps": 0,
    "llm_calls": 0,
    "llm_fallbacks": 0,
    "pg_fallback_used": False,
}
```

#### 1.2 修改 finalize() 方法
在 `finalize()` 中聚合统计：
```python
def finalize(self, function_list, selected_relations, sexpr):
    # 现有逻辑...
    
    # 聚合工具使用统计
    tool_usage = self.data["tool_usage_stats"]
    action_dist = self.data["action_distribution"]
    
    for step in self.data["steps"]:
        # 统计工具使用
        if "tool_call" in step:
            tool_name = step["tool_call"].get("tool", "")
            if tool_name in tool_usage:
                tool_usage[tool_name] += 1
        
        # 统计动作分布
        action = step.get("action", "")
        if action in action_dist:
            action_dist[action] += 1
    
    # 统计多工具调用
    tool_usage["multi_tool_calls"] = self.data["multi_tool_chain_stats"]["total_chains"]
    
    # 统计 KB 使用
    kb_stats = self.data["kb_stats"]
    self.data["kb_usage_stats"]["passive_injections"] = kb_stats["passive_injections"]
    self.data["kb_usage_stats"]["active_consultations"] = kb_stats["active_consultations"]
    
    # 统计性能
    self.data["performance_stats"]["total_steps"] = len(self.data["steps"])
```

### 2. 修改 `test_closed_loop.py`

#### 2.1 增加统计输出
在测试结束后输出详细统计：
```python
def print_aggregated_stats(all_traces):
    """输出聚合统计信息。"""
    total_samples = len(all_traces)
    
    # 工具使用统计
    tool_usage = defaultdict(int)
    action_dist = defaultdict(int)
    multi_tool_calls = 0
    active_consultations = 0
    
    for trace in all_traces:
        # 累加工具使用
        for tool, count in trace.get("tool_usage_stats", {}).items():
            tool_usage[tool] += count
        
        # 累加动作分布
        for action, count in trace.get("action_distribution", {}).items():
            action_dist[action] += count
        
        # 累加多工具调用
        multi_tool_calls += trace.get("tool_usage_stats", {}).get("multi_tool_calls", 0)
        
        # 累加主动 consult
        active_consultations += trace.get("kb_usage_stats", {}).get("active_consultations", 0)
    
    print("\n=== 聚合统计 ({} samples) ===".format(total_samples))
    print("\n工具使用:")
    for tool, count in sorted(tool_usage.items()):
        print("  {}: {}/{} ({:.1f}%)".format(tool, count, total_samples, count/total_samples*100))
    
    print("\n动作分布:")
    for action, count in sorted(action_dist.items()):
        if count > 0:
            print("  {}: {}".format(action, count))
    
    print("\n关键指标:")
    print("  多工具调用次数: {}".format(multi_tool_calls))
    print("  主动 KB consult 次数: {}".format(active_consultations))
    print("  多工具调用率: {:.1f}%".format(multi_tool_calls/total_samples*100 if total_samples > 0 else 0))
```

#### 2.2 增加 200 样本测试模式
```python
def run_200_sample_test(args):
    """运行 200 样本规模测试。"""
    # 加载数据集
    dataset = load_dataset()  # 假设有这个函数
    
    # 运行测试
    all_traces = []
    for i, sample in enumerate(dataset[:200]):
        # 运行 agent
        trace = run_agent(sample, args)
        all_traces.append(trace)
        
        # 每 50 个样本输出一次进度
        if (i + 1) % 50 == 0:
            print("已完成 {}/200 样本".format(i + 1))
    
    # 输出聚合统计
    print_aggregated_stats(all_traces)
    
    # 保存结果
    save_results(all_traces, args.output_file)
```

### 3. 创建 `scripts/analyze_traces.py`

#### 3.1 创建 trace 分析脚本
```python
#!/usr/bin/env python3
"""分析 trace 文件，输出整改前后对比。"""

import json
import sys
from collections import defaultdict

def analyze_trace_file(trace_file):
    """分析 trace 文件。"""
    with open(trace_file, 'r') as f:
        traces = json.load(f)
    
    print("=== Trace 分析报告 ===")
    print("样本数: {}".format(len(traces)))
    
    # 分析工具使用
    tool_usage = defaultdict(int)
    action_dist = defaultdict(int)
    multi_tool_calls = 0
    active_consultations = 0
    
    for trace in traces:
        # 累加工具使用
        for tool, count in trace.get("tool_usage_stats", {}).items():
            tool_usage[tool] += count
        
        # 累加动作分布
        for action, count in trace.get("action_distribution", {}).items():
            action_dist[action] += count
        
        # 累加多工具调用
        multi_tool_calls += trace.get("tool_usage_stats", {}).get("multi_tool_calls", 0)
        
        # 累加主动 consult
        active_consultations += trace.get("kb_usage_stats", {}).get("active_consultations", 0)
    
    # 输出分析结果
    print("\n工具使用统计:")
    for tool, count in sorted(tool_usage.items()):
        print("  {}: {} 次".format(tool, count))
    
    print("\n动作分布:")
    for action, count in sorted(action_dist.items()):
        if count > 0:
            print("  {}: {} 次".format(action, count))
    
    print("\n关键指标:")
    print("  多工具调用次数: {}".format(multi_tool_calls))
    print("  主动 KB consult 次数: {}".format(active_consultations))
    
    # 计算整改效果
    total_samples = len(traces)
    print("\n整改效果评估:")
    print("  工具覆盖率: {:.1f}%".format(len([t for t in tool_usage.values() if t > 0]) / 5 * 100))
    print("  多工具调用率: {:.1f}%".format(multi_tool_calls / total_samples * 100 if total_samples > 0 else 0))
    print("  主动 consult 率: {:.1f}%".format(active_consultations / total_samples * 100 if total_samples > 0 else 0))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python analyze_traces.py <trace_file.json>")
        sys.exit(1)
    
    analyze_trace_file(sys.argv[1])
```

### 4. 创建 `scripts/run_scale_test.sh`

#### 4.1 创建规模测试脚本
```bash
#!/bin/bash
# 运行 200 样本规模测试

set -e

cd /data/gt/omgv2-o1

# 1. 语法检查
echo "=== 语法检查 ==="
/data/gt/envs/lf_gjq/bin/python3 -m py_compile reasoning/agent.py
/data/gt/envs/lf_gjq/bin/python3 -m py_compile reasoning/llm_agent.py
echo "语法检查通过"

# 2. 单元测试
echo "=== 单元测试 ==="
/data/gt/envs/lf_gjq/bin/python3 test_t12_experience_kb.py
/data/gt/envs/lf_gjq/bin/python3 test_t13_non_join_actions.py
/data/gt/envs/lf_gjq/bin/python3 test_t14_multi_tool_chains.py
echo "单元测试通过"

# 3. 小样本验证 (20 samples)
echo "=== 小样本验证 (20 samples) ==="
/data/gt/envs/lf_gjq/bin/python3 test_closed_loop.py --llm-first --llm --trace-export /data/gt/t15_smoke_20.json 20 > /data/gt/t15_smoke_20.log 2>&1
echo "小样本验证完成"

# 4. 中等样本验证 (50 samples)
echo "=== 中等样本验证 (50 samples) ==="
/data/gt/envs/lf_gjq/bin/python3 test_closed_loop.py --llm-first --llm --trace-export /data/gt/t15_validation_50.json 50 > /data/gt/t15_validation_50.log 2>&1
echo "中等样本验证完成"

# 5. 规模测试 (200 samples)
echo "=== 规模测试 (200 samples) ==="
/data/gt/envs/lf_gjq/bin/python3 test_closed_loop.py --llm-first --llm --trace-export /data/gt/t15_scale_200.json 200 > /data/gt/t15_scale_200.log 2>&1
echo "规模测试完成"

# 6. 分析结果
echo "=== 分析结果 ==="
/data/gt/envs/lf_gjq/bin/python3 scripts/analyze_traces.py /data/gt/t15_scale_200.json > /data/gt/t15_analysis_report.txt 2>&1
echo "分析完成，报告保存到 /data/gt/t15_analysis_report.txt"

echo "=== 测试完成 ==="
```

## 验收标准

### 功能验收
1. TraceCollector 正确记录 tool_usage_stats、action_distribution、kb_usage_stats、performance_stats
2. 200 样本统计中：`consult_experience > 0`，且 `verify_expression` 或 `explore_neighbors` 至少一个 > 0
3. 多工具调用次数 > 0
4. 50 样本回归不低于当前已知基线：Syntax OK 50/50，Merged hit 不低于 35/50

### 输出验收
1. 产出可复用的统计输出，供后续 T6/T8/T9/T10 继续使用
2. 生成整改前后对比报告
3. 保存 trace 文件供后续分析

## 风险控制

### 高风险点
1. **200 样本测试耗时过长** → 先跑 20/50 样本验证方向
2. **LLM 环境不可用** → 提供纯 heuristic 模式作为备选
3. **统计输出格式不兼容** → 使用标准 JSON 格式

### 回滚策略
- 每个步骤独立提交，可单独回滚
- 保留 `phase6-t12-experience-kb` 分支作为基线

## 时间估算
- 修改 TraceCollector 统计: 2 小时
- 修改 test_closed_loop.py 输出: 1.5 小时
- 创建分析脚本: 1.5 小时
- 创建规模测试脚本: 1 小时
- 运行 200 样本测试: 4 小时
- 分析结果和生成报告: 2 小时
- **总计: 12 小时**

## 下一步行动
1. 修改 `reasoning/agent.py` 扩展 TraceCollector 统计
2. 修改 `test_closed_loop.py` 增加统计输出
3. 创建 `scripts/analyze_traces.py` 分析脚本
4. 创建 `scripts/run_scale_test.sh` 规模测试脚本
5. 运行 20/50/200 样本测试
6. 分析结果，生成整改前后对比报告
7. 完成后通知 PM 审查