#!/usr/bin/env python3
"""分析 trace 文件，输出整改前后对比。"""

import json
import sys
from collections import defaultdict

def analyze_trace_file(trace_file):
    """分析 trace 文件。"""
    try:
        with open(trace_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("错误: 文件 '{}' 不存在".format(trace_file))
        return
    except json.JSONDecodeError:
        print("错误: 文件 '{}' 不是有效的 JSON 格式".format(trace_file))
        return
    
    traces = data.get("traces", [])
    if not traces:
        print("错误: 文件中没有找到 trace 数据")
        return
    
    print("=== Trace 分析报告 ===")
    print("文件: {}".format(trace_file))
    print("样本数: {}".format(len(traces)))
    print("模式: {}".format(data.get("mode", "unknown")))
    
    # 分析工具使用
    tool_usage = defaultdict(int)
    action_dist = defaultdict(int)
    multi_tool_calls = 0
    active_consultations = 0
    passive_injections = 0
    
    for trace_entry in traces:
        trace_data = trace_entry.get("trace", {})
        
        # 累加工具使用
        tool_stats = trace_data.get("tool_usage_stats", {})
        for tool, count in tool_stats.items():
            tool_usage[tool] += count
        
        # 累加动作分布
        action_stats = trace_data.get("action_distribution", {})
        for action, count in action_stats.items():
            action_dist[action] += count
        
        # 累加多工具调用
        multi_tool_calls += trace_data.get("multi_tool_chain_stats", {}).get("total_chains", 0)
        
        # 累加 KB 使用
        kb_stats = trace_data.get("kb_usage_stats", {})
        active_consultations += kb_stats.get("active_consultations", 0)
        passive_injections += kb_stats.get("passive_injections", 0)
    
    # 输出分析结果
    print("\n工具使用统计:")
    for tool, count in sorted(tool_usage.items()):
        if count > 0:
            print("  {}: {} 次".format(tool, count))
    
    print("\n动作分布:")
    for action, count in sorted(action_dist.items()):
        if count > 0:
            print("  {}: {} 次".format(action, count))
    
    print("\n关键指标:")
    print("  多工具调用次数: {}".format(multi_tool_calls))
    print("  主动 KB consult 次数: {}".format(active_consultations))
    print("  被动 KB 注入次数: {}".format(passive_injections))
    
    # 计算整改效果
    total_samples = len(traces)
    print("\n整改效果评估:")
    
    # 工具覆盖率
    tools_used = len([t for t in tool_usage.values() if t > 0])
    total_tools = 5  # extend_expression, explore_neighbors, verify_expression, consult_experience, inspect_path
    print("  工具覆盖率: {:.1f}% ({}/{})".format(tools_used / total_tools * 100, tools_used, total_tools))
    
    # 多工具调用率
    print("  多工具调用率: {:.1f}%".format(multi_tool_calls / total_samples * 100 if total_samples > 0 else 0))
    
    # 主动 consult 率
    print("  主动 consult 率: {:.1f}%".format(active_consultations / total_samples * 100 if total_samples > 0 else 0))
    
    # 非 JOIN 动作统计
    non_join_actions = sum(count for action, count in action_dist.items() 
                          if action not in ["join_forward", "join_reverse"] and count > 0)
    print("  非 JOIN 动作次数: {}".format(non_join_actions))
    
    # 输出摘要信息
    summary = data.get("summary", {})
    if summary:
        print("\n测试摘要:")
        print("  语法正确: {}/{}".format(summary.get("syntax_ok", 0), summary.get("total", 0)))
        print("  LF 命中率: {}/{} ({:.1f}%)".format(
            summary.get("lf_hit_count", 0), 
            summary.get("evaluable", 0),
            summary.get("lf_hit_count", 0) / summary.get("evaluable", 1) * 100 if summary.get("evaluable", 0) > 0 else 0))
        print("  Merged 命中率: {}/{} ({:.1f}%)".format(
            summary.get("merged_hit_count", 0), 
            summary.get("evaluable", 0),
            summary.get("merged_hit_count", 0) / summary.get("evaluable", 1) * 100 if summary.get("evaluable", 0) > 0 else 0))

def main():
    if len(sys.argv) < 2:
        print("用法: python analyze_traces.py <trace_file.json>")
        print("示例: python analyze_traces.py /data/gt/t15_scale_200.json")
        sys.exit(1)
    
    trace_file = sys.argv[1]
    analyze_trace_file(trace_file)

if __name__ == "__main__":
    main()