# T15 简化实施计划

## 当前状态
- TraceCollector.__init__() 已增加 tool_usage_stats、action_distribution、kb_usage_stats、performance_stats
- finalize() 方法尚未更新统计逻辑
- 缺少测试脚本和分析工具

## 分步实施

### 第一步：完成 TraceCollector 统计逻辑
1. 修改 finalize() 方法，聚合工具使用和动作分布统计
2. 增加性能统计（llm_calls、llm_fallbacks）

### 第二步：修改 test_closed_loop.py
1. 在测试结束后输出聚合统计信息
2. 增加 --stats 参数控制统计输出

### 第三步：创建分析脚本
1. 创建 scripts/analyze_traces.py
2. 实现 trace 文件分析功能

### 第四步：运行测试验证
1. 运行 20 样本小规模测试
2. 验证统计输出正确性
3. 运行 50 样本中等规模测试

## 简化验收标准
1. TraceCollector 正确记录统计信息
2. 测试脚本输出聚合统计
3. 20 样本测试通过
4. 50 样本测试通过

## 下一步行动
1. 首先完成 TraceCollector 的 finalize() 方法修改
2. 然后修改 test_closed_loop.py 增加统计输出
3. 创建分析脚本
4. 运行测试验证