#!/bin/bash
# 运行 200 样本规模测试

set -e

cd /data/gt/omgv2-o1

echo "=== T15: 指标监控与 200 样本回归测试 ==="
echo "开始时间: $(date)"

# 1. 语法检查
echo ""
echo "=== 1. 语法检查 ==="
/data/gt/envs/lf_gjq/bin/python3 -m py_compile reasoning/agent.py
/data/gt/envs/lf_gjq/bin/python3 -m py_compile reasoning/llm_agent.py
/data/gt/envs/lf_gjq/bin/python3 -m py_compile reasoning/beam_agent.py
echo "语法检查通过"

# 2. 单元测试
echo ""
echo "=== 2. 单元测试 ==="
/data/gt/envs/lf_gjq/bin/python3 test_t12_experience_kb.py
/data/gt/envs/lf_gjq/bin/python3 test_t13_non_join_actions.py
/data/gt/envs/lf_gjq/bin/python3 test_t14_multi_tool_chains.py
echo "单元测试通过"

# 3. 小样本验证 (20 samples)
echo ""
echo "=== 3. 小样本验证 (20 samples) ==="
/data/gt/envs/lf_gjq/bin/python3 test_closed_loop.py --llm-first --llm --stats --trace-export /data/gt/t15_smoke_20.json 20 > /data/gt/t15_smoke_20.log 2>&1
echo "小样本验证完成，结果保存到 /data/gt/t15_smoke_20.json"

# 4. 中等样本验证 (50 samples)
echo ""
echo "=== 4. 中等样本验证 (50 samples) ==="
/data/gt/envs/lf_gjq/bin/python3 test_closed_loop.py --llm-first --llm --stats --trace-export /data/gt/t15_validation_50.json 50 > /data/gt/t15_validation_50.log 2>&1
echo "中等样本验证完成，结果保存到 /data/gt/t15_validation_50.json"

# 5. 分析小样本结果
echo ""
echo "=== 5. 分析小样本结果 ==="
/data/gt/envs/lf_gjq/bin/python3 scripts/analyze_traces.py /data/gt/t15_smoke_20.json > /data/gt/t15_smoke_20_analysis.txt 2>&1
echo "小样本分析完成，报告保存到 /data/gt/t15_smoke_20_analysis.txt"

# 6. 分析中等样本结果
echo ""
echo "=== 6. 分析中等样本结果 ==="
/data/gt/envs/lf_gjq/bin/python3 scripts/analyze_traces.py /data/gt/t15_validation_50.json > /data/gt/t15_validation_50_analysis.txt 2>&1
echo "中等样本分析完成，报告保存到 /data/gt/t15_validation_50_analysis.txt"

echo ""
echo "=== 测试完成 ==="
echo "结束时间: $(date)"
echo ""
echo "生成的文件:"
echo "  - /data/gt/t15_smoke_20.json (20样本trace)"
echo "  - /data/gt/t15_smoke_20.log (20样本日志)"
echo "  - /data/gt/t15_smoke_20_analysis.txt (20样本分析报告)"
echo "  - /data/gt/t15_validation_50.json (50样本trace)"
echo "  - /data/gt/t15_validation_50.log (50样本日志)"
echo "  - /data/gt/t15_validation_50_analysis.txt (50样本分析报告)"
echo ""
echo "如需运行200样本测试，请手动执行:"
echo "  /data/gt/envs/lf_gjq/bin/python3 test_closed_loop.py --llm-first --llm --stats --trace-export /data/gt/t15_scale_200.json 200"
echo "  /data/gt/envs/lf_gjq/bin/python3 scripts/analyze_traces.py /data/gt/t15_scale_200.json"