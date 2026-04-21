#!/usr/bin/env python3
"""Test extend_expression statistics."""

import sys
sys.path.insert(0, '/data/gt/omgv2-o1')

from reasoning.agent import GreedyAgent, TraceCollector

def test_extend_expression_stats():
    """Test that extend_expression statistics are counted correctly."""
    trace_collector = TraceCollector()
    agent = GreedyAgent(
        question="test",
        entities=["m.02jxk"],
        subgraph=None,
        trace_collector=trace_collector,
    )
    
    # Simulate some steps with non-observation actions
    agent.function_list = ['expression = START("m.02jxk")']
    agent.current_entity = "m.02jxk"
    agent.expression_id = ""
    
    # Simulate a join action (non-observation)
    filtered_out = {"some.relation": ["m.0abc"]}
    result = agent._dispatch_action("join", filtered_out, {})
    
    # Add step to trace using new_step()
    step1 = trace_collector.new_step()
    step1.update({
        "step_num": 1,
        "action": "join_forward",
        "chosen_relation": "some.relation",
        "current_entity_before": "m.02jxk",
        "current_entity_after": "m.0abc",
    })
    
    # Simulate another join action
    filtered_out2 = {"another.relation": ["m.0def"]}
    result2 = agent._dispatch_action("join", filtered_out2, {})
    
    # Add another step to trace
    step2 = trace_collector.new_step()
    step2.update({
        "step_num": 2,
        "action": "join_forward",
        "chosen_relation": "another.relation",
        "current_entity_before": "m.0abc",
        "current_entity_after": "m.0def",
    })
    
    # Finalize trace
    trace_collector.finalize(agent.function_list, agent.selected_relations, "")
    
    # Check statistics
    tool_usage = trace_collector.data.get("tool_usage_stats", {})
    extend_expr_count = tool_usage.get("extend_expression", 0)
    
    print(f"Tool usage stats: {tool_usage}")
    print(f"extend_expression count: {extend_expr_count}")
    
    if extend_expr_count > 0:
        print("SUCCESS: extend_expression statistics > 0")
        return True
    else:
        print("ERROR: extend_expression statistics = 0")
        return False

def test_kb_usage_stats_sync():
    """Test that kb_usage_stats is correctly synced from kb_stats."""
    trace_collector = TraceCollector()
    
    # Simulate kb_stats
    trace_collector.data["kb_stats"] = {
        "passive_injections": 2,
        "active_consultations": 3,
        "consultation_results": [
            {"confidence": 0.8},
            {"confidence": 0.6},
            {"confidence": 0.3},
        ]
    }
    
    # Finalize trace
    trace_collector.finalize([], [], "")
    
    # Check kb_usage_stats
    kb_usage_stats = trace_collector.data.get("kb_usage_stats", {})
    passive_injections = kb_usage_stats.get("passive_injections", 0)
    active_consultations = kb_usage_stats.get("active_consultations", 0)
    success_rate = kb_usage_stats.get("consultation_success_rate", 0.0)
    
    print(f"\nkb_usage_stats: {kb_usage_stats}")
    print(f"passive_injections: {passive_injections}")
    print(f"active_consultations: {active_consultations}")
    print(f"consultation_success_rate: {success_rate}")
    
    # Check if values are synced correctly
    if (passive_injections == 2 and active_consultations == 3 and 
        abs(success_rate - 2/3) < 0.01):  # 2 out of 3 have confidence > 0.5
        print("SUCCESS: kb_usage_stats correctly synced")
        return True
    else:
        print("ERROR: kb_usage_stats not correctly synced")
        return False

if __name__ == "__main__":
    print("Testing extend_expression statistics...")
    extend_ok = test_extend_expression_stats()
    
    print("\nTesting kb_usage_stats sync...")
    kb_ok = test_kb_usage_stats_sync()
    
    if extend_ok and kb_ok:
        print("\nAll statistics tests passed!")
        sys.exit(0)
    else:
        print("\nSome statistics tests failed!")
        sys.exit(1)