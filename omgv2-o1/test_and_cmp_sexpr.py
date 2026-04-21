#!/usr/bin/env python3
"""Test AND/CMP S-expression conversion."""

import sys
sys.path.insert(0, '/data/gt/omgv2-o1')

from reasoning.agent import GreedyAgent, TraceCollector
from skills.lf_construction import function_list_to_sexpr

def test_and_sexpr():
    """Test that AND generates valid S-expression."""
    agent = GreedyAgent(
        question="test",
        entities=["m.02jxk"],
        subgraph=None,
        trace_collector=TraceCollector(),
    )
    agent.function_list = ['expression = START("m.02jxk")']
    agent.current_entity = "m.02jxk"
    agent.expression_id = ""
    
    # Execute AND operation
    filtered_out = {"some.relation": ["m.0abc"]}
    result = agent._dispatch_action("and", filtered_out, {})
    
    print("Function list after AND:")
    for fl in agent.function_list:
        print(f"  {fl}")
    
    # Convert to S-expression
    sexpr = function_list_to_sexpr(agent.function_list)
    print(f"S-expression: {sexpr}")
    
    if sexpr == "@BAD_EXPRESSION":
        print("ERROR: AND generated @BAD_EXPRESSION")
        return False
    else:
        print("SUCCESS: AND generated valid S-expression")
        return True

def test_cmp_sexpr():
    """Test that CMP generates valid S-expression."""
    agent = GreedyAgent(
        question="test",
        entities=["m.02jxk"],
        subgraph=None,
        trace_collector=TraceCollector(),
    )
    agent.function_list = ['expression = START("m.02jxk")']
    agent.current_entity = "m.02jxk"
    agent.expression_id = ""
    
    # Execute CMP operation
    filtered_out = {"location.country.population": ["m.01234"]}
    result = agent._dispatch_action("cmp", filtered_out, {})
    
    print("\nFunction list after CMP:")
    for fl in agent.function_list:
        print(f"  {fl}")
    
    # Convert to S-expression
    sexpr = function_list_to_sexpr(agent.function_list)
    print(f"S-expression: {sexpr}")
    
    if sexpr == "@BAD_EXPRESSION":
        print("ERROR: CMP generated @BAD_EXPRESSION")
        return False
    else:
        print("SUCCESS: CMP generated valid S-expression")
        return True

if __name__ == "__main__":
    print("Testing AND/CMP S-expression conversion...")
    and_ok = test_and_sexpr()
    cmp_ok = test_cmp_sexpr()
    
    if and_ok and cmp_ok:
        print("\nAll AND/CMP tests passed!")
        sys.exit(0)
    else:
        print("\nSome AND/CMP tests failed!")
        sys.exit(1)