"""T13: Non-JOIN action coverage test.

Verifies that _decide_action() correctly dispatches to count/argmax/argmin/time_filter/cmp/and
without requiring step_count >= 2, and that _dispatch_action() handles them properly.
"""
import sys


def test_decide_action_count():
    """Count should be returned on first step when question contains 'how many'."""
    from reasoning.agent import GreedyAgent
    from reasoning.agent import TraceCollector

    agent = GreedyAgent(
        question="how many countries are in europe",
        entities=["m.02jxk"],
        subgraph=None,
        trace_collector=TraceCollector(),
    )
    out_rels = {"location.country.continent": ["m.0ge6k"]}
    action = agent._decide_action(out_rels, {})
    assert action == "count", f"Expected 'count', got '{action}'"
    print("  [PASS] count on step 1 (no gate)")


def test_decide_action_argmax():
    """Argmax should be returned when question contains argmax keywords and literal rel exists."""
    from reasoning.agent import GreedyAgent
    from reasoning.agent import TraceCollector

    agent = GreedyAgent(
        question="which country has the largest population",
        entities=["m.02jxk"],
        subgraph=None,
        trace_collector=TraceCollector(),
    )
    out_rels = {"location.country.population": ["m.01234"], "location.country.continent": ["m.0ge6k"]}
    action = agent._decide_action(out_rels, {})
    assert action == "argmax", f"Expected 'argmax', got '{action}'"
    print("  [PASS] argmax on step 1 (no gate)")


def test_decide_action_argmin():
    """Argmin should be returned when question contains argmin keywords and literal rel exists."""
    from reasoning.agent import GreedyAgent
    from reasoning.agent import TraceCollector

    agent = GreedyAgent(
        question="which country has the smallest population",
        entities=["m.02jxk"],
        subgraph=None,
        trace_collector=TraceCollector(),
    )
    out_rels = {"location.country.population": ["m.01234"], "location.country.continent": ["m.0ge6k"]}
    action = agent._decide_action(out_rels, {})
    assert action == "argmin", f"Expected 'argmin', got '{action}'"
    print("  [PASS] argmin on step 1 (no gate)")


def test_decide_action_time_filter():
    """Time filter should be returned when question contains 'when' and literal rel exists."""
    from reasoning.agent import GreedyAgent
    from reasoning.agent import TraceCollector

    agent = GreedyAgent(
        question="when was the company founded",
        entities=["m.02jxk"],
        subgraph=None,
        trace_collector=TraceCollector(),
    )
    out_rels = {"organization.organization.date_founded": ["m.01234"]}
    action = agent._decide_action(out_rels, {})
    assert action == "time_filter", f"Expected 'time_filter', got '{action}'"
    print("  [PASS] time_filter on step 1 (no gate)")


def test_decide_action_and():
    """AND should be returned when question contains 'and' and >= 2 out_rels."""
    from reasoning.agent import GreedyAgent
    from reasoning.agent import TraceCollector

    agent = GreedyAgent(
        question="countries in europe and asia",
        entities=["m.02jxk"],
        subgraph=None,
        trace_collector=TraceCollector(),
    )
    out_rels = {"location.country.continent": ["m.0ge6k"], "location.country.region": ["m.0f2s9"]}
    action = agent._decide_action(out_rels, {})
    assert action == "and", f"Expected 'and', got '{action}'"
    print("  [PASS] and on step 1 (no gate)")


def test_decide_action_cmp():
    """CMP should be returned when question contains comparison keywords and literal rel exists."""
    from reasoning.agent import GreedyAgent
    from reasoning.agent import TraceCollector

    agent = GreedyAgent(
        question="countries with population greater than 1000000",
        entities=["m.02jxk"],
        subgraph=None,
        trace_collector=TraceCollector(),
    )
    out_rels = {"location.country.population": ["m.01234"]}
    action = agent._decide_action(out_rels, {})
    assert action == "cmp", f"Expected 'cmp', got '{action}'"
    print("  [PASS] cmp on step 1 (no gate)")


def test_dispatch_and():
    """_dispatch_action should handle 'and' action."""
    from reasoning.agent import GreedyAgent, TraceCollector

    agent = GreedyAgent(
        question="test",
        entities=["m.02jxk"],
        subgraph=None,
        trace_collector=TraceCollector(),
    )
    agent.function_list = ['expression = START("m.02jxk")']
    agent.current_entity = "m.02jxk"
    agent.expression_id = ""
    filtered_out = {"some.relation": ["m.0abc"]}
    result = agent._dispatch_action("and", filtered_out, {})
    assert result is not None, "Expected result dict"
    assert result["relation"].startswith("AND:"), f"Expected AND: prefix, got {result}"
    print("  [PASS] dispatch and")


def test_dispatch_cmp():
    """_dispatch_action should handle 'cmp' action."""
    from reasoning.agent import GreedyAgent, TraceCollector

    agent = GreedyAgent(
        question="test",
        entities=["m.02jxk"],
        subgraph=None,
        trace_collector=TraceCollector(),
    )
    agent.function_list = ['expression = START("m.02jxk")']
    agent.current_entity = "m.02jxk"
    agent.expression_id = ""
    filtered_out = {"location.country.population": ["m.01234"]}
    result = agent._dispatch_action("cmp", filtered_out, {})
    assert result is not None, "Expected result dict"
    assert result["relation"].startswith("CMP:"), f"Expected CMP: prefix, got {result}"
    print("  [PASS] dispatch cmp")


def test_dispatch_count():
    """_dispatch_action should handle 'count' action."""
    from reasoning.agent import GreedyAgent, TraceCollector

    agent = GreedyAgent(
        question="test",
        entities=["m.02jxk"],
        subgraph=None,
        trace_collector=TraceCollector(),
    )
    agent.function_list = ['expression = START("m.02jxk")']
    agent.current_entity = "m.02jxk"
    agent.expression_id = ""
    result = agent._dispatch_action("count", {}, {})
    assert result is not None, "Expected result dict"
    assert result["relation"] == "COUNT", f"Expected COUNT, got {result}"
    print("  [PASS] dispatch count")


def test_replay_skips_non_join():
    """_replay_path_answer should stop at non-JOIN actions (verify prefix check logic)."""
    from reasoning.agent import GreedyAgent, TraceCollector

    agent = GreedyAgent(
        question="test",
        entities=["m.02jxk"],
        subgraph=None,
        trace_collector=TraceCollector(),
    )
    # Verify that the prefix check in _replay_path_answer covers new prefixes
    # Set first relation as AND: — should break immediately without needing subgraph
    agent.selected_relations = ["AND:something"]
    agent.entities = ["m.02jxk"]
    result = agent._replay_path_answer()
    assert isinstance(result, set), "Expected set result"
    # When first rel is non-JOIN, current stays as start entity
    assert result == {"m.02jxk"}, f"Expected start entity, got {result}"
    print("  [PASS] replay stops at non-JOIN actions")


if __name__ == "__main__":
    tests = [
        test_decide_action_count,
        test_decide_action_argmax,
        test_decide_action_argmin,
        test_decide_action_time_filter,
        test_decide_action_and,
        test_decide_action_cmp,
        test_dispatch_and,
        test_dispatch_cmp,
        test_dispatch_count,
        test_replay_skips_non_join,
    ]
    print(f"Running {len(tests)} T13 tests...")
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed:
        sys.exit(1)
    print("All T13 tests passed!")
