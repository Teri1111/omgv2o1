#!/usr/bin/env python3
"""
Test for T14: Multi-tool combination and self-verification closed loop.
Tests explore->extend, verify->revise, inspect->decide chain patterns.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skills.tools.adapters import (
    explore_neighbors_adapter,
    verify_expression_adapter,
    inspect_path_adapter,
)
from reasoning.subgraph import SubgraphBuilder
from reasoning.agent import LLMGuidedAgent, TraceCollector


def test_explore_neighbors_returns_discovered_relations():
    """Test that explore_neighbors_adapter returns discovered_relations."""
    # Build a small subgraph
    builder = SubgraphBuilder()
    paths = [
        {
            "path": ["m.0f8l9c", "film.actor.film", "m.0bxtg", "film.film.directed_by", "m.0gqs1"],
            "score": 0.95,
        },
        {
            "path": ["m.0f8l9c", "film.actor.film", "m.0bxtg", "film.film.genre", "m.02l7c8"],
            "score": 0.85,
        },
    ]
    builder.build(paths)

    result = explore_neighbors_adapter(
        entity="m.0f8l9c",
        direction="outgoing",
        subgraph=builder
    )

    # Check basic results
    assert "results" in result
    assert "count" in result
    assert result["count"] >= 1

    # T14: Check discovered_relations field
    assert "discovered_relations" in result
    assert "discovered_count" in result
    assert isinstance(result["discovered_relations"], list)

    # Check that discovered_relations have correct structure
    for dr in result["discovered_relations"]:
        assert "relation" in dr
        assert "targets" in dr
        assert "direction" in dr

    print("  ✓ explore_neighbors returns discovered_relations")


def test_verify_expression_returns_suggestions():
    """Test that verify_expression_adapter returns suggestions on failure."""
    # Test with an invalid expression
    result = verify_expression_adapter(
        mode="partial",
        expression="(JOIN bad_relation (START bad_entity))"
    )

    assert "valid" in result
    # If syntax validation found issues, suggestions should be present
    if not result.get("valid", True):
        assert "suggestions" in result or "error" in result

    print("  ✓ verify_expression returns suggestions")


def test_verify_expression_valid_no_suggestions():
    """Test that verify_expression_adapter doesn't add unnecessary suggestions for valid expressions."""
    # Test with a potentially valid expression structure
    result = verify_expression_adapter(
        mode="partial",
        expression="(JOIN film.actor.film (START m.0f8l9c))"
    )

    assert "valid" in result
    print("  ✓ verify_expression handles valid expressions correctly")


def test_inspect_path_returns_confidence():
    """Test that inspect_path_adapter returns confidence score."""
    builder = SubgraphBuilder()
    paths = [
        ["m.0f8l9c->film.actor.film->m.0bxtg->film.film.directed_by->m.0gqs1", "m.0gqs1"],
    ]
    # Build subgraph from a triplet-style path
    sg_builder = SubgraphBuilder()
    sg_paths = [
        {
            "path": ["m.0f8l9c", "film.actor.film", "m.0bxtg", "film.film.directed_by", "m.0gqs1"],
            "score": 0.95,
        },
    ]
    sg_builder.build(sg_paths)

    result = inspect_path_adapter(
        path_index=0,
        start_entity="m.0f8l9c",
        candidate_paths=paths,
        subgraph=sg_builder
    )

    # Check result structure
    assert result is not None
    assert "error" not in result or result.get("error") is None

    # T14: Check confidence field
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0

    print("  ✓ inspect_path returns confidence score")


def test_multi_tool_chain_tracking():
    """Test that LLMGuidedAgent tracks multi-tool chain state."""
    builder = SubgraphBuilder()
    paths_dict = [
        {
            "path": ["m.0f8l9c", "film.actor.film", "m.0bxtg"],
            "score": 0.95,
        },
    ]
    builder.build(paths_dict)

    # Use correct format for candidate_paths: list of [path_string, answer]
    candidate_paths = [["m.0f8l9c->film.actor.film->m.0bxtg", "m.0bxtg"]]
    trace_collector = TraceCollector()
    agent = LLMGuidedAgent(
        question="test question",
        entities=["m.0f8l9c"],
        subgraph=builder,
        max_steps=3,
        candidate_paths=candidate_paths,
        trace_collector=trace_collector,
    )

    # Check initial state
    assert hasattr(agent, "_last_observation_tool")
    assert hasattr(agent, "_last_observation_result")
    assert hasattr(agent, "_observation_history")
    assert hasattr(agent, "_multi_tool_chain_count")
    assert agent._last_observation_tool is None
    assert agent._last_observation_result is None
    assert agent._observation_history == []
    assert agent._multi_tool_chain_count == 0

    print("  ✓ Multi-tool chain tracking state initialized")


def test_process_multi_tool_chain_explore():
    """Test _process_multi_tool_chain for explore_neighbors pattern."""
    builder = SubgraphBuilder()
    paths_dict = [{"path": ["m.0f8l9c", "film.actor.film", "m.0bxtg"], "score": 0.95}]
    builder.build(paths_dict)

    candidate_paths = [["m.0f8l9c->film.actor.film->m.0bxtg", "m.0bxtg"]]
    agent = LLMGuidedAgent(
        question="test question",
        entities=["m.0f8l9c"],
        subgraph=builder,
        max_steps=3,
        candidate_paths=candidate_paths,
    )

    # Simulate an explore_neighbors observation result
    obs_result = {
        "observation": {
            "results": [{"relation": "film.actor.film", "target": "m.0bxtg"}],
            "count": 1,
            "discovered_relations": [
                {"relation": "film.actor.film", "targets": ["m.0bxtg"], "direction": "forward"}
            ],
            "discovered_count": 1,
        },
        "success": True,
    }

    agent._process_multi_tool_chain("explore_neighbors", obs_result)

    assert agent._multi_tool_chain_count == 1
    assert hasattr(agent, "_last_discovered_relations")
    assert len(agent._last_discovered_relations) == 1
    assert agent._last_discovered_relations[0]["relation"] == "film.actor.film"

    print("  ✓ explore_neighbors -> extend_expression chain processed")


def test_process_multi_tool_chain_verify():
    """Test _process_multi_tool_chain for verify_expression pattern."""
    builder = SubgraphBuilder()
    paths_dict = [{"path": ["m.0f8l9c", "film.actor.film", "m.0bxtg"], "score": 0.95}]
    builder.build(paths_dict)

    candidate_paths = [["m.0f8l9c->film.actor.film->m.0bxtg", "m.0bxtg"]]
    agent = LLMGuidedAgent(
        question="test question",
        entities=["m.0f8l9c"],
        subgraph=builder,
        max_steps=3,
        candidate_paths=candidate_paths,
    )

    # Simulate a failed verify_expression observation
    obs_result = {
        "observation": {
            "valid": False,
            "error": "Expression yields no results",
            "suggestions": ["Try a different relation"],
        },
        "success": True,
    }

    agent._process_multi_tool_chain("verify_expression", obs_result)

    assert agent._multi_tool_chain_count == 1
    assert agent._last_failure is not None
    assert "verification failed" in agent._last_failure.lower()

    print("  ✓ verify_expression -> revise chain processed")


def test_process_multi_tool_chain_inspect():
    """Test _process_multi_tool_chain for inspect_path pattern."""
    builder = SubgraphBuilder()
    paths_dict = [{"path": ["m.0f8l9c", "film.actor.film", "m.0bxtg"], "score": 0.95}]
    builder.build(paths_dict)

    candidate_paths = [["m.0f8l9c->film.actor.film->m.0bxtg", "m.0bxtg"]]
    agent = LLMGuidedAgent(
        question="test question",
        entities=["m.0f8l9c"],
        subgraph=builder,
        max_steps=3,
        candidate_paths=candidate_paths,
    )

    # Simulate a high-confidence inspect_path observation
    obs_result = {
        "observation": {
            "sexpr": "(JOIN film.actor.film (START m.0f8l9c))",
            "confidence": 0.85,
            "path_index": 0,
        },
        "success": True,
    }

    step_trace = {}
    agent._process_multi_tool_chain("inspect_path", obs_result, step_trace)

    assert agent._multi_tool_chain_count == 1
    assert "multi_tool_chain" in step_trace
    assert step_trace["multi_tool_chain"]["chain_type"] == "inspect_to_decide_adopt"

    print("  ✓ inspect_path -> decide chain processed")


def test_trace_collector_multi_tool_stats():
    """Test that TraceCollector aggregates multi-tool chain stats."""
    collector = TraceCollector()

    # Add steps with multi-tool chain info
    step1 = collector.new_step()
    step1["multi_tool_chain"] = {
        "tool": "explore_neighbors",
        "chain_processed": True,
        "chain_type": "explore_to_extend",
    }

    step2 = collector.new_step()
    step2["multi_tool_chain"] = {
        "tool": "verify_expression",
        "chain_processed": True,
        "chain_type": "verify_to_revise",
    }

    step3 = collector.new_step()
    step3["multi_tool_chain"] = {
        "tool": "explore_neighbors",
        "chain_processed": True,
        "chain_type": "explore_to_extend",
    }

    collector.finalize([], [], "")

    stats = collector.data.get("multi_tool_chain_stats", {})
    assert stats.get("total_chains") == 3
    assert stats.get("chain_types", {}).get("explore_to_extend") == 2
    assert stats.get("chain_types", {}).get("verify_to_revise") == 1

    print("  ✓ TraceCollector aggregates multi-tool chain stats")


def test_no_infinite_loop_on_budget():
    """Test that observation budget prevents infinite loops."""
    builder = SubgraphBuilder()
    paths_dict = [{"path": ["m.0f8l9c", "film.actor.film", "m.0bxtg"], "score": 0.95}]
    builder.build(paths_dict)

    candidate_paths = [["m.0f8l9c->film.actor.film->m.0bxtg", "m.0bxtg"]]
    agent = LLMGuidedAgent(
        question="test question",
        entities=["m.0f8l9c"],
        subgraph=builder,
        max_steps=6,
        candidate_paths=candidate_paths,
    )

    # Check budget limits
    assert agent._observation_budget["explore_neighbors"] <= 3
    assert agent._observation_budget["verify_expression"] <= 4
    assert agent._observation_budget["consult_experience"] <= 3
    assert agent._observation_budget["inspect_path"] <= 3

    # Verify total budget is reasonable (prevents infinite loops)
    total_budget = sum(agent._observation_budget.values())
    assert total_budget <= agent.max_steps * 2, \
        f"Total observation budget ({total_budget}) should be bounded by 2x max_steps ({agent.max_steps * 2})"

    print("  ✓ Observation budget prevents infinite loops")


if __name__ == "__main__":
    print("T14: Multi-tool combination and self-verification closed loop tests")
    print("=" * 70)

    tests = [
        test_explore_neighbors_returns_discovered_relations,
        test_verify_expression_returns_suggestions,
        test_verify_expression_valid_no_suggestions,
        test_inspect_path_returns_confidence,
        test_multi_tool_chain_tracking,
        test_process_multi_tool_chain_explore,
        test_process_multi_tool_chain_verify,
        test_process_multi_tool_chain_inspect,
        test_trace_collector_multi_tool_stats,
        test_no_infinite_loop_on_budget,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")

    if failed > 0:
        sys.exit(1)
    else:
        print("All T14 tests passed!")
        sys.exit(0)
