"""Direct tests for ToolRegistry — calling each Tool with schema-shaped kwargs.

Covers the two gaps identified in the 2026-04-18 re-review:
  1. verify_expression with only expression param (no func_list injection)
  2. extend_expression with expression_id="expression1" string input
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# Issue 1: verify_expression must work with expression param alone
# ============================================================

def test_verify_expression_with_expression_only():
    """Schema-shaped call: only mode + expression, no func_list injection."""
    from skills.skill_registry import TOOLS
    tool = TOOLS.get("verify_expression")

    # Call exactly as LLM would: just mode and expression
    result = tool.callable(
        mode="partial",
        expression="(JOIN film.actor.film m.0d6lp2)",
    )
    print(f"  result keys: {list(result.keys())}")
    assert "valid" in result, "Missing 'valid' key"
    assert "error" not in result or result.get("valid") is not None, "Unexpected error"
    print(f"  valid={result.get('valid')}, syntax_ok={result.get('syntax_ok')}")
    return result


def test_verify_expression_invalid_sexpr():
    """Schema-shaped call: invalid S-expression should return valid=False."""
    from skills.skill_registry import TOOLS
    tool = TOOLS.get("verify_expression")

    result = tool.callable(
        mode="partial",
        expression="(JOIN (bad nesting",
    )
    assert result["valid"] is False, "Should be invalid for bad expression"
    print(f"  correctly rejected: valid={result['valid']}")
    return result


def test_verify_expression_fallback_to_func_list():
    """When expression is omitted, should fall back to func_list from kwargs."""
    from skills.skill_registry import TOOLS
    tool = TOOLS.get("verify_expression")

    func_list = ['expression = START("m.0d6lp2")']
    result = tool.callable(
        mode="partial",
        func_list=func_list,
    )
    print(f"  fallback result: valid={result.get('valid')}")
    return result


def test_verify_expression_nothing_provided():
    """When neither expression nor func_list is provided, should return error."""
    from skills.skill_registry import TOOLS
    tool = TOOLS.get("verify_expression")

    result = tool.callable(mode="partial")
    assert "error" in result, "Should have error when nothing provided"
    assert result["valid"] is False
    print(f"  correctly errored: {result['error']}")
    return result


# ============================================================
# Issue 2: extend_expression expression_id string parsing
# ============================================================

def test_extend_expr_id_string_expression1():
    """expression_id='expression1' should resolve to 1, not fallback to cur_id."""
    from skills.tools.extend_expression_tool import _parse_expr_id, _build_step

    # Test _parse_expr_id directly
    assert _parse_expr_id("expression1", cur_id=0) == 1, "expression1 → 1"
    assert _parse_expr_id("expression2", cur_id=1) == 2, "expression2 → 2"
    assert _parse_expr_id("1", cur_id=0) == 1, "1 → 1"
    assert _parse_expr_id("", cur_id=3) == 3, "empty → cur_id"
    assert _parse_expr_id("expression", cur_id=3) == 3, "expression → cur_id"

    # Test through _build_step with expression_id="expression1"
    func_list = ['expression = START("m.0d6lp2")', 'expression1 = JOIN("rel1", expression)']
    step = _build_step(
        action="join",
        func_list=func_list,
        expression_id="expression1",
        relation="rel2",
        direction="forward",
    )
    # Should target expression1, not expression (which would be cur_id fallback)
    assert "expression1" in step and "expression2" in step, \
        f"step should reference expression1 and create expression2, got: {step}"
    print(f"  step: {step}")
    return step


def test_extend_expr_sub_expression_id_string():
    """sub_expression_id='expression2' should resolve to 2."""
    from skills.tools.extend_expression_tool import _build_step

    func_list = [
        'expression = START("m.0d6lp2")',
        'expression1 = JOIN("rel1", expression)',
        'expression2 = JOIN("rel2", expression)',
    ]
    step = _build_step(
        action="and",
        func_list=func_list,
        expression_id="expression1",
        sub_expression_id="expression2",
    )
    assert "AND(expression1, expression2)" in step, \
        f"step should AND expression1 and expression2, got: {step}"
    print(f"  step: {step}")
    return step


def test_extend_expr_through_tool():
    """Full Tool call with expression_id='expression1' string."""
    from skills.skill_registry import TOOLS
    tool = TOOLS.get("extend_expression")

    func_list = ['expression = START("m.0d6lp2")', 'expression1 = JOIN("film.actor.film", expression)']
    result = tool.callable(
        action="join",
        func_list=func_list,
        expression_id="expression1",
        relation="film.film.genre",
        direction="forward",
    )
    step = result.get("step", "")
    # The new step should target expression1, creating expression2
    assert "expression2" in step, f"Should create expression2, got: {step}"
    assert "expression1" in step, f"Should target expression1, got: {step}"
    print(f"  step: {step}")
    return result


# ============================================================
# Existing tests (regression)
# ============================================================

def test_extend_expression_basic():
    """Basic extend_expression call."""
    from skills.skill_registry import TOOLS
    tool = TOOLS.get("extend_expression")
    func_list = ['expression = START("m.0d6lp2")']
    result = tool.callable(
        action="join",
        func_list=func_list,
        relation="film.actor.film",
        direction="forward",
    )
    print(f"  success={result.get('success')}, step={result.get('step')}")
    return result


def test_current_expr_id_patterns():
    """Test _current_expr_id handles both numbered and unnumbered patterns."""
    from skills.tools.extend_expression_tool import _current_expr_id
    assert _current_expr_id(['expression = START("m.0d6lp2")']) == 0
    assert _current_expr_id(['expression1 = JOIN("rel", expression1)']) == 1
    assert _current_expr_id(['expression2 = AND(expression1, expression2)']) == 2
    assert _current_expr_id([]) == 0
    print("  ALL PASSED")


def test_explore_neighbors_with_mock():
    """explore_neighbors with mock subgraph."""
    from skills.skill_registry import TOOLS
    tool = TOOLS.get("explore_neighbors")
    class MockSubgraph:
        def get_outgoing(self, ent):
            return [("rel1", "target1"), ("rel2", "target2")]
        def get_incoming(self, ent):
            return [("rel3", "source1")]
        def __contains__(self, ent):
            return True
    result = tool.callable(
        entity="m.0d6lp2",
        direction="both",
        subgraph=MockSubgraph(),
    )
    print(f"  count={result.get('count')}")
    return result


def test_consult_experience_basic():
    """consult_experience basic call."""
    from skills.skill_registry import TOOLS
    tool = TOOLS.get("consult_experience")
    result = tool.callable(
        state_description="Looking for film actors",
        last_error="",
        current_expression='expression = START("m.0d6lp2")',
        available_relations=["film.actor.film"],
        top_k=2,
    )
    print(f"  type={type(result).__name__}, len={len(result) if isinstance(result, str) else 'N/A'}")
    return result


def test_inspect_path_with_mock():
    """inspect_path with mock data."""
    from skills.skill_registry import TOOLS
    tool = TOOLS.get("inspect_path")
    mock_paths = [["m.0d6lp2->film.actor.film->m.0b_g8d", "some_answer"]]
    class MockSubgraph:
        def get_outgoing(self, ent):
            return [("film.actor.film", "m.0b_g8d")]
        def get_incoming(self, ent):
            return []
        def __contains__(self, ent):
            return True
    result = tool.callable(
        path_index=0,
        start_entity="m.0d6lp2",
        candidate_paths=mock_paths,
        subgraph=MockSubgraph(),
    )
    print(f"  keys={list(result.keys()) if isinstance(result, dict) else type(result)}")
    return result



# ============================================================
# Issue 3: verify_expression(mode=full, expression=...) must indicate no execution
# ============================================================

def test_verify_expression_full_no_execution():
    """mode=full with raw expression should return execution_attempted=False."""
    from skills.skill_registry import TOOLS
    tool = TOOLS.get("verify_expression")

    # Syntax-correct but potentially non-executable expression
    result = tool.callable(
        mode="full",
        expression="(COUNT m.0d6lp2)",
    )
    assert result.get("execution_attempted") is False,         f"Should have execution_attempted=False, got: {result}"
    assert "warning" in result, "Should have warning field"
    print(f"  valid={result.get('valid')}, execution_attempted={result.get('execution_attempted')}")
    return result


def test_verify_expression_full_with_func_list():
    """mode=full with func_list should actually execute (no execution_attempted field)."""
    from skills.skill_registry import TOOLS
    tool = TOOLS.get("verify_expression")

    func_list = ['expression = START("m.0d6lp2")']
    result = tool.callable(
        mode="full",
        func_list=func_list,
    )
    # Should NOT have execution_attempted=False since it actually executed
    assert result.get("execution_attempted") is not False,         f"Full execution with func_list should not have execution_attempted=False"
    print(f"  result keys: {list(result.keys())}")
    return result


# ============================================================
# Issue 4: _parse_expr_id should warn on invalid input like "expressionX"
# ============================================================

def test_parse_expr_id_warns_on_invalid():
    """_parse_expr_id('expressionX') should warn and fallback to cur_id."""
    import warnings
    from skills.tools.extend_expression_tool import _parse_expr_id

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _parse_expr_id("expressionX", cur_id=3)
        assert result == 3, f"Should fall back to cur_id=3, got {result}"
        assert len(w) == 1, f"Should have 1 warning, got {len(w)}"
        assert "expressionX" in str(w[0].message), f"Warning should mention 'expressionX'"
        print(f"  correctly warned: {w[0].message}")
    return result


def test_parse_expr_id_no_warn_on_valid():
    """_parse_expr_id('expression1') should NOT warn."""
    import warnings
    from skills.tools.extend_expression_tool import _parse_expr_id

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _parse_expr_id("expression1", cur_id=0)
        assert result == 1
        assert len(w) == 0, f"Should have no warnings for valid input, got {len(w)}"
        print(f"  correctly parsed without warning: {result}")
    return result


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    tests = [
        # Issue 1: verify_expression expression param
        ("verify_expression_with_expression_only", test_verify_expression_with_expression_only),
        ("verify_expression_invalid_sexpr", test_verify_expression_invalid_sexpr),
        ("verify_expression_fallback_to_func_list", test_verify_expression_fallback_to_func_list),
        ("verify_expression_nothing_provided", test_verify_expression_nothing_provided),
        # Issue 3: verify_expression full mode semantics
        ("verify_expression_full_no_execution", test_verify_expression_full_no_execution),
        ("verify_expression_full_with_func_list", test_verify_expression_full_with_func_list),
        # Issue 2: extend_expression expression_id parsing
        ("extend_expr_id_string_expression1", test_extend_expr_id_string_expression1),
        ("extend_expr_sub_expression_id_string", test_extend_expr_sub_expression_id_string),
        ("extend_expr_through_tool", test_extend_expr_through_tool),
        # Issue 4: _parse_expr_id warning on invalid input
        ("parse_expr_id_warns_on_invalid", test_parse_expr_id_warns_on_invalid),
        ("parse_expr_id_no_warn_on_valid", test_parse_expr_id_no_warn_on_valid),
        # Regression
        ("extend_expression_basic", test_extend_expression_basic),
        ("current_expr_id_patterns", test_current_expr_id_patterns),
        ("explore_neighbors_with_mock", test_explore_neighbors_with_mock),
        ("consult_experience_basic", test_consult_experience_basic),
        ("inspect_path_with_mock", test_inspect_path_with_mock),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
