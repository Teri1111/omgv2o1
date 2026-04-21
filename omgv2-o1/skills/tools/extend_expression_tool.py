import warnings
"""extend_expression Tool — unified LF step extension with automatic validation.

Wraps the 8 LF primitives (START, JOIN, AND, ARG, CMP, TC, COUNT, STOP)
and validates each step via execute_partial().
"""

from typing import Dict, Any, Optional, List


# Import existing primitives
from skills.lf_construction import (
    START, JOIN, AND, ARG, CMP, TC, COUNT, STOP,
    function_list_to_sexpr,
)
from skills.execution_feedback import execute_partial


# ============================================================
# Step builders — produce valid Python step strings for exec()
# ============================================================

def _current_expr_id(func_list: List[str]) -> int:
    """Parse the current expression counter from func_list.
    
    Handles two patterns:
      - 'expression = START(...)'  → 0 (unnumbered)
      - 'expression1 = JOIN(...)'  → 1 (numbered)
    """
    if not func_list:
        return 0
    last = func_list[-1]
    if "=" not in last:
        return 0
    id_part = last.split("=")[0].strip()
    suffix = id_part.replace("expression", "").strip()
    if suffix == "":
        return 0
    try:
        return int(suffix)
    except ValueError:
        return 0


def _expr_ref(expr_id: int) -> str:
    """Return expression reference string. 0 → 'expression', N → 'expressionN'."""
    return "expression" if expr_id == 0 else f"expression{expr_id}"


def _parse_expr_id(raw: str, cur_id: int) -> int:
    """Parse an expression_id string into an integer.
    
    Handles:
      - "" or "expression" → cur_id (default to current)
      - "expression1" → 1 (strip prefix)
      - "1" → 1 (direct int)
      - "expressionX" or other junk → cur_id + WARNING logged
    """
    if not raw or raw == "expression":
        return cur_id
    # Strip "expression" prefix if present
    clean = raw.replace("expression", "", 1).strip()
    if not clean:
        return cur_id
    try:
        return int(clean)
    except ValueError:
        import warnings
        warnings.warn(
            f"_parse_expr_id: invalid expression_id '{raw}', falling back to cur_id={cur_id}",
            stacklevel=2,
        )
        return cur_id


def _build_join_step(new_id: int, relation: str, target_id: int,
                     reverse: bool = False) -> str:
    rel = "(R {})".format(relation) if reverse else relation
    return '{0} = JOIN("{1}", {2})'.format(_expr_ref(new_id), rel, _expr_ref(target_id))


def _build_and_step(new_id: int, main_id: int, sub_id: int) -> str:
    return "{0} = AND({1}, {2})".format(_expr_ref(new_id), _expr_ref(main_id), _expr_ref(sub_id))


def _build_arg_step(new_id: int, operator: str, target_id: int,
                    relation: str, reverse: bool = False) -> str:
    rel = "(R {})".format(relation) if reverse else relation
    return '{0} = ARG("{1}", {2}, "{3}")'.format(_expr_ref(new_id), operator, _expr_ref(target_id), rel)


def _build_cmp_step(new_id: int, operator: str, relation: str,
                    target_id: int, reverse: bool = False) -> str:
    rel = "(R {})".format(relation) if reverse else relation
    return '{0} = CMP("{1}", "{2}", {3})'.format(_expr_ref(new_id), operator, rel, _expr_ref(target_id))


def _build_tc_step(new_id: int, target_id: int, relation: str,
                   entity: str, reverse: bool = False) -> str:
    rel = "(R {})".format(relation) if reverse else relation
    return '{0} = TC({1}, "{2}", "{3}")'.format(_expr_ref(new_id), _expr_ref(target_id), rel, entity)


def _build_count_step(new_id: int, target_id: int) -> str:
    return "{0} = COUNT({1})".format(_expr_ref(new_id), _expr_ref(target_id))


def _build_step(
    action: str,
    func_list: List[str],
    expression_id: str = "",
    relation: str = None,
    direction: str = "forward",
    sub_expression_id: str = None,
    operator: str = None,
    entity: str = None,
) -> str:
    """Build a single function_list step string from tool parameters.

    Returns a valid Python assignment string compatible with exec().
    """
    cur_id = _current_expr_id(func_list)
    new_id = cur_id + 1
    reverse = direction == "reverse"

    # Resolve target expression ID — use _parse_expr_id for robust parsing
    target_id = _parse_expr_id(expression_id, cur_id)

    if action == "join":
        assert relation is not None, "join requires relation"
        return _build_join_step(new_id, relation, target_id, reverse)

    elif action == "and":
        sub_id = _parse_expr_id(sub_expression_id, cur_id) if sub_expression_id is not None else cur_id
        return _build_and_step(new_id, target_id, sub_id)

    elif action in ("argmax", "argmin"):
        op = action.upper()
        assert relation is not None, "{0} requires relation".format(action)
        return _build_arg_step(new_id, op, target_id, relation, reverse)

    elif action == "cmp":
        assert operator is not None, "cmp requires operator"
        assert relation is not None, "cmp requires relation"
        return _build_cmp_step(new_id, operator, relation, target_id, reverse)

    elif action == "tc":
        assert relation is not None, "tc requires relation"
        assert entity is not None, "tc requires entity"
        return _build_tc_step(new_id, target_id, relation, entity, reverse)

    elif action == "count":
        return _build_count_step(new_id, target_id)

    else:
        raise ValueError("Unknown action: {0}".format(action))


# ============================================================
# Public API
# ============================================================

def extend_expression(
    action: str,
    func_list: List[str],
    expression_id: str = "",
    relation: str = None,
    direction: str = "forward",
    sub_expression_id: str = None,
    operator: str = None,
    entity: str = None,
    target_expr: str = None,
) -> Dict[str, Any]:
    """Extend current LF expression by one step, with automatic partial validation.

    This is the unified Tool interface wrapping the LF primitives.
    Dispatches to the correct primitive based on ``action``, builds the
    function_list step, then validates via execute_partial().

    Args:
        action: One of join/and/argmax/argmin/cmp/tc/count.
        func_list: Current function_list (will NOT be modified in-place).
        expression_id: Target expression ID (defaults to current).
            Accepts: "", "expression", "expression1", "1" — all resolve correctly.
        relation: Freebase relation (required for join/arg/cmp/tc).
        direction: "forward" or "reverse".
        sub_expression_id: Second expression ID for AND merge.
        operator: Comparison operator for CMP (le/lt/ge/gt).
        entity: Target entity MID for TC.
        target_expr: Full S-expression (reserved for alternative AND merge).

    Returns:
        dict with keys:
            success (bool):           Whether the step passed validation.
            new_expression (str):     The resulting S-expression.
            new_func_list (list):     Updated function_list (copy, not modified).
            step (str):               The function_list step string.
            num_answers (int):        Number of intermediate answers.
            answers (list):           Actual answer entities.
            execution_score (float):  Score in [0, 1].
            error (str | None):       Error message if any.
    """
    # Build the step string
    step = _build_step(
        action=action,
        func_list=func_list,
        expression_id=expression_id,
        relation=relation,
        direction=direction,
        sub_expression_id=sub_expression_id,
        operator=operator,
        entity=entity,
    )

    # Create new func_list (copy + append)
    new_func_list = list(func_list) + [step]

    # Map action to step_type for execute_partial
    # "join" actions need non-empty results; others tolerate syntax-only
    if action in ("join", "and"):
        step_type = "join"
    elif action in ("argmax", "argmin"):
        step_type = "arg"
    else:
        step_type = action  # "cmp", "tc", "count"

    # Validate
    validation = execute_partial(new_func_list, step_type=step_type)

    return {
        "success": validation["valid"],
        "new_expression": validation["sexpr"],
        "new_func_list": new_func_list,
        "step": step,
        "num_answers": validation.get("num_answers", 0),
        "answers": validation.get("answers", []),
        "execution_score": validation.get("execution_score", 0.0),
        "error": validation.get("error"),
    }
