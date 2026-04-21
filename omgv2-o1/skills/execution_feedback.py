"""
Execution Feedback Skills for OMGv2 — Phase 3 refactor.

Provides three tiers of execution validation:
  1. execute_partial() — single-step validation: syntax + partial SPARQL exec
  2. execute_final()   — full LF execution: convert complete function_list → SPARQL → answers
  3. evaluate_candidate_relation() — tries both LF orientations for a relation, picks best

Also retains legacy helpers: build_candidate_func_list, validate_tentative_step.
"""

from skills.validate_syntax import validate_syntax


def _clean_answers(raw_answers):
    return [
        answer for answer in raw_answers
        if answer and str(answer).strip() and answer != "None"
    ]


# ============================================================
# Tier 1: Execute_partial — single-step validation
# ============================================================

def execute_partial(func_list, step_type="join"):
    """Validate a tentative step: syntax check + partial SPARQL execution.

    This is the standard single-step validation interface for Beam Search pruning.

    Args:
        func_list: Current function_list with one tentative step appended.
        step_type: "join" (reject empty results), "arg"/"count"/"tc" (syntax only).

    Returns:
        dict with keys:
            valid (bool):       Whether the step is acceptable.
            sexpr (str):        The generated S-expression.
            syntax_ok (bool):   Whether syntax check passed.
            exec_ok (bool|None): True=has answers, False=empty, None=exec error.
            num_answers (int):  Number of non-empty answers returned.
            answers (list):     The actual answer entities.
            error (str|None):   Error message if any.
            execution_score (float): Score in [0,1] for Beam Search ranking.
    """
    from skills.lf_construction import function_list_to_sexpr

    sexpr = function_list_to_sexpr(func_list)
    if sexpr == "@BAD_EXPRESSION":
        return {
            "valid": False, "sexpr": sexpr, "syntax_ok": False,
            "syntax_error": "bad expression", "exec_ok": None,
            "num_answers": 0, "answers": [], "error": "bad expression",
            "execution_score": 0.0,
        }

    syn = validate_syntax(sexpr)
    if not syn["valid"]:
        return {
            "valid": False, "sexpr": sexpr, "syntax_ok": False,
            "syntax_error": syn["error"], "exec_ok": None,
            "num_answers": 0, "answers": [], "error": syn["error"],
            "execution_score": 0.0,
        }

    # Syntax OK — try execution
    exec_ok = None
    num_answers = 0
    answers = []
    error = None

    try:
        from executor.sparql_executor import execute_function_list
        exec_result = execute_function_list(func_list)
        if exec_result.get("error"):
            error = exec_result["error"]
            exec_ok = None
        else:
            answers = _clean_answers(exec_result.get("answers", []))
            num_answers = len(answers)
            exec_ok = num_answers > 0
            if not exec_ok and step_type == "join":
                error = "empty execution result"
    except Exception as exc:
        error = str(exc)
        exec_ok = None

    # Validity decision
    if step_type == "join" and exec_ok is False:
        valid = False
    elif step_type == "join" and exec_ok is None and error:
        valid = False  # exec error is NOT acceptable for join validation
    elif error and exec_ok is None:
        valid = True  # non-join: syntax-only steps tolerate exec errors
    else:
        valid = True

    # Execution score for ranking (0.0–1.0)
    if exec_ok is True:
        execution_score = min(1.0, num_answers / 100.0)
    elif exec_ok is False:
        execution_score = 0.0
    elif exec_ok is None and error:
        execution_score = 0.0  # exec error → zero score, not 0.5
    else:
        execution_score = 0.5  # unknown (syntax-only step, no exec attempted)

    return {
        "valid": valid, "sexpr": sexpr, "syntax_ok": True,
        "syntax_error": None, "exec_ok": exec_ok,
        "num_answers": num_answers, "answers": answers, "error": error,
        "execution_score": execution_score,
    }


# ============================================================
# Tier 2: Execute_final — complete LF execution
# ============================================================

def execute_final(func_list):
    """Execute a complete function_list end-to-end.

    Converts the full function_list to S-expression, validates syntax,
    executes via SPARQL, and returns answers.

    Args:
        func_list: Complete function_list (ending with STOP).

    Returns:
        dict with keys:
            valid (bool):       Whether syntax is valid.
            sexpr (str):        The generated S-expression.
            answers (list):     Clean answer entities.
            num_answers (int):  Count of answers.
            error (str|None):   Error message if any.
    """
    from skills.lf_construction import function_list_to_sexpr
    from executor.sparql_executor import execute_lf

    sexpr = function_list_to_sexpr(func_list)
    if sexpr == "@BAD_EXPRESSION":
        return {
            "valid": False, "sexpr": sexpr,
            "answers": [], "num_answers": 0, "error": "bad expression",
        }

    syn = validate_syntax(sexpr)
    if not syn["valid"]:
        return {
            "valid": False, "sexpr": sexpr,
            "answers": [], "num_answers": 0, "error": syn["error"],
        }

    try:
        result = execute_lf(sexpr)
        if result.get("error"):
            return {
                "valid": True, "sexpr": sexpr,
                "answers": [], "num_answers": 0, "error": result["error"],
            }
        answers = _clean_answers(result.get("answers", []))
        return {
            "valid": True, "sexpr": sexpr,
            "answers": answers, "num_answers": len(answers), "error": None,
        }
    except Exception as exc:
        return {
            "valid": True, "sexpr": sexpr,
            "answers": [], "num_answers": 0, "error": str(exc),
        }


# ============================================================
# Tier 3: evaluate_candidate_relation — orientation selection
# ============================================================

def build_candidate_func_list(current_func_list, relation, reverse=False):
    """Build a tentative function list with one additional JOIN step."""
    if not current_func_list:
        return []
    eid = current_func_list[-1].split(" = ")[0].replace("expression", "")
    full_rel = "(R " + relation + ")" if reverse else relation
    candidate = list(current_func_list)
    candidate.append(
        'expression' + eid + ' = JOIN("' + full_rel + '", expression' + eid + ')'
    )
    return candidate


def evaluate_candidate_relation(
    current_func_list, relation, available_rels, reverse=False, expected_entities=None
):
    """Try both LF orientations for one traversal relation and keep the best.

    Uses execute_partial() for each orientation.
    """
    if relation not in available_rels:
        return {
            "relation": relation, "valid": False, "error": "not in available",
            "sexpr": None, "accepted": False, "lf_relation": None,
            "orientation_results": [], "expected_entities": list(expected_entities or []),
        }

    expected = set(expected_entities or [])
    default_lf_relation = "(R " + relation + ")" if reverse else relation
    orientation_results = []

    for lf_reverse in (False, True):
        candidate_list = build_candidate_func_list(
            current_func_list, relation, reverse=lf_reverse
        )
        result = execute_partial(candidate_list, step_type="join")
        answers = result.get("answers", [])
        overlap = len(expected.intersection(answers))
        lf_relation = "(R " + relation + ")" if lf_reverse else relation
        orientation_results.append({
            "lf_relation": lf_relation, "lf_reverse": lf_reverse,
            "sexpr": result.get("sexpr"), "syntax_ok": result.get("syntax_ok"),
            "exec_ok": result.get("exec_ok"),
            "num_answers": result.get("num_answers", 0),
            "answers": answers, "valid": result.get("valid", False),
            "error": result.get("error"), "overlap": overlap,
            "execution_score": result.get("execution_score", 0.0),
        })

    best = max(
        orientation_results,
        key=lambda item: (
            item["overlap"],
            int(item["valid"]),
            item["num_answers"],
            int(item["lf_relation"] == default_lf_relation),
        ),
    )

    accepted = best["valid"]
    error = best.get("error")
    if not accepted and not error:
        error = "validation failed"

    return {
        "relation": relation, "lf_relation": best["lf_relation"],
        "valid": best["valid"], "accepted": accepted,
        "sexpr": best.get("sexpr"), "syntax_ok": best.get("syntax_ok"),
        "exec_ok": best.get("exec_ok"),
        "num_answers": best.get("num_answers", 0),
        "answers": best.get("answers", []),
        "overlap": best.get("overlap", 0), "error": error,
        "orientation_results": orientation_results,
        "expected_entities": sorted(expected),
        "execution_score": best.get("execution_score", 0.0),
    }


# ============================================================
# Legacy alias (backward compat)
# ============================================================

def validate_tentative_step(func_list, step_type="join"):
    """Legacy alias for execute_partial()."""
    return execute_partial(func_list, step_type=step_type)
