"""
Path_to_LF_draft skill for OMGv2.

Converts a T5 candidate path directly into a complete function_list + S-expression.
Used for Path-Guided mode: when T5 top-1 path is confident, skip step-by-step reasoning.

Path format: "entity_or_name->rel1->entity_or_name->rel2->answer"
Note: entities in the path may be readable names, not mids.
We only use the RELATIONS from the path and determine direction by probing the subgraph.

Direction strategy: For each hop, try BOTH forward and reverse orientations,
build a partial function_list, execute it via SPARQL, and pick the one with results.
This ensures we match the actual KB direction, not just the subgraph (which may
have bidirectional edges for traversal convenience).
"""

from typing import List, Optional


def parse_relations(path_string: str) -> List[str]:
    """Extract the relation sequence from a path string.

    Input: "e0->r1->e1->r2->e2"
    Output: ["r1", "r2"]
    """
    parts = path_string.split("->")
    return [parts[i] for i in range(1, len(parts), 2)]


def _build_and_test(func_list_base, rel, is_reverse):
    """Build a candidate function_list with one more JOIN and test via SPARQL.

    Returns (updated_func_list, sexpr, num_answers) or None on failure.
    """
    from skills.skill_registry import get_skill

    _exec_partial = get_skill("execute_partial").callable

    lf_rel = "(R " + rel + ")" if is_reverse else rel
    candidate = list(func_list_base)
    candidate.append('expression = JOIN("' + lf_rel + '", expression)')
    result = _exec_partial(candidate, step_type="join")

    return candidate, result.get("sexpr", ""), result.get("num_answers", 0)


def path_to_lf_draft(
    start_entity: str,
    candidate_paths: list,
    subgraph,
) -> Optional[dict]:
    """Generate a complete function_list from the top-1 T5 candidate path.

    For each relation in the path, tries both forward and reverse orientations,
    executes the partial SPARQL, and picks the one that returns results.

    Args:
        start_entity: The starting entity mid.
        candidate_paths: T5 output, list of [path_string, answer_string].
        subgraph: SubgraphBuilder instance.

    Returns:
        dict with keys: function_list, sexpr, relations, directions
        or None if path is invalid or no orientation works.
    """
    if not candidate_paths or not candidate_paths[0]:
        return None

    top1 = candidate_paths[0]
    path_str = top1[0] if isinstance(top1, list) else top1
    if not path_str:
        return None

    relations = parse_relations(path_str)
    if not relations:
        return None

    # Build function_list hop by hop, trying both directions
    function_list = ['expression = START("' + start_entity + '")']
    directions = []
    current_sexpr = None

    for rel in relations:
        # Try forward first
        fwd_list, fwd_sexpr, fwd_count = _build_and_test(function_list, rel, is_reverse=False)
        # Try reverse
        rev_list, rev_sexpr, rev_count = _build_and_test(function_list, rel, is_reverse=True)

        if fwd_count > 0 and fwd_count >= rev_count:
            function_list = fwd_list
            current_sexpr = fwd_sexpr
            directions.append("forward")
        elif rev_count > 0:
            function_list = rev_list
            current_sexpr = rev_sexpr
            directions.append("reverse")
        else:
            # Neither produced results — pick forward as default
            # (might work in full path context even if partial doesn't)
            function_list = fwd_list
            current_sexpr = fwd_sexpr
            directions.append("forward")

    # Finalize
    function_list.append('expression = STOP(expression)')

    from skills.lf_construction import function_list_to_sexpr
    sexpr = function_list_to_sexpr(function_list)

    return {
        "function_list": function_list,
        "sexpr": sexpr,
        "relations": relations,
        "directions": directions,
        "path_string": path_str,
    }
