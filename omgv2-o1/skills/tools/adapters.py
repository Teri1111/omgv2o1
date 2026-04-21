"""Adapter functions bridging Tool JSON Schema parameters to internal callable signatures."""

from typing import Dict, Any, Optional, List, Union


def explore_neighbors_adapter(
    entity: str,
    direction: str = "both",
    filter_pattern: str = None,
    **kwargs
) -> Dict[str, Any]:
    """Adapter for explore_neighbors Tool.
    
    Schema: entity, direction, filter_pattern
    Actual: explore_subgraph(subgraph, entity_id, max_hops=2)
    
    This adapter is called from the agent context where subgraph is available.
    The agent must inject subgraph via kwargs or a context manager.
    """
    # Import here to avoid circular imports
    from skills.explore_subgraph import explore_subgraph
    
    subgraph = kwargs.get("subgraph")
    if subgraph is None:
        return {"error": "subgraph not available in context", "results": []}
    
    max_hops = kwargs.get("max_hops", 2)
    results = explore_subgraph(subgraph, entity, max_hops=max_hops)
    
    # Apply filter if provided
    if filter_pattern:
        import re
        pattern = re.compile(filter_pattern)
        results = [r for r in results if pattern.search(r.get("relation", ""))]
    
    # Filter by direction
    if direction != "both":
        if direction == "outgoing":
            results = [r for r in results if not r.get("relation", "").startswith("(R ")]
        elif direction == "incoming":
            results = [r for r in results if r.get("relation", "").startswith("(R ")]
    
    # T14: Build discovered_relations for multi-tool chaining (explore -> extend)
    discovered_relations = []
    seen_rels = set()
    for r in results:
        rel = r.get("relation", "")
        if rel and rel not in seen_rels:
            seen_rels.add(rel)
            targets = [r.get("target", "")]
            discovered_relations.append({
                "relation": rel,
                "targets": targets,
                "direction": "reverse" if rel.startswith("(R ") else "forward"
            })
    
    return {
        "results": results,
        "count": len(results),
        "discovered_relations": discovered_relations,
        "discovered_count": len(discovered_relations),
    }


def verify_expression_adapter(
    mode: str = "partial",
    expression: str = None,
    **kwargs
) -> Dict[str, Any]:
    """Adapter for verify_expression Tool.
    
    Schema: mode, expression
    Actual: validate_syntax(expression) / execute_partial(func_list) / execute_final(func_list)
    
    If expression is provided (S-expression string), validates it directly.
    If expression is omitted, falls back to func_list from kwargs for execution-based validation.
    """
    from skills.validate_syntax import validate_syntax
    
    # Priority 1: If expression is provided, validate it directly
    if expression is not None:
        syn = validate_syntax(expression)
        if mode == "full":
            # Full execution from raw S-expression is not supported without func_list.
            # Return syntax result with execution_attempted=False so callers can distinguish.
            return {
                "valid": syn["valid"],
                "syntax_ok": syn["valid"],
                "sexpr": expression,
                "error": syn.get("error"),
                "execution_attempted": False,
                "warning": "full execution requires func_list; only syntax was checked",
            }
        else:
            # Partial mode: syntax check is the expected behavior
            result = syn.copy()
            result["execution_attempted"] = False
            return result
    
    # Priority 2: Fall back to func_list for execution-based validation
    func_list = kwargs.get("func_list")
    if func_list is not None:
        from skills.execution_feedback import execute_partial, execute_final
        if mode == "full":
            result = execute_final(func_list)
        else:
            result = execute_partial(func_list, step_type=kwargs.get("step_type", "join"))
        # T14: Enrich with suggestions for verify->revise chain
        if not result.get("valid", True) and not result.get("suggestions"):
            suggestions = []
            error = result.get("error", "")
            if "empty" in error.lower() or result.get("num_answers", 0) == 0:
                suggestions.append("Current expression yields no results; try a different relation or direction")
            if "syntax" in error.lower():
                suggestions.append("Syntax error detected; check relation names and expression structure")
            result["suggestions"] = suggestions
        return result

    return {"error": "Neither expression nor func_list provided", "valid": False}


def consult_experience_adapter(
    state_description: str = "",
    last_error: str = "",
    current_expression: str = "",
    available_relations: List[str] = None,
    top_k: int = 3,
    query_type: str = "passive",  # "passive", "active", or "skill_md"
    **kwargs
) -> Union[str, dict]:
    """Adapter for consult_experience Tool.
    
    Schema: state_description, last_error, current_expression, available_relations, top_k
    Actual: search_experience_rules(question, current_entity, current_expression, 
            available_relations, last_failure, top_k, threshold)
    
    Maps schema param names to actual param names.
    
    query_type:
        "passive"  - returns formatted string (backward compatible)
        "active"   - returns structured dict with matched_rules, guidance_text, etc.
        "skill_md" - returns SKILL.md documents matching the query
    """
    from skills.experience_kb_skill import (
        search_experience_rules,
        consult_experience_active,
        search_skill_docs,
    )
    
    if query_type == "active":
        return consult_experience_active(
            state_description=state_description,
            last_error=last_error,
            current_expr=current_expression,
            available_relations=available_relations or [],
            top_k=top_k,
            threshold=0.3
        )
    elif query_type == "skill_md":
        # T6: Search SKILL.md documents
        query = state_description or last_error or current_expression
        if not query:
            return {
                "matched_skills": [],
                "skill_contents": [],
                "guidance_text": "",
                "query_type": "skill_md",
            }
        results = search_skill_docs(query, top_k=top_k)
        skill_contents = []
        for r in results:
            skill_contents.append({
                "rule_id": r["rule_id"],
                "title": r["title"],
                "rule_type": r["rule_type"],
                "confidence": r["confidence"],
                "score": r["score"],
                "full_content": r["full_content"],
            })
        # Build combined guidance text from skill docs
        if skill_contents:
            lines = [f"[Experience Skills — {len(skill_contents)} skill docs matched]"]
            for i, sc in enumerate(skill_contents, 1):
                lines.append(f"--- SKILL {i}: {sc['title']} (rule_id={sc['rule_id']}, score={sc['score']:.1f}) ---")
                lines.append(sc["full_content"])
                lines.append("")
            guidance_text = "\n".join(lines)
        else:
            guidance_text = ""
        return {
            "matched_skills": skill_contents,
            "skill_contents": skill_contents,
            "guidance_text": guidance_text,
            "rule_ids": [sc["rule_id"] for sc in skill_contents],
            "confidence": sum(sc["confidence"] for sc in skill_contents) / max(len(skill_contents), 1),
            "query_type": "skill_md",
        }
    else:
        return search_experience_rules(
            question=state_description,
            current_entity=kwargs.get("current_entity", ""),
            current_expression=current_expression,
            available_relations=available_relations or [],
            last_failure=last_error,
            top_k=top_k,
            threshold=kwargs.get("threshold", 0.4),
        )


def inspect_path_adapter(
    path_index: int = 0,
    **kwargs
) -> Optional[Dict[str, Any]]:
    """Adapter for inspect_path Tool.
    
    Schema: path_index (int)
    Actual: path_to_lf_draft(start_entity, candidate_paths, subgraph)
    
    The agent must provide start_entity, candidate_paths, subgraph via kwargs.
    Returns the draft for the specified path_index.
    """
    from skills.path_to_lf import path_to_lf_draft
    
    start_entity = kwargs.get("start_entity")
    candidate_paths = kwargs.get("candidate_paths")
    subgraph = kwargs.get("subgraph")
    
    if not all([start_entity, candidate_paths, subgraph]):
        return {"error": "Missing required context (start_entity, candidate_paths, subgraph)"}
    
    if path_index >= len(candidate_paths):
        return {"error": f"path_index {path_index} out of range (have {len(candidate_paths)} paths)"}
    
    # path_to_lf_draft processes the top path; to inspect a specific index,
    # we pass a single-element list
    single_path = [candidate_paths[path_index]]
    result = path_to_lf_draft(start_entity, single_path, subgraph)
    
    if result is None:
        return {"error": f"Could not generate LF draft for path index {path_index}"}
    
    result["path_index"] = path_index
    
    # T14: Add confidence score for inspect_path -> decide chain
    if "confidence" not in result:
        # Compute confidence based on draft quality
        confidence = 0.0
        relations = result.get("relations", [])
        directions = result.get("directions", [])
        if relations and directions and len(relations) == len(directions):
            confidence += 0.3
        sexpr = result.get("sexpr", "")
        if sexpr and "BAD" not in sexpr:
            confidence += 0.3
        func_list = result.get("function_list", [])
        if func_list:
            confidence += 0.2
        path_string = result.get("path_string", "")
        if path_string:
            confidence += 0.2
        result["confidence"] = min(1.0, confidence)
    
    return result
