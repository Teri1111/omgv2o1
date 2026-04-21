"""LLM helpers for stepwise action selection in OMGv2.

Provides both the original next-relation chooser and a more tool-like action
chooser that selects constrained agent actions step by step.

P1 Refactor: Chat API + KBQA-o1 pure-text format + think-block stripping.
"""

import json
import re
import requests
from typing import List, Optional, Tuple


LLM_ENDPOINT = "http://127.0.0.1:8002/v1/chat/completions"
LLM_MODEL = "deepseek-r1:14b"


def _request_chat_completion(system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> Tuple[str, bool, Optional[str]]:
    """Request a chat completion from the LLM endpoint.
    
    Returns (raw_text, http_called, fallback_reason).
    Strips <think> blocks from the response before returning.
    """
    http_called = False
    raw_text = ""
    fallback_reason = None

    try:
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        resp = requests.post(LLM_ENDPOINT, json=payload, timeout=60)
        http_called = True

        if resp.status_code == 200:
            data = resp.json()
            raw_text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        else:
            fallback_reason = "HTTP " + str(resp.status_code)
    except requests.exceptions.Timeout as e:
        raise RuntimeError(f"LLM request timed out after 60s ({LLM_ENDPOINT}): {e}") from e
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"LLM connection failed ({LLM_ENDPOINT}): {e}") from e
    except Exception as e:
        fallback_reason = str(e)

    # P1-T2: Strip <think> blocks from response
    if raw_text:
        raw_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()

    if not raw_text and fallback_reason is None:
        fallback_reason = "empty LLM response"

    return raw_text, http_called, fallback_reason


def _extract_json_object(raw_text: str):
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    snippet = raw_text[start:end + 1]
    try:
        return json.loads(snippet)
    except Exception:
        return None


# --- P1-T4: Structured argument parsers ---

def _parse_compare_argument(argument: str) -> dict:
    """Parse compare argument into {operator, relation}.
    
    Formats:
        "GT | film.film.revenue" -> {operator: "gt", relation: "film.film.revenue"}
        "film.film.revenue"      -> {operator: "gt", relation: "film.film.revenue"}  (default gt)
    """
    if not argument:
        return {"operator": "gt", "relation": ""}
    parts = [p.strip() for p in argument.split("|")]
    _op_map = {"gt": "gt", "lt": "lt", "ge": "ge", "le": "le",
               "greater": "gt", "less": "lt", ">": "gt", "<": "lt",
               ">=": "ge", "<=": "le"}
    if len(parts) == 2 and parts[0].lower() in _op_map:
        return {"operator": _op_map[parts[0].lower()], "relation": parts[1]}
    # No operator prefix — use raw argument as relation, default gt
    return {"operator": "gt", "relation": argument}


def _parse_time_constraint_argument(argument: str) -> dict:
    """Parse time_constraint argument into {relation, time_value}.
    
    Formats:
        "film.film.date | 2000" -> {relation: "film.film.date", time_value: "2000"}
        "film.film.date"        -> {relation: "film.film.date", time_value: ""}  (no value)
    """
    if not argument:
        return {"relation": "", "time_value": ""}
    parts = [p.strip() for p in argument.split("|")]
    if len(parts) == 2:
        return {"relation": parts[0], "time_value": parts[1]}
    return {"relation": argument, "time_value": ""}


# --- P1-T4: Unified Action Parser ---

def parse_action(raw_text: str, available_fwd: list, available_rev: list,
                 allow_count=False, allow_finish=False,
                 allow_argmax=False, allow_argmin=False,
                 allow_time_filter=False, allow_and=False, allow_cmp=False) -> Optional[dict]:
    """Parse KBQA-o1 style action from LLM output.

    Expected format:
        Thought{N}: <reasoning>
        Action{N}: ActionType [ argument ]

    Returns: {"action": ..., "relation": ..., "thought": ...} or None
    """
    # 1. Strip <think> blocks (redundant but safe)
    text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()

    # 2. Extract Thought
    thought_match = re.search(r'Thought\s*\d*\s*:\s*(.*?)(?=Action\s*\d*\s*:|$)', text, re.DOTALL | re.IGNORECASE)
    thought = thought_match.group(1).strip() if thought_match else ""

    # 3. Extract Action line
    action_match = re.search(
        r'Action\s*\d*\s*:\s*'
        r'(Find_relation|Count|Finish|Argmax|Argmin|Extract_entity|Merge|Compare|Time_constraint)'
        r'\s*\[\s*([^\]]+)\s*\]',
        text, re.DOTALL | re.IGNORECASE
    )
    if not action_match:
        # Fuzzy fallback: scan for relation names in text
        all_rels = available_fwd + available_rev
        for rel in sorted(all_rels, key=len, reverse=True):
            if rel in text:
                return {"action": "join_forward" if rel in available_fwd else "join_reverse",
                        "relation": rel, "thought": thought}
        return None

    action_type = action_match.group(1).strip().lower()
    argument = action_match.group(2).strip().strip('"\'')

    # 4. Map action type
    if action_type == "find_relation":
        # Match relation
        if argument in available_fwd:
            return {"action": "join_forward", "relation": argument, "thought": thought}
        elif argument in available_rev:
            return {"action": "join_reverse", "relation": argument, "thought": thought}
        # Fuzzy match
        for rel in available_fwd:
            if rel in argument or argument in rel:
                return {"action": "join_forward", "relation": rel, "thought": thought}
        for rel in available_rev:
            if rel in argument or argument in rel:
                return {"action": "join_reverse", "relation": rel, "thought": thought}
        return None
    elif action_type == "count" and allow_count:
        return {"action": "count", "relation": None, "thought": thought}
    elif action_type == "finish" and allow_finish:
        return {"action": "finish", "relation": None, "thought": thought}
    elif action_type == "argmax" and allow_argmax:
        return {"action": "argmax", "relation": argument, "thought": thought}
    elif action_type == "argmin" and allow_argmin:
        return {"action": "argmin", "relation": argument, "thought": thought}
    elif action_type == "merge" and allow_and:
        return {"action": "and", "relation": argument, "thought": thought}
    elif action_type == "compare" and allow_cmp:
        # Parse structured args: "GT | film.film.revenue" -> {operator: gt, relation: film.film.revenue}
        parsed = _parse_compare_argument(argument)
        return {"action": "cmp", "relation": parsed["relation"], "operator": parsed["operator"], "thought": thought}
    elif action_type == "time_constraint" and allow_time_filter:
        # Parse structured args: "film.film.date | 2000" -> {relation: film.film.date, time_value: "2000"}
        parsed = _parse_time_constraint_argument(argument)
        return {"action": "time_filter", "relation": parsed["relation"], "time_value": parsed["time_value"], "thought": thought}
    return None


# --- P1-T1: New unified choose_next_step ---

def choose_next_step(
    question,
    scratchpad,
    available_forward_relations,
    available_reverse_relations,
    allow_count=False,
    allow_finish=False,
    candidate_hint=None,
    last_failure=None,
    experience_guidance="",
    t5_draft_hint=None,
    trace=None,
    allow_argmax=False,
    allow_argmin=False,
    allow_time_filter=False,
    allow_and=False,
    allow_cmp=False,
    last_observation_tool=None,
    last_observation_result=None,
):
    """Unified KBQA-o1 style step chooser using Chat API.
    
    Builds a system prompt describing available actions in KBQA-o1 format,
    and a user prompt with question, available relations, draft hint, and scratchpad.
    Returns (parsed_choice_or_None, http_called).
    """
    # Build action instructions for system prompt
    action_lines = []
    if available_forward_relations or available_reverse_relations:
        action_lines.append("- Find_relation [ relation_name ] -> expression = JOIN('relation', expression)")
    if allow_count:
        action_lines.append("- Count [ expression ] -> expression = COUNT(expression)")
    if allow_finish:
        action_lines.append("- Finish [ expression ] -> expression = STOP(expression)")
    if allow_argmax:
        action_lines.append("- Argmax [ relation ] -> expression = ARGMAX(expression, 'relation')")
    if allow_argmin:
        action_lines.append("- Argmin [ relation ] -> expression = ARGMIN(expression, 'relation')")
    if allow_time_filter:
        action_lines.append("- Time_constraint [ relation | time_value ] -> expression = TC(expression, 'relation', 'time')")
    if allow_and:
        action_lines.append("- Merge [ relation ] -> expression = AND(JOIN('relation', expression), expression)")
    if allow_cmp:
        action_lines.append("- Compare [ GT/LT/GE/LE | relation ] -> expression = CMP('mode', 'relation', expression)")

    action_block = "\n".join(action_lines) if action_lines else "- Find_relation [ relation_name ]"

    system_prompt = (
        "You are a KBQA agent that constructs S-expressions step by step.\n"
        "\n"
        "Available actions:\n"
        + action_block + "\n"
        "\n"
        "Rules:\n"
        "1. Reply with exactly one Thought and one Action per turn\n"
        "2. Only use relations from the Available Relations list\n"
        "3. Format: Thought{N}: <reasoning>\\nAction{N}: <ActionType> [ <argument> ]\n"
        "4. Do NOT wrap your response in JSON or code fences\n"
        "5. Never answer the question directly — always output an Action to build the query\n"
    )

    # Build user prompt
    user_lines = [
        "Question: " + question,
    ]

    # State description
    if available_forward_relations:
        user_lines.append("Available forward relations: " + ", ".join(sorted(available_forward_relations)))
    if available_reverse_relations:
        user_lines.append("Available reverse relations: " + ", ".join(sorted(available_reverse_relations)))
    if candidate_hint:
        user_lines.append("Hint from path search: " + candidate_hint)
    if t5_draft_hint:
        draft_sexpr = t5_draft_hint.get("sexpr", "N/A")
        user_lines.append("T5 Draft S-expression: " + draft_sexpr)
        draft_path = t5_draft_hint.get("path_string", "")
        if draft_path:
            user_lines.append("T5 Draft path: " + draft_path)
        if t5_draft_hint.get("relations"):
            rels = []
            for rel, direction in zip(t5_draft_hint["relations"], t5_draft_hint["directions"]):
                rels.append("(R " + rel + ")" if direction == "reverse" else rel)
            user_lines.append("T5 Draft relations: " + " -> ".join(rels))
    if last_failure:
        user_lines.append("Last step failed: " + last_failure)
    # T14: Add observation context for multi-tool chaining
    if last_observation_tool and last_observation_result:
        user_lines.append("Last observation tool: " + last_observation_tool)
        obs_result_str = str(last_observation_result)[:200]
        user_lines.append("Last observation result: " + obs_result_str)

    # Guidance block
    if experience_guidance:
        user_lines.append("")
        user_lines.append(experience_guidance)

    user_lines.append("")
    user_lines.append(scratchpad)

    # Calculate step number
    step_n = scratchpad.count("Thought") + 1
    user_lines.append("Thought" + str(step_n) + ":")

    user_prompt = "\n".join(user_lines)

    if trace is not None:
        trace["prompt_context"] = user_prompt
        trace["system_prompt"] = system_prompt
        trace["available_forward_relations"] = sorted(available_forward_relations)
        trace["available_reverse_relations"] = sorted(available_reverse_relations)
        trace["candidate_hint"] = candidate_hint
        if t5_draft_hint:
            trace["t5_draft_hint_sexpr"] = t5_draft_hint.get("sexpr")
        if last_failure:
            trace["last_failure"] = last_failure
        if experience_guidance:
            trace["experience_guidance"] = experience_guidance
        trace["mode"] = "chat_o1"

    raw_text, http_called, fallback_reason = _request_chat_completion(system_prompt, user_prompt)

    parsed = None
    thought = ""

    if raw_text:
        parsed = parse_action(
            raw_text,
            available_fwd=available_forward_relations,
            available_rev=available_reverse_relations,
            allow_count=allow_count,
            allow_finish=allow_finish,
            allow_argmax=allow_argmax,
            allow_argmin=allow_argmin,
            allow_time_filter=allow_time_filter,
            allow_and=allow_and,
            allow_cmp=allow_cmp,
        )
        if parsed:
            thought = parsed.get("thought", "")

    # Fallback: try JSON extraction if parse_action failed
    if parsed is None and raw_text:
        tool_payload = _extract_json_object(raw_text)
        if tool_payload and "tool" in tool_payload:
            tool_name = str(tool_payload.get("tool", "")).strip()
            tool_args = tool_payload.get("args", {})
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except Exception:
                    tool_args = {}
            tool_thought = tool_payload.get("thought", "")

            if tool_name == "extend_expression":
                action_arg = str(tool_args.get("action", "")).strip().lower()
                direction = str(tool_args.get("direction", "forward")).strip().lower()
                relation = tool_args.get("relation")

                if action_arg == "join" and relation:
                    if direction == "reverse" and relation in available_reverse_relations:
                        parsed = {"action": "join_reverse", "relation": relation, "thought": tool_thought}
                    elif direction == "forward" and relation in available_forward_relations:
                        parsed = {"action": "join_forward", "relation": relation, "thought": tool_thought}
                    else:
                        # Invalid direction — record warning
                        if direction not in ("forward", "reverse"):
                            fallback_reason = "Invalid direction '%s', corrected by fuzzy match" % direction
                        # Fuzzy match
                        for rel in available_forward_relations:
                            if rel in str(relation) or str(relation) in rel:
                                parsed = {"action": "join_forward", "relation": rel, "thought": tool_thought}
                                break
                        if parsed is None:
                            for rel in available_reverse_relations:
                                if rel in str(relation) or str(relation) in rel:
                                    parsed = {"action": "join_reverse", "relation": rel, "thought": tool_thought}
                                    break
                elif action_arg == "count" and allow_count:
                    parsed = {"action": "count", "relation": None, "thought": tool_thought}
                elif action_arg == "finish" and allow_finish:
                    parsed = {"action": "finish", "relation": None, "thought": tool_thought}
                elif action_arg == "argmax" and allow_argmax:
                    parsed = {"action": "argmax", "relation": tool_args.get("relation"), "thought": tool_thought}
                elif action_arg == "argmin" and allow_argmin:
                    parsed = {"action": "argmin", "relation": tool_args.get("relation"), "thought": tool_thought}
            elif tool_name in ("explore_neighbors", "verify_expression", "consult_experience", "inspect_path"):
                parsed = {
                    "type": "observation",
                    "tool": tool_name,
                    "args": tool_args,
                    "thought": tool_thought or thought,
                    "action": "observation"
                }

    # Final fallback: text-based heuristic
    if parsed is None and raw_text:
        cleaned = raw_text.strip().lower()
        if "count" in cleaned and allow_count:
            parsed = {"action": "count", "relation": None, "thought": ""}
        elif "finish" in cleaned and allow_finish:
            parsed = {"action": "finish", "relation": None, "thought": ""}
        elif "argmax" in cleaned and allow_argmax:
            parsed = {"action": "argmax", "relation": None, "thought": ""}
        elif "argmin" in cleaned and allow_argmin:
            parsed = {"action": "argmin", "relation": None, "thought": ""}
        else:
            # Try to find a relation in the raw text
            for rel in available_forward_relations:
                if rel in raw_text:
                    parsed = {"action": "join_forward", "relation": rel, "thought": ""}
                    break
            if parsed is None:
                for rel in available_reverse_relations:
                    if rel in raw_text:
                        parsed = {"action": "join_reverse", "relation": rel, "thought": ""}
                        break

    if trace is not None:
        trace["raw_text"] = raw_text
        trace["thought"] = parsed.get("thought", thought) if parsed else thought
        trace["parsed_action"] = parsed.get("action") if parsed else None
        trace["parsed_relation"] = parsed.get("relation") if parsed else None
        trace["http_called"] = http_called
        if fallback_reason:
            trace["fallback_reason"] = fallback_reason
        if not parsed and raw_text:
            trace["parse_failure"] = "Could not parse action from: " + raw_text[:200]

    return parsed, http_called


# --- Backward-compatible wrappers ---

def choose_next_step_function_call(
    question,
    scratchpad,
    available_forward_relations,
    available_reverse_relations,
    allow_count=False,
    allow_finish=False,
    candidate_hint=None,
    last_failure=None,
    experience_guidance="",
    t5_draft_hint=None,
    trace=None,
    allow_argmax=False,
    allow_argmin=False,
    allow_time_filter=False,
    allow_and=False,
    allow_cmp=False,
    last_observation_tool=None,
    last_observation_result=None,
):
    """Backward-compatible wrapper that delegates to choose_next_step().
    
    This replaces the old function-calling protocol with the new KBQA-o1
    chat-based approach. Includes ReAct fallback for backward compatibility.
    """
    parsed, http_called = choose_next_step(
        question=question,
        scratchpad=scratchpad,
        available_forward_relations=available_forward_relations,
        available_reverse_relations=available_reverse_relations,
        allow_count=allow_count,
        allow_finish=allow_finish,
        candidate_hint=candidate_hint,
        last_failure=last_failure,
        experience_guidance=experience_guidance,
        t5_draft_hint=t5_draft_hint,
        trace=trace,
        allow_argmax=allow_argmax,
        allow_argmin=allow_argmin,
        allow_time_filter=allow_time_filter,
        allow_and=allow_and,
        allow_cmp=allow_cmp,
        last_observation_tool=last_observation_tool,
        last_observation_result=last_observation_result,
    )

    # If primary parse failed, try ReAct fallback
    if parsed is None:
        if trace is not None:
            trace["function_call_failed"] = True
        react_parsed, react_http = choose_next_step_react(
            question=question,
            scratchpad=scratchpad,
            available_forward_relations=available_forward_relations,
            available_reverse_relations=available_reverse_relations,
            allow_count=allow_count,
            allow_finish=allow_finish,
            candidate_hint=candidate_hint,
            last_failure=last_failure,
            experience_guidance=experience_guidance,
            t5_draft_hint=t5_draft_hint,
            trace=None,  # Don't pass trace to ReAct to avoid cross-contamination
            allow_argmax=allow_argmax,
            allow_argmin=allow_argmin,
            allow_time_filter=allow_time_filter,
            allow_and=allow_and,
            allow_cmp=allow_cmp,
        )
        if react_parsed is not None:
            if trace is not None:
                trace["function_call_recovered_by_react"] = True
                trace["recovered_action"] = react_parsed.get("action")
            return react_parsed, http_called or react_http

    return parsed, http_called


def choose_next_step_react(
    question,
    scratchpad,
    available_forward_relations,
    available_reverse_relations,
    allow_count=False,
    allow_finish=False,
    candidate_hint=None,
    last_failure=None,
    experience_guidance="",
    t5_draft_hint=None,
    trace=None,
    allow_argmax=False,
    allow_argmin=False,
    allow_time_filter=False,
    allow_and=False,
    allow_cmp=False,
):
    """Backward-compatible ReAct wrapper that delegates to choose_next_step().
    
    Kept for imports in agent.py. Now uses the same Chat API under the hood.
    """
    return choose_next_step(
        question=question,
        scratchpad=scratchpad,
        available_forward_relations=available_forward_relations,
        available_reverse_relations=available_reverse_relations,
        allow_count=allow_count,
        allow_finish=allow_finish,
        candidate_hint=candidate_hint,
        last_failure=last_failure,
        experience_guidance=experience_guidance,
        t5_draft_hint=t5_draft_hint,
        trace=trace,
        allow_argmax=allow_argmax,
        allow_argmin=allow_argmin,
        allow_time_filter=allow_time_filter,
        allow_and=allow_and,
        allow_cmp=allow_cmp,
    )


def choose_next_relation(
    question: str,
    entity: str,
    available_relations: List[str],
    candidate_hint: Optional[str] = None,
    trace: Optional[dict] = None,
) -> Tuple[Optional[str], bool]:
    """Ask the LLM to pick the next relation from available_relations.

    Returns (chosen_relation_or_None, http_called).
    When trace dict is provided, records prompt_context, raw_text, parsed_choice.
    """
    if not available_relations:
        return None, False

    rel_list_str = ", ".join(sorted(available_relations))
    hint_line = ""
    if candidate_hint:
        hint_line = "\nSuggested relation from path search: " + candidate_hint

    system_prompt = "You are a KBQA agent. Reply with ONLY the relation name, nothing else."
    user_prompt = (
        "Question: " + question + "\n"
        "Current entity: " + entity + "\n"
        "Available relations: " + rel_list_str + "\n"
        "Which relation should I follow next to answer the question?"
        + hint_line
    )

    if trace is not None:
        trace["prompt_context"] = user_prompt
        trace["available_relations"] = sorted(available_relations)
        trace["candidate_hint"] = candidate_hint

    parsed_choice = None
    raw_text, http_called, fallback_reason = _request_chat_completion(system_prompt, user_prompt)

    if raw_text:
        cleaned = raw_text.strip().strip("\"'\n")
        if cleaned in available_relations:
            parsed_choice = cleaned
        else:
            for rel in available_relations:
                if rel in cleaned:
                    parsed_choice = rel
                    break
            if parsed_choice is None:
                fallback_reason = "No relation match in: " + cleaned[:100]
    else:
        fallback_reason = "empty LLM response"

    if trace is not None:
        trace["raw_text"] = raw_text
        trace["parsed_choice"] = parsed_choice
        trace["http_called"] = http_called
        if fallback_reason:
            trace["fallback_reason"] = fallback_reason

    return parsed_choice, http_called


def choose_next_tool_action(
    question: str,
    entity: str,
    current_expression: str,
    available_forward_relations: List[str],
    available_reverse_relations: List[str],
    allow_count: bool = False,
    allow_finish: bool = False,
    allow_argmax: bool = False,
    allow_argmin: bool = False,
    allow_time_filter: bool = False,
    candidate_hint: Optional[str] = None,
    trace: Optional[dict] = None,
):
    """Ask the LLM to choose the next tool action and optional relation.

    Returns ({"action": ..., "relation": ...} or None, http_called).
    The response is constrained to a small tool-action space so the model is
    choosing tools, not free-form LF text.
    """
    action_space = []
    if available_forward_relations:
        action_space.append("join_forward")
    if available_reverse_relations:
        action_space.append("join_reverse")
    if allow_count:
        action_space.append("count")
    if allow_finish:
        action_space.append("finish")
    if allow_argmax:
        action_space.append("argmax")
    if allow_argmin:
        action_space.append("argmin")
    if allow_time_filter:
        action_space.append("time_filter")

    if not action_space:
        return None, False

    system_prompt = "You are building a KBQA logical form step by step by calling one tool at a time. Reply with JSON only."

    prompt_lines = [
        "Question: " + question,
        "Current entity: " + entity,
        "Current expression: " + current_expression,
        "Allowed actions: " + ", ".join(action_space),
    ]
    if available_forward_relations:
        prompt_lines.append(
            "Forward relations for join_forward: " + ", ".join(sorted(available_forward_relations))
        )
    if available_reverse_relations:
        prompt_lines.append(
            "Reverse relations for join_reverse: " + ", ".join(sorted(available_reverse_relations))
        )
    if candidate_hint:
        prompt_lines.append("Suggested next relation from path search: " + candidate_hint)
    prompt_lines.extend([
        "Use exactly this schema:",
        '{"action": "join_forward|join_reverse|count|finish|argmax|argmin|time_filter", "relation": "relation_name_or_null", "time_relation": "time_relation_or_null"}',
        "Set relation to null for count, finish, argmax, or argmin.",
        "For argmax/argmin, the system will auto-detect a literal relation.",
        "For time_filter, set relation or time_relation to a date/time relation from the forward list.",
    ])
    user_prompt = "\n".join(prompt_lines)

    if trace is not None:
        trace["prompt_context"] = user_prompt
        trace["available_forward_relations"] = sorted(available_forward_relations)
        trace["available_reverse_relations"] = sorted(available_reverse_relations)
        trace["candidate_hint"] = candidate_hint

    raw_text, http_called, fallback_reason = _request_chat_completion(system_prompt, user_prompt)
    parsed_action = None
    parsed_relation = None

    payload = _extract_json_object(raw_text) if raw_text else None
    if payload:
        action = str(payload.get("action", "")).strip().lower()
        relation = payload.get("relation")
        if relation is not None:
            relation = str(relation).strip()

        if action == "join_forward" and relation in available_forward_relations:
            parsed_action = action
            parsed_relation = relation
        elif action == "join_reverse" and relation in available_reverse_relations:
            parsed_action = action
            parsed_relation = relation
        elif action == "count" and allow_count:
            parsed_action = action
        elif action == "finish" and allow_finish:
            parsed_action = action
        elif action == "argmax" and allow_argmax:
            parsed_action = action
        elif action == "argmin" and allow_argmin:
            parsed_action = action
        elif action == "time_filter" and allow_time_filter:
            time_rel = payload.get("time_relation") or relation
            if time_rel is not None:
                time_rel = str(time_rel).strip()
            if time_rel and time_rel in available_forward_relations:
                parsed_action = action
                parsed_relation = time_rel
            else:
                fallback_reason = "time_filter without valid time_relation"
        else:
            fallback_reason = "Invalid action payload"
    elif raw_text:
        cleaned = raw_text.strip().lower()
        if "count" in cleaned and allow_count:
            parsed_action = "count"
        elif "finish" in cleaned and allow_finish:
            parsed_action = "finish"
        elif "argmax" in cleaned and allow_argmax:
            parsed_action = "argmax"
        elif "argmin" in cleaned and allow_argmin:
            parsed_action = "argmin"
        elif "time_filter" in cleaned and allow_time_filter:
            for rel in available_forward_relations:
                if rel in raw_text and any(t in rel.lower() for t in ["date", "year", "time", "start_date", "end_date"]):
                    parsed_action = "time_filter"
                    parsed_relation = rel
                    break
            if parsed_action is None:
                fallback_reason = "time_filter text fallback without valid time relation"
        else:
            for rel in available_forward_relations:
                if rel in raw_text:
                    parsed_action = "join_forward"
                    parsed_relation = rel
                    break
            if parsed_action is None:
                for rel in available_reverse_relations:
                    if rel in raw_text:
                        parsed_action = "join_reverse"
                        parsed_relation = rel
                        break
            if parsed_action is None:
                fallback_reason = "No valid tool action match"

    if trace is not None:
        trace["raw_text"] = raw_text
        trace["parsed_action"] = parsed_action
        trace["parsed_relation"] = parsed_relation
        trace["http_called"] = http_called
        if fallback_reason:
            trace["fallback_reason"] = fallback_reason

    if parsed_action is None:
        return None, http_called

    return {"action": parsed_action, "relation": parsed_relation}, http_called
