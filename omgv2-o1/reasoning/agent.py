"""Greedy Reasoning Agent for OMGv2 v8 - T5 path-guided with tracing."""
from typing import List, Tuple, Optional, Dict
from skills.skill_registry import get_skill
from skills.lf_construction import function_list_to_sexpr
from skills.path_to_lf import path_to_lf_draft

# Lazy-loaded skill callables (resolved once, cached)
_EXPLORE_FN = None
_EVAL_FN = None
_EXEC_FINAL_FN = None


def _explore():
    global _EXPLORE_FN
    if _EXPLORE_FN is None:
        _EXPLORE_FN = get_skill("explore_subgraph").callable
    return _EXPLORE_FN


def _eval_relation():
    global _EVAL_FN
    if _EVAL_FN is None:
        _EVAL_FN = get_skill("evaluate_candidate_relation").callable
    return _EVAL_FN


def _exec_final():
    global _EXEC_FINAL_FN
    if _EXEC_FINAL_FN is None:
        _EXEC_FINAL_FN = get_skill("execute_final").callable
    return _EXEC_FINAL_FN


class TraceCollector:
    """Collects structured trace data for a single agent run."""

    def __init__(self):
        self.data: Dict = {
            "raw_sample": None,
            "t5_alignment": None,
            "candidate_relations": [],
            "subgraph_build": {},
            "subgraph_snapshot": None,
            "steps": [],
            "path_replay": [],
            "path_replay_steps": [],
            "final_function_list": [],
            "final_selected_relations": [],
            "sexpr": None,
            "syntax_validation": None,
            "lf_execution": None,
            "path_answers": [],
            "merged_answers": [],
            "lf_hit": None,
            "merged_hit": None,
            "path_guided_used": None,
            "path_guided_draft": None,
            "final_execution": None,
            "execution_summary": None,
            "llm_first": False,
            "pg_fallback": False,
            "kb_stats": {
                "passive_injections": 0,
                "active_consultations": 0,
                "consultation_results": [],
                "retrieved_rule_ids": [],  # T8: all retrieved rule IDs (deduplicated)
            },
            # T14: Multi-tool chain statistics
            "multi_tool_chain_stats": {
                "total_chains": 0,
                "chain_types": {},
            },
            # T15: Enhanced statistics
            "tool_usage_stats": {
                "extend_expression": 0,
                "explore_neighbors": 0,
                "verify_expression": 0,
                "consult_experience": 0,
                "inspect_path": 0,
                "multi_tool_calls": 0,  # 多工具调用次数
            },
            "action_distribution": {
                "join_forward": 0,
                "join_reverse": 0,
                "count": 0,
                "argmax": 0,
                "argmin": 0,
                "time_filter": 0,
                "and": 0,
                "cmp": 0,
                "finish": 0,
                "observation": 0,
            },
            "kb_usage_stats": {
                "passive_injections": 0,
                "active_consultations": 0,
                "consultation_success_rate": 0.0,
            },
            "performance_stats": {
                "total_steps": 0,
                "llm_calls": 0,
                "llm_fallbacks": 0,
                "pg_fallback_used": False,
            },
        }

    def record_raw_sample(self, question, start_entity, triplets, golden_answers, sample_index):
        self.data["raw_sample"] = {
            "question": question,
            "start_entity": start_entity,
            "triplets": triplets,
            "triplets_preview": str(triplets)[:2000],
            "golden_answers": list(golden_answers),
            "sample_index": sample_index,
        }

    def record_t5_input(self, candidate_paths):
        if not candidate_paths:
            self.data["t5_alignment"] = {"top1_path": None, "top1_answer": None, "num_candidates": 0, "all_candidates": []}
            return
        top1 = candidate_paths[0]
        if isinstance(top1, list):
            top1_path = top1[0]
            top1_answer = top1[1] if len(top1) > 1 else None
        else:
            top1_path = top1
            top1_answer = None
        all_cands = []
        for c in candidate_paths:
            if isinstance(c, list):
                all_cands.append({"path": c[0], "answer": c[1] if len(c) > 1 else None})
            else:
                all_cands.append({"path": c, "answer": None})
        self.data["t5_alignment"] = {
            "top1_path": top1_path,
            "top1_answer": top1_answer,
            "num_candidates": len(candidate_paths),
            "all_candidates": all_cands,
        }

    def record_candidate_relations(self, rels):
        self.data["candidate_relations"] = list(rels)

    def record_subgraph_build(self, sg_trace: dict):
        self.data["subgraph_build"] = dict(sg_trace)

    def record_subgraph_snapshot(self, snapshot: dict):
        self.data["subgraph_snapshot"] = snapshot

    def new_step(self) -> dict:
        step = {}
        self.data["steps"].append(step)
        return step

    def record_step_break_reason(self, step: dict, reason: str):
        step["break_reason"] = reason

    def record_path_replay(self, path_answer):
        self.data["path_replay"] = list(path_answer) if path_answer else []

    def record_path_replay_steps(self, transitions):
        self.data["path_replay_steps"] = list(transitions)

    def finalize(self, function_list, selected_relations, sexpr, agent_stats=None):
        self.data["final_function_list"] = list(function_list)
        self.data["final_selected_relations"] = list(selected_relations)
        self.data["sexpr"] = sexpr
        # Compute execution summary from step traces (step-level only)
        steps = self.data.get("steps", [])
        exec_scores = []
        exec_steps_with_answers = 0
        # T14: Aggregate multi-tool chain stats from steps
        total_chains = 0
        chain_types = {}
        # T15: Aggregate tool usage and action distribution from steps
        tool_usage = {
            "extend_expression": 0,
            "explore_neighbors": 0,
            "verify_expression": 0,
            "consult_experience": 0,
            "inspect_path": 0,
            "multi_tool_calls": 0,
        }
        action_dist = {
            "join_forward": 0,
            "join_reverse": 0,
            "count": 0,
            "argmax": 0,
            "argmin": 0,
            "time_filter": 0,
            "and": 0,
            "cmp": 0,
            "finish": 0,
            "observation": 0,
        }
        for s in steps:
            ev = s.get("execution_validation")
            if ev:
                exec_scores.append(ev.get("execution_score", 0.0))
                if ev.get("exec_ok"):
                    exec_steps_with_answers += 1
            # Count multi-tool chains
            mc = s.get("multi_tool_chain", {})
            if mc.get("chain_processed"):
                total_chains += 1
                ctype = mc.get("chain_type", "unknown")
                chain_types[ctype] = chain_types.get(ctype, 0) + 1
                # Count multi-tool calls
                tool_usage["multi_tool_calls"] += 1
            
            # Aggregate tool usage from observation_tool field
            obs_tool = s.get("observation_tool")
            if obs_tool and obs_tool in tool_usage:
                tool_usage[obs_tool] += 1
            
            # Aggregate action distribution
            action = s.get("action")
            if action and action in action_dist:
                action_dist[action] += 1
            # Count extend_expression calls (all non-observation actions)
            if action and action != "observation":
                tool_usage["extend_expression"] += 1
        
        self.data["execution_summary"] = {
            "num_steps_with_execution": len(exec_scores),
            "avg_execution_score": sum(exec_scores) / len(exec_scores) if exec_scores else 0.0,
            "steps_with_answers": exec_steps_with_answers,
        }
        self.data["multi_tool_chain_stats"] = {
            "total_chains": total_chains,
            "chain_types": chain_types,
        }
        # Update tool usage stats
        self.data["tool_usage_stats"] = tool_usage
        # Update action distribution
        self.data["action_distribution"] = action_dist
        
        # Sync kb_stats to kb_usage_stats
        kb_stats = self.data.get("kb_stats", {})
        kb_usage_stats = self.data.get("kb_usage_stats", {})
        kb_usage_stats["passive_injections"] = kb_stats.get("passive_injections", 0)
        kb_usage_stats["active_consultations"] = kb_stats.get("active_consultations", 0)
        # Calculate consultation success rate
        consultation_results = kb_stats.get("consultation_results", [])
        if consultation_results:
            successful_consultations = sum(1 for r in consultation_results if r.get("confidence", 0) > 0.5)
            kb_usage_stats["consultation_success_rate"] = successful_consultations / len(consultation_results)
        else:
            kb_usage_stats["consultation_success_rate"] = 0.0
        self.data["kb_usage_stats"] = kb_usage_stats
        
        # Update performance stats if agent_stats provided
        if agent_stats:
            self.data["performance_stats"].update(agent_stats)


class GreedyAgent:
    def __init__(self, question, entities, subgraph, max_steps=6, candidate_paths=None,
                 trace_collector=None, llm_first=False):
        self.question = question.lower()
        self.entities = entities
        self.subgraph = subgraph
        self.max_steps = max_steps
        self.function_list = []
        self.current_entity = None
        self.expression_id = ""
        self.step_count = 0
        self.visited_edges = set()
        self.visited_entities = set()
        # Exposed after run()
        self.selected_relations = []
        self.path_answer = set()
        # T5 path-guided bias
        self.candidate_rels = self._parse_candidate_rels(candidate_paths)
        # Tracing
        self._trace_collector = trace_collector
        self._trace = trace_collector.data if trace_collector else None
        self._candidate_paths_for_draft = candidate_paths
        # LLM-First mode
        self.llm_first = llm_first
        self._t5_draft_hint = None
        # T15: Performance statistics
        self._total_steps = 0
        self._llm_calls = 0
        self._llm_fallbacks = 0
        self._pg_fallback_used = False
        if trace_collector:
            trace_collector.record_t5_input(candidate_paths)
            trace_collector.record_candidate_relations(self.candidate_rels)

    def _parse_candidate_rels(self, candidate_paths):
        """Extract ordered relation list from top-1 candidate path.

        candidate_paths: list of [path_string, answer_string] from T5.
        path_string format: "entity->rel->entity->rel->answer"
        """
        if not candidate_paths or not candidate_paths[0]:
            return []
        path_str = candidate_paths[0][0] if isinstance(candidate_paths[0], list) else candidate_paths[0]
        parts = path_str.split("->")
        return [parts[i] for i in range(1, len(parts), 2)]

    def _select_path_guided_relation(self, available_rels):
        """Prefer the next unvisited relation from the candidate path sequence."""
        if not self.candidate_rels or not available_rels:
            return None
        matched = 0
        for sel in self.selected_relations:
            if matched >= len(self.candidate_rels):
                break
            if sel.startswith("(R ") and sel.endswith(")"):
                plain = sel[3:-1].strip()
            elif ":" in sel:
                continue
            else:
                plain = sel
            if plain == self.candidate_rels[matched]:
                matched += 1
        if matched < len(self.candidate_rels):
            next_rel = self.candidate_rels[matched]
            if next_rel in available_rels:
                return next_rel
        return None

    def _discover_one_hop(self, entity):
        """Use explore_subgraph skill to discover one-hop neighborhood."""
        edges = _explore()(self.subgraph, entity, max_hops=1)
        outgoing = []
        incoming = []
        for e in edges:
            if e["hop"] != 1:
                continue
            if e["source"] == entity:
                outgoing.append((e["relation"], e["target"]))
            elif e["target"] == entity:
                rel = e["relation"]
                if rel.startswith("(R ") and rel.endswith(")"):
                    rel = rel[3:-1].strip()
                incoming.append((rel, e["source"]))
        return outgoing, incoming

    def _dedup_by_relation(self, edge_list):
        """Group (rel, target) pairs into {rel: [targets]}."""
        result = {}
        for rel, tgt in edge_list:
            if rel not in result:
                result[rel] = []
            result[rel].append(tgt)
        return result

    def _has_literal_relation(self, out_rels):
        return self._find_literal_relation(out_rels) is not None

    def _evaluate_relation_candidate(self, relation, available_rels, reverse):
        expected_entities = available_rels.get(relation, [])
        return _eval_relation()(
            self.function_list,
            relation,
            available_rels,
            reverse=reverse,
            expected_entities=expected_entities,
        )

    def _build_validation_entry(self, result):
        return {
            "candidate": result["relation"],
            "lf_relation": result.get("lf_relation"),
            "sexpr": result.get("sexpr", ""),
            "syntax_ok": result.get("syntax_ok"),
            "exec_ok": result.get("exec_ok"),
            "num_answers": result.get("num_answers", 0),
            "execution_score": result.get("execution_score", 0.0),
            "overlap": result.get("overlap", 0),
            "expected_entities": result.get("expected_entities", []),
            "accepted": result["accepted"],
            "reason": result.get("error") or ("ok" if result["accepted"] else "rejected"),
            "orientation_results": [
                {
                    "lf_relation": item.get("lf_relation"),
                    "syntax_ok": item.get("syntax_ok"),
                    "exec_ok": item.get("exec_ok"),
                    "num_answers": item.get("num_answers", 0),
                    "execution_score": item.get("execution_score", 0.0),
                    "overlap": item.get("overlap", 0),
                    "accepted": item.get("valid", False),
                    "reason": item.get("error") or ("ok" if item.get("valid") else "rejected"),
                }
                for item in result.get("orientation_results", [])
            ],
        }

    def _choose_join_relation(self, available_rels, reverse):
        """Choose next relation with validation feedback.

        Tries path-guided first, then sorted order. Each candidate is validated
        (syntax + partial execution) before acceptance. First valid candidate wins.
        Falls back to first candidate if all are rejected.
        """
        guided = self._select_path_guided_relation(available_rels)
        rels_sorted = sorted(available_rels.keys())
        candidates = []
        if guided:
            candidates.append(guided)
        for r in rels_sorted:
            if r not in candidates:
                candidates.append(r)

        validation_trace = []
        candidate_results = []
        for rel in candidates:
            result = self._evaluate_relation_candidate(rel, available_rels, reverse)
            candidate_results.append(result)
            validation_trace.append(self._build_validation_entry(result))
            if result["accepted"]:
                if self._trace_collector and self._trace_collector.data["steps"]:
                    self._trace_collector.data["steps"][-1]["validation"] = validation_trace
                return result

        # All rejected -- fall back to first candidate
        if self._trace_collector and self._trace_collector.data["steps"]:
            self._trace_collector.data["steps"][-1]["validation"] = validation_trace
        return candidate_results[0] if candidate_results else None

    def run(self):
        self.function_list = []
        self.step_count = 0
        self.visited_edges = set()
        self.visited_entities = set()
        self.selected_relations = []
        self.path_answer = set()
        # T15: Reset performance statistics
        self._total_steps = 0
        self._llm_calls = 0
        self._llm_fallbacks = 0
        self._pg_fallback_used = False
        if not self.entities:
            return [], "@BAD_EXPRESSION"
        entity = self.entities[0]
        self._do_start(entity)

        # Record llm_first in trace
        if self._trace_collector and self.llm_first:
            self._trace_collector.data["llm_first"] = True

        # --- Path-Guided mode: try direct LF from T5 top-1 path ---
        draft = path_to_lf_draft(entity, self._candidate_paths_for_draft, self.subgraph)
        if draft is not None:
            from skills.validate_syntax import validate_syntax
            syn = validate_syntax(draft["sexpr"])
            if syn["valid"]:
                if not self.llm_first:
                    # Original PG short-circuit behavior
                    try:
                        lf_result = _exec_final()(draft["function_list"])
                        lf_answers = lf_result.get("answers", [])
                        if lf_answers:
                            # Path-Guided succeeded — short-circuit
                            self.function_list = draft["function_list"]
                            self.selected_relations = [
                                ("(R " + r + ")" if d == "reverse" else r)
                                for r, d in zip(draft["relations"], draft["directions"])
                            ]
                            self.path_answer = self._replay_path_answer()
                            sexpr = draft["sexpr"]
                            if self._trace_collector:
                                self._trace_collector.data["path_guided_used"] = True
                                self._trace_collector.data["path_guided_draft"] = draft
                                self._trace_collector.data["final_execution"] = {
                                    "sexpr": sexpr, "answers": lf_answers,
                                    "num_answers": len(lf_answers), "error": None,
                                }
                                # T15: Collect agent stats for finalize
                                agent_stats = {
                                    "total_steps": self.step_count,
                                    "llm_calls": self._llm_calls,
                                    "llm_fallbacks": self._llm_fallbacks,
                                    "pg_fallback_used": self._pg_fallback_used,
                                }
                                self._trace_collector.finalize(
                                    self.function_list, self.selected_relations, sexpr, agent_stats=agent_stats)
                            return self.function_list, sexpr
                    except Exception:
                        pass  # Fall through to step-by-step
                else:
                    # LLM-First: save draft as hint, don't short-circuit
                    self._t5_draft_hint = draft
                    if self._trace_collector:
                        self._trace_collector.data["llm_first"] = True
                        self._trace_collector.data["path_guided_draft"] = draft

        while self.step_count < self.max_steps:
            self.step_count += 1
            step_trace = None
            if self._trace_collector:
                step_trace = self._trace_collector.new_step()
                step_trace["step_num"] = self.step_count
                step_trace["current_entity_before"] = self.current_entity

            outgoing, incoming = self._discover_one_hop(self.current_entity)

            out_rels = self._dedup_by_relation(outgoing)
            in_rels = self._dedup_by_relation(incoming)

            if step_trace is not None:
                step_trace["outgoing_raw_edges"] = [
                    {"rel": r, "tgt": t} for r, t in outgoing
                ]
                step_trace["incoming_raw_edges"] = [
                    {"rel": r, "src": s} for r, s in incoming
                ]
                step_trace["out_rels_deduped"] = {
                    r: list(tgts) for r, tgts in out_rels.items()
                }
                step_trace["in_rels_deduped"] = {
                    r: list(srcs) for r, srcs in in_rels.items()
                }

            unvisited_out = {
                rel: tgts for rel, tgts in out_rels.items()
                if (self.current_entity, rel) not in self.visited_edges
            }
            unvisited_in = {
                rel: srcs for rel, srcs in in_rels.items()
                if (self.current_entity, "(R " + rel + ")") not in self.visited_edges
            }

            filtered_out = {}
            for rel, tgts in unvisited_out.items():
                remaining = [t for t in tgts if t not in self.visited_entities]
                if remaining:
                    filtered_out[rel] = remaining
            filtered_in = {}
            for rel, srcs in unvisited_in.items():
                remaining = [s for s in srcs if s not in self.visited_entities]
                if remaining:
                    filtered_in[rel] = remaining

            if step_trace is not None:
                step_trace["unvisited_out"] = {
                    r: list(tgts) for r, tgts in unvisited_out.items()
                }
                step_trace["unvisited_in"] = {
                    r: list(srcs) for r, srcs in unvisited_in.items()
                }
                step_trace["filtered_out"] = {
                    r: list(tgts) for r, tgts in filtered_out.items()
                }
                step_trace["filtered_in"] = {
                    r: list(srcs) for r, srcs in filtered_in.items()
                }
                step_trace["function_list_snapshot"] = list(self.function_list)
                step_trace["selected_relations_snapshot"] = list(self.selected_relations)

            if not filtered_out and not filtered_in:
                if step_trace is not None:
                    self._trace_collector.record_step_break_reason(step_trace, "no_unvisited_edges")
                break

            action = self._decide_action(filtered_out, filtered_in)

            if step_trace is not None:
                step_trace["action"] = action

            chosen_rel = None
            chosen_lf_rel = None

            result = self._dispatch_action(action, filtered_out, filtered_in)
            if result == "break" or result is None:
                break
            chosen_rel = result.get("relation")
            chosen_lf_rel = result.get("lf_relation")

            if step_trace is not None:
                step_trace["chosen_relation"] = chosen_rel
                step_trace["chosen_lf_relation"] = chosen_lf_rel
                step_trace["current_entity_after"] = self.current_entity

        self._do_stop()

        if self.selected_relations:
            self.path_answer = self._replay_path_answer()

        # PG fallback for LLM-First mode
        _has_answers = False
        if self.function_list:
            try:
                _tmp = _exec_final()(self.function_list)
                _has_answers = bool(_tmp.get("answers"))
            except Exception:
                pass
        if self.llm_first and (not self.function_list or not _has_answers):
            if self._t5_draft_hint is not None:
                try:
                    fallback_result = _exec_final()(self._t5_draft_hint["function_list"])
                    fallback_answers = fallback_result.get("answers", [])
                    if fallback_answers:
                        self.function_list = self._t5_draft_hint["function_list"]
                        self.selected_relations = [
                            ("(R " + r + ")" if d == "reverse" else r)
                            for r, d in zip(self._t5_draft_hint["relations"], self._t5_draft_hint["directions"])
                        ]
                        self.path_answer = self._replay_path_answer()
                        if self._trace_collector:
                            self._trace_collector.data["pg_fallback"] = True
                            self._trace_collector.data["llm_first"] = True
                            sexpr = self._t5_draft_hint["sexpr"]
                            self._trace_collector.data["final_execution"] = {
                                "sexpr": sexpr, "answers": fallback_answers,
                                "num_answers": len(fallback_answers), "error": None,
                            }
                            # T15: Collect agent stats for finalize
                            agent_stats = {
                                "total_steps": self.step_count,
                                "llm_calls": self._llm_calls,
                                "llm_fallbacks": self._llm_fallbacks,
                                "pg_fallback_used": self._pg_fallback_used,
                            }
                            # T8: Online feedback loop for LLM-First fallback path
                            try:
                                from skills.experience_kb_skill import _get_kb, get_last_retrieved_rule_ids
                                _kb = _get_kb()
                                if _kb is not None:
                                    _all_rule_ids = set()
                                    _all_rule_ids.update(get_last_retrieved_rule_ids())
                                    _kbs = self._trace_collector.data.get("kb_stats", {})
                                    for _rid in _kbs.get("retrieved_rule_ids", []):
                                        _all_rule_ids.add(_rid)
                                    for _cr in _kbs.get("consultation_results", []):
                                        for _rid in _cr.get("rule_ids", []):
                                            _all_rule_ids.add(_rid)
                                    if _all_rule_ids:
                                        _is_correct = False
                                        try:
                                            _golden = set(str(a) for a in self._trace_collector.data.get("raw_sample", {}).get("golden_answers", []))
                                            _final_ans = set(str(a) for a in fallback_answers)
                                            if _golden and _final_ans:
                                                _is_correct = bool(_final_ans & _golden)
                                        except Exception:
                                            pass
                                        for _rid in _all_rule_ids:
                                            if _rid:
                                                if _is_correct:
                                                    _kb.update_success(_rid)
                                                else:
                                                    _kb.update_failure(_rid)
                                        _fb = self._trace_collector.data.get("kb_stats", {})
                                        _fb["feedback_applied"] = True
                                        _fb["feedback_is_correct"] = _is_correct
                                        _fb["feedback_rule_count"] = len(_all_rule_ids)
                                        self._trace_collector.data["kb_stats"] = _fb
                            except Exception:
                                pass
                            self._trace_collector.finalize(
                                self.function_list, self.selected_relations, sexpr, agent_stats=agent_stats)
                        return self.function_list, self._t5_draft_hint["sexpr"]
                except Exception:
                    pass

        if self._trace_collector:
            sexpr = function_list_to_sexpr(self.function_list)
            self._trace_collector.record_path_replay(self.path_answer)
            transitions = self._replay_path_transitions()
            self._trace_collector.record_path_replay_steps(transitions)
            # Record full subgraph snapshot
            sg_snap = self.subgraph.snapshot()
            self._trace_collector.record_subgraph_snapshot(sg_snap)
            # Record final execution
            if not self._trace_collector.data.get("final_execution"):
                try:
                    final_result = _exec_final()(self.function_list)
                    self._trace_collector.data["final_execution"] = {
                        "sexpr": sexpr,
                        "answers": final_result.get("answers", []),
                        "num_answers": final_result.get("num_answers", 0),
                        "error": final_result.get("error"),
                    }
                except Exception:
                    pass
            # T15: Update performance statistics
            self._total_steps = self.step_count
            agent_stats = {
                "total_steps": self._total_steps,
                "llm_calls": self._llm_calls,
                "llm_fallbacks": self._llm_fallbacks,
                "pg_fallback_used": self._pg_fallback_used,
            }
            self._trace_collector.finalize(
                self.function_list, self.selected_relations, sexpr, agent_stats=agent_stats)
        else:
            sexpr = function_list_to_sexpr(self.function_list)

        return self.function_list, sexpr

    def _slim_observation(self, observation_text, tool_name=""):
        """P1-T3: Slim down observation text for scratchpad.
        
        - For KB consultation results, only keep rule_title and suggested_action
        - For explore_neighbors results, only keep the relation list
        - Truncate to 200 chars max
        """
        obs_str = str(observation_text)
        
        # Try to parse as dict/JSON for structured slimming
        if isinstance(observation_text, dict):
            obs = observation_text
        elif obs_str.startswith("{"):
            try:
                import json
                obs = json.loads(obs_str)
            except Exception:
                obs = None
        else:
            obs = None
        
        if obs and isinstance(obs, dict):
            # KB consultation results: only keep rule_title and suggested_action
            if tool_name == "consult_experience" or "guidance_text" in obs or "matched_rules" in obs:
                parts = []
                if "guidance_text" in obs and obs["guidance_text"]:
                    parts.append(str(obs["guidance_text"])[:150])
                if "suggested_action" in obs and obs["suggested_action"]:
                    parts.append("Suggested: " + str(obs["suggested_action"])[:100])
                if "rule_title" in obs and obs["rule_title"]:
                    parts.append("Rule: " + str(obs["rule_title"])[:80])
                if parts:
                    return "; ".join(parts)[:200]
            
            # explore_neighbors: only keep the relation list
            if tool_name == "explore_neighbors" or "discovered_relations" in obs:
                rels = obs.get("discovered_relations", [])
                if rels:
                    rel_names = [r.get("relation", str(r)) if isinstance(r, dict) else str(r) for r in rels[:10]]
                    return "Relations: " + ", ".join(rel_names)[:200]
                # Try forward_relations or other relation keys
                for key in ["forward_relations", "backward_relations", "relations"]:
                    if key in obs and obs[key]:
                        return key + ": " + ", ".join(str(r) for r in obs[key][:10])[:200]
        
        # Default: truncate to 200 chars
        return obs_str[:200]

    def _execute_observation(self, tool_name, tool_args, thought):
        """Execute an observation tool and return structured result."""
        result = {
            "tool": tool_name,
            "args": tool_args,
            "thought": thought,
            "observation": None,
            "success": False
        }

        try:
            if tool_name == "explore_neighbors":
                from skills.tools.adapters import explore_neighbors_adapter
                # Inject subgraph context
                tool_args_with_context = {**tool_args, "subgraph": self.subgraph}
                observation = explore_neighbors_adapter(**tool_args_with_context)
                result["observation"] = observation
                result["success"] = True

            elif tool_name == "verify_expression":
                from skills.tools.adapters import verify_expression_adapter
                # Inject current expression context
                from skills.lf_construction import function_list_to_sexpr
                current_expr = function_list_to_sexpr(self.function_list) if self.function_list else ""
                tool_args_with_context = {**tool_args, "expression": current_expr}
                observation = verify_expression_adapter(**tool_args_with_context)
                result["observation"] = observation
                result["success"] = True

            elif tool_name == "consult_experience":
                from skills.tools.adapters import consult_experience_adapter
                # Inject current state context
                from skills.lf_construction import function_list_to_sexpr
                current_expr = function_list_to_sexpr(self.function_list) if self.function_list else ""
                # For skill_md mode, preserve LLM-provided state_description (it IS the query);
                # only inject default context if state_description is missing.
                # For passive/active modes, inject entity context as before.
                existing_sd = tool_args.get("state_description", "")
                if tool_args.get("query_type") == "skill_md":
                    default_sd = existing_sd  # preserve LLM value
                else:
                    default_sd = existing_sd or f"Current entity: {self.current_entity}"
                tool_args_with_context = {
                    **tool_args,
                    "state_description": default_sd,
                    "last_error": tool_args.get("last_error", "") or (self._last_failure or ""),
                    "current_expression": current_expr
                }
                observation = consult_experience_adapter(**tool_args_with_context)
                result["observation"] = observation
                result["success"] = True

            elif tool_name == "inspect_path":
                from skills.tools.adapters import inspect_path_adapter
                # Inject T5 draft hint context and other required context
                tool_args_with_context = {
                    **tool_args,
                    "start_entity": self.entities[0] if self.entities else "",
                    "candidate_paths": self._candidate_paths_for_draft or [],
                    "subgraph": self.subgraph
                }
                observation = inspect_path_adapter(**tool_args_with_context)
                result["observation"] = observation
                result["success"] = True

            else:
                result["observation"] = f"Unknown observation tool: {tool_name}"
        except Exception as e:
            result["observation"] = f"Error executing {tool_name}: {str(e)}"

        # T14: Store observation result for multi-tool chaining
        self._last_observation_tool = tool_name
        self._last_observation_result = result.get("observation")
        self._observation_history.append({
            "tool": tool_name,
            "result": result.get("observation"),
            "success": result.get("success", False),
            "step": self.step_count
        })

        return result

    def _dispatch_action(self, action, filtered_out, filtered_in):
        """Dispatch an action name to the appropriate skill-backed method.

        Returns dict with 'relation' and 'lf_relation', or None on failure,
                or string "break" to signal loop termination.
        """
        if action == "finish":
            return "break"

        if action == "join_forward" and filtered_out:
            choice = self._choose_join_relation(filtered_out, reverse=False)
            if choice:
                rel = choice["relation"]
                lf_rel = choice.get("lf_relation")
                self._do_join(rel, reverse=False, lf_relation=lf_rel)
                return {"relation": rel, "lf_relation": lf_rel}

        if action == "join_reverse" and filtered_in:
            choice = self._choose_join_relation(filtered_in, reverse=True)
            if choice:
                rel = choice["relation"]
                lf_rel = choice.get("lf_relation")
                self._do_join(rel, reverse=True, lf_relation=lf_rel)
                return {"relation": "(R " + rel + ")", "lf_relation": lf_rel}

        if action == "argmax":
            rel = None
            pending = getattr(self, "_pending_llm_choice", None)
            if pending is not None:
                cand = pending.get("relation")
                if cand and cand in filtered_out:
                    rel = cand
                elif cand:
                    # Fuzzy match: pending relation partially matches available
                    for r in filtered_out:
                        if cand in r or r in cand:
                            rel = r
                            break
            if rel is None:
                rel = self._find_literal_relation(filtered_out)
            if rel:
                self._do_arg("ARGMAX", rel)
                return {"relation": "ARGMAX:" + rel}

        if action == "argmin":
            rel = None
            pending = getattr(self, "_pending_llm_choice", None)
            if pending is not None:
                cand = pending.get("relation")
                if cand and cand in filtered_out:
                    rel = cand
                elif cand:
                    for r in filtered_out:
                        if cand in r or r in cand:
                            rel = r
                            break
            if rel is None:
                rel = self._find_literal_relation(filtered_out)
            if rel:
                self._do_arg("ARGMIN", rel)
                return {"relation": "ARGMIN:" + rel}

        if action == "time_filter" and filtered_out:
            time_rel = None
            pending = getattr(self, "_pending_llm_choice", None)
            if pending is not None:
                cand = pending.get("relation") or pending.get("time_relation")
                if cand and cand in filtered_out:
                    time_rel = cand
                elif cand:
                    # Fuzzy match
                    for r in filtered_out:
                        if cand in r or r in cand:
                            time_rel = r
                            break
            if time_rel is None:
                time_rel = self._find_literal_relation(filtered_out)
            if time_rel and time_rel in filtered_out:
                tc_targets = filtered_out[time_rel]
                # Use time_value from pending if available, else fall back to first target
                tc_entity = None
                if pending is not None:
                    tc_entity = pending.get("time_value")
                if not tc_entity:
                    tc_entity = tc_targets[0] if tc_targets else self.current_entity
                self._do_tc(time_rel, tc_entity)
                if hasattr(self, "_pending_llm_choice"):
                    self._pending_llm_choice = None
                return {"relation": "TC:" + time_rel}

        if action == "count":
            self._do_count()
            return {"relation": "COUNT"}

        if action == "and":
            rel = None
            pending = getattr(self, "_pending_llm_choice", None)
            if pending is not None:
                cand = pending.get("relation")
                if cand and cand in filtered_out:
                    rel = cand
            if rel is None and filtered_out:
                # Pick first relation for simplified AND
                rel = next(iter(filtered_out.keys()))
            if rel:
                self._do_and(rel)
                if hasattr(self, "_pending_llm_choice"):
                    self._pending_llm_choice = None
                return {"relation": "AND:" + rel}

        if action == "cmp":
            rel = None
            operator = None
            pending = getattr(self, "_pending_llm_choice", None)
            if pending is not None:
                cand = pending.get("relation")
                operator = pending.get("operator")
                if cand and cand in filtered_out:
                    rel = cand
                elif cand:
                    for r in filtered_out:
                        if cand in r or r in cand:
                            rel = r
                            break
            if rel is None:
                rel = self._find_literal_relation(filtered_out)
            if rel:
                self._do_cmp(rel, operator=operator)
                if hasattr(self, "_pending_llm_choice"):
                    self._pending_llm_choice = None
                return {"relation": "CMP:" + rel}

        return None

    def _replay_path_answer(self):
        """Replay selected JOIN relations from start entity in subgraph."""
        if not self.entities or not self.selected_relations:
            return set()
        current = {self.entities[0]}
        for rel in self.selected_relations:
            if rel.startswith("ARGMAX:") or rel.startswith("ARGMIN:") or rel.startswith("TC:") or rel.startswith("AND:") or rel.startswith("CMP:") or rel == "COUNT":
                break
            next_entities = set()
            for ent in current:
                if rel.startswith("(R ") and rel.endswith(")"):
                    orig_rel = rel[3:-1].strip()
                    for r, src in self.subgraph.get_incoming(ent):
                        if r == orig_rel:
                            next_entities.add(src)
                else:
                    for tgt in self.subgraph.get_targets(ent, rel):
                        next_entities.add(tgt)
            current = next_entities
            if not current:
                break
        return current

    def _replay_path_transitions(self):
        """Replay path as list of step-by-step transition dicts."""
        if not self.entities or not self.selected_relations:
            return []
        transitions = []
        current = {self.entities[0]}
        for rel in self.selected_relations:
            if rel.startswith("ARGMAX:") or rel.startswith("ARGMIN:") or rel.startswith("TC:") or rel.startswith("AND:") or rel.startswith("CMP:") or rel == "COUNT":
                transitions.append({"relation": rel, "from": sorted(current), "to": sorted(current), "type": "aggregation"})
                break
            next_entities = set()
            for ent in current:
                if rel.startswith("(R ") and rel.endswith(")"):
                    orig_rel = rel[3:-1].strip()
                    for r, src in self.subgraph.get_incoming(ent):
                        if r == orig_rel:
                            next_entities.add(src)
                else:
                    for tgt in self.subgraph.get_targets(ent, rel):
                        next_entities.add(tgt)
            transitions.append({"relation": rel, "from": sorted(current), "to": sorted(next_entities), "type": "join"})
            current = next_entities
            if not current:
                break
        return transitions

    def _decide_action(self, out_rels, in_rels):
        """Decide next action based on question and available relations.

        T13: Non-JOIN action coverage — no step_count gate.
        """
        argmax_words = ["largest", "biggest", "highest", "greatest", "longest",
                        "last", "newest", "latest", "first", "oldest", "most recent",
                        "most famous", "most popular"]
        argmin_words = ["least", "smallest", "fewest", "lowest", "shortest", "earliest"]
        is_count = "how many" in self.question or "number of" in self.question
        is_argmax = any(w in self.question for w in argmax_words)
        is_argmin = any(w in self.question for w in argmin_words)
        is_tc = any(w in self.question for w in ["when", "time", "date"])
        is_cmp = any(w in self.question for w in ["greater than", "less than", "more than",
                                                    "older than", "younger than"])
        is_and = " and " in self.question or "both" in self.question

        if out_rels:
            # T13: prioritize non-JOIN actions (no step_count gate)
            if is_count:
                return "count"
            if is_argmax and self._has_literal_relation(out_rels):
                return "argmax"
            if is_argmin and self._has_literal_relation(out_rels):
                return "argmin"
            if is_tc and self._has_literal_relation(out_rels):
                return "time_filter"
            if is_cmp and self._has_literal_relation(out_rels):
                return "cmp"
            if is_and and len(out_rels) >= 2:
                return "and"
            return "join_forward"

        if in_rels:
            return "join_reverse"

        return "finish"

    def _find_literal_relation(self, out_rels):
        """Return first relation matching literal/scalar property patterns, or None."""
        patterns = ["date", "year", "time", "start_date", "end_date",
                     "population", "age", "height", "weight",
                     "revenue", "price", "rating"]
        for rel in out_rels:
            for p in patterns:
                if p in rel.lower():
                    return rel
        return None

    def _do_start(self, entity):
        if self.expression_id != "":
            self.expression_id = str(int(self.expression_id) + 1) if self.expression_id else "1"
        eid = self.expression_id
        fl = 'expression' + eid + ' = START("' + entity + '")'
        self.function_list.append(fl)
        self.current_entity = entity
        self.visited_entities.add(entity)

    def _do_join(self, relation, reverse=False, lf_relation=None):
        traversal_rel = "(R " + relation + ")" if reverse else relation
        lf_rel = lf_relation if lf_relation is not None else (
            "(R " + relation + ")" if reverse else relation
        )
        eid = self.expression_id
        fl = 'expression' + eid + ' = JOIN("' + lf_rel + '", expression' + eid + ')'
        self.function_list.append(fl)
        self.visited_edges.add((self.current_entity, traversal_rel))
        self.selected_relations.append(lf_rel)
        if not reverse:
            targets = self.subgraph.get_targets(self.current_entity, relation)
            for t in targets:
                if t not in self.visited_entities:
                    self.current_entity = t
                    self.visited_entities.add(t)
                    break
        else:
            for rel, src in self.subgraph.get_incoming(self.current_entity):
                if rel == relation and src not in self.visited_entities:
                    self.current_entity = src
                    self.visited_entities.add(src)
                    break

    def _do_arg(self, mode, relation):
        eid = self.expression_id
        fl = 'expression' + eid + ' = ARG("' + mode + '", expression' + eid + ', "' + relation + '")'
        self.function_list.append(fl)
        self.selected_relations.append(mode + ":" + relation)

    def _do_count(self):
        eid = self.expression_id
        fl = 'expression' + eid + ' = COUNT(expression' + eid + ')'
        self.function_list.append(fl)
        self.selected_relations.append("COUNT")


    def _do_tc(self, relation, entity):
        eid = self.expression_id
        fl = 'expression' + eid + ' = TC(expression' + eid + ', "' + relation + '", "' + entity + '")'
        self.function_list.append(fl)
        self.selected_relations.append("TC:" + relation)

    def _do_and(self, relation):
        """Execute AND operation — intersect current expression with a join on relation.
        
        Creates a sub-expression by joining on the relation, then ANDs with current expression.
        """
        # Save current expression ID as main_id
        main_id = self.expression_id
        
        # Increment for sub-expression (join on relation)
        if main_id == "":
            sub_id = "1"
        else:
            sub_id = str(int(main_id) + 1)
        
        # Create sub-expression: expressionN = JOIN(relation, expressionM)
        lf_rel = relation
        fl_sub = 'expression' + sub_id + ' = JOIN("' + lf_rel + '", expression' + main_id + ')'
        self.function_list.append(fl_sub)
        
        # Increment for AND result
        result_id = str(int(sub_id) + 1)
        
        # Create AND result: expressionK = AND(expressionM, expressionN)
        fl_and = 'expression' + result_id + ' = AND(expression' + main_id + ', expression' + sub_id + ')'
        self.function_list.append(fl_and)
        
        # Update current expression ID to result
        self.expression_id = result_id
        self.selected_relations.append("AND:" + relation)

    def _do_cmp(self, relation, operator=None):
        """Execute CMP (comparison) operation on a literal relation.
        
        Args:
            relation: The literal relation to compare (e.g., "population", "date")
            operator: Comparison operator (gt/lt/ge/le). If None, inferred from question.
        """
        if operator is None:
            operator = self._infer_cmp_operator()
        
        eid = self.expression_id
        fl = 'expression' + eid + ' = CMP("' + operator + '", "' + relation + '", expression' + eid + ')'
        self.function_list.append(fl)
        self.selected_relations.append("CMP:" + relation)
    
    def _infer_cmp_operator(self):
        """Infer comparison operator from question text."""
        q = self.question.lower()
        # Check longer phrases first to avoid substring matches
        if any(w in q for w in ["greater than or equal", "at least"]):
            return "ge"
        if any(w in q for w in ["less than or equal", "at most"]):
            return "le"
        if any(w in q for w in ["greater than", "more than", "older than"]):
            return "gt"
        if any(w in q for w in ["less than", "younger than"]):
            return "lt"
        # Default to greater than
        return "gt"

    def _do_stop(self):
        eid = self.expression_id
        fl = 'expression' + eid + ' = STOP(expression' + eid + ')'
        self.function_list.append(fl)


class LLMGuidedAgent(GreedyAgent):
    """GreedyAgent variant that asks an LLM to choose constrained tool actions.

    Falls back to the parent's heuristic/path-guided policy whenever the LLM
    output is invalid, ambiguous, empty, or rejected by validation.
    """

    def __init__(self, question, entities, subgraph, max_steps=6,
                 candidate_paths=None, trace_collector=None, llm_first=False):
        super().__init__(question, entities, subgraph, max_steps, candidate_paths,
                         trace_collector=trace_collector, llm_first=llm_first)
        self._llm_chooser_calls = 0
        self._llm_http_requests = 0
        self._llm_chooser_fallbacks = 0
        self._pending_llm_choice = None
        self.scratchpad = ""
        self._last_failure = None
        self._observation_budget = {
            "explore_neighbors": 2,
            "verify_expression": 3,
            "consult_experience": 2,
            "inspect_path": 2,
        }
        self._observation_counts = {
            "explore_neighbors": 0,
            "verify_expression": 0,
            "consult_experience": 0,
            "inspect_path": 0,
        }
        # T14: Multi-tool chain tracking
        self._last_observation_tool = None
        self._last_observation_result = None
        self._observation_history = []  # Record all observation results for multi-tool chaining
        self._multi_tool_chain_count = 0  # Track how many multi-tool chains occurred

    def _should_consult_experience(self, action, step_result):
        """判断是否需要主动查询 Experience KB。"""
        # 检查失败条件
        if self._last_failure:
            return True

        if step_result and step_result.get("error"):
            return True

        # 检查空结果（非 finish 动作）
        if action != "finish":
            step_answers = step_result.get("answers", []) if step_result else []
            if not step_answers and step_result.get("error") is None:
                return True

        return False

    def _process_multi_tool_chain(self, tool_name, obs_result, step_trace=None):
        """T14: Process multi-tool chain results after observation execution.
        
        Handles three chain patterns:
        1. explore_neighbors -> extend_expression: Store discovered relations
        2. verify_expression -> revise: Set last_failure if verification failed
        3. inspect_path -> decide: Adopt path-guided draft if confidence is high
        """
        observation = obs_result.get("observation")
        if not obs_result.get("success") or observation is None:
            return

        # Handle string observations (e.g., from consult_experience)
        if isinstance(observation, str):
            chain_info = {"tool": tool_name, "chain_processed": False}
            if step_trace is not None:
                step_trace["multi_tool_chain"] = chain_info
            return

        chain_info = {"tool": tool_name, "chain_processed": False}

        if tool_name == "explore_neighbors":
            # Pattern 1: explore_neighbors -> extend_expression
            # Store discovered relations for potential use in next step
            discovered = observation.get("discovered_relations", [])
            if discovered:
                self._last_discovered_relations = discovered
                chain_info["chain_type"] = "explore_to_extend"
                chain_info["discovered_relations_count"] = len(discovered)
                chain_info["chain_processed"] = True
                self._multi_tool_chain_count += 1
                # Write hint to scratchpad so LLM can use these relations
                rel_names = [r["relation"] for r in discovered[:5]]
                hint = "[T14 explore->extend] Discovered relations: " + ", ".join(rel_names)
                obs_n = self.scratchpad.count("Observation") + 1
                self.scratchpad += "Observation" + str(obs_n) + ": " + hint + "\n"

        elif tool_name == "verify_expression":
            # Pattern 2: verify_expression -> revise
            if not observation.get("valid", True):
                error = observation.get("error", "Verification failed")
                self._last_failure = "Expression verification failed: " + str(error)
                chain_info["chain_type"] = "verify_to_revise"
                chain_info["verification_error"] = error
                chain_info["chain_processed"] = True
                self._multi_tool_chain_count += 1
                # Write error feedback to scratchpad
                suggestions = observation.get("suggestions", [])
                feedback = "[T14 verify->revise] Verification failed: " + str(error)
                if suggestions:
                    feedback += " Suggestions: " + "; ".join(suggestions)
                obs_n = self.scratchpad.count("Observation") + 1
                self.scratchpad += "Observation" + str(obs_n) + ": " + feedback + "\n"
            else:
                # Verification passed - clear any previous failure
                chain_info["chain_type"] = "verify_passed"
                chain_info["chain_processed"] = True

        elif tool_name == "inspect_path":
            # Pattern 3: inspect_path -> decide
            confidence = observation.get("confidence", 0.0)
            if confidence > 0.7:
                # High confidence path - consider adopting it
                path_sexpr = observation.get("sexpr", "")
                if path_sexpr and "BAD" not in path_sexpr:
                    chain_info["chain_type"] = "inspect_to_decide_adopt"
                    chain_info["confidence"] = confidence
                    chain_info["chain_processed"] = True
                    self._multi_tool_chain_count += 1
                    # Write recommendation to scratchpad
                    feedback = "[T14 inspect->decide] Path confidence %.1f: %s" % (confidence, path_sexpr[:100])
                    obs_n = self.scratchpad.count("Observation") + 1
                    self.scratchpad += "Observation" + str(obs_n) + ": " + feedback + "\n"
            else:
                chain_info["chain_type"] = "inspect_to_decide_skip"
                chain_info["confidence"] = confidence
                chain_info["chain_processed"] = True

        # Record chain info in trace
        if step_trace is not None:
            step_trace["multi_tool_chain"] = chain_info

    def _decide_action(self, out_rels, in_rels):
        """Ask the LLM in ReAct scratchpad format, with heuristic fallback."""
        from reasoning.llm_agent import choose_next_step_function_call

        heuristic_action = super()._decide_action(out_rels, in_rels)
        allow_count = heuristic_action == "count"
        allow_finish = not out_rels and not in_rels
        allow_argmax = heuristic_action == "argmax"
        allow_argmin = heuristic_action == "argmin"
        allow_time_filter = heuristic_action == "time_filter"
        allow_and = heuristic_action == "and"
        allow_cmp = heuristic_action == "cmp"

        guided_source = out_rels if out_rels else in_rels
        candidate_hint = self._select_path_guided_relation(guided_source) if guided_source else None

        # Retrieve experience guidance
        try:
            from skills.experience_kb_skill import search_experience_rules, get_last_retrieved_rule_ids
            from skills.lf_construction import function_list_to_sexpr
            current_expr = function_list_to_sexpr(self.function_list) if self.function_list else ""
            exp_guidance = search_experience_rules(
                question=self.question,
                current_entity=self.current_entity if hasattr(self, 'current_entity') else "",
                current_expression=current_expr,
                available_relations=list(out_rels.keys()) + list(in_rels.keys()),
                last_failure=self._last_failure or "",
            )
            # T8: Track retrieved rule_ids for feedback loop
            if self._trace_collector:
                _retrieved = get_last_retrieved_rule_ids()
                if _retrieved:
                    _kbs = self._trace_collector.data.get("kb_stats", {})
                    _existing = set(_kbs.get("retrieved_rule_ids", []))
                    _existing.update(_retrieved)
                    _kbs["retrieved_rule_ids"] = list(_existing)
                    _kbs["passive_injections"] = _kbs.get("passive_injections", 0) + 1
                    self._trace_collector.data["kb_stats"] = _kbs
        except Exception:
            exp_guidance = ""

        llm_trace = {}
        self._llm_chooser_calls += 1
        choice, http_called = choose_next_step_function_call(
            question=self.question,
            scratchpad=self.scratchpad,
            available_forward_relations=list(out_rels.keys()),
            available_reverse_relations=list(in_rels.keys()),
            allow_count=allow_count,
            allow_finish=allow_finish,
            allow_argmax=allow_argmax,
            allow_argmin=allow_argmin,
            allow_time_filter=allow_time_filter,
            allow_and=allow_and,
            allow_cmp=allow_cmp,
            candidate_hint=candidate_hint,
            last_failure=self._last_failure,
            experience_guidance=exp_guidance,
            t5_draft_hint=self._t5_draft_hint,
            trace=llm_trace,
            last_observation_tool=self._last_observation_tool,
            last_observation_result=self._last_observation_result,
        )
        if http_called:
            self._llm_http_requests += 1

        if self._trace_collector and self._trace_collector.data["steps"]:
            current_step = self._trace_collector.data["steps"][-1]
            current_step["llm_trace"] = llm_trace

        self._pending_llm_choice = None
        self._last_failure = None

        if choice is not None:
            action = choice.get("action")
            relation = choice.get("relation")
            thought = choice.get("thought", "")

            step_n = self.scratchpad.count("Thought") + 1
            self.scratchpad += "Thought" + str(step_n) + ": " + thought + "\n"

            # Budget check BEFORE writing Action line (avoids scratchpad pollution on fallback)
            if action == "observation":
                obs_tool_name = choice.get("tool", "")
                if obs_tool_name in self._observation_budget:
                    if self._observation_counts[obs_tool_name] >= self._observation_budget[obs_tool_name]:
                        self._llm_chooser_fallbacks += 1
                        self._last_failure = "Observation budget exhausted for " + obs_tool_name
                        if self._trace_collector and self._trace_collector.data["steps"]:
                            current_step = self._trace_collector.data["steps"][-1]
                            if "llm_trace" in current_step:
                                current_step["llm_trace"]["fallback_used"] = True
                                current_step["llm_trace"]["fallback_reason"] = "observation_budget_exhausted:" + obs_tool_name
                        # Write the heuristic fallback action to keep scratchpad consistent
                        ha = heuristic_action
                        if ha in ("join_forward", "join_reverse"):
                            self.scratchpad += "Action" + str(step_n) + ": Find_relation [ fallback ]\n"
                        elif ha == "count":
                            self.scratchpad += "Action" + str(step_n) + ": Count [ expression ]\n"
                        elif ha == "finish":
                            self.scratchpad += "Action" + str(step_n) + ": Finish [ expression ]\n"
                        elif ha == "argmax":
                            self.scratchpad += "Action" + str(step_n) + ": Argmax [ expression ]\n"
                        elif ha == "argmin":
                            self.scratchpad += "Action" + str(step_n) + ": Argmin [ expression ]\n"
                        elif ha == "time_filter":
                            self.scratchpad += "Action" + str(step_n) + ": TimeFilter [ auto ]\n"
                        elif ha == "and":
                            self.scratchpad += "Action" + str(step_n) + ": And [ auto ]\n"
                        elif ha == "cmp":
                            self.scratchpad += "Action" + str(step_n) + ": Cmp [ expression ]\n"
                        return heuristic_action

            # Write Action lines (only reached if budget check passed)
            if action in ("join_forward", "join_reverse"):
                self.scratchpad += "Action" + str(step_n) + ": Find_relation [ " + relation + " ]\n"
            elif action == "count":
                self.scratchpad += "Action" + str(step_n) + ": Count [ expression ]\n"
            elif action == "finish":
                self.scratchpad += "Action" + str(step_n) + ": Finish [ expression ]\n"
            elif action == "observation":
                obs_tool = choice.get("tool", "unknown")
                self.scratchpad += "Action" + str(step_n) + ": Observe [ " + obs_tool + " ]\n"
            elif action == "argmax":
                self.scratchpad += "Action" + str(step_n) + ": Argmax [ expression ]\n"
            elif action == "argmin":
                self.scratchpad += "Action" + str(step_n) + ": Argmin [ expression ]\n"
            elif action == "time_filter":
                self.scratchpad += "Action" + str(step_n) + ": TimeFilter [ " + (relation or "auto") + " ]\n"
            elif action == "and":
                self.scratchpad += "Action" + str(step_n) + ": And [ " + (relation or "auto") + " ]\n"
            elif action == "cmp":
                self.scratchpad += "Action" + str(step_n) + ": Cmp [ expression ]\n"

            if action == "join_forward" and relation in out_rels:
                self._pending_llm_choice = choice
                return "join_forward"
            if action == "join_reverse" and relation in in_rels:
                self._pending_llm_choice = choice
                return "join_reverse"
            if action == "count" and allow_count:
                return "count"
            if action == "finish" and allow_finish:
                return "finish"
            if action == "argmax":
                self._pending_llm_choice = choice
                return "argmax"
            if action == "argmin":
                self._pending_llm_choice = choice
                return "argmin"
            if action == "time_filter":
                self._pending_llm_choice = choice
                return "time_filter"
            if action == "and":
                self._pending_llm_choice = choice
                return "and"
            if action == "cmp":
                self._pending_llm_choice = choice
                return "cmp"
            if action == "observation":
                self._pending_llm_choice = choice
                return "observation"

        self._llm_chooser_fallbacks += 1
        self._last_failure = "LLM output invalid, using heuristic: " + heuristic_action
        if self._trace_collector and self._trace_collector.data["steps"]:
            current_step = self._trace_collector.data["steps"][-1]
            if "llm_trace" in current_step:
                current_step["llm_trace"]["fallback_used"] = True
                current_step["llm_trace"]["fallback_reason"] = current_step["llm_trace"].get(
                    "fallback_reason", "invalid tool action, used heuristic fallback"
                )
        return heuristic_action
    def _choose_join_relation(self, available_rels, reverse):
        """Validate relations after the LLM has already chosen a tool action.

        The pending LLM relation is treated as a preferred candidate, but the
        environment still validates and can override it.
        """
        guided = self._select_path_guided_relation(available_rels)
        rels_sorted = sorted(available_rels.keys())
        candidates = []
        llm_choice = self._pending_llm_choice
        pending_rel = None
        if llm_choice is not None:
            expected_action = "join_reverse" if reverse else "join_forward"
            if llm_choice.get("action") == expected_action:
                rel = llm_choice.get("relation")
                if rel in available_rels:
                    pending_rel = rel
                    candidates.append(rel)
        if guided and guided not in candidates:
            candidates.append(guided)
        for r in rels_sorted:
            if r not in candidates:
                candidates.append(r)

        # Validate each candidate
        validation_trace = []
        candidate_results = []
        accepted_result = None
        for rel in candidates:
            result = self._evaluate_relation_candidate(rel, available_rels, reverse)
            candidate_results.append(result)
            validation_trace.append(self._build_validation_entry(result))
            if result["accepted"] and accepted_result is None:
                accepted_result = result

        # Record validation trace
        if self._trace_collector and self._trace_collector.data["steps"]:
            current_step = self._trace_collector.data["steps"][-1]
            current_step["validation"] = validation_trace

        self._pending_llm_choice = None
        if accepted_result is not None:
            if pending_rel is not None and accepted_result["relation"] != pending_rel:
                self._llm_chooser_fallbacks += 1
                if self._trace_collector and self._trace_collector.data["steps"]:
                    cs = self._trace_collector.data["steps"][-1]
                    if "llm_trace" in cs:
                        cs["llm_trace"]["fallback_used"] = True
                        cs["llm_trace"]["fallback_reason"] = "LLM tool relation rejected by validation"
            return accepted_result

        # All candidates rejected by validation -- fall back to first
        if pending_rel is not None:
            self._llm_chooser_fallbacks += 1
        if self._trace_collector and self._trace_collector.data["steps"]:
            cs = self._trace_collector.data["steps"][-1]
            if "llm_trace" in cs:
                cs["llm_trace"]["fallback_used"] = True
                cs["llm_trace"]["fallback_reason"] = "all candidates rejected by validation"
        return candidate_results[0] if candidate_results else None

    def run(self):
        """Run with scratchpad Observation updates and A1 failure feedback."""
        from skills.validate_syntax import validate_syntax

        self.function_list = []
        self.step_count = 0
        self.visited_edges = set()
        self.visited_entities = set()
        self.selected_relations = []
        self.path_answer = set()
        # Reset observation counts per sample
        for k in self._observation_counts:
            self._observation_counts[k] = 0
        # T14: Reset multi-tool chain tracking
        self._last_observation_tool = None
        self._last_observation_result = None
        self._observation_history = []
        self._last_discovered_relations = []
        self._multi_tool_chain_count = 0
        if not self.entities:
            return [], "@BAD_EXPRESSION"
        entity = self.entities[0]
        self._do_start(entity)

        # Record llm_first in trace
        if self._trace_collector and self.llm_first:
            self._trace_collector.data["llm_first"] = True

        # --- Path-Guided mode: try direct LF from T5 top-1 path ---
        draft = path_to_lf_draft(entity, self._candidate_paths_for_draft, self.subgraph)
        if draft is not None:
            from skills.validate_syntax import validate_syntax
            syn = validate_syntax(draft["sexpr"])
            if syn["valid"]:
                if not self.llm_first:
                    # Original PG short-circuit behavior
                    try:
                        lf_result = _exec_final()(draft["function_list"])
                        lf_answers = lf_result.get("answers", [])
                        if lf_answers:
                            self.function_list = draft["function_list"]
                            self.selected_relations = [
                                ("(R " + r + ")" if d == "reverse" else r)
                                for r, d in zip(draft["relations"], draft["directions"])
                            ]
                            self.path_answer = self._replay_path_answer()
                            sexpr = draft["sexpr"]
                            if self._trace_collector:
                                self._trace_collector.data["path_guided_used"] = True
                                self._trace_collector.data["path_guided_draft"] = draft
                                self._trace_collector.data["final_execution"] = {
                                    "sexpr": sexpr, "answers": lf_answers,
                                    "num_answers": len(lf_answers), "error": None,
                                }
                                # T15: Collect agent stats for finalize
                                agent_stats = {
                                    "total_steps": self.step_count,
                                    "llm_calls": self._llm_calls,
                                    "llm_fallbacks": self._llm_fallbacks,
                                    "pg_fallback_used": self._pg_fallback_used,
                                }
                                self._trace_collector.finalize(
                                    self.function_list, self.selected_relations, sexpr, agent_stats=agent_stats)
                            return self.function_list, sexpr
                    except Exception:
                        pass
                else:
                    # LLM-First: save draft as hint, don't short-circuit
                    self._t5_draft_hint = draft
                    if self._trace_collector:
                        self._trace_collector.data["llm_first"] = True
                        self._trace_collector.data["path_guided_draft"] = draft

        self.scratchpad = "Question: " + self.question + "\n"
        self.scratchpad += "Thought1: I need to identify the starting entity.\n"
        self.scratchpad += "Action1: Extract_entity [ " + entity + " ]\n"
        self.scratchpad += "Observation1: " + function_list_to_sexpr(self.function_list) + "\n"

        while self.step_count < self.max_steps:
            self.step_count += 1
            step_trace = None
            if self._trace_collector:
                step_trace = self._trace_collector.new_step()
                step_trace["step_num"] = self.step_count
                step_trace["current_entity_before"] = self.current_entity

            outgoing, incoming = self._discover_one_hop(self.current_entity)
            out_rels = self._dedup_by_relation(outgoing)
            in_rels = self._dedup_by_relation(incoming)

            if step_trace is not None:
                step_trace["outgoing_raw_edges"] = [{"rel": r, "tgt": t} for r, t in outgoing]
                step_trace["incoming_raw_edges"] = [{"rel": r, "src": s} for r, s in incoming]
                step_trace["out_rels_deduped"] = {r: list(tgts) for r, tgts in out_rels.items()}
                step_trace["in_rels_deduped"] = {r: list(srcs) for r, srcs in in_rels.items()}

            unvisited_out = {rel: tgts for rel, tgts in out_rels.items()
                            if (self.current_entity, rel) not in self.visited_edges}
            unvisited_in = {rel: srcs for rel, srcs in in_rels.items()
                           if (self.current_entity, "(R " + rel + ")") not in self.visited_edges}
            filtered_out = {}
            for rel, tgts in unvisited_out.items():
                remaining = [t for t in tgts if t not in self.visited_entities]
                if remaining:
                    filtered_out[rel] = remaining
            filtered_in = {}
            for rel, srcs in unvisited_in.items():
                remaining = [s for s in srcs if s not in self.visited_entities]
                if remaining:
                    filtered_in[rel] = remaining

            if step_trace is not None:
                step_trace["unvisited_out"] = {r: list(tgts) for r, tgts in unvisited_out.items()}
                step_trace["unvisited_in"] = {r: list(srcs) for r, srcs in unvisited_in.items()}
                step_trace["filtered_out"] = {r: list(tgts) for r, tgts in filtered_out.items()}
                step_trace["filtered_in"] = {r: list(srcs) for r, srcs in filtered_in.items()}
                step_trace["function_list_snapshot"] = list(self.function_list)
                step_trace["selected_relations_snapshot"] = list(self.selected_relations)

            if not filtered_out and not filtered_in:
                if step_trace is not None:
                    self._trace_collector.record_step_break_reason(step_trace, "no_unvisited_edges")
                break

            # T12: 添加重试机制变量
            retry_after_consult = False
            max_retries = 1  # 最多重试一次
            retry_count = 0
            
            # 保存状态快照，用于重试时回滚
            saved_function_list = list(self.function_list)
            saved_selected_relations = list(self.selected_relations)
            saved_current_entity = self.current_entity
            saved_visited_edges = set(self.visited_edges)
            saved_visited_entities = set(self.visited_entities)
            saved_expression_id = self.expression_id
            
            while retry_count <= max_retries:
                if retry_count == 0:
                    # 正常执行
                    action = self._decide_action(filtered_out, filtered_in)
                else:
                    # 重试：重新调用 _decide_action
                    action = self._decide_action(filtered_out, filtered_in)
                
                if step_trace is not None:
                    if retry_count == 0:
                        step_trace["action"] = action
                    else:
                        step_trace["action_retry"] = action

                chosen_rel = None
                chosen_lf_rel = None

                # Handle observation action separately
                if action == "observation":
                    # Get observation details from pending LLM choice
                    obs_choice = self._pending_llm_choice or {}
                    tool_name = obs_choice.get("tool", "unknown")
                    tool_args = obs_choice.get("args", {})
                    thought = obs_choice.get("thought", "")

                    # Execute observation tool
                    obs_result = self._execute_observation(tool_name, tool_args, thought)
                    observation_text = obs_result.get("observation", "No observation")

                    # Add observation to scratchpad (P1-T3: slim it down)
                    obs_n = self.scratchpad.count("Observation") + 1
                    slim_obs = self._slim_observation(observation_text, tool_name=tool_name)
                    self.scratchpad += "Observation" + str(obs_n) + ": " + slim_obs + "\n"

                    # Record in trace
                    if step_trace is not None:
                        if retry_count == 0:
                            step_trace["observation_tool"] = tool_name
                            step_trace["observation_args"] = tool_args
                            step_trace["observation_result"] = observation_text
                            step_trace["observation_success"] = obs_result.get("success", False)
                        else:
                            step_trace["observation_tool_retry"] = tool_name
                            step_trace["observation_args_retry"] = tool_args
                            step_trace["observation_result_retry"] = observation_text
                            step_trace["observation_success_retry"] = obs_result.get("success", False)

                    # Increment observation count for budget tracking
                    if tool_name in self._observation_counts:
                        self._observation_counts[tool_name] += 1

                    # T14: Multi-tool chain processing
                    self._process_multi_tool_chain(tool_name, obs_result, step_trace)

                    # Continue to next step (observation doesn't advance LF)
                    # 如果是重试，也需要继续
                    if retry_count == 0:
                        break  # 跳出内部循环，继续外部while循环
                    else:
                        # 重试后的observation，也继续
                        retry_count = max_retries + 1  # 确保不再重试
                        break

                result = self._dispatch_action(action, filtered_out, filtered_in)
                if result == "break" or result is None:
                    # 如果是重试，也需要break
                    if retry_count == 0:
                        break  # 跳出内部循环
                    else:
                        retry_count = max_retries + 1  # 确保不再重试
                        break
                
                chosen_rel = result.get("relation")
                chosen_lf_rel = result.get("lf_relation")

                if step_trace is not None:
                    if retry_count == 0:
                        step_trace["chosen_relation"] = chosen_rel
                        step_trace["chosen_lf_relation"] = chosen_lf_rel
                        step_trace["current_entity_after"] = self.current_entity
                    else:
                        step_trace["chosen_relation_retry"] = chosen_rel
                        step_trace["chosen_lf_relation_retry"] = chosen_lf_rel
                        step_trace["current_entity_after_retry"] = self.current_entity

                obs_n = self.scratchpad.count("Observation") + 1
                current_sexpr = function_list_to_sexpr(self.function_list)
                self.scratchpad += "Observation" + str(obs_n) + ": " + current_sexpr + "\n"

                # 检查是否需要consult
                need_consult = False
                step_result = None
                if chosen_rel and chosen_rel not in ("COUNT",) and not chosen_rel.startswith(("ARGMAX:", "ARGMIN:", "TC:", "AND:", "CMP:")):
                    try:
                        step_result = _exec_final()(self.function_list)
                        step_answers = step_result.get("answers", [])
                        if not step_answers and step_result.get("error") is None:
                            self._last_failure = "Previous step " + str(action) + " with " + str(chosen_rel) + " produced empty SPARQL results."
                        # Record per-step execution in trace
                        if step_trace is not None:
                            if retry_count == 0:
                                step_trace["execution_validation"] = {
                                    "exec_ok": len(step_answers) > 0,
                                    "num_answers": len(step_answers),
                                    "error": step_result.get("error"),
                                    "execution_score": min(1.0, len(step_answers) / 100.0) if step_answers else 0.0,
                                }
                            else:
                                step_trace["execution_validation_retry"] = {
                                    "exec_ok": len(step_answers) > 0,
                                    "num_answers": len(step_answers),
                                    "error": step_result.get("error"),
                                    "execution_score": min(1.0, len(step_answers) / 100.0) if step_answers else 0.0,
                                }
                    except Exception:
                        step_result = {"error": "execution exception"}

                    # T12: 主动查询 Experience KB after failure
                    if self._should_consult_experience(action, step_result):
                        need_consult = True

                # 如果需要consult，并且还没有重试过，则执行consult
                if need_consult and retry_count == 0:
                    # 检查 consult 预算（每个 sample 最多 2 次）
                    if self._observation_counts["consult_experience"] < self._observation_budget["consult_experience"]:
                        # 触发 consult_experience observation
                        from skills.lf_construction import function_list_to_sexpr as fl2sexpr
                        current_expr = fl2sexpr(self.function_list) if self.function_list else ""
                        consult_args = {
                            "state_description": f"Current entity: {self.current_entity}",
                            "last_error": self._last_failure or "",
                            "current_expr": current_expr,
                            "available_relations": list(filtered_out.keys()) + list(filtered_in.keys()),
                            "query_type": "active",
                        }
                        consult_result = self._execute_observation("consult_experience", consult_args, "Consulting KB after failure")

                        if consult_result.get("success"):
                            # 将 consultation 结果写入 scratchpad (P1-T3: slim it down)
                            obs_n = self.scratchpad.count("Observation") + 1
                            slim_obs = self._slim_observation(consult_result["observation"], tool_name="consult_experience")
                            self.scratchpad += "Observation" + str(obs_n) + ": " + slim_obs + "\n"

                            # 更新 trace
                            if step_trace is not None:
                                step_trace["kb_consultation"] = {
                                    "triggered": True,
                                    "args": consult_args,
                                    "result": consult_result["observation"],
                                    "success": True
                                }

                            # 增加 consult 计数
                            self._observation_counts["consult_experience"] += 1

                            # 更新 kb_stats
                            if self._trace_collector:
                                kb_stats = self._trace_collector.data.get("kb_stats", {})
                                kb_stats["active_consultations"] = kb_stats.get("active_consultations", 0) + 1
                                consultation_result = consult_result.get("observation", {})
                                if isinstance(consultation_result, dict):
                                    kb_stats.get("consultation_results", []).append({
                                        "step": self.step_count,
                                        "guidance_text": consultation_result.get("guidance_text", ""),
                                        "rule_ids": consultation_result.get("rule_ids", []),
                                        "confidence": consultation_result.get("confidence", 0.0),
                                    })
                                else:
                                    kb_stats.get("consultation_results", []).append({
                                        "step": self.step_count,
                                        "guidance_text": str(consultation_result),
                                        "rule_ids": [],
                                        "confidence": 0.0,
                                    })
                                self._trace_collector.data["kb_stats"] = kb_stats

                            # 清除 last_failure，允许重试
                            self._last_failure = None
                            
                            # T12: 设置重试标志，准备重试
                            # 回滚失败动作的状态
                            self.function_list = saved_function_list
                            self.selected_relations = saved_selected_relations
                            self.current_entity = saved_current_entity
                            self.visited_edges = saved_visited_edges
                            self.visited_entities = saved_visited_entities
                            self.expression_id = saved_expression_id
                            
                            retry_after_consult = True
                            retry_count += 1
                            continue  # 继续内部循环，进行重试
                        else:
                            # consult失败，不重试
                            break
                    else:
                        # consult预算不足，不重试
                        break
                else:
                    # 不需要consult，或者已经重试过，跳出内部循环
                    break
            
            # 如果进行了重试，在trace中记录
            if step_trace is not None and retry_count > 0:
                step_trace["retry_performed"] = True
                step_trace["retry_count"] = retry_count

        self._do_stop()
        if self.selected_relations:
            self.path_answer = self._replay_path_answer()

        # PG fallback for LLM-First mode
        _has_answers = False
        if self.function_list:
            try:
                _tmp = _exec_final()(self.function_list)
                _has_answers = bool(_tmp.get("answers"))
            except Exception:
                pass
        if self.llm_first and (not self.function_list or not _has_answers):
            if self._t5_draft_hint is not None:
                try:
                    fallback_result = _exec_final()(self._t5_draft_hint["function_list"])
                    fallback_answers = fallback_result.get("answers", [])
                    if fallback_answers:
                        self.function_list = self._t5_draft_hint["function_list"]
                        self.selected_relations = [
                            ("(R " + r + ")" if d == "reverse" else r)
                            for r, d in zip(self._t5_draft_hint["relations"], self._t5_draft_hint["directions"])
                        ]
                        self.path_answer = self._replay_path_answer()
                        if self._trace_collector:
                            self._trace_collector.data["pg_fallback"] = True
                            self._trace_collector.data["llm_first"] = True
                            sexpr = self._t5_draft_hint["sexpr"]
                            self._trace_collector.data["final_execution"] = {
                                "sexpr": sexpr, "answers": fallback_answers,
                                "num_answers": len(fallback_answers), "error": None,
                            }
                            # T15: Collect agent stats for finalize
                            agent_stats = {
                                "total_steps": self.step_count,
                                "llm_calls": self._llm_calls,
                                "llm_fallbacks": self._llm_fallbacks,
                                "pg_fallback_used": self._pg_fallback_used,
                            }
                            # T8: Online feedback loop for LLM-First fallback path
                            try:
                                from skills.experience_kb_skill import _get_kb, get_last_retrieved_rule_ids
                                _kb = _get_kb()
                                if _kb is not None:
                                    _all_rule_ids = set()
                                    _all_rule_ids.update(get_last_retrieved_rule_ids())
                                    _kbs = self._trace_collector.data.get("kb_stats", {})
                                    for _rid in _kbs.get("retrieved_rule_ids", []):
                                        _all_rule_ids.add(_rid)
                                    for _cr in _kbs.get("consultation_results", []):
                                        for _rid in _cr.get("rule_ids", []):
                                            _all_rule_ids.add(_rid)
                                    if _all_rule_ids:
                                        _is_correct = False
                                        try:
                                            _golden = set(str(a) for a in self._trace_collector.data.get("raw_sample", {}).get("golden_answers", []))
                                            _final_ans = set(str(a) for a in fallback_answers)
                                            if _golden and _final_ans:
                                                _is_correct = bool(_final_ans & _golden)
                                        except Exception:
                                            pass
                                        for _rid in _all_rule_ids:
                                            if _rid:
                                                if _is_correct:
                                                    _kb.update_success(_rid)
                                                else:
                                                    _kb.update_failure(_rid)
                                        _fb = self._trace_collector.data.get("kb_stats", {})
                                        _fb["feedback_applied"] = True
                                        _fb["feedback_is_correct"] = _is_correct
                                        _fb["feedback_rule_count"] = len(_all_rule_ids)
                                        self._trace_collector.data["kb_stats"] = _fb
                            except Exception:
                                pass
                            self._trace_collector.finalize(
                                self.function_list, self.selected_relations, sexpr, agent_stats=agent_stats)
                        return self.function_list, self._t5_draft_hint["sexpr"]
                except Exception:
                    pass

        sexpr = function_list_to_sexpr(self.function_list)
        syn = validate_syntax(sexpr)
        if self._trace_collector:
            # Record final execution
            if not self._trace_collector.data.get("final_execution"):
                try:
                    final_result = _exec_final()(self.function_list)
                    self._trace_collector.data["final_execution"] = {
                        "sexpr": sexpr,
                        "answers": final_result.get("answers", []),
                        "num_answers": final_result.get("num_answers", 0),
                        "error": final_result.get("error"),
                    }
                except Exception:
                    pass
            # T15: Collect agent stats for finalize
            agent_stats = {
                "total_steps": self.step_count,
                "llm_calls": self._llm_calls,
                "llm_fallbacks": self._llm_fallbacks,
                "pg_fallback_used": self._pg_fallback_used,
            }
            # T8: Online feedback loop — update success/fail counts for retrieved rules
            try:
                from skills.experience_kb_skill import _get_kb, get_last_retrieved_rule_ids
                kb = _get_kb()
                if kb is not None:
                    all_rule_ids = set()
                    all_rule_ids.update(get_last_retrieved_rule_ids())
                    if self._trace_collector:
                        kbs = self._trace_collector.data.get("kb_stats", {})
                        for rid in kbs.get("retrieved_rule_ids", []):
                            all_rule_ids.add(rid)
                        for cr in kbs.get("consultation_results", []):
                            for rid in cr.get("rule_ids", []):
                                all_rule_ids.add(rid)
                    if all_rule_ids:
                        is_correct = False
                        try:
                            golden = set()
                            if self._trace_collector:
                                raw = self._trace_collector.data.get("raw_sample", {})
                                golden = set(str(a) for a in raw.get("golden_answers", []))
                            final_ans = set()
                            fe = self._trace_collector.data.get("final_execution", {}) if self._trace_collector else {}
                            final_ans = set(str(a) for a in fe.get("answers", []))
                            if golden and final_ans:
                                is_correct = bool(final_ans & golden)
                        except Exception:
                            pass
                        for rule_id in all_rule_ids:
                            if rule_id:
                                if is_correct:
                                    kb.update_success(rule_id)
                                else:
                                    kb.update_failure(rule_id)
                        if self._trace_collector:
                            fb = self._trace_collector.data.get("kb_stats", {})
                            fb["feedback_applied"] = True
                            fb["feedback_is_correct"] = is_correct
                            fb["feedback_rule_count"] = len(all_rule_ids)
                            self._trace_collector.data["kb_stats"] = fb
            except Exception as e:
                pass
            self._trace_collector.finalize(self.function_list, self.selected_relations, sexpr, agent_stats=agent_stats)
        return self.function_list, sexpr