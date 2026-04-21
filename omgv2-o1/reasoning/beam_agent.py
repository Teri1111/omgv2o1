"""Beam Search Reasoning Agent for OMGv2 - Phase 4.

Explores multiple candidate branches in parallel instead of a single greedy path.
Uses execute_partial() for pruning at each step, execution_score for ranking.
Supports COUNT/ARGMAX/ARGMIN/TC actions alongside JOIN.
Trace-compatible with TraceCollector (steps/execution_validation/path_replay).
"""
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
from skills.skill_registry import get_skill
from skills.lf_construction import function_list_to_sexpr
from skills.path_to_lf import path_to_lf_draft

# Lazy-loaded skill callables
_EXPLORE_FN = None
_EXEC_PARTIAL_FN = None
_EXEC_FINAL_FN = None


def _explore():
    global _EXPLORE_FN
    if _EXPLORE_FN is None:
        _EXPLORE_FN = get_skill("explore_subgraph").callable
    return _EXPLORE_FN


def _exec_partial():
    global _EXEC_PARTIAL_FN
    if _EXEC_PARTIAL_FN is None:
        _EXEC_PARTIAL_FN = get_skill("execute_partial").callable
    return _EXEC_PARTIAL_FN


def _exec_final():
    global _EXEC_FINAL_FN
    if _EXEC_FINAL_FN is None:
        _EXEC_FINAL_FN = get_skill("execute_final").callable
    return _EXEC_FINAL_FN


_LITERAL_PATTERNS = ["date", "year", "time", "start_date", "end_date",
                     "population", "age", "height", "weight",
                     "revenue", "price", "rating"]


def _find_literal_relation(out_rels):
    for rel in out_rels:
        for p in _LITERAL_PATTERNS:
            if p in rel.lower():
                return rel
    return None


@dataclass
class BeamBranch:
    """Represents a single candidate branch in the beam."""
    function_list: List[str]
    selected_relations: List[str]
    current_entity: str
    visited_edges: Set[Tuple[str, str]]
    visited_entities: Set[str]
    expression_id: str
    step_count: int
    execution_score: float = 0.0
    num_answers: int = 0
    last_sexpr: str = ""
    is_terminal: bool = False
    # Per-step trace data accumulated during expansion
    step_traces: list = field(default_factory=list)
    step_type: str = "join"  # "join", "count", "argmax", "argmin", "tc"

    def copy(self) -> "BeamBranch":
        return BeamBranch(
            function_list=list(self.function_list),
            selected_relations=list(self.selected_relations),
            current_entity=self.current_entity,
            visited_edges=set(self.visited_edges),
            visited_entities=set(self.visited_entities),
            expression_id=self.expression_id,
            step_count=self.step_count,
            execution_score=self.execution_score,
            num_answers=self.num_answers,
            last_sexpr=self.last_sexpr,
            is_terminal=self.is_terminal,
            step_traces=list(self.step_traces),
            step_type=self.step_type,
        )


class BeamAgent:
    """Beam Search agent: explores top-k branches in parallel."""

    def __init__(self, question, entities, subgraph, max_steps=6,
                 beam_width=3, candidate_paths=None, trace_collector=None,
                 llm_first=False):
        self.question = question.lower()
        self.entities = entities
        self.subgraph = subgraph
        self.max_steps = max_steps
        self.beam_width = beam_width
        self.candidate_paths = candidate_paths
        self._trace_collector = trace_collector
        self.llm_first = llm_first

        self.candidate_rels = self._parse_candidate_rels(candidate_paths)

        # Final results
        self.function_list = []
        self.selected_relations = []
        self.path_answer = set()
        self.best_branch = None

        if trace_collector:
            trace_collector.record_t5_input(candidate_paths)
            trace_collector.record_candidate_relations(self.candidate_rels)

    def _parse_candidate_rels(self, candidate_paths):
        if not candidate_paths or not candidate_paths[0]:
            return []
        path_str = candidate_paths[0][0] if isinstance(candidate_paths[0], list) else candidate_paths[0]
        parts = path_str.split("->")
        return [parts[i] for i in range(1, len(parts), 2)]

    def _replay_draft_state(self, start_entity, draft_fl):
        """Replay draft function_list to reconstruct terminal entity and visited state."""
        import re

        current_entity = start_entity
        visited_edges = set()
        visited_entities = {start_entity}

        for line in draft_fl:
            line = line.strip()
            match = re.match(r'(expression\d*)\s*=\s*(.+)', line)
            if not match:
                continue
            body = match.group(2)

            start_match = re.match(r'START\("([^"]+)"\)', body)
            if start_match:
                current_entity = start_match.group(1)
                visited_entities.add(current_entity)
                continue

            join_match = re.match(r'JOIN\("([^"]+)",\s*expression\d*\)', body)
            if not join_match:
                continue

            lf_rel = join_match.group(1)
            is_reverse = lf_rel.startswith("(R ") and lf_rel.endswith(")")
            raw_rel = lf_rel[3:-1] if is_reverse else lf_rel
            traversal_rel = "(R " + raw_rel + ")" if is_reverse else raw_rel
            visited_edges.add((current_entity, traversal_rel))

            if not is_reverse:
                targets = self.subgraph.get_targets(current_entity, raw_rel)
                for target in targets:
                    if target not in visited_entities:
                        current_entity = target
                        visited_entities.add(target)
                        break
            else:
                for rel, src in self.subgraph.get_incoming(current_entity):
                    if rel == raw_rel and src not in visited_entities:
                        current_entity = src
                        visited_entities.add(src)
                        break

        return current_entity, visited_edges, visited_entities

    def _discover_one_hop(self, entity):
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
        result = {}
        for rel, tgt in edge_list:
            if rel not in result:
                result[rel] = []
            result[rel].append(tgt)
        return result

    def _path_guided_hint(self, branch):
        if not self.candidate_rels:
            return None
        matched = 0
        for sel in branch.selected_relations:
            if matched >= len(self.candidate_rels):
                break
            if sel.startswith("(R ") and sel.endswith(")"):
                plain = sel[3:-1].strip()
            elif ":" in sel or sel == "COUNT":
                continue
            else:
                plain = sel
            if plain == self.candidate_rels[matched]:
                matched += 1
        if matched < len(self.candidate_rels):
            return self.candidate_rels[matched]
        return None

    def _do_start(self, entity):
        branch = BeamBranch(
            function_list=[],
            selected_relations=[],
            current_entity=entity,
            visited_edges=set(),
            visited_entities={entity},
            expression_id="",
            step_count=0,
        )
        fl = 'expression = START("' + entity + '")'
        branch.function_list.append(fl)
        branch.last_sexpr = function_list_to_sexpr(branch.function_list)
        return branch

    def _has_literal_relation(self, out_rels):
        return _find_literal_relation(out_rels) is not None

    def _is_count_question(self):
        return "how many" in self.question or "number of" in self.question

    def _is_argmax_question(self):
        words = ["largest", "biggest", "highest", "greatest", "longest",
                 "last", "newest", "latest", "first", "oldest", "most recent",
                 "most famous", "most popular"]
        return any(w in self.question for w in words)

    def _is_argmin_question(self):
        words = ["least", "smallest", "fewest", "lowest", "shortest", "earliest"]
        return any(w in self.question for w in words)

    def _question_intent(self):
        """Return the terminal action type the question implies, or None."""
        if self._is_count_question():
            return "count"
        if self._is_argmax_question():
            return "argmax"
        if self._is_argmin_question():
            return "argmin"
        return None

    def _expand_branch(self, branch):
        """Expand one branch by discovering all candidate next steps.

        Returns list of (new_branch, validation_result) tuples.
        Includes JOIN (forward/reverse), COUNT, ARGMAX, ARGMIN, TC candidates.
        """
        if branch.is_terminal:
            return [(branch, None)]

        outgoing, incoming = self._discover_one_hop(branch.current_entity)
        out_rels = self._dedup_by_relation(outgoing)
        in_rels = self._dedup_by_relation(incoming)

        # Filter unvisited edges
        unvisited_out = {}
        for rel, tgts in out_rels.items():
            edge_key = (branch.current_entity, rel)
            remaining = [t for t in tgts if t not in branch.visited_entities
                         and edge_key not in branch.visited_edges]
            if remaining:
                unvisited_out[rel] = remaining

        unvisited_in = {}
        for rel, srcs in in_rels.items():
            traversal_rel = "(R " + rel + ")"
            edge_key = (branch.current_entity, traversal_rel)
            remaining = [s for s in srcs if s not in branch.visited_entities
                         and edge_key not in branch.visited_edges]
            if remaining:
                unvisited_in[rel] = remaining

        if not unvisited_out and not unvisited_in:
            term = branch.copy()
            term.is_terminal = True
            return [(term, None)]

        results = []
        eid = branch.expression_id

        # --- Action candidates ---

        # 1. JOIN candidates (forward + reverse)
        hint = self._path_guided_hint(branch)
        join_candidates = []

        if hint:
            if hint in unvisited_out:
                join_candidates.append((hint, unvisited_out[hint], False, hint))
            if hint in unvisited_in:
                join_candidates.append((hint, unvisited_in[hint], True, "(R " + hint + ")"))

        for rel, tgts in unvisited_out.items():
            if not any(c[0] == rel and not c[2] for c in join_candidates):
                join_candidates.append((rel, tgts, False, rel))

        for rel, srcs in unvisited_in.items():
            if not any(c[0] == rel and c[2] for c in join_candidates):
                join_candidates.append((rel, srcs, True, "(R " + rel + ")"))

        for rel, targets, reverse, lf_rel in join_candidates:
            new_branch = branch.copy()
            new_branch.step_count += 1
            fl = 'expression' + eid + ' = JOIN("' + lf_rel + '", expression' + eid + ')'
            new_branch.function_list.append(fl)
            traversal_rel = "(R " + rel + ")" if reverse else rel
            new_branch.selected_relations.append(traversal_rel)

            val = _exec_partial()(new_branch.function_list, step_type="join")
            new_branch.last_sexpr = val.get("sexpr", "")
            new_branch.execution_score = val.get("execution_score", 0.0)
            new_branch.num_answers = val.get("num_answers", 0)

            step_trace = {
                "action": "join_reverse" if reverse else "join_forward",
                "chosen_relation": rel,
                "chosen_lf_relation": lf_rel,
                "execution_validation": {
                    "exec_ok": val.get("exec_ok"),
                    "num_answers": val.get("num_answers", 0),
                    "error": val.get("error"),
                    "execution_score": val.get("execution_score", 0.0),
                },
            }
            new_branch.step_traces.append(step_trace)

            if val.get("valid"):
                for t in targets:
                    if t not in new_branch.visited_entities:
                        new_branch.current_entity = t
                        new_branch.visited_entities.add(t)
                        break
                new_branch.visited_edges.add((branch.current_entity, traversal_rel))
                results.append((new_branch, val))

        # 2. COUNT candidate (after at least 1 JOIN step)
        if branch.step_count >= 1 and unvisited_out and self._is_count_question():
            new_branch = branch.copy()
            new_branch.step_count += 1
            fl = 'expression' + eid + ' = COUNT(expression' + eid + ')'
            new_branch.function_list.append(fl)
            new_branch.selected_relations.append("COUNT")
            new_branch.is_terminal = True
            new_branch.step_type = "count"

            val = _exec_partial()(new_branch.function_list, step_type="count")
            new_branch.last_sexpr = val.get("sexpr", "")
            new_branch.execution_score = val.get("execution_score", 0.0)
            new_branch.num_answers = val.get("num_answers", 0)

            step_trace = {
                "action": "count",
                "chosen_relation": "COUNT",
                "execution_validation": {
                    "exec_ok": val.get("exec_ok"),
                    "num_answers": val.get("num_answers", 0),
                    "error": val.get("error"),
                    "execution_score": val.get("execution_score", 0.0),
                },
            }
            new_branch.step_traces.append(step_trace)

            if val.get("valid"):
                results.append((new_branch, val))

        # 3. ARGMAX/ARGMIN candidates (after at least 1 JOIN step, with literal rel)
        if branch.step_count >= 1 and unvisited_out:
            lit_rel = _find_literal_relation(unvisited_out)
            if lit_rel:
                for mode, check in [("ARGMAX", self._is_argmax_question()),
                                     ("ARGMIN", self._is_argmin_question())]:
                    if not check:
                        continue
                    new_branch = branch.copy()
                    new_branch.step_count += 1
                    fl = 'expression' + eid + ' = ARG("' + mode + '", expression' + eid + ', "' + lit_rel + '")'
                    new_branch.function_list.append(fl)
                    new_branch.selected_relations.append(mode + ":" + lit_rel)
                    new_branch.is_terminal = True
                    new_branch.step_type = mode.lower()

                    val = _exec_partial()(new_branch.function_list, step_type="arg")
                    new_branch.last_sexpr = val.get("sexpr", "")
                    new_branch.execution_score = val.get("execution_score", 0.0)
                    new_branch.num_answers = val.get("num_answers", 0)

                    step_trace = {
                        "action": mode.lower(),
                        "chosen_relation": mode + ":" + lit_rel,
                        "execution_validation": {
                            "exec_ok": val.get("exec_ok"),
                            "num_answers": val.get("num_answers", 0),
                            "error": val.get("error"),
                            "execution_score": val.get("execution_score", 0.0),
                        },
                    }
                    new_branch.step_traces.append(step_trace)

                    if val.get("valid"):
                        results.append((new_branch, val))

        # 4. TC (time constraint) candidate
        if branch.step_count >= 1 and unvisited_out:
            lit_rel = _find_literal_relation(unvisited_out)
            if lit_rel and lit_rel in unvisited_out:
                tc_targets = unvisited_out[lit_rel]
                tc_entity = tc_targets[0] if tc_targets else branch.current_entity
                new_branch = branch.copy()
                new_branch.step_count += 1
                fl = 'expression' + eid + ' = TC(expression' + eid + ', "' + lit_rel + '", "' + tc_entity + '")'
                new_branch.function_list.append(fl)
                new_branch.selected_relations.append("TC:" + lit_rel)
                new_branch.step_type = "tc"

                val = _exec_partial()(new_branch.function_list, step_type="tc")
                new_branch.last_sexpr = val.get("sexpr", "")
                new_branch.execution_score = val.get("execution_score", 0.0)
                new_branch.num_answers = val.get("num_answers", 0)

                step_trace = {
                    "action": "time_filter",
                    "chosen_relation": "TC:" + lit_rel,
                    "execution_validation": {
                        "exec_ok": val.get("exec_ok"),
                        "num_answers": val.get("num_answers", 0),
                        "error": val.get("error"),
                        "execution_score": val.get("execution_score", 0.0),
                    },
                }
                new_branch.step_traces.append(step_trace)

                if val.get("valid"):
                    results.append((new_branch, val))

        return results

    def _populate_step_traces(self, branch, step_num):
        """Convert a branch's accumulated step_traces into TraceCollector steps."""
        if not self._trace_collector:
            return
        for st in branch.step_traces:
            step = self._trace_collector.new_step()
            step["step_num"] = step_num
            step_num += 1
            step["action"] = st.get("action", "")
            step["chosen_relation"] = st.get("chosen_relation", "")
            step["chosen_lf_relation"] = st.get("chosen_lf_relation", "")
            if st.get("execution_validation"):
                step["execution_validation"] = st["execution_validation"]

    def _finalize_trace(self, sexpr):
        """Populate standard TraceCollector fields for analysis compatibility."""
        if not self._trace_collector:
            return

        # path_replay: replay selected relations from start entity
        self._trace_collector.record_path_replay(self.path_answer)

        # path_replay_steps
        transitions = self._replay_path_transitions()
        self._trace_collector.record_path_replay_steps(transitions)

        # subgraph_snapshot
        try:
            sg_snap = self.subgraph.snapshot()
            self._trace_collector.record_subgraph_snapshot(sg_snap)
        except Exception:
            pass

        # final_execution
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

        # finalize (computes execution_summary from steps)
        self._trace_collector.finalize(
            self.function_list, self.selected_relations, sexpr)

    def run(self):
        if not self.entities:
            self.function_list = []
            return [], "@BAD_EXPRESSION"

        entity = self.entities[0]

        # --- Path-Guided short-circuit or LLM-First draft branch ---
        draft = None
        draft_branch = None
        pg_syn = None

        draft = path_to_lf_draft(entity, self.candidate_paths, self.subgraph)
        if draft is not None:
            from skills.validate_syntax import validate_syntax
            pg_syn = validate_syntax(draft["sexpr"])
            if pg_syn["valid"]:
                if not self.llm_first:
                    # Original PG short-circuit
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
                                self._trace_collector.data["beam_info"] = {
                                    "beam_width": self.beam_width,
                                    "mode": "path_guided_short_circuit",
                                }
                                # Populate steps for execution_summary compatibility
                                step = self._trace_collector.new_step()
                                step["step_num"] = 1
                                step["action"] = "path_guided"
                                step["chosen_relation"] = "->".join(self.selected_relations)
                                step["execution_validation"] = {
                                    "exec_ok": True,
                                    "num_answers": len(lf_answers),
                                    "error": None,
                                    "execution_score": min(1.0, len(lf_answers) / 100.0),
                                }
                                self._finalize_trace(sexpr)
                            return self.function_list, sexpr
                    except Exception:
                        pass
                else:
                    # LLM-First: PG draft becomes an initial beam branch
                    # Strip STOP from draft function list so beam search handles it normally
                    draft_fl = list(draft["function_list"])
                    if draft_fl and draft_fl[-1].strip().endswith("STOP(expression)"):
                        draft_fl = draft_fl[:-1]

                    draft_entity, draft_edges, draft_visited = self._replay_draft_state(entity, draft_fl)

                    draft_branch = BeamBranch(
                        function_list=draft_fl,
                        selected_relations=[
                            ("(R " + r + ")" if d == "reverse" else r)
                            for r, d in zip(draft["relations"], draft["directions"])
                        ],
                        current_entity=draft_entity,
                        visited_edges=draft_edges,
                        visited_entities=draft_visited,
                        expression_id="",
                        step_count=len(draft_fl) - 1,  # minus START
                        execution_score=1.0,  # PG draft gets top score as initial hint
                        num_answers=0,
                        is_terminal=False,  # Let beam search expand/compete
                    )
                    draft_branch.last_sexpr = draft["sexpr"]
                    if self._trace_collector:
                        self._trace_collector.data["path_guided_draft"] = draft
                        self._trace_collector.data["llm_first"] = True

        # --- Beam Search ---
        initial_branch = self._do_start(entity)
        if self.llm_first and draft_branch is not None:
            # Start beam with both the normal initial branch AND the PG draft branch
            beam = [initial_branch, draft_branch]
            # Deduplicate if initial_branch is identical to draft_branch
            if (initial_branch.function_list == draft_branch.function_list and
                initial_branch.selected_relations == draft_branch.selected_relations):
                beam = [initial_branch]
        else:
            beam = [initial_branch]
        beam_trace = []

        for step in range(1, self.max_steps + 1):
            all_candidates = []
            step_info = {"step": step, "branches": []}

            for branch in beam:
                if branch.is_terminal:
                    all_candidates.append(branch)
                    continue
                expanded = self._expand_branch(branch)
                for new_branch, val in expanded:
                    all_candidates.append(new_branch)

            # Boost terminal candidates matching question intent
            intent = self._question_intent()
            _BOOST = 2.0  # additive boost to execution_score for matching terminals
            def _sort_key(b):
                base = b.execution_score
                if b.is_terminal and intent and b.step_type == intent:
                    base += _BOOST
                return (base, b.num_answers)
            all_candidates.sort(key=_sort_key, reverse=True)
            # Force-keep at least one matching terminal candidate
            if intent:
                terminal_matching = [b for b in all_candidates
                                     if b.is_terminal and b.step_type == intent]
                if terminal_matching and terminal_matching[0] not in all_candidates[:self.beam_width]:
                    beam = all_candidates[:self.beam_width - 1] + [terminal_matching[0]]
                else:
                    beam = all_candidates[:self.beam_width]
            else:
                beam = all_candidates[:self.beam_width]

            if self._trace_collector:
                for b in beam:
                    step_info["branches"].append({
                        "current_entity": b.current_entity,
                        "selected_relations": list(b.selected_relations),
                        "execution_score": b.execution_score,
                        "num_answers": b.num_answers,
                        "is_terminal": b.is_terminal,
                        "function_list_len": len(b.function_list),
                    })
                beam_trace.append(step_info)

            if all(b.is_terminal for b in beam):
                break

        # Select best branch
        if not beam:
            self.function_list = []
            sexpr = "@BAD_EXPRESSION"
        else:
            # Prefer terminal candidates matching question intent
            intent = self._question_intent()
            def _best_key(b):
                base = b.execution_score
                if b.is_terminal and intent and b.step_type == intent:
                    base += 2.0
                return (base, b.num_answers)
            best = max(beam, key=_best_key)
            self.best_branch = best

            # Add STOP (skip if the branch already has STOP, e.g. from draft)
            eid = best.expression_id
            last_fl = best.function_list[-1].strip() if best.function_list else ""
            if not last_fl.endswith("STOP(expression" + eid + ")"):
                stop_fl = 'expression' + eid + ' = STOP(expression' + eid + ')'
                best.function_list.append(stop_fl)

            self.function_list = best.function_list
            self.selected_relations = best.selected_relations
            self.path_answer = self._replay_path_answer()
            sexpr = function_list_to_sexpr(self.function_list)

            # Populate step traces from the winning branch
            if self._trace_collector:
                self._populate_step_traces(best, 1)

        # Trace finalization
        if self._trace_collector:
            self._trace_collector.data["beam_info"] = {
                "beam_width": self.beam_width,
                "mode": "beam_search",
                "beam_trace": beam_trace,
                "final_beam_size": len(beam),
                "best_execution_score": beam[0].execution_score if beam else 0.0,
                "best_num_answers": beam[0].num_answers if beam else 0,
            }
            self._finalize_trace(sexpr)

        return self.function_list, sexpr

    def _replay_path_answer(self):
        if not self.entities or not self.selected_relations:
            return set()
        current = {self.entities[0]}
        for rel in self.selected_relations:
            if rel.startswith("ARGMAX:") or rel.startswith("ARGMIN:") or rel.startswith("TC:") or rel == "COUNT":
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
        if not self.entities or not self.selected_relations:
            return []
        transitions = []
        current = {self.entities[0]}
        for rel in self.selected_relations:
            if rel.startswith("ARGMAX:") or rel.startswith("ARGMIN:") or rel.startswith("TC:") or rel == "COUNT":
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
