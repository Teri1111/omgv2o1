"""T7 review unit tests — v3 (P2F-2 fix).

Covers:
1. argmax/argmin contract — function calling chooser returns argmax/argmin,
   agent._decide_action passes it through (not heuristic fallback)
2. invalid direction trace — direction="sideways" triggers fallback_reason
   in trace even when join still succeeds (via JSON fallback path)
3. fenced JSON trailing thought — ```json {..} extracts via JSON fallback
4. function_call_recovered_by_react — FC fails, ReAct recovers, trace marks recovered
5. Complex actions (Merge, Compare, Time_constraint) wired through chooser
"""
import sys
import os
import json
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, "/data/gt/omgv2-o1")


def _make_subgraph():
    """Build a minimal SubgraphBuilder for testing."""
    from reasoning.subgraph import SubgraphBuilder
    sub = SubgraphBuilder()
    triplets = {
        "0": [("m.01", "film.actor.film", "m.02")],
        "1": [("m.02", "film.film.initial_release_date", "2000")]
    }
    sub.build_from_triplets(triplets)
    return sub


class TestT7ArgmaxArgminContract(unittest.TestCase):
    """Issue 1: argmax/argmin from function calling must be accepted by _decide_action."""

    def test_argmax_returns_argmax_action(self):
        from reasoning.agent import LLMGuidedAgent
        sub = _make_subgraph()
        agent = LLMGuidedAgent(
            question="test", entities=["m.01"], subgraph=sub, max_steps=3, llm_first=True,
        )
        mock_choice = {"action": "argmax", "relation": "film.film.initial_release_date", "thought": "test"}
        with patch("reasoning.llm_agent.choose_next_step_function_call", return_value=(mock_choice, True)):
            result = agent._decide_action({"film.actor.film": ["m.02"]}, {})
            self.assertEqual(result, "argmax",
                             "argmax from function calling should pass through, not fall back to heuristic")
            self.assertIn("Argmax", agent.scratchpad,
                         "argmax action must write Argmax to scratchpad")

    def test_argmin_returns_argmin_action(self):
        from reasoning.agent import LLMGuidedAgent
        sub = _make_subgraph()
        agent = LLMGuidedAgent(
            question="test", entities=["m.01"], subgraph=sub, max_steps=3, llm_first=True,
        )
        mock_choice = {"action": "argmin", "relation": "film.film.initial_release_date", "thought": "test"}
        with patch("reasoning.llm_agent.choose_next_step_function_call", return_value=(mock_choice, True)):
            result = agent._decide_action({"film.actor.film": ["m.02"]}, {})
            self.assertEqual(result, "argmin")
            self.assertIn("Argmin", agent.scratchpad,
                         "argmin action must write Argmin to scratchpad")


class TestT7DirectionValidationTrace(unittest.TestCase):
    """Issue 1b: invalid direction should produce fallback_reason in trace even on success."""

    def test_invalid_direction_writes_trace_warning(self):
        """JSON fallback with sideways direction should produce trace warning.
        We force parse_action to return None so JSON fallback is reached."""
        from reasoning.llm_agent import choose_next_step_function_call
        mock_raw = json.dumps({
            "tool": "extend_expression",
            "args": {"action": "join", "relation": "film.actor.film", "direction": "sideways"},
            "thought": "test thought"
        })
        trace = {}
        with patch("reasoning.llm_agent._request_chat_completion", return_value=(mock_raw, True, None)):
            with patch("reasoning.llm_agent.parse_action", return_value=None):
                parsed, http = choose_next_step_function_call(
                    question="test", scratchpad="",
                    available_forward_relations=["film.actor.film"],
                    available_reverse_relations=[], trace=trace,
                )
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["action"], "join_forward")
        self.assertIn("fallback_reason", trace,
                      "Invalid direction correction must appear in trace fallback_reason")
        self.assertIn("sideways", trace["fallback_reason"])


class TestT7TrailingThought(unittest.TestCase):
    """Issue 3: fenced JSON followed by trailing natural language thought (JSON fallback path)."""

    def test_fenced_json_trailing_thought(self):
        """JSON fallback should parse fenced JSON and extract thought from it.
        We force parse_action to return None so JSON fallback is reached."""
        from reasoning.llm_agent import choose_next_step_function_call
        mock_raw = (
            '```json\n'
            '{"tool": "extend_expression", "args": {"action": "join", "relation": "film.actor.film", '
            '"direction": "forward"}, "thought": "I chose film.actor.film because the question asks about actors."}\n'
            '```'
        )
        trace = {}
        with patch("reasoning.llm_agent._request_chat_completion", return_value=(mock_raw, True, None)):
            with patch("reasoning.llm_agent.parse_action", return_value=None):
                parsed, http = choose_next_step_function_call(
                    question="test", scratchpad="",
                    available_forward_relations=["film.actor.film"],
                    available_reverse_relations=[], trace=trace,
                )
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["action"], "join_forward")
        self.assertTrue(len(trace.get("thought", "")) > 0,
                        "Thought from JSON fallback should not be empty, got: " + repr(trace.get("thought")))

    def test_code_fence_stripped_from_prefix(self):
        """JSON fallback should strip code fence markers from thought.
        We force parse_action to return None so JSON fallback is reached."""
        from reasoning.llm_agent import choose_next_step_function_call
        mock_raw = (
            '```json\n'
            '{"tool": "extend_expression", "args": {"action": "count"}, '
            '"thought": "inner thought"}\n'
            '```'
        )
        trace = {}
        with patch("reasoning.llm_agent._request_chat_completion", return_value=(mock_raw, True, None)):
            with patch("reasoning.llm_agent.parse_action", return_value=None):
                parsed, http = choose_next_step_function_call(
                    question="test", scratchpad="",
                    available_forward_relations=[], available_reverse_relations=[],
                    allow_count=True, trace=trace,
                )
        self.assertIsNotNone(parsed)
        self.assertNotIn("```", trace.get("thought", ""),
                         "Code fence markers should not appear in thought")


class TestT7RecoveryTrace(unittest.TestCase):
    """Issue 2: function_call_recovered_by_react must be written when ReAct rescues."""

    def test_react_recovery_marks_trace(self):
        """When JSON fallback also fails (unrecognized tool), ReAct should recover."""
        from reasoning.llm_agent import choose_next_step_function_call
        mock_raw = json.dumps({"tool": "unknown_tool", "args": {}, "thought": ""})
        react_parsed = {"action": "join_forward", "relation": "film.actor.film", "thought": "Let me join."}
        trace = {}
        with patch("reasoning.llm_agent._request_chat_completion", return_value=(mock_raw, True, None)):
            with patch("reasoning.llm_agent.parse_action", return_value=None):
                with patch("reasoning.llm_agent.choose_next_step_react", return_value=(react_parsed, True)):
                    parsed, http = choose_next_step_function_call(
                        question="test", scratchpad="",
                        available_forward_relations=["film.actor.film"],
                        available_reverse_relations=[], trace=trace,
                    )
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["action"], "join_forward")
        self.assertTrue(trace.get("function_call_failed"),
                        "function_call_failed should be True when JSON tool is unrecognized")
        self.assertTrue(trace.get("function_call_recovered_by_react"),
                        "function_call_recovered_by_react should be True when ReAct rescues")
        self.assertEqual(trace.get("recovered_action"), "join_forward")

    def test_both_fail_no_false_recovery(self):
        """When both JSON fallback and ReAct fail, recovered should be False."""
        from reasoning.llm_agent import choose_next_step_function_call
        mock_raw = json.dumps({"tool": "unknown_tool", "args": {}})
        trace = {}
        with patch("reasoning.llm_agent._request_chat_completion", return_value=(mock_raw, True, None)):
            with patch("reasoning.llm_agent.parse_action", return_value=None):
                with patch("reasoning.llm_agent.choose_next_step_react", return_value=(None, True)):
                    parsed, http = choose_next_step_function_call(
                        question="test", scratchpad="",
                        available_forward_relations=["film.actor.film"],
                        available_reverse_relations=[], trace=trace,
                    )
        self.assertIsNone(parsed)
        self.assertTrue(trace.get("function_call_failed"))
        self.assertNotIn("function_call_recovered_by_react", trace,
                         "recovered key should not be set when both JSON and ReAct fail")


class TestT7Regression(unittest.TestCase):
    """Smoke: existing test_closed_loop.py 3 should still pass."""

    def test_greedy_baseline(self):
        import subprocess
        env = os.environ.copy()
        env["TOKENIZERS_PARALLELISM"] = "false"
        r = subprocess.run(
            "/data/gt/envs/lf_gjq/bin/python3 test_closed_loop.py 3 2>&1 | grep 'LF hit'",
            shell=True, capture_output=True, text=True, timeout=60,
            cwd="/data/gt/omgv2-o1", env=env,
        )
        self.assertIn("3/3", r.stdout)

    def test_llm_first_baseline(self):
        import subprocess
        env = os.environ.copy()
        env["TOKENIZERS_PARALLELISM"] = "false"
        r = subprocess.run(
            "/data/gt/envs/lf_gjq/bin/python3 test_closed_loop.py --llm-first 3 2>&1 | grep 'LF hit'",
            shell=True, capture_output=True, text=True, timeout=60,
            cwd="/data/gt/omgv2-o1", env=env,
        )
        self.assertIn("3/3", r.stdout)



class TestT7ReactTraceNoCrash(unittest.TestCase):
    """Blocking bug: choose_next_step_react must not reference undefined tool_call."""

    def test_react_trace_no_nameerror(self):
        """choose_next_step_react with trace must not raise NameError on tool_call."""
        from reasoning.llm_agent import choose_next_step_react
        mock_raw = "Thought2: Joining.\nAction2: Find_relation [ film.actor.film ]"
        trace = {}
        with patch("reasoning.llm_agent._request_chat_completion", return_value=(mock_raw, True, None)):
            parsed, http = choose_next_step_react(
                question="test",
                scratchpad="Thought1: Start.\nAction1: Start [ m.01 ]\nObservation1: m.01\n",
                available_forward_relations=["film.actor.film"],
                available_reverse_relations=[],
                trace=trace,
            )
        # Must not crash with NameError
        self.assertIn("thought", trace)
        self.assertIsNotNone(parsed)

    def test_react_then_fc_no_cross_contamination(self):
        """ReAct trace must not leak tool_call from function_call scope."""
        from reasoning.llm_agent import choose_next_step_react
        mock_raw = "Thought1: Explore.\nAction1: Find_relation [ music.artist.genre ]"
        trace = {}
        with patch("reasoning.llm_agent._request_chat_completion", return_value=(mock_raw, True, None)):
            choose_next_step_react(
                question="q", scratchpad="",
                available_forward_relations=["music.artist.genre"],
                available_reverse_relations=[],
                trace=trace,
            )
        # tool_call key should NOT exist in ReAct trace
        self.assertNotIn("tool_call", trace,
                         "ReAct trace should not have tool_call key from function_call scope")


class TestT7ComplexActions(unittest.TestCase):
    """P2F-2: Merge, Compare, Time_constraint actions wired through chooser."""

    def test_merge_action_parsed(self):
        """Merge action should be parsed when allow_and=True."""
        from reasoning.llm_agent import parse_action
        raw = "Thought1: Need to combine.\nAction1: Merge [ film.actor.film ]"
        result = parse_action(raw, available_fwd=["film.actor.film"], available_rev=[],
                              allow_and=True)
        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "and")
        self.assertEqual(result["relation"], "film.actor.film")

    def test_merge_action_blocked_without_flag(self):
        """Merge action should not be parsed when allow_and=False."""
        from reasoning.llm_agent import parse_action
        raw = "Thought1: Need to combine.\nAction1: Merge [ film.actor.film ]"
        result = parse_action(raw, available_fwd=["film.actor.film"], available_rev=[],
                              allow_and=False)
        self.assertIsNone(result)

    def test_compare_action_parsed(self):
        """Compare action should be parsed when allow_cmp=True."""
        from reasoning.llm_agent import parse_action
        raw = "Thought1: Compare dates.\nAction1: Compare [ GT | film.film.initial_release_date ]"
        result = parse_action(raw, available_fwd=["film.film.initial_release_date"], available_rev=[],
                              allow_cmp=True)
        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "cmp")

    def test_compare_action_blocked_without_flag(self):
        """Compare action should not be parsed when allow_cmp=False."""
        from reasoning.llm_agent import parse_action
        raw = "Thought1: Compare dates.\nAction1: Compare [ GT | film.film.initial_release_date ]"
        result = parse_action(raw, available_fwd=["film.film.initial_release_date"], available_rev=[],
                              allow_cmp=False)
        self.assertIsNone(result)

    def test_time_constraint_action_parsed(self):
        """Time_constraint action should be parsed when allow_time_filter=True."""
        from reasoning.llm_agent import parse_action
        raw = "Thought1: Filter by year.\nAction1: Time_constraint [ film.film.initial_release_date | 2000 ]"
        result = parse_action(raw, available_fwd=["film.film.initial_release_date"], available_rev=[],
                              allow_time_filter=True)
        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "time_filter")

    def test_time_constraint_blocked_without_flag(self):
        """Time_constraint action should not be parsed when allow_time_filter=False."""
        from reasoning.llm_agent import parse_action
        raw = "Thought1: Filter by year.\nAction1: Time_constraint [ film.film.initial_release_date | 2000 ]"
        result = parse_action(raw, available_fwd=["film.film.initial_release_date"], available_rev=[],
                              allow_time_filter=False)
        self.assertIsNone(result)

    def test_complex_actions_in_system_prompt(self):
        """choose_next_step system prompt should include complex action descriptions."""
        from reasoning.llm_agent import choose_next_step
        trace = {}
        mock_raw = "Thought1: Join.\nAction1: Find_relation [ film.actor.film ]"
        with patch("reasoning.llm_agent._request_chat_completion", return_value=(mock_raw, True, None)):
            choose_next_step(
                question="test", scratchpad="",
                available_forward_relations=["film.actor.film"],
                available_reverse_relations=[],
                allow_and=True, allow_cmp=True, allow_time_filter=True,
                trace=trace,
            )
        sys_prompt = trace.get("system_prompt", "")
        self.assertIn("Merge", sys_prompt, "System prompt should mention Merge when allow_and=True")
        self.assertIn("Compare", sys_prompt, "System prompt should mention Compare when allow_cmp=True")
        self.assertIn("Time_constraint", sys_prompt,
                       "System prompt should mention Time_constraint when allow_time_filter=True")

    def test_choose_next_step_function_call_passes_complex_flags(self):
        """choose_next_step_function_call should pass allow_and, allow_cmp, allow_time_filter."""
        from reasoning.llm_agent import choose_next_step_function_call
        mock_raw = "Thought1: Merge actors.\nAction1: Merge [ film.actor.film ]"
        trace = {}
        with patch("reasoning.llm_agent._request_chat_completion", return_value=(mock_raw, True, None)):
            parsed, http = choose_next_step_function_call(
                question="test", scratchpad="",
                available_forward_relations=["film.actor.film"],
                available_reverse_relations=[],
                allow_and=True, trace=trace,
            )
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["action"], "and")
        self.assertEqual(parsed["relation"], "film.actor.film")

    def test_choose_next_step_function_call_blocks_complex_without_flags(self):
        """choose_next_step_function_call should block complex actions when flags are False."""
        from reasoning.llm_agent import choose_next_step_function_call
        mock_raw = "Thought1: Merge actors.\nAction1: Merge [ film.actor.film ]"
        trace = {}
        with patch("reasoning.llm_agent._request_chat_completion", return_value=(mock_raw, True, None)):
            parsed, http = choose_next_step_function_call(
                question="test", scratchpad="",
                available_forward_relations=["film.actor.film"],
                available_reverse_relations=[],
                allow_and=False, trace=trace,
            )
        # Merge action should not be returned when allow_and=False
        if parsed is not None:
            self.assertNotEqual(parsed["action"], "and",
                                "Merge action should not be returned when allow_and=False")

    def test_agent_decide_action_passes_complex_flags(self):
        """LLMGuidedAgent._decide_action should wire complex flags to chooser."""
        from reasoning.agent import LLMGuidedAgent
        sub = _make_subgraph()
        agent = LLMGuidedAgent(
            question="test", entities=["m.01"], subgraph=sub, max_steps=3, llm_first=True,
        )
        # "and" action maps to "and" internal action (written as "And" in scratchpad)
        mock_choice = {"action": "and", "relation": "film.actor.film", "thought": "merge"}
        with patch("reasoning.llm_agent.choose_next_step_function_call", return_value=(mock_choice, True)) as mock_call:
            # Mock heuristic to return "and" so allow_and=True
            with patch.object(type(agent).__bases__[0], '_decide_action', return_value="and"):
                result = agent._decide_action({"film.actor.film": ["m.02"]}, {})
        self.assertEqual(result, "and")
        # The agent writes the action capitalized in scratchpad
        self.assertIn("And", agent.scratchpad)


class TestT7ParameterExecution(unittest.TestCase):
    """Verify that argmax/argmin/cmp/time_filter parameters from _pending_llm_choice
    are actually used in function_list and S-expression output."""

    def _make_agent_with_subgraph(self):
        """Create an LLMGuidedAgent with a minimal subgraph for testing."""
        from reasoning.agent import LLMGuidedAgent
        sub = _make_subgraph()
        agent = LLMGuidedAgent(
            question="test", entities=["m.01"], subgraph=sub, max_steps=3, llm_first=True,
        )
        # Bootstrap: simulate _do_start so function_list and expression_id are set
        agent._do_start("m.01")
        return agent

    def test_argmax_uses_pending_relation(self):
        """argmax should use the relation from _pending_llm_choice, not heuristic fallback."""
        from skills.lf_construction import function_list_to_sexpr
        agent = self._make_agent_with_subgraph()
        target_relation = "film.film.initial_release_date"
        filtered_out = {target_relation: ["2000"]}
        # Simulate LLM choosing argmax with a specific relation
        agent._pending_llm_choice = {"action": "argmax", "relation": target_relation, "thought": "find latest"}
        result = agent._dispatch_action("argmax", filtered_out, {})
        self.assertIsNotNone(result, "argmax dispatch should return a result dict")
        self.assertEqual(result["relation"], "ARGMAX:" + target_relation)
        # Verify function_list contains the target relation
        fl_str = "\n".join(agent.function_list)
        self.assertIn(target_relation, fl_str,
                      "function_list should contain the LLM-chosen relation, not heuristic fallback")
        # Verify S-expression
        sexpr = function_list_to_sexpr(agent.function_list)
        self.assertIn(target_relation, sexpr,
                      "S-expression should contain the LLM-chosen argmax relation")

    def test_argmin_uses_pending_relation(self):
        """argmin should use the relation from _pending_llm_choice, not heuristic fallback."""
        from skills.lf_construction import function_list_to_sexpr
        agent = self._make_agent_with_subgraph()
        target_relation = "film.film.initial_release_date"
        filtered_out = {target_relation: ["2000"]}
        agent._pending_llm_choice = {"action": "argmin", "relation": target_relation, "thought": "find earliest"}
        result = agent._dispatch_action("argmin", filtered_out, {})
        self.assertIsNotNone(result, "argmin dispatch should return a result dict")
        self.assertEqual(result["relation"], "ARGMIN:" + target_relation)
        fl_str = "\n".join(agent.function_list)
        self.assertIn(target_relation, fl_str,
                      "function_list should contain the LLM-chosen relation")
        sexpr = function_list_to_sexpr(agent.function_list)
        self.assertIn(target_relation, sexpr,
                      "S-expression should contain the LLM-chosen argmin relation")

    def test_cmp_passes_operator(self):
        """cmp should pass the operator from _pending_llm_choice into the S-expression."""
        from skills.lf_construction import function_list_to_sexpr
        agent = self._make_agent_with_subgraph()
        target_relation = "film.film.revenue"
        filtered_out = {target_relation: ["1000"]}
        # LLM chooses "lt" operator explicitly
        agent._pending_llm_choice = {"action": "cmp", "relation": target_relation, "operator": "lt", "thought": "less than"}
        result = agent._dispatch_action("cmp", filtered_out, {})
        self.assertIsNotNone(result, "cmp dispatch should return a result dict")
        self.assertEqual(result["relation"], "CMP:" + target_relation)
        # Verify function_list contains "lt" operator (not default "gt")
        fl_str = "\n".join(agent.function_list)
        self.assertIn('"lt"', fl_str,
                      "function_list should contain the LLM-chosen operator 'lt', not default 'gt'")
        # Verify S-expression contains lt
        sexpr = function_list_to_sexpr(agent.function_list)
        self.assertIn("lt", sexpr,
                      "S-expression should contain the LLM-chosen operator 'lt'")
        self.assertNotIn("gt", sexpr.split(target_relation)[0] if target_relation in sexpr else sexpr,
                         "S-expression should not contain default 'gt' operator")

    def test_cmp_default_operator_when_none(self):
        """cmp should use heuristic-inferred operator when _pending_llm_choice has no operator."""
        from skills.lf_construction import function_list_to_sexpr
        agent = self._make_agent_with_subgraph()
        target_relation = "film.film.revenue"
        filtered_out = {target_relation: ["1000"]}
        # No operator in pending choice — should fall back to _infer_cmp_operator (default "gt")
        agent._pending_llm_choice = {"action": "cmp", "relation": target_relation, "thought": "compare"}
        # Remove 'operator' key if present
        agent._pending_llm_choice.pop("operator", None)
        result = agent._dispatch_action("cmp", filtered_out, {})
        self.assertIsNotNone(result)
        sexpr = function_list_to_sexpr(agent.function_list)
        # Default operator is "gt" (from _infer_cmp_operator fallback)
        self.assertIn("gt", sexpr,
                      "S-expression should use default 'gt' operator when none specified")

    def test_time_filter_uses_time_value(self):
        """time_filter should use time_value from _pending_llm_choice in the TC S-expression."""
        from skills.lf_construction import function_list_to_sexpr
        agent = self._make_agent_with_subgraph()
        target_relation = "film.film.initial_release_date"
        filtered_out = {target_relation: ["1999", "2000", "2001"]}
        # LLM specifies time_value = "2000"
        agent._pending_llm_choice = {
            "action": "time_filter", "relation": target_relation,
            "time_value": "2000", "thought": "filter by year 2000"
        }
        result = agent._dispatch_action("time_filter", filtered_out, {})
        self.assertIsNotNone(result, "time_filter dispatch should return a result dict")
        self.assertEqual(result["relation"], "TC:" + target_relation)
        # Verify function_list contains "2000"
        fl_str = "\n".join(agent.function_list)
        self.assertIn("2000", fl_str,
                      "function_list should contain the LLM-chosen time_value '2000'")
        # Verify S-expression
        sexpr = function_list_to_sexpr(agent.function_list)
        self.assertIn("2000", sexpr,
                      "S-expression should contain the LLM-chosen time_value '2000'")
        self.assertIn(target_relation, sexpr,
                      "S-expression should contain the time relation")

    def test_time_filter_fallback_to_first_target(self):
        """time_filter should fall back to first target entity when time_value is missing."""
        from skills.lf_construction import function_list_to_sexpr
        agent = self._make_agent_with_subgraph()
        target_relation = "film.film.initial_release_date"
        filtered_out = {target_relation: ["1999", "2000"]}
        # No time_value — should use first target
        agent._pending_llm_choice = {"action": "time_filter", "relation": target_relation, "thought": "filter"}
        agent._pending_llm_choice.pop("time_value", None)
        result = agent._dispatch_action("time_filter", filtered_out, {})
        self.assertIsNotNone(result)
        sexpr = function_list_to_sexpr(agent.function_list)
        # Should contain the first target "1999" from filtered_out
        self.assertIn("1999", sexpr,
                      "S-expression should contain first target entity '1999' as fallback time_value")

    def test_parse_action_compare_structured(self):
        """parse_action should extract operator and relation from Compare [ LT | relation ]."""
        from reasoning.llm_agent import parse_action
        raw = "Thought1: Compare revenue.\nAction1: Compare [ LT | film.film.revenue ]"
        result = parse_action(raw, available_fwd=["film.film.revenue"], available_rev=[], allow_cmp=True)
        self.assertIsNotNone(result, "parse_action should parse Compare action")
        self.assertEqual(result["action"], "cmp")
        self.assertEqual(result["operator"], "lt")
        self.assertEqual(result["relation"], "film.film.revenue")

    def test_parse_action_compare_default_operator(self):
        """parse_action should default to 'gt' when Compare has no operator prefix."""
        from reasoning.llm_agent import parse_action
        raw = "Thought1: Compare.\nAction1: Compare [ film.film.revenue ]"
        result = parse_action(raw, available_fwd=["film.film.revenue"], available_rev=[], allow_cmp=True)
        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "cmp")
        self.assertEqual(result["operator"], "gt", "Should default to 'gt' when no operator given")

    def test_parse_action_time_constraint_structured(self):
        """parse_action should extract relation and time_value from Time_constraint [ rel | value ]."""
        from reasoning.llm_agent import parse_action
        raw = "Thought1: Filter by year.\nAction1: Time_constraint [ film.film.initial_release_date | 2000 ]"
        result = parse_action(raw, available_fwd=["film.film.initial_release_date"], available_rev=[],
                              allow_time_filter=True)
        self.assertIsNotNone(result, "parse_action should parse Time_constraint action")
        self.assertEqual(result["action"], "time_filter")
        self.assertEqual(result["relation"], "film.film.initial_release_date")
        self.assertEqual(result["time_value"], "2000")

    def test_parse_action_time_constraint_no_value(self):
        """parse_action should handle Time_constraint with no time_value."""
        from reasoning.llm_agent import parse_action
        raw = "Thought1: Filter.\nAction1: Time_constraint [ film.film.date ]"
        result = parse_action(raw, available_fwd=["film.film.date"], available_rev=[], allow_time_filter=True)
        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "time_filter")
        self.assertEqual(result["relation"], "film.film.date")
        self.assertEqual(result["time_value"], "", "time_value should be empty when not specified")

    def test_parse_action_compare_blocked_without_flag(self):
        """parse_action should NOT parse Compare when allow_cmp=False."""
        from reasoning.llm_agent import parse_action
        raw = "Thought1: Compare.\nAction1: Compare [ LT | film.film.revenue ]"
        result = parse_action(raw, available_fwd=["film.film.revenue"], available_rev=[], allow_cmp=False)
        self.assertIsNone(result, "Compare should be blocked when allow_cmp=False")

    def test_parse_action_time_constraint_blocked_without_flag(self):
        """parse_action should NOT parse Time_constraint when allow_time_filter=False."""
        from reasoning.llm_agent import parse_action
        raw = "Thought1: Filter.\nAction1: Time_constraint [ film.film.date | 2000 ]"
        result = parse_action(raw, available_fwd=["film.film.date"], available_rev=[], allow_time_filter=False)
        self.assertIsNone(result, "Time_constraint should be blocked when allow_time_filter=False")


class TestT7EvalEngineering(unittest.TestCase):
    """Evaluate engineering: compute_f1, --output JSONL, --resume skip."""

    @classmethod
    def setUpClass(cls):
        cls.fixture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_fixtures")
        os.makedirs(cls.fixture_dir, exist_ok=True)
        
        # 内联 3 条样本，不依赖外部数据
        cls.data_path = os.path.join(cls.fixture_dir, "cwq_3samples.json")
        mini_data = [
            {
                "index": "0",
                "question": "Lou Seal is the mascot for the team that last won the World Series when?",
                "start_entity": "m.03_dwn",
                "triplets": {
                    "0": [["m.03_dwn", "sports.sports_team.team_mascot", ["m.0713r"]]],
                    "1": [["m.0713r", "sports.sports_team.championships", ["m.0ds8qct", "m.09gnk2r", "m.0117q3yz"]]],
                },
                "answer_id": ["m.0117q3yz"]
            },
            {
                "index": "1",
                "question": "Where did the concert artist go to college?",
                "start_entity": "m.010qhfmm",
                "triplets": {
                    "0": [["m.010qhfmm", "music.artist.concert_tours", ["m.03gr7w"]]],
                    "1": [["m.03gr7w", "music.artist.concert_tours", ["m.0wfrv9g", "m.05yxq0p", "m.010qhfmm"]]],
                    "2": [["m.0h3d7qb", "education.education.institution", ["m.0267yb_"]], ["m.0n1dd_6", "education.education.institution", ["m.04lbv7"]], ["m.0h3d7qj", "education.education.institution", ["m.01qdhx"]]],
                },
                "answer_id": ["m.01qdhx"]
            },
            {
                "index": "2",
                "question": "What is the name of the battle?",
                "start_entity": "m.06w4g",
                "triplets": {
                    "0": [["m.06w4g", "military.military_person.participated_in_battles", ["m.013zny"]]],
                },
                "answer_id": ["m.013zny"]
            },
        ]
        with open(cls.data_path, "w") as f:
            json.dump(mini_data, f)
        
        cls.t5_path = os.path.join(cls.fixture_dir, "t5_3samples.json")
        mini_t5 = [
            {"index": "0", "candidate_paths": [["sports.sports_team.team_mascot", "sports.sports_team.championships"]]},
            {"index": "1", "candidate_paths": [["music.artist.concert_tours", "people.person.education"]]},
            {"index": "2", "candidate_paths": [["people.person.places_lived", "location.location.people_born_here"]]},
        ]
        with open(cls.t5_path, "w") as f:
            json.dump(mini_t5, f)

    @classmethod
    def tearDownClass(cls):
        for path in [cls.data_path, cls.t5_path]:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(cls.fixture_dir) and not os.listdir(cls.fixture_dir):
            os.rmdir(cls.fixture_dir)

    # ---- compute_f1 ----------------------------------------------------------
    def test_compute_f1_both_empty(self):
        """Two empty sets → precision=1, recall=1, f1=1."""
        from evaluate import compute_f1
        r = compute_f1(set(), set())
        self.assertEqual(r["precision"], 1.0)
        self.assertEqual(r["recall"], 1.0)
        self.assertEqual(r["f1"], 1.0)

    def test_compute_f1_pred_nonempty_golden_empty(self):
        """Predicted non-empty but golden empty → f1=0."""
        from evaluate import compute_f1
        r = compute_f1({"m.01"}, set())
        self.assertEqual(r["f1"], 0.0)
        self.assertEqual(r["precision"], 0.0)
        self.assertEqual(r["recall"], 0.0)

    def test_compute_f1_partial_match(self):
        """Partial overlap → precision=0.5, recall=0.5, f1=0.5."""
        from evaluate import compute_f1
        r = compute_f1({"m.01", "m.02"}, {"m.01", "m.03"})
        self.assertAlmostEqual(r["precision"], 0.5)
        self.assertAlmostEqual(r["recall"], 0.5)
        self.assertAlmostEqual(r["f1"], 0.5)

    def test_compute_f1_exact_match(self):
        """Full overlap → precision=1, recall=1, f1=1."""
        from evaluate import compute_f1
        r = compute_f1({"m.01", "m.02"}, {"m.01", "m.02"})
        self.assertEqual(r["precision"], 1.0)
        self.assertEqual(r["recall"], 1.0)
        self.assertEqual(r["f1"], 1.0)

    # ---- --output JSONL ------------------------------------------------------
    def test_output_jsonl_written(self):
        """--output should produce a JSONL file with required fields."""
        import subprocess, tempfile, os
        fd, out_path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            env = os.environ.copy()
            env["TOKENIZERS_PARALLELISM"] = "false"
            env["TEST_DATA_PATH"] = self.__class__.data_path
            env["TEST_T5_PATH"] = self.__class__.t5_path
            result = subprocess.run(
                ["/data/gt/envs/lf_gjq/bin/python3", "test_closed_loop.py",
                 "--eval-mode", "full", "--output", out_path],
                capture_output=True, text=True, timeout=300,
                cwd="/data/gt/omgv2-o1", env=env,
            )
            self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")
            self.assertTrue(os.path.exists(out_path), "Output file should exist")
            with open(out_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            self.assertGreater(len(lines), 0, "JSONL should not be empty")
            for line in lines:
                obj = json.loads(line)
                for key in ("question_id", "f1", "precision", "recall", "em", "hit"):
                    self.assertIn(key, obj, f"Missing field '{key}' in JSONL line")
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    # ---- --resume skip -------------------------------------------------------
    def test_resume_skips_completed(self):
        """--resume should skip samples already present in output file."""
        import subprocess, tempfile, os
        fd, out_path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            # Pre-populate with one completed sample (question_id=0)
            seed = {
                "question_id": 0, "question": "fake", "predicted_answers": [],
                "golden_answers": [], "lf_sexpr": "", "merged_sexpr": "",
                "f1": 0.0, "precision": 0.0, "recall": 0.0, "em": False, "hit": False,
            }
            with open(out_path, "w") as f:
                f.write(json.dumps(seed) + "\n")

            env = os.environ.copy()
            env["TOKENIZERS_PARALLELISM"] = "false"
            env["TEST_DATA_PATH"] = self.__class__.data_path
            env["TEST_T5_PATH"] = self.__class__.t5_path
            r = subprocess.run(
                ["/data/gt/envs/lf_gjq/bin/python3", "test_closed_loop.py",
                 "--resume", "--eval-mode", "full", "--output", out_path],
                capture_output=True, text=True, timeout=300,
                cwd="/data/gt/omgv2-o1", env=env,
            )
            self.assertEqual(r.returncode, 0, f"stderr: {r.stderr}")
            combined = r.stdout + r.stderr
            self.assertIn("already completed", combined,
                          "Should print 'already completed' when resuming")
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    # ---- --full-test sample count -------------------------------------------
    def test_full_test_loads_all_samples(self):
        """--full-test should load full dataset, not default 5."""
        import subprocess, re
        env = os.environ.copy()
        env["TOKENIZERS_PARALLELISM"] = "false"
        # Do NOT set TEST_DATA_PATH — use real data to verify full load
        proc = subprocess.Popen(
            ["/data/gt/envs/lf_gjq/bin/python3", "test_closed_loop.py",
             "--full-test"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            cwd="/data/gt/omgv2-o1", env=env,
        )
        output_lines = []
        try:
            for line in proc.stdout:
                output_lines.append(line)
                if "Loading" in line and "samples" in line:
                    proc.kill()
                    break
        finally:
            proc.kill()
            proc.stdout.close()
            proc.wait()
        combined = "".join(output_lines)
        self.assertNotIn("Loading 5 samples", combined)
        match = re.search(r"Loading (\d+) samples", combined)
        self.assertIsNotNone(match, f"No 'Loading N samples' found in:\n{combined}")
        self.assertGreater(int(match.group(1)), 100,
                           "Full test should load >100 samples")


if __name__ == "__main__":
    unittest.main()
