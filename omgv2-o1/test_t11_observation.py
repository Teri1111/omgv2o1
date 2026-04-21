"""T11 observation unit tests.

Covers:
1. ObservationDecideAction — observation action dispatch from _decide_action
2. ObservationBudget — budget mechanism blocks/falls back correctly
3. ObservationDispatch — observation execution branch in run()
"""
import sys
import os
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


def _make_observation_choice(tool="explore_neighbors", args=None, thought="test"):
    """Return a mock LLM chooser result for an observation action."""
    if args is None:
        args = {"entity": "m.01", "direction": "outbound"}
    return ({
        "action": "observation",
        "tool": tool,
        "args": args,
        "thought": thought,
    }, True)


# ---------------------------------------------------------------------------
# Test class 1: observation action dispatch
# ---------------------------------------------------------------------------
class TestObservationDecideAction(unittest.TestCase):
    """Observation action should be returned by _decide_action when LLM
    returns an observation tool call."""

    def test_observation_returns_observation_action(self):
        from reasoning.agent import LLMGuidedAgent
        sub = _make_subgraph()
        agent = LLMGuidedAgent(
            question="test", entities=["m.01"], subgraph=sub,
            max_steps=3, llm_first=True,
        )
        mock_choice, _ = _make_observation_choice()
        with patch("reasoning.llm_agent.choose_next_step_function_call",
                   return_value=(mock_choice, True)):
            result = agent._decide_action(
                {"film.actor.film": ["m.02"]}, {})
        self.assertEqual(result, "observation")

    def test_observation_scratchpad_has_action_line(self):
        from reasoning.agent import LLMGuidedAgent
        sub = _make_subgraph()
        agent = LLMGuidedAgent(
            question="test", entities=["m.01"], subgraph=sub,
            max_steps=3, llm_first=True,
        )
        mock_choice, _ = _make_observation_choice(tool="explore_neighbors")
        with patch("reasoning.llm_agent.choose_next_step_function_call",
                   return_value=(mock_choice, True)):
            agent._decide_action({"film.actor.film": ["m.02"]}, {})
        self.assertIn("Observe [ explore_neighbors ]", agent.scratchpad)

    def test_observation_scratchpad_has_thought_line(self):
        from reasoning.agent import LLMGuidedAgent
        sub = _make_subgraph()
        agent = LLMGuidedAgent(
            question="test", entities=["m.01"], subgraph=sub,
            max_steps=3, llm_first=True,
        )
        mock_choice, _ = _make_observation_choice(thought="my reasoning here")
        with patch("reasoning.llm_agent.choose_next_step_function_call",
                   return_value=(mock_choice, True)):
            agent._decide_action({"film.actor.film": ["m.02"]}, {})
        self.assertIn("Thought", agent.scratchpad)
        self.assertIn("my reasoning here", agent.scratchpad)


# ---------------------------------------------------------------------------
# Test class 2: observation budget
# ---------------------------------------------------------------------------
class TestObservationBudget(unittest.TestCase):
    """Budget mechanism for observation tools."""

    def test_budget_allows_within_limit(self):
        from reasoning.agent import LLMGuidedAgent
        sub = _make_subgraph()
        agent = LLMGuidedAgent(
            question="test", entities=["m.01"], subgraph=sub,
            max_steps=3, llm_first=True,
        )
        # Default budget for explore_neighbors is 2, count is 0
        self.assertEqual(agent._observation_counts["explore_neighbors"], 0)
        mock_choice, _ = _make_observation_choice(tool="explore_neighbors")
        with patch("reasoning.llm_agent.choose_next_step_function_call",
                   return_value=(mock_choice, True)):
            result = agent._decide_action(
                {"film.actor.film": ["m.02"]}, {})
        self.assertEqual(result, "observation")

    def test_budget_blocks_at_limit(self):
        from reasoning.agent import LLMGuidedAgent
        sub = _make_subgraph()
        agent = LLMGuidedAgent(
            question="test", entities=["m.01"], subgraph=sub,
            max_steps=3, llm_first=True,
        )
        # Set count to budget limit
        agent._observation_counts["explore_neighbors"] = agent._observation_budget["explore_neighbors"]
        mock_choice, _ = _make_observation_choice(tool="explore_neighbors")
        with patch("reasoning.llm_agent.choose_next_step_function_call",
                   return_value=(mock_choice, True)):
            result = agent._decide_action(
                {"film.actor.film": ["m.02"]}, {})
        # Should fall back to heuristic (join_forward since out_rels present)
        self.assertNotEqual(result, "observation")
        self.assertEqual(result, "join_forward")

    def test_budget_counter_increments(self):
        from reasoning.agent import LLMGuidedAgent
        sub = _make_subgraph()
        agent = LLMGuidedAgent(
            question="test", entities=["m.01"], subgraph=sub,
            max_steps=2, llm_first=True,
        )
        initial = agent._observation_counts["explore_neighbors"]

        # First call: LLM returns observation, then dispatch executes it
        obs_choice = _make_observation_choice(tool="explore_neighbors")[0]
        join_choice = (
            {"action": "join_forward", "relation": "film.actor.film", "thought": "go"}, True)

        call_iter = iter([
            (obs_choice, True),   # first _decide_action -> observation
            join_choice,          # second _decide_action -> join_forward
        ])

        def mock_chooser(*a, **kw):
            return next(call_iter)

        with patch("reasoning.agent.LLMGuidedAgent._should_consult_experience", return_value=False):
            with patch("reasoning.llm_agent.choose_next_step_function_call", side_effect=mock_chooser):
                with patch("reasoning.agent.LLMGuidedAgent._execute_observation",
                           return_value={"observation": "neighbors of m.01", "success": True}):
                    agent.run()

        # After one observation execution, count should have incremented
        self.assertEqual(
            agent._observation_counts["explore_neighbors"], initial + 1)

    def test_budget_fallback_writes_trace(self):
        from reasoning.agent import LLMGuidedAgent
        from reasoning.agent import TraceCollector
        tc = TraceCollector()
        sub = _make_subgraph()
        agent = LLMGuidedAgent(
            question="test", entities=["m.01"], subgraph=sub,
            max_steps=3, llm_first=True, trace_collector=tc,
        )
        # Exhaust budget
        tool = "explore_neighbors"
        agent._observation_counts[tool] = agent._observation_budget[tool]

        mock_choice, _ = _make_observation_choice(tool=tool)
        with patch("reasoning.llm_agent.choose_next_step_function_call",
                   return_value=(mock_choice, True)):
            # Need a step trace for fallback recording
            tc.new_step()
            agent._decide_action({"film.actor.film": ["m.02"]}, {})

        step = tc.data["steps"][-1]
        self.assertIn("llm_trace", step)
        self.assertIn("fallback_reason", step["llm_trace"])
        self.assertIn("observation_budget_exhausted",
                       step["llm_trace"]["fallback_reason"])


# ---------------------------------------------------------------------------
# Test class 3: observation execution dispatch
# ---------------------------------------------------------------------------
class TestObservationDispatch(unittest.TestCase):
    """Observation execution branch in run()."""

    def test_observation_writes_to_scratchpad(self):
        from reasoning.agent import LLMGuidedAgent
        sub = _make_subgraph()
        agent = LLMGuidedAgent(
            question="test", entities=["m.01"], subgraph=sub,
            max_steps=2, llm_first=True,
        )

        obs_choice = _make_observation_choice(tool="explore_neighbors")[0]
        join_choice = (
            {"action": "join_forward", "relation": "film.actor.film", "thought": "go"}, True)

        call_iter = iter([
            (obs_choice, True),
            join_choice,
        ])

        def mock_chooser(*a, **kw):
            return next(call_iter)

        with patch("reasoning.agent.LLMGuidedAgent._should_consult_experience", return_value=False):
            with patch("reasoning.llm_agent.choose_next_step_function_call", side_effect=mock_chooser):
                with patch("reasoning.agent.LLMGuidedAgent._execute_observation",
                           return_value={"observation": "m.01 -> film.actor.film -> m.02", "success": True}):
                    agent.run()

        # Scratchpad should contain an ObservationN: ... line
        self.assertIn("Observation", agent.scratchpad)
        self.assertIn("m.01 -> film.actor.film -> m.02", agent.scratchpad)

    def test_observation_does_not_advance_function_list(self):
        from reasoning.agent import LLMGuidedAgent
        sub = _make_subgraph()
        agent = LLMGuidedAgent(
            question="test", entities=["m.01"], subgraph=sub,
            max_steps=2, llm_first=True,
        )

        obs_choice = _make_observation_choice(tool="explore_neighbors")[0]
        join_choice = (
            {"action": "join_forward", "relation": "film.actor.film", "thought": "go"}, True)

        call_iter = iter([
            (obs_choice, True),
            join_choice,
        ])

        def mock_chooser(*a, **kw):
            return next(call_iter)

        with patch("reasoning.agent.LLMGuidedAgent._should_consult_experience", return_value=False):
            with patch("reasoning.llm_agent.choose_next_step_function_call", side_effect=mock_chooser):
                with patch("reasoning.agent.LLMGuidedAgent._execute_observation",
                           return_value={"observation": "neighbors", "success": True}):
                    fl, sexpr = agent.run()

        # The function_list should contain START + one JOIN + STOP (observation does not add LF)
        join_lines = [l for l in fl if "JOIN" in l]
        self.assertEqual(len(join_lines), 1,
                         "Observation step should not add extra JOIN to function_list")


# ---------------------------------------------------------------------------
# Test class 4: budget edge cases
# ---------------------------------------------------------------------------
class TestObservationBudgetEdgeCases(unittest.TestCase):
    """Edge cases for observation budget and count reset."""

    def test_budget_fallback_scratchpad_no_observe(self):
        from reasoning.agent import LLMGuidedAgent
        sub = _make_subgraph()
        agent = LLMGuidedAgent(
            question="test", entities=["m.01"], subgraph=sub,
            max_steps=3, llm_first=True,
        )
        # Exhaust budget: set count to budget limit
        agent._observation_counts["explore_neighbors"] = 2
        agent._observation_budget["explore_neighbors"] = 2
        mock_choice, _ = _make_observation_choice(tool="explore_neighbors")
        with patch("reasoning.llm_agent.choose_next_step_function_call",
                   return_value=(mock_choice, True)):
            result = agent._decide_action(
                {"film.actor.film": ["m.02"]}, {})
        # Should fall back to heuristic, not return "observation"
        self.assertNotEqual(result, "observation")
        self.assertEqual(result, "join_forward")
        # Scratchpad should NOT contain "Observe" line
        self.assertNotIn("Observe", agent.scratchpad)
        # Scratchpad must contain the heuristic fallback action, not be empty
        has_fallback_action = ("Find_relation" in agent.scratchpad or
                               "Count" in agent.scratchpad or
                               "Finish" in agent.scratchpad or
                               "Argmax" in agent.scratchpad or
                               "Argmin" in agent.scratchpad)
        self.assertTrue(has_fallback_action,
                        f"Budget fallback must write heuristic Action line to scratchpad, got: {agent.scratchpad!r}")

    def test_counts_reset_per_run(self):
        from reasoning.agent import LLMGuidedAgent
        sub = _make_subgraph()
        agent = LLMGuidedAgent(
            question="test", entities=["m.01"], subgraph=sub,
            max_steps=1, llm_first=True,
        )
        # Simulate exhausted counts from a previous run
        agent._observation_counts["explore_neighbors"] = 5
        # Mock _decide_action to return "finish" so the loop completes in 1 step
        with patch.object(agent, "_decide_action", return_value="finish"):
            agent.run()
        # run() should have reset observation counts at lines 987-989
        self.assertEqual(agent._observation_counts["explore_neighbors"], 0)


if __name__ == "__main__":
    unittest.main()
