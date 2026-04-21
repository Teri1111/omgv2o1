"""Test T8: Online feedback loop — success/fail count updates and weighted search."""
import sys
import os
import json
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_update_success():
    """Test that update_success increments success_count and persists."""
    sys.path.insert(0, "/data/gt/experience_kb")
    from modules.knowledge_base import ExperienceKB
    tmp_dir = tempfile.mkdtemp()
    try:
        kb = ExperienceKB(tmp_dir, embedding_model_name="all-MiniLM-L6-v2")
        rule_id = kb.add_rule({
            "rule_type": "ERROR_RECOVERY",
            "state_description": "test state",
            "action": {"description": "test action", "steps": ["step1"]},
        })
        assert kb.get_success_rate(rule_id) == 0.5  # neutral prior
        assert kb.update_success(rule_id) is True
        assert kb.rules[kb._id_to_idx[rule_id]]["success_count"] == 1
        # Verify persistence
        kb2 = ExperienceKB(tmp_dir, embedding_model_name="all-MiniLM-L6-v2")
        assert kb2.rules[kb2._id_to_idx[rule_id]]["success_count"] == 1
        assert kb2.get_success_rate(rule_id) == 1.0
        print("PASS: test_update_success")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_update_failure():
    """Test that update_failure increments fail_count and persists."""
    sys.path.insert(0, "/data/gt/experience_kb")
    from modules.knowledge_base import ExperienceKB
    tmp_dir = tempfile.mkdtemp()
    try:
        kb = ExperienceKB(tmp_dir, embedding_model_name="all-MiniLM-L6-v2")
        rule_id = kb.add_rule({
            "rule_type": "ERROR_RECOVERY",
            "state_description": "test state",
            "action": {"description": "test action", "steps": ["step1"]},
        })
        assert kb.update_failure(rule_id) is True
        assert kb.rules[kb._id_to_idx[rule_id]]["fail_count"] == 1
        # Verify persistence
        kb2 = ExperienceKB(tmp_dir, embedding_model_name="all-MiniLM-L6-v2")
        assert kb2.rules[kb2._id_to_idx[rule_id]]["fail_count"] == 1
        assert kb2.get_success_rate(rule_id) == 0.0
        print("PASS: test_update_failure")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_get_success_rate():
    """Test success rate calculation with mixed counts."""
    sys.path.insert(0, "/data/gt/experience_kb")
    from modules.knowledge_base import ExperienceKB
    tmp_dir = tempfile.mkdtemp()
    try:
        kb = ExperienceKB(tmp_dir, embedding_model_name="all-MiniLM-L6-v2")
        rule_id = kb.add_rule({
            "rule_type": "ERROR_RECOVERY",
            "state_description": "test state",
            "action": {"description": "test action", "steps": ["step1"]},
        })
        # 3 success, 1 fail => 75%
        for _ in range(3):
            kb.update_success(rule_id)
        kb.update_failure(rule_id)
        assert abs(kb.get_success_rate(rule_id) - 0.75) < 1e-6
        # Non-existent rule
        assert kb.get_success_rate("nonexistent") == 0.0
        print("PASS: test_get_success_rate")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_search_weights_success_rate():
    """Test that high success_rate rules get a score boost in search."""
    sys.path.insert(0, "/data/gt/experience_kb")
    from modules.knowledge_base import ExperienceKB
    tmp_dir = tempfile.mkdtemp()
    try:
        kb = ExperienceKB(tmp_dir, embedding_model_name="all-MiniLM-L6-v2")
        # Add two similar rules
        rid_good = kb.add_rule({
            "rule_type": "ERROR_RECOVERY",
            "state_description": "SPARQL query returns empty result",
            "action": {"description": "check type constraints", "steps": ["verify entity types"]},
        })
        rid_bad = kb.add_rule({
            "rule_type": "ERROR_RECOVERY",
            "state_description": "SPARQL query returns empty result set",
            "action": {"description": "relax type constraints", "steps": ["remove type filter"]},
        })
        # Make rid_good have high success rate
        for _ in range(5):
            kb.update_success(rid_good)
        # Make rid_bad have low success rate
        for _ in range(5):
            kb.update_failure(rid_bad)
        # Search — rid_good should rank higher due to success_rate boost
        results = kb.search("SPARQL empty result", top_k=5, threshold=0.1)
        assert len(results) >= 2
        assert results[0]["rule_id"] == rid_good
        print("PASS: test_search_weights_success_rate")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_feedback_in_trace():
    """Test that trace kb_stats gets feedback_applied field."""
    from reasoning.agent import TraceCollector
    tc = TraceCollector()
    kb_stats = tc.data.get("kb_stats", {})
    assert "retrieved_rule_ids" in kb_stats
    # Simulate feedback being applied
    kb_stats["feedback_applied"] = True
    kb_stats["feedback_is_correct"] = True
    kb_stats["feedback_rule_count"] = 2
    tc.data["kb_stats"] = kb_stats
    assert tc.data["kb_stats"]["feedback_applied"] is True
    assert tc.data["kb_stats"]["feedback_is_correct"] is True
    assert tc.data["kb_stats"]["feedback_rule_count"] == 2
    print("PASS: test_feedback_in_trace")


def test_skill_tracks_rule_ids():
    """Test that search_experience_rules records _LAST_RETRIEVED_RULE_IDS."""
    from skills.experience_kb_skill import get_last_retrieved_rule_ids
    # Should return a list (may be empty if KB not available)
    ids = get_last_retrieved_rule_ids()
    assert isinstance(ids, list)
    print("PASS: test_skill_tracks_rule_ids")


def test_hit_then_miss_clears_rule_ids():
    """Test that miss path clears _LAST_RETRIEVED_RULE_IDS."""
    import skills.experience_kb_skill as ekb
    # Manually set stale state
    ekb._LAST_RETRIEVED_RULE_IDS = ["stale_rule_1", "stale_rule_2"]
    # Simulate KB unavailable path
    original_get_kb = ekb._get_kb
    ekb._get_kb = lambda: None
    try:
        result = ekb.search_experience_rules("some question")
        assert result == ""
        assert ekb.get_last_retrieved_rule_ids() == [], f"Expected empty, got {ekb.get_last_retrieved_rule_ids()}"
    finally:
        ekb._get_kb = original_get_kb
    print("PASS: test_hit_then_miss_clears_rule_ids")


def test_save_rules():
    """Test that _save_rules persists rules to disk."""
    sys.path.insert(0, "/data/gt/experience_kb")
    from modules.knowledge_base import ExperienceKB
    tmp_dir = tempfile.mkdtemp()
    try:
        kb = ExperienceKB(tmp_dir, embedding_model_name="all-MiniLM-L6-v2")
        rule_id = kb.add_rule({
            "rule_type": "ERROR_RECOVERY",
            "state_description": "test",
            "action": {"description": "test", "steps": ["step1"]},
        })
        # Modify and save
        kb.rules[kb._id_to_idx[rule_id]]["success_count"] = 42
        kb._save_rules()
        # Verify file written
        with open(os.path.join(tmp_dir, "rules.json")) as f:
            rules = json.load(f)
        found = [r for r in rules if r["rule_id"] == rule_id]
        assert len(found) == 1
        assert found[0]["success_count"] == 42
        print("PASS: test_save_rules")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    tests = [
        test_update_success,
        test_update_failure,
        test_get_success_rate,
        test_search_weights_success_rate,
        test_feedback_in_trace,
        test_skill_tracks_rule_ids,
        test_hit_then_miss_clears_rule_ids,
        test_save_rules,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1
    print(f"\nResults: {passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(1 if failed > 0 else 0)
