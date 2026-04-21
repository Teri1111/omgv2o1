#!/usr/bin/env python3
"""
Test T6: Experience KB SKILL.md conversion and retrieval.

Tests:
1. SKILL.md files exist and have valid frontmatter
2. search_skill_docs() returns relevant results
3. consult_experience_adapter with query_type="skill_md" works
4. Backward compatibility: passive and active modes still work
5. Coverage: all rule types are represented in SKILL.md files
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DOMAIN_SKILLS_DIR = "/data/gt/omgv2-o1/skills/domain_skills"


def test_skill_md_files_exist():
    """Verify 10-15 SKILL.md files exist with .md extension."""
    files = [f for f in os.listdir(DOMAIN_SKILLS_DIR) if f.endswith(".md")]
    assert 10 <= len(files) <= 15, f"Expected 10-15 SKILL.md files, got {len(files)}"
    print(f"✓ Found {len(files)} SKILL.md files")


def test_skill_md_frontmatter():
    """Verify each SKILL.md has valid frontmatter with required fields."""
    import yaml
    files = [f for f in os.listdir(DOMAIN_SKILLS_DIR) if f.endswith(".md")]
    required_keys = {"rule_id", "name", "rule_type", "confidence", "tags", "triggers"}

    for fname in files:
        fpath = os.path.join(DOMAIN_SKILLS_DIR, fname)
        with open(fpath, "r") as f:
            content = f.read()

        # Check frontmatter delimiters
        assert content.startswith("---"), f"{fname}: missing frontmatter start"
        parts = content.split("---", 2)
        assert len(parts) >= 3, f"{fname}: incomplete frontmatter"

        # Parse frontmatter
        fm_text = parts[1].strip()
        metadata = {}
        for line in fm_text.split("\n"):
            if ":" in line:
                key, val = line.split(":", 1)
                metadata[key.strip()] = val.strip().strip('"').strip("'")

        # Check required keys
        for rk in required_keys:
            assert rk in metadata, f"{fname}: missing required key '{rk}'"

        # Check body exists (after frontmatter)
        body = parts[2].strip()
        assert len(body) > 50, f"{fname}: body too short ({len(body)} chars)"

    print(f"✓ All {len(files)} SKILL.md files have valid frontmatter")


def test_skill_md_rule_types_coverage():
    """Verify all 5 rule types are represented."""
    import json
    files = [f for f in os.listdir(DOMAIN_SKILLS_DIR) if f.endswith(".md")]
    types_found = set()

    for fname in files:
        fpath = os.path.join(DOMAIN_SKILLS_DIR, fname)
        with open(fpath, "r") as f:
            content = f.read()
        # Extract rule_type from frontmatter
        for line in content.split("\n"):
            if line.strip().startswith("rule_type:"):
                rt = line.split(":", 1)[1].strip().strip('"').strip("'")
                types_found.add(rt)
                break

    expected_types = {"RELATION_DIRECTION", "MULTI_HOP_PATTERN", "CONSTRAINT_PATTERN",
                      "CVT_NAVIGATION", "ERROR_RECOVERY"}
    missing = expected_types - types_found
    assert not missing, f"Missing rule types: {missing}"
    print(f"✓ All 5 rule types covered: {types_found}")


def test_search_skill_docs():
    """Test search_skill_docs() returns relevant results."""
    from skills.experience_kb_skill import search_skill_docs, _load_skill_docs

    # Force reload
    import skills.experience_kb_skill as ekb
    ekb._SKILL_DOCS = None

    docs = _load_skill_docs()
    assert len(docs) >= 10, f"Expected >= 10 docs loaded, got {len(docs)}"

    # Search for SPARQL-related docs
    results = search_skill_docs("SPARQL syntax error ORDER BY", top_k=3)
    assert len(results) > 0, "No results for SPARQL query"
    assert results[0]["score"] > 0, "Top result has 0 score"

    # The ORDER BY rule should be in top results
    titles = [r["title"] for r in results]
    found_orderby = any("ORDER BY" in t for t in titles)
    assert found_orderby, f"Expected ORDER BY rule in results, got: {titles}"
    print(f"✓ search_skill_docs returned {len(results)} results for SPARQL query")

    # Search for error recovery
    results2 = search_skill_docs("syntax error arrow notation", top_k=3)
    assert len(results2) > 0, "No results for syntax error query"
    print(f"✓ search_skill_docs returned {len(results2)} results for syntax error query")

    # Search for entity binding
    results3 = search_skill_docs("VALUES entity binding MID", top_k=3)
    assert len(results3) > 0, "No results for VALUES query"
    print(f"✓ search_skill_docs returned {len(results3)} results for VALUES query")


def test_get_skill_doc_content():
    """Test get_skill_doc_content() returns full SKILL.md content."""
    from skills.experience_kb_skill import get_skill_doc_content

    # Use rule_18f217f5 which contains SearchGraphPatterns
    content = get_skill_doc_content("rule_18f217f5")
    assert len(content) > 100, f"Content too short: {len(content)}"
    assert "SearchGraphPatterns" in content, "Missing expected content"
    assert "---" in content, "Missing frontmatter delimiter"
    print(f"✓ get_skill_doc_content returned {len(content)} chars for rule_18f217f5")

    # Non-existent rule
    empty = get_skill_doc_content("rule_nonexistent")
    assert empty == "", f"Expected empty for nonexistent rule, got: {empty}"
    print("✓ get_skill_doc_content returns empty for nonexistent rule")


def test_list_skill_docs():
    """Test list_skill_docs() returns all docs."""
    from skills.experience_kb_skill import list_skill_docs

    docs = list_skill_docs()
    assert len(docs) >= 10, f"Expected >= 10 docs, got {len(docs)}"

    rule_ids = [d["rule_id"] for d in docs]
    assert "rule_18f217f5" in rule_ids, "Missing rule_18f217f5"
    assert "rule_066e1183" in rule_ids, "Missing rule_066e1183"
    print(f"✓ list_skill_docs returned {len(docs)} docs")


def test_consult_experience_adapter_skill_md_mode():
    """Test consult_experience_adapter with query_type='skill_md'."""
    from skills.tools.adapters import consult_experience_adapter

    # skill_md mode
    result = consult_experience_adapter(
        state_description="SPARQL ORDER BY syntax error",
        query_type="skill_md",
        top_k=3
    )
    assert isinstance(result, dict), "Should return dict"
    assert result["query_type"] == "skill_md", f"Expected skill_md, got {result['query_type']}"
    assert "matched_skills" in result, "Missing matched_skills"
    assert "guidance_text" in result, "Missing guidance_text"
    assert "rule_ids" in result, "Missing rule_ids"
    assert len(result["matched_skills"]) > 0, "No matched skills for SPARQL query"

    # Verify full_content is included
    first_skill = result["matched_skills"][0]
    assert "full_content" in first_skill, "Missing full_content in matched skill"
    assert len(first_skill["full_content"]) > 50, "full_content too short"

    print(f"✓ consult_experience(skill_md) returned {len(result['matched_skills'])} skills")
    print(f"  Top skill: {first_skill['title']} (score={first_skill['score']:.1f})")

    # Empty query
    result2 = consult_experience_adapter(query_type="skill_md")
    assert result2["query_type"] == "skill_md"
    print("✓ consult_experience(skill_md) handles empty query gracefully")


def test_backward_compatibility_passive():
    """Test that passive mode still works (backward compatible)."""
    from skills.tools.adapters import consult_experience_adapter

    result = consult_experience_adapter(
        state_description="test entity",
        last_error="test error",
        current_expression="test expr",
        available_relations=["rel1", "rel2"],
        top_k=3
    )
    # Passive mode returns a string (formatted rules or empty)
    assert isinstance(result, str), f"Passive mode should return str, got {type(result)}"
    print("✓ Passive mode returns string (backward compatible)")


def test_backward_compatibility_active():
    """Test that active mode still works (backward compatible)."""
    from skills.tools.adapters import consult_experience_adapter

    result = consult_experience_adapter(
        state_description="test entity",
        last_error="test error",
        current_expression="test expr",
        available_relations=["rel1", "rel2"],
        top_k=5,
        query_type="active"
    )
    assert isinstance(result, dict), "Active mode should return dict"
    assert result["query_type"] == "active_consult"
    assert "matched_rules" in result
    print("✓ Active mode returns structured dict (backward compatible)")


def test_skill_md_content_quality():
    """Verify SKILL.md content includes steps and avoid sections."""
    from skills.experience_kb_skill import _load_skill_docs

    docs = _load_skill_docs()
    for rule_id, doc in docs.items():
        body = doc["body"]
        # Should have Steps section
        assert "## Steps" in body or "## steps" in body.lower(), \
            f"{rule_id}: missing Steps section"
        # Should have Avoid section
        assert "## Avoid" in body or "## avoid" in body.lower(), \
            f"{rule_id}: missing Avoid section"
        # Should have When to Apply section
        assert "## When to Apply" in body or "## when to apply" in body.lower(), \
            f"{rule_id}: missing When to Apply section"
        # Should have Common Cases section
        assert "## Common Cases" in body or "## common cases" in body.lower(), \
            f"{rule_id}: missing Common Cases section"

    print(f"✓ All {len(docs)} SKILL.md files have required sections")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    tests = [
        test_skill_md_files_exist,
        test_skill_md_frontmatter,
        test_skill_md_rule_types_coverage,
        test_search_skill_docs,
        test_get_skill_doc_content,
        test_list_skill_docs,
        test_consult_experience_adapter_skill_md_mode,
        test_backward_compatibility_passive,
        test_backward_compatibility_active,
        test_skill_md_content_quality,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"FAILED: {failed} test(s)")
        sys.exit(1)
