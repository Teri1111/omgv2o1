"""
Smoke test for OMGv2 subgraph builder.
Tests basic functionality of SubgraphBuilder.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reasoning.subgraph import SubgraphBuilder


def test_basic_path():
    """Test basic path processing."""
    builder = SubgraphBuilder()
    
    # Test single path: e0 -> r1 -> e1 -> r2 -> e2
    paths = [
        {
            "path": ["m.0f8l9c", "film.actor.film", "m.0bxtg", "film.film.directed_by", "m.0gqs1"],
            "score": 0.95
        }
    ]
    
    builder.build(paths)
    
    # Check entities
    entities = builder.get_all_entities()
    assert "m.0f8l9c" in entities
    assert "m.0bxtg" in entities
    assert "m.0gqs1" in entities
    assert len(entities) == 3
    
    # Check relations
    relations = builder.get_all_relations()
    assert "film.actor.film" in relations
    assert "film.film.directed_by" in relations
    assert len(relations) == 2
    
    # Check outgoing edges
    outgoing_from_f8l9c = builder.get_outgoing("m.0f8l9c")
    assert len(outgoing_from_f8l9c) == 1
    assert outgoing_from_f8l9c[0] == ("film.actor.film", "m.0bxtg")
    
    outgoing_from_bxtg = builder.get_outgoing("m.0bxtg")
    assert len(outgoing_from_bxtg) == 1
    assert outgoing_from_bxtg[0] == ("film.film.directed_by", "m.0gqs1")
    
    outgoing_from_gqs1 = builder.get_outgoing("m.0gqs1")
    assert len(outgoing_from_gqs1) == 0
    
    print("✓ Basic path test passed")


def test_reverse_relation():
    """Test reverse relation handling."""
    builder = SubgraphBuilder()
    
    # Test path with reverse relation: e0 -> (R r1) -> e1
    # This should create edge: e1 -> r1 -> e0
    paths = [
        {
            "path": ["m.0f8l9c", "(R film.actor.film)", "m.0bxtg"],
            "score": 0.9
        }
    ]
    
    builder.build(paths)
    
    # Check that reverse relation creates reversed edge
    outgoing_from_bxtg = builder.get_outgoing("m.0bxtg")
    assert len(outgoing_from_bxtg) == 1
    assert outgoing_from_bxtg[0] == ("film.actor.film", "m.0f8l9c")
    
    outgoing_from_f8l9c = builder.get_outgoing("m.0f8l9c")
    assert len(outgoing_from_f8l9c) == 0
    
    print("✓ Reverse relation test passed")


def test_multiple_paths():
    """Test merging multiple paths."""
    builder = SubgraphBuilder()
    
    paths = [
        {
            "path": ["m.0f8l9c", "film.actor.film", "m.0bxtg", "film.film.directed_by", "m.0gqs1"],
            "score": 0.95
        },
        {
            "path": ["m.0f8l9c", "film.actor.film", "m.0bxtg", "film.film.genre", "m.02l7c8"],
            "score": 0.85
        }
    ]
    
    builder.build(paths)
    
    # Check that m.0bxtg has two outgoing edges
    outgoing_from_bxtg = builder.get_outgoing("m.0bxtg")
    assert len(outgoing_from_bxtg) == 2
    
    relations_from_bxtg = [rel for rel, _ in outgoing_from_bxtg]
    assert "film.film.directed_by" in relations_from_bxtg
    assert "film.film.genre" in relations_from_bxtg
    
    print("✓ Multiple paths test passed")


def test_empty_paths():
    """Test with empty or invalid paths."""
    builder = SubgraphBuilder()
    
    # Test empty list
    builder.build([])
    assert len(builder.get_all_entities()) == 0
    assert len(builder.get_all_relations()) == 0
    
    # Test invalid paths
    builder.build([{"path": [], "score": 0.5}])
    assert len(builder.get_all_entities()) == 0
    
    builder.build([{"path": ["m.0f8l9c"], "score": 0.5}])
    assert len(builder.get_all_entities()) == 0
    
    print("✓ Empty paths test passed")


if __name__ == "__main__":
    test_basic_path()
    test_reverse_relation()
    test_multiple_paths()
    test_empty_paths()
    print("\nAll smoke tests passed!")
