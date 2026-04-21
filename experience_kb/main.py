"""
KBQA Experience Knowledge Base - Main Entry Point.

Provides CLI interface for:
    1. Building KB from trajectories:  python main.py build --trajectory_dir <dir>
    2. Searching the KB:               python main.py search "SPARQL returns empty result"
    3. Running guided pipeline demo:   python main.py demo
    4. Consolidating rules:            python main.py consolidate
    5. Showing KB stats:               python main.py stats
"""

import os
import sys
import json
import argparse

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from modules.trajectory_collector import TrajectoryCollector
from modules.experience_extractor import ExperienceExtractor, LLMClient
from modules.knowledge_base import ExperienceKB
from modules.rule_retriever import RuleRetriever


def cmd_build(args):
    """Build the KB from trajectory files."""
    print(f"[Build] Building experience KB from trajectories in: {args.trajectory_dir}")

    # Initialize components
    collector = TrajectoryCollector()
    kb = ExperienceKB(args.kb_dir, args.embedding_model)

    # Collect trajectories
    if os.path.isdir(args.trajectory_dir):
        trajectories = collector.load_trajectory_dir(args.trajectory_dir)
    elif args.trajectory_dir.endswith(".jsonl"):
        trajectories = collector.load_trajectory_jsonl(args.trajectory_dir)
    elif args.trajectory_dir.endswith(".json"):
        t = collector.load_trajectory(args.trajectory_dir)
        trajectories = [t] if t else []
    else:
        print(f"[Build] Unknown input format: {args.trajectory_dir}")
        return

    if not trajectories:
        print("[Build] No trajectories loaded.")
        return

    print(f"[Build] Loaded {len(trajectories)} trajectories.")

    # Extract episodes
    all_episodes = []
    for traj in trajectories:
        episodes = collector.extract_episodes(traj)
        all_episodes.extend(episodes)

    print(f"[Build] Extracted {len(all_episodes)} step-level episodes.")

    # Extract rules with LLM
    if args.use_llm:
        extractor = ExperienceExtractor(
            llm_base_url=args.llm_url,
            llm_api_key=args.llm_key,
            llm_model_name=args.llm_model,
        )
        rules = extractor.extract_rules_batch(all_episodes)
        print(f"[Build] Extracted {len(rules)} rules via LLM.")
    else:
        # Create rules directly from episodes (no LLM)
        rules = _episodes_to_rules_simple(all_episodes)
        print(f"[Build] Created {len(rules)} rules (simple mode, no LLM).")

    if not rules:
        print("[Build] No rules to add.")
        return

    # Add to KB
    rule_ids = kb.add_rules_batch(rules)
    kb.save()

    print(f"[Build] Added {len(rule_ids)} rules to KB at: {args.kb_dir}")
    print(f"[Build] KB stats: {json.dumps(kb.get_stats(), indent=2)}")


def cmd_search(args):
    """Search the KB for relevant rules."""
    kb = ExperienceKB(args.kb_dir, args.embedding_model)
    retriever = RuleRetriever(kb)

    if not kb.rules:
        print("[Search] KB is empty. Run 'build' first.")
        return

    print(f"[Search] Query: {args.query}")
    print(f"[Search] Top-{args.top_k} results (threshold={args.threshold}):\n")

    rules = kb.search(args.query, top_k=args.top_k, threshold=args.threshold)

    if not rules:
        print("  No matching rules found.")
        return

    for i, rule in enumerate(rules, 1):
        print(f"  [{i}] Score: {rule['score']:.3f} | Type: {rule['rule_type']} | {rule.get('title', 'N/A')}")
        print(f"      State: {rule.get('state_description', 'N/A')[:100]}")
        action = rule.get("action", {})
        print(f"      Action: {action.get('description', 'N/A')[:100]}")
        print()

    if args.format_guidance:
        print("--- Formatted Guidance ---")
        print(retriever.format_as_guidance(rules))


def cmd_stats(args):
    """Show KB statistics."""
    kb = ExperienceKB(args.kb_dir, args.embedding_model)
    stats = kb.get_stats()

    print("=== Experience KB Statistics ===")
    print(f"  Total rules: {stats['total_rules']}")
    print(f"  Consolidated: {stats['consolidated']}")
    print(f"  Unconsolidated: {stats['unconsolidated']}")
    print(f"  Embedding dim: {stats['embedding_dim']}")
    print(f"  Total successes: {stats['total_success']}")
    print(f"  Total failures: {stats['total_fail']}")
    print(f"\n  By type:")
    for rt, count in stats["by_type"].items():
        print(f"    {rt}: {count}")


def cmd_consolidate(args):
    """Run consolidation on unconsolidated rules."""
    kb = ExperienceKB(args.kb_dir, args.embedding_model)
    stats = kb.get_stats()

    print(f"[Consolidate] KB has {stats['unconsolidated']} unconsolidated rules.")

    if stats["unconsolidated"] < 2:
        print("[Consolidate] Not enough unconsolidated rules. Need at least 2.")
        return

    llm = LLMClient(
        base_url=args.llm_url,
        api_key=args.llm_key,
        model_name=args.llm_model,
    )

    kb.consolidate(
        llm_client=llm,
        cluster_sim_threshold=args.sim_threshold,
        min_unconsolidated=args.min_rules,
    )

    new_stats = kb.get_stats()
    print(f"[Consolidate] Done. New stats: {new_stats['total_rules']} total, {new_stats['consolidated']} consolidated.")


def cmd_demo(args):
    """Run a quick demo of the full pipeline."""
    print("=== Experience KB Demo ===\n")

    kb = ExperienceKB(args.kb_dir, args.embedding_model)
    retriever = RuleRetriever(kb)

    if kb.embed_model is None:
        print("No embedding model. Install sentence-transformers: pip install sentence-transformers")
        return

    # Add sample rules
    sample_rules = [
        {
            "title": "Empty SPARQL Result Recovery",
            "description": "When SPARQL returns 0 results with entity constraint, verify entity type.",
            "rule_type": "ERROR_RECOVERY",
            "state_description": "SPARQL query with specific entity returns empty results",
            "state_keywords": ["empty", "zero", "entity", "sparql"],
            "action": {
                "description": "Verify entity type and try broader search",
                "steps": [
                    "Check entity type with SearchTypes",
                    "Try broader entity search",
                    "Relax the problematic constraint",
                ],
            },
        },
        {
            "title": "How Many Requires COUNT",
            "description": "Questions asking 'how many' need COUNT aggregate.",
            "rule_type": "CONSTRAINT_GUIDE",
            "state_description": "Question asks 'how many' requiring aggregation",
            "state_keywords": ["how_many", "count", "total"],
            "action": {
                "description": "Use COUNT() with GROUP BY",
                "steps": ["Wrap in COUNT()", "Add GROUP BY", "Remove LIMIT"],
            },
        },
    ]

    if len(kb.rules) == 0:
        kb.add_rules_batch(sample_rules)
        kb.save()
        print(f"Added {len(sample_rules)} sample rules.\n")

    # Demo retrieval
    test_queries = [
        ("SPARQL returns empty set for entity query", "Entity query failure"),
        ("How many states are in the United States?", "COUNT question"),
        ("What is the capital of France?", "Simple property query"),
    ]

    for query, desc in test_queries:
        print(f"--- {desc}: '{query}' ---")
        guidance = retriever.get_guidance_for_prompt(
            question=query,
            current_sparql="SELECT ?x WHERE { ... }",
            last_error="empty result" if "empty" in desc.lower() else None,
        )
        if guidance:
            print(guidance)
        else:
            print("  (No matching rules)")
        print()

    print("=== Demo Complete ===")


def _episodes_to_rules_simple(episodes):
    """Convert episodes to rules without LLM (simple template-based)."""
    rules = []
    for ep in episodes:
        state = ep.get("state", {})
        action = ep.get("action", {})
        error_type = state.get("error_type", "unknown")
        outcome = ep.get("outcome", "unknown")

        if outcome == "failure" and not ep.get("is_recovery"):
            # Skip non-recovery failures (not useful as rules)
            continue

        rule_type = "ERROR_RECOVERY" if ep.get("is_recovery") else "SUCCESS_SHORTCUT"

        rule = {
            "rule_type": rule_type,
            "state_description": f"{error_type}: {state.get('error_message', 'N/A')}",
            "state_keywords": [error_type, outcome, action.get("type", "")],
            "title": f"{'Recovery' if ep.get('is_recovery') else 'Success'} for {error_type}",
            "description": f"From trajectory {ep.get('source_trajectory', 'unknown')}, step {ep.get('step_index', 0)}",
            "action": {
                "description": action.get("reasoning", "No reasoning recorded"),
                "steps": [action.get("new_sparql", "")],
            },
            "source_trajectories": [ep.get("source_trajectory", "")],
        }
        rules.append(rule)

    return rules


def main():
    parser = argparse.ArgumentParser(
        description="KBQA Experience Knowledge Base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py build --trajectory_dir data/trajectories/
  python main.py search "SPARQL returns empty result"
  python main.py stats
  python main.py consolidate
  python main.py demo
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--kb_dir", default=Config.KB_DIR, help="KB directory")
    common.add_argument("--embedding_model", default=Config.EMBEDDING_MODEL, help="Embedding model name")
    common.add_argument("--llm_url", default=Config.LLM_BASE_URL, help="LLM API base URL")
    common.add_argument("--llm_key", default=Config.LLM_API_KEY, help="LLM API key")
    common.add_argument("--llm_model", default=Config.LLM_MODEL_NAME, help="LLM model name")

    # build
    p_build = subparsers.add_parser("build", parents=[common], help="Build KB from trajectories")
    p_build.add_argument("trajectory_dir", help="Path to trajectory files (dir, .json, or .jsonl)")
    p_build.add_argument("--use_llm", action="store_true", help="Use LLM for rule extraction (default: simple mode)")

    # search
    p_search = subparsers.add_parser("search", parents=[common], help="Search KB for rules")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--top_k", type=int, default=5, help="Number of results")
    p_search.add_argument("--threshold", type=float, default=0.4, help="Min similarity threshold")
    p_search.add_argument("--format_guidance", action="store_true", help="Show formatted guidance")

    # stats
    subparsers.add_parser("stats", parents=[common], help="Show KB statistics")

    # consolidate
    p_cons = subparsers.add_parser("consolidate", parents=[common], help="Consolidate episodic rules into meta-rules")
    p_cons.add_argument("--sim_threshold", type=float, default=0.75, help="Cluster similarity threshold")
    p_cons.add_argument("--min_rules", type=int, default=3, help="Min unconsolidated rules to trigger")

    # demo
    subparsers.add_parser("demo", parents=[common], help="Run quick demo")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    Config.print_config()
    print()

    commands = {
        "build": cmd_build,
        "search": cmd_search,
        "stats": cmd_stats,
        "consolidate": cmd_consolidate,
        "demo": cmd_demo,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
