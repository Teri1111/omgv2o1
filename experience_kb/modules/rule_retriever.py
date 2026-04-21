"""
KBQA Experience Knowledge Base - Rule Retriever.

Runtime retrieval interface that queries the KB for relevant error-correction rules
and formats them for LLM prompt injection.

Adapted from vkbqa/modules/iterative_reasoner.py Step 0 (memory retrieval)
and the prompt injection pattern in _reason_step.

Key design:
    - Build semantic query from current SPARQL generation state
    - Retrieve top-K rules via vector similarity
    - Re-rank based on error type match, confidence, usage stats
    - Format as structured guidance text for LLM prompt
"""

import re
from typing import List, Dict, Optional

from .knowledge_base import ExperienceKB


class RuleRetriever:
    """
    Retrieves and formats error-correction rules for LLM prompt injection.
    
    Usage:
        kb = ExperienceKB("/path/to/kb")
        retriever = RuleRetriever(kb)
        guidance = retriever.get_guidance_for_prompt(
            question="What college did Obama attend?",
            current_sparql="SELECT ?x WHERE { ns:m.02mjmr ns:people.person.education ?e . ?e ns:education.education.institution ?x }",
            last_error="Empty result set"
        )
        # Inject guidance into LLM prompt
    """

    def __init__(self, knowledge_base: ExperienceKB):
        self.kb = knowledge_base

    # ------------------------------------------------------------------
    # Main retrieval methods
    # ------------------------------------------------------------------

    def retrieve_for_state(
        self,
        question: str,
        current_sparql: str = "",
        last_error: Optional[str] = None,
        entity_links: Optional[List[str]] = None,
        step_history: Optional[List[Dict]] = None,
        top_k: int = 3,
        rule_types: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Retrieve relevant rules for the current SPARQL generation state.
        
        Args:
            question: The natural language question
            current_sparql: Current SPARQL query (partial or full)
            last_error: Last error message from SPARQL execution (if any)
            entity_links: List of linked entity names/IDs
            step_history: Previous generation steps for context
            top_k: Number of rules to retrieve
            rule_types: Optional filter by rule types
            
        Returns:
            List of rule dicts with score, sorted by relevance
        """
        # Build state query
        query = self._build_state_query(
            question, current_sparql, last_error, entity_links, step_history
        )

        # Search KB
        rules = self.kb.search(query, top_k=top_k * 2, rule_types=rule_types, threshold=0.4)

        # Re-rank
        rules = self._rank_rules(
            rules, question, current_sparql, last_error, entity_links, step_history
        )

        return rules[:top_k]

    def format_as_guidance(self, rules: List[Dict]) -> str:
        """
        Format rules as detailed prompt guidance text.
        
        Returns structured text like:
            [Experience Knowledge Base - Retrieved Guidance]
            Rule 1 (ERROR_RECOVERY, confidence: 0.92):
              Pattern: When SPARQL returns empty results
              Action: 1. Verify entity type ...
        """
        if not rules:
            return ""

        lines = ["[Experience Knowledge Base - Retrieved Guidance]"]
        for i, rule in enumerate(rules, 1):
            rule_type = rule.get("rule_type", "UNKNOWN")
            score = rule.get("score", 0.0)
            confidence = rule.get("confidence", 0.0)
            title = rule.get("title", "Untitled")
            state_desc = rule.get("state_description", "N/A")

            action = rule.get("action", {})
            action_desc = action.get("description", "N/A")
            steps = action.get("steps", [])

            lines.append(f"\nRule {i} ({rule_type}, relevance: {score:.2f}, confidence: {confidence:.2f}):")
            lines.append(f"  Title: {title}")
            lines.append(f"  Pattern: {state_desc}")
            lines.append(f"  Action: {action_desc}")
            if steps:
                lines.append(f"  Steps:")
                for step in steps:
                    lines.append(f"    - {step}")

            # Show usage stats if available
            sc = rule.get("success_count", 0)
            fc = rule.get("fail_count", 0)
            if sc + fc > 0:
                lines.append(f"  Validated: {sc} successes, {fc} failures ({sc}/{sc+fc} = {sc/(sc+fc):.0%} success rate)")

        return "\n".join(lines)

    def format_as_compact_guidance(self, rules: List[Dict]) -> str:
        """
        Format rules as compact one-liners for context-limited situations.
        
        Returns:
            [Experience Guidance]
            - Empty result? Check entity type, try broader search (0.92)
            - Type mismatch? Re-link entity, verify predicate (0.87)
        """
        if not rules:
            return ""

        lines = ["[Experience Guidance]"]
        for rule in rules:
            title = rule.get("title", "Untitled")
            score = rule.get("score", 0.0)
            action = rule.get("action", {})
            steps = action.get("steps", [])
            # Take first step as summary
            hint = steps[0] if steps else action.get("description", "")
            # Truncate if too long
            if len(hint) > 100:
                hint = hint[:97] + "..."
            lines.append(f"  - {title} -> {hint} ({score:.2f})")

        return "\n".join(lines)

    def get_guidance_for_prompt(
        self,
        question: str,
        current_sparql: str = "",
        last_error: Optional[str] = None,
        entity_links: Optional[List[str]] = None,
        step_history: Optional[List[Dict]] = None,
        top_k: int = 3,
        compact: bool = False,
    ) -> str:
        """
        Convenience method: retrieve + format in one call.
        
        Args:
            compact: If True, use compact format (fewer tokens)
            
        Returns:
            Formatted guidance string ready for prompt injection
        """
        rules = self.retrieve_for_state(
            question, current_sparql, last_error, entity_links, step_history, top_k
        )
        if compact:
            return self.format_as_compact_guidance(rules)
        return self.format_as_guidance(rules)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_state_query(
        self,
        question: str,
        current_sparql: str,
        last_error: Optional[str],
        entity_links: Optional[List[str]],
        step_history: Optional[List[Dict]],
    ) -> str:
        """
        Construct a search query from the current SPARQL generation state.
        
        Weight: error > SPARQL > question > entities
        """
        parts = []

        # Error message (highest weight - most diagnostic)
        if last_error:
            parts.append(f"Error: {last_error}")

        # Current SPARQL structure
        if current_sparql:
            # Extract key structural elements
            sparql_summary = self._summarize_sparql(current_sparql)
            parts.append(f"SPARQL: {sparql_summary}")

        # Question (medium weight)
        if question:
            parts.append(f"Question: {question}")

        # Entity links (lower weight)
        if entity_links:
            parts.append(f"Entities: {', '.join(entity_links[:5])}")

        # Step history context
        if step_history:
            last_step = step_history[-1] if step_history else {}
            if last_step.get("error"):
                parts.append(f"Last step error: {last_step['error']}")

        return " | ".join(parts) if parts else "general KBQA state"

    def _summarize_sparql(self, sparql: str) -> str:
        """Extract key features from SPARQL for search."""
        features = []

        # Check for common patterns
        if "FILTER" in sparql.upper():
            features.append("uses FILTER")
        if "OPTIONAL" in sparql.upper():
            features.append("uses OPTIONAL")
        if "COUNT" in sparql.upper():
            features.append("uses COUNT")
        if "ORDER BY" in sparql.upper():
            features.append("uses ORDER BY")
        if "LIMIT" in sparql.upper():
            features.append("uses LIMIT")
        if "NOT EXISTS" in sparql.upper():
            features.append("uses NOT EXISTS")
        if "UNION" in sparql.upper():
            features.append("uses UNION")

        # Count triple patterns
        triple_count = sparql.count(".")
        features.append(f"{triple_count} triple patterns")

        # Check for empty result indicators
        if len(sparql.strip()) < 20:
            features.append("minimal query")

        return f"({', '.join(features)}) {sparql[:200]}"

    def _rank_rules(
        self,
        rules: List[Dict],
        question: str,
        current_sparql: str,
        last_error: Optional[str],
        entity_links: Optional[List[str]],
        step_history: Optional[List[Dict]],
    ) -> List[Dict]:
        """
        Re-rank rules based on contextual relevance.
        
        Boosting factors:
        1. Error type keyword match: +0.15
        2. High success rate (success / total > 0.7): +0.10
        3. Semantic_Rule type: +0.05 (already boosted in search, extra here)
        4. Low usage penalty (total < 2): -0.10
        5. SPARQL pattern keyword match: +0.10
        """
        for rule in rules:
            boost = 0.0

            # 1. Error type match
            if last_error:
                error_lower = last_error.lower()
                state_keywords = rule.get("state_keywords", [])
                state_desc = rule.get("state_description", "").lower()

                # Check keyword overlap
                keyword_hits = sum(1 for kw in state_keywords if kw.lower() in error_lower)
                if keyword_hits > 0:
                    boost += 0.05 * min(keyword_hits, 3)

                # Check state description overlap
                if any(word in error_lower for word in state_desc.split() if len(word) > 4):
                    boost += 0.10

            # 2. Success rate boost
            sc = rule.get("success_count", 0)
            fc = rule.get("fail_count", 0)
            total_usage = sc + fc
            if total_usage >= 2:
                success_rate = sc / total_usage
                if success_rate > 0.7:
                    boost += 0.10
                elif success_rate < 0.3:
                    boost -= 0.05
            else:
                # Low usage penalty
                boost -= 0.10

            # 3. SPARQL keyword match
            if current_sparql:
                sparql_lower = current_sparql.lower()
                action_desc = rule.get("action", {}).get("description", "").lower()
                action_keywords = ["filter", "optional", "count", "order", "exists", "union", "type"]
                for kw in action_keywords:
                    if kw in sparql_lower and kw in action_desc:
                        boost += 0.05
                        break

            # Apply boost
            rule["score"] = min(rule.get("score", 0.0) + boost, 1.0)

        # Re-sort
        rules.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return rules


# ------------------------------------------------------------------
# Standalone demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import tempfile
    from .knowledge_base import ExperienceKB

    print("=== RuleRetriever Demo ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        kb = ExperienceKB(tmpdir)
        retriever = RuleRetriever(kb)

        if kb.embed_model is None:
            print("No embedding model. Install sentence-transformers.")
            exit(0)

        # Add sample rules
        rules = [
            {
                "title": "Empty SPARQL Result Recovery",
                "description": "When SPARQL returns 0 results, entity may be incorrectly linked.",
                "rule_type": "ERROR_RECOVERY",
                "state_description": "SPARQL query with specific entity returns empty results",
                "state_keywords": ["empty", "zero", "entity", "sparql"],
                "action": {
                    "description": "Verify entity type, try broader search",
                    "steps": [
                        "Check entity type with SearchTypes",
                        "Try broader entity search",
                        "Relax constraint on problematic triple",
                    ],
                },
                "success_count": 5,
                "fail_count": 1,
            },
            {
                "title": "Birth Date Direct Lookup",
                "description": "For birth date questions, direct property lookup suffices.",
                "rule_type": "SUCCESS_SHORTCUT",
                "state_description": "Question asks for birth date of a person",
                "state_keywords": ["birth", "date", "born", "person"],
                "action": {
                    "description": "Use direct property path",
                    "steps": ["Link to person entity", "Use date_of_birth property directly"],
                },
                "success_count": 10,
                "fail_count": 0,
            },
            {
                "title": "COUNT Aggregation Guide",
                "description": "For 'how many' questions, use COUNT with GROUP BY.",
                "rule_type": "CONSTRAINT_GUIDE",
                "state_description": "Question asks 'how many' requiring aggregation",
                "state_keywords": ["how_many", "count", "total", "number"],
                "action": {
                    "description": "Use COUNT() aggregate with GROUP BY",
                    "steps": [
                        "Wrap result variable in COUNT()",
                        "Add GROUP BY for aggregation",
                        "Remove LIMIT if present",
                    ],
                },
                "success_count": 8,
                "fail_count": 2,
            },
        ]
        kb.add_rules_batch(rules)

        # Demo 1: Empty result retrieval
        print("--- Retrieval: Empty SPARQL result ---")
        guidance = retriever.get_guidance_for_prompt(
            question="What college did Obama attend?",
            current_sparql="SELECT ?x WHERE { ns:m.02mjmr ns:people.person.education ?e }",
            last_error="Empty result set returned",
            entity_links=["Barack_Obama"],
        )
        print(guidance)

        # Demo 2: Birth date retrieval
        print("\n--- Retrieval: Birth date question ---")
        guidance = retriever.get_guidance_for_prompt(
            question="What year was Einstein born?",
            current_sparql="SELECT ?date WHERE { ns:m.01hp9 ns:people.person.date_of_birth ?date }",
        )
        print(guidance)

        # Demo 3: Compact format
        print("\n--- Compact guidance ---")
        guidance = retriever.get_guidance_for_prompt(
            question="How many states are in the US?",
            current_sparql="SELECT ?s WHERE { ?s ns:type.object.type ns:location.us_state }",
            last_error="Result too large, need COUNT",
            compact=True,
        )
        print(guidance)

        print("\n=== Demo Complete ===")
