"""
KBQA Experience Knowledge Base - Pipeline Integration.

Provides adapters to integrate the experience KB into existing KBQA pipelines.
Supports:
    1. Standalone SPARQL generation with experience-guided correction loop
    2. Drop-in adapter for existing LLM-based SPARQL generators
    3. Trajectory logging for continuous learning

Adapted from vkbqa/modules/iterative_reasoner.py run_loop pattern.
"""

import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Callable

from .knowledge_base import ExperienceKB
from .rule_retriever import RuleRetriever
from .experience_extractor import ExperienceExtractor, LLMClient


class SPARQLExecutor:
    """
    Abstract SPARQL executor interface.
    Users should subclass this and implement execute().
    """

    def execute(self, sparql: str) -> Dict:
        """
        Execute a SPARQL query and return results.
        
        Returns:
            {
                "success": bool,
                "result_count": int,
                "results": list,
                "error": str or None
            }
        """
        raise NotImplementedError("Subclass must implement execute()")


class ExperienceGuidedPipeline:
    """
    SPARQL generation pipeline with experience-guided error correction loop.
    
    Flow:
        1. Generate initial SPARQL from question + entities
        2. Execute SPARQL
        3. If error/empty: retrieve relevant correction rules from KB
        4. Inject rules into LLM prompt, regenerate SPARQL
        5. Repeat until success or max steps
        6. Log trajectory for continuous learning
    """

    def __init__(
        self,
        sparql_generator_fn: Callable,
        executor: SPARQLExecutor,
        experience_kb: ExperienceKB,
        llm_client: Optional[LLMClient] = None,
        max_steps: int = 5,
        experience_top_k: int = 3,
        compact_guidance: bool = False,
    ):
        """
        Args:
            sparql_generator_fn: Function(question, entities, current_sparql, 
                                          error, guidance, history) -> dict
                Must return {"sparql": str, "reasoning": str}
            executor: SPARQLExecutor instance
            experience_kb: ExperienceKB instance
            llm_client: Optional LLMClient for trajectory extraction
            max_steps: Max correction steps
            experience_top_k: Number of rules to retrieve per step
            compact_guidance: Use compact rule format
        """
        self.generate = sparql_generator_fn
        self.executor = executor
        self.kb = experience_kb
        self.retriever = RuleRetriever(experience_kb)
        self.llm_client = llm_client
        self.max_steps = max_steps
        self.experience_top_k = experience_top_k
        self.compact_guidance = compact_guidance

        # Trajectory logging
        self._last_trajectory = None

    def run(
        self,
        question: str,
        entities: List[str],
        gold_sparql: Optional[str] = None,
        gold_answer: Optional[str] = None,
    ) -> Dict:
        """
        Run the experience-guided SPARQL generation pipeline.
        
        Args:
            question: Natural language question
            entities: Linked entities
            gold_sparql: Ground truth SPARQL (for logging, not used in generation)
            gold_answer: Ground truth answer (for evaluation)
            
        Returns:
            {
                "final_sparql": str,
                "final_result": dict,
                "steps": list,
                "trajectory_id": str,
                "is_correct": bool or None
            }
        """
        print(f"\n[Pipeline] Starting for: {question[:80]}...")
        print(f"[Pipeline] Entities: {entities}")

        trajectory_id = f"traj_{uuid.uuid4().hex[:8]}"
        steps = []
        current_sparql = ""
        last_error = None

        for step_idx in range(self.max_steps):
            print(f"\n[Pipeline] Step {step_idx + 1}/{self.max_steps}")

            # Retrieve experience guidance
            guidance = ""
            retrieved_rule_ids = []
            if step_idx > 0:  # Skip guidance on first step
                guidance = self.retriever.get_guidance_for_prompt(
                    question=question,
                    current_sparql=current_sparql,
                    last_error=last_error,
                    entity_links=entities,
                    step_history=steps,
                    top_k=self.experience_top_k,
                    compact=self.compact_guidance,
                )
                if guidance:
                    print(f"[Pipeline] Retrieved guidance:\n{guidance[:500]}...")
                    # Track which rules were used
                    retrieved_rules = self.retriever.retrieve_for_state(
                        question, current_sparql, last_error, entities, steps, self.experience_top_k
                    )
                    retrieved_rule_ids = [r.get("rule_id") for r in retrieved_rules if r.get("rule_id")]

            # Generate SPARQL
            gen_result = self.generate(
                question=question,
                entities=entities,
                current_sparql=current_sparql,
                error=last_error,
                guidance=guidance,
                history=steps,
            )
            new_sparql = gen_result.get("sparql", "")
            reasoning = gen_result.get("reasoning", "")

            if not new_sparql:
                print("[Pipeline] Generator returned empty SPARQL. Stopping.")
                break

            print(f"[Pipeline] Generated SPARQL: {new_sparql[:200]}...")
            print(f"[Pipeline] Reasoning: {reasoning[:200]}...")

            # Execute SPARQL
            exec_result = self.executor.execute(new_sparql)
            success = exec_result.get("success", False)
            result_count = exec_result.get("result_count", 0)
            error = exec_result.get("error")

            print(f"[Pipeline] Execution: success={success}, count={result_count}, error={error}")

            # Classify outcome
            if success and result_count > 0:
                outcome = "success"
            elif error:
                outcome = "failure"
            else:
                outcome = "partial"  # Empty but no error

            # Log step
            step_record = {
                "step_id": step_idx,
                "state": {
                    "question": question,
                    "linked_entities": entities,
                    "current_sparql": current_sparql,
                    "new_sparql": new_sparql,
                    "sparql_result_count": result_count,
                    "error_type": self._classify_error(error, result_count, exec_result),
                    "error_message": error,
                },
                "action": {
                    "type": "revise" if step_idx > 0 else "generate",
                    "reasoning": reasoning,
                    "new_sparql": new_sparql,
                    "guidance_used": bool(guidance),
                    "retrieved_rule_ids": retrieved_rule_ids,
                },
                "outcome": outcome,
                "is_recovery": outcome == "success" and len(steps) > 0 and
                              any(s.get("outcome") == "failure" for s in steps),
            }
            steps.append(step_record)

            # Update current state
            current_sparql = new_sparql

            if outcome == "success":
                print(f"[Pipeline] Success at step {step_idx + 1}!")
                # Update rule stats (positive)
                for rid in retrieved_rule_ids:
                    self.kb.update_stats(rid, success=True)
                break
            elif outcome == "failure":
                last_error = error or f"Empty result set (count={result_count})"
                # Update rule stats (negative if rules were used)
                if retrieved_rule_ids:
                    for rid in retrieved_rule_ids:
                        self.kb.update_stats(rid, success=False)
            else:
                last_error = f"Empty result (count={result_count})"
        else:
            print(f"[Pipeline] Max steps ({self.max_steps}) reached.")

        # Evaluate correctness
        is_correct = None
        if gold_answer is not None:
            final_result = steps[-1]["state"]["new_sparql"] if steps else ""
            is_correct = self._check_correctness(final_result, gold_answer, steps)

        # Build trajectory record
        trajectory = {
            "trajectory_id": trajectory_id,
            "question": question,
            "entities": entities,
            "gold_sparql": gold_sparql,
            "gold_answer": gold_answer,
            "steps": steps,
            "final_sparql": current_sparql,
            "is_correct": is_correct,
            "total_steps": len(steps),
            "recovery_steps": sum(1 for s in steps if s.get("is_recovery")),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._last_trajectory = trajectory

        return {
            "final_sparql": current_sparql,
            "final_result": steps[-1] if steps else None,
            "steps": steps,
            "trajectory_id": trajectory_id,
            "is_correct": is_correct,
        }

    def get_last_trajectory(self) -> Optional[Dict]:
        """Get the trajectory from the last run."""
        return self._last_trajectory

    def save_trajectory(self, output_dir: str):
        """Save last trajectory to disk for future learning."""
        if self._last_trajectory is None:
            print("[Pipeline] No trajectory to save.")
            return
        import os
        os.makedirs(output_dir, exist_ok=True)
        tid = self._last_trajectory["trajectory_id"]
        path = os.path.join(output_dir, f"{tid}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._last_trajectory, f, ensure_ascii=False, indent=2)
        print(f"[Pipeline] Trajectory saved to {path}")

    def extract_and_store_rules(self):
        """
        Extract error-correction rules from the last trajectory
        and add them to the KB. Requires LLM client.
        """
        if self._last_trajectory is None:
            print("[Pipeline] No trajectory to extract from.")
            return []
        if self.llm_client is None:
            print("[Pipeline] No LLM client. Cannot extract rules.")
            return []

        extractor = ExperienceExtractor(
            llm_base_url=self.llm_client.base_url,
            llm_api_key=self.llm_client.api_key,
            llm_model_name=self.llm_client.model_name,
        )
        rules = extractor.extract_from_trajectory(self._last_trajectory)
        if rules:
            rule_ids = self.kb.add_rules_batch(rules)
            self.kb.save()
            print(f"[Pipeline] Extracted and stored {len(rule_ids)} rules from trajectory.")
            return rule_ids
        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_error(self, error: Optional[str], result_count: int, exec_result: Dict) -> str:
        """Classify the type of error from SPARQL execution."""
        if error:
            error_lower = error.lower()
            if "syntax" in error_lower or "parse" in error_lower:
                return "syntax_error"
            if "type" in error_lower and "mismatch" in error_lower:
                return "type_mismatch"
            if "timeout" in error_lower:
                return "timeout"
            if "empty" in error_lower or "no result" in error_lower:
                return "empty_result"
            return "execution_error"
        if result_count == 0:
            return "empty_result"
        return "none"

    def _check_correctness(self, final_sparql: str, gold_answer: str, steps: List[Dict]) -> Optional[bool]:
        """Check if the final result matches gold answer."""
        if not steps:
            return None
        last_result = steps[-1].get("state", {})
        results = last_result.get("results", [])
        if not results:
            return False
        # Simple string match (customize per dataset)
        result_strs = [str(r) for r in results]
        return any(gold_answer.lower() in r.lower() for r in result_strs)


class KBQAPipelineAdapter:
    """
    Drop-in adapter that wraps an existing KBQA pipeline with experience guidance.
    
    Usage:
        existing_pipeline = YourExistingKBPipeline(...)
        adapter = KBQAPipelineAdapter(existing_pipeline, "/path/to/experience_kb")
        result = adapter.answer(question, entities)
    """

    def __init__(self, base_pipeline, experience_kb_path: str, **kb_kwargs):
        """
        Args:
            base_pipeline: Existing pipeline with .generate_sparql(question, entities) method
            experience_kb_path: Path to experience KB directory
            **kb_kwargs: Additional kwargs for ExperienceKB
        """
        self.base = base_pipeline
        self.kb = ExperienceKB(experience_kb_path, **kb_kwargs)
        self.retriever = RuleRetriever(self.kb)

    def answer(self, question: str, entities: List[str], **kwargs) -> Dict:
        """
        Answer a question with experience-guided correction.
        
        Intercepts the base pipeline's generation loop to inject guidance.
        """
        history = []
        current_sparql = ""
        last_error = None
        max_steps = getattr(self.base, "max_steps", 5)

        for step in range(max_steps):
            # Retrieve guidance
            guidance = ""
            if step > 0 and last_error:
                guidance = self.retriever.get_guidance_for_prompt(
                    question, current_sparql, last_error, entities, history
                )

            # Generate with base pipeline
            result = self.base.generate_sparql(
                question=question,
                entities=entities,
                current_sparql=current_sparql,
                error=last_error,
                guidance=guidance,
                **kwargs,
            )

            new_sparql = result.get("sparql", "")
            exec_result = result.get("execution", {})
            error = exec_result.get("error")
            count = exec_result.get("result_count", 0)

            history.append({
                "step": step,
                "sparql": new_sparql,
                "error": error,
                "count": count,
            })

            current_sparql = new_sparql

            if not error and count > 0:
                return {"sparql": current_sparql, "result": exec_result, "steps": history}
            last_error = error or f"Empty result (count={count})"

        return {"sparql": current_sparql, "result": None, "steps": history, "max_steps_reached": True}


# ------------------------------------------------------------------
# Standalone demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("=== Pipeline Integration Demo ===\n")

    # Mock SPARQL executor
    class MockExecutor(SPARQLExecutor):
        def __init__(self):
            self.call_count = 0

        def execute(self, sparql: str) -> Dict:
            self.call_count += 1
            # Simulate: first call fails, second succeeds with guidance
            if self.call_count == 1:
                return {"success": False, "result_count": 0, "results": [], "error": "Empty result set"}
            else:
                return {"success": True, "result_count": 1, "results": ["Harvard_University"], "error": None}

    # Mock SPARQL generator
    def mock_generator(question, entities, current_sparql, error, guidance, history):
        if not current_sparql:
            return {
                "sparql": "SELECT ?x WHERE { ns:m.02mjmr ns:people.person.education ?e }",
                "reasoning": "Initial query for education",
            }
        else:
            # With guidance, generate improved query
            return {
                "sparql": "SELECT ?x WHERE { ns:m.02mjmr ns:people.person.education ?e . ?e ns:education.education.institution ?x }",
                "reasoning": "Added institution join based on guidance",
            }

    with tempfile.TemporaryDirectory() as tmpdir:
        kb = ExperienceKB(tmpdir)
        if kb.embed_model is None:
            print("No embedding model. Install sentence-transformers.")
            exit(0)

        # Add a relevant rule
        kb.add_rule({
            "title": "Empty Education Query Recovery",
            "description": "Education queries need institution join",
            "rule_type": "ERROR_RECOVERY",
            "state_description": "Education query returns empty, need institution join",
            "state_keywords": ["education", "empty", "institution", "join"],
            "action": {
                "description": "Add institution join to education query",
                "steps": [
                    "Keep the education triple pattern",
                    "Add join: ?e ns:education.education.institution ?x",
                    "Select the institution variable",
                ],
            },
        })

        executor = MockExecutor()
        pipeline = ExperienceGuidedPipeline(
            sparql_generator_fn=mock_generator,
            executor=executor,
            experience_kb=kb,
            max_steps=3,
        )

        result = pipeline.run(
            question="What college did Obama attend?",
            entities=["Barack_Obama"],
            gold_answer="Harvard_University",
        )

        print(f"\nFinal SPARQL: {result['final_sparql']}")
        print(f"Steps taken: {len(result['steps'])}")
        print(f"Is correct: {result['is_correct']}")

        # Save trajectory
        pipeline.save_trajectory(tmpdir)
        print(f"\nTrajectory saved.")

        print("\n=== Demo Complete ===")
