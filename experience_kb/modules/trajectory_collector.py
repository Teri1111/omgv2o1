"""
Trajectory Collector for KBQA Experience Knowledge Base.

Parses step-level KBQA/SPARQL trajectories into structured error-correction episodes.
Supports JSON trajectory files, JSONL queue files, and batch directory loading.

Each trajectory step is converted into an episode capturing:
- State: question, linked entities, current SPARQL, execution result, error
- Action: type (revise/relink/relax/generate), reasoning, new SPARQL
- Outcome: success, failure, or partial

Episodes are classified as error recovery steps, successful shortcuts, or failed attempts.
"""

import json
import os
import uuid
import glob
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger("TrajectoryCollector")


# Episode classification constants
EPISODE_TYPES = {
    "ERROR_RECOVERY": "error_recovery",
    "SUCCESS_SHORTCUT": "success_shortcut",
    "FAILED_ATTEMPT": "failed_attempt",
    "PARTIAL_PROGRESS": "partial_progress",
    "INITIAL_GENERATION": "initial_generation"
}

# Action type mapping for normalization
ACTION_TYPE_MAP = {
    "revise": "revise",
    "relink": "relink",
    "relax": "relax",
    "generate": "generate",
    "entity_relink": "relink",
    "constraint_relax": "relax",
    "sparql_revise": "revise",
    "initial": "generate",
    "refine": "revise"
}


class TrajectoryCollector:
    """
    Collects and parses KBQA/SPARQL trajectories into structured episodes.
    
    Supports multiple input formats:
    - Single JSON trajectory files
    - JSONL trajectory queue files (one JSON object per line)
    - Batch directory loading for multiple trajectory files
    
    Each trajectory step is extracted as a standalone episode with
    error-correction pair classification.
    """
    
    def __init__(self, trajectory_id_prefix: str = "traj"):
        """
        Initialize the TrajectoryCollector.
        
        Args:
            trajectory_id_prefix: Prefix for generated trajectory IDs
        """
        self.trajectory_id_prefix = trajectory_id_prefix
        logger.info("[TrajectoryCollector] Initialized")
    
    def load_trajectory(self, file_path: str) -> Dict[str, Any]:
        """
        Load a single trajectory from a JSON file.
        
        Args:
            file_path: Path to the JSON trajectory file
            
        Returns:
            Dictionary containing the trajectory data
            
        Raises:
            FileNotFoundError: If file does not exist
            json.JSONDecodeError: If file contains invalid JSON
            ValueError: If trajectory format is invalid
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Trajectory file not found: {file_path}")
        
        logger.info(f"[TrajectoryCollector] Loading trajectory from: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                trajectory = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"[TrajectoryCollector] Invalid JSON in {file_path}: {e}")
            raise
        
        # Validate and normalize the trajectory
        trajectory = self._normalize_trajectory(trajectory, source_file=file_path)
        
        logger.info(f"[TrajectoryCollector] Loaded trajectory with {len(trajectory.get('steps', []))} steps")
        return trajectory
    
    def load_trajectory_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load trajectories from a JSONL file (one JSON object per line).
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            List of trajectory dictionaries
            
        Raises:
            FileNotFoundError: If file does not exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSONL file not found: {file_path}")
        
        logger.info(f"[TrajectoryCollector] Loading JSONL trajectories from: {file_path}")
        
        trajectories = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    trajectory = json.loads(line)
                    trajectory = self._normalize_trajectory(trajectory, source_file=file_path)
                    trajectories.append(trajectory)
                except json.JSONDecodeError as e:
                    logger.warning(f"[TrajectoryCollector] Skipping invalid JSON on line {line_num}: {e}")
                    continue
        
        logger.info(f"[TrajectoryCollector] Loaded {len(trajectories)} trajectories from JSONL")
        return trajectories
    
    def load_trajectory_dir(self, dir_path: str, pattern: str = "*.json") -> List[Dict[str, Any]]:
        """
        Load all trajectory files from a directory.
        
        Args:
            dir_path: Path to the directory containing trajectory files
            pattern: Glob pattern for matching files (default: "*.json")
            
        Returns:
            List of trajectory dictionaries
            
        Raises:
            FileNotFoundError: If directory does not exist
        """
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Trajectory directory not found: {dir_path}")
        
        logger.info(f"[TrajectoryCollector] Loading trajectories from directory: {dir_path}")
        
        file_pattern = os.path.join(dir_path, pattern)
        file_paths = sorted(glob.glob(file_pattern))
        
        if not file_paths:
            logger.warning(f"[TrajectoryCollector] No files matching '{pattern}' in {dir_path}")
            return []
        
        trajectories = []
        for file_path in file_paths:
            try:
                if file_path.endswith('.jsonl'):
                    trajectories.extend(self.load_trajectory_jsonl(file_path))
                else:
                    trajectories.append(self.load_trajectory(file_path))
            except Exception as e:
                logger.warning(f"[TrajectoryCollector] Failed to load {file_path}: {e}")
                continue
        
        logger.info(f"[TrajectoryCollector] Loaded {len(trajectories)} trajectories from directory")
        return trajectories
    
    def extract_episodes(self, trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract step-level episodes from a trajectory.
        
        Each step in the trajectory becomes a standalone episode with:
        - Unique episode_id
        - State snapshot (question, entities, SPARQL, error)
        - Action taken
        - Outcome classification
        
        Args:
            trajectory: A normalized trajectory dictionary
            
        Returns:
            List of episode dictionaries
        """
        traj_id = trajectory.get("trajectory_id", f"{self.trajectory_id_prefix}_{uuid.uuid4().hex[:8]}")
        question = trajectory.get("question", "")
        steps = trajectory.get("steps", [])
        gold_sparql = trajectory.get("gold_sparql", "")
        gold_answer = trajectory.get("gold_answer", "")
        
        if not steps:
            logger.warning(f"[TrajectoryCollector] Trajectory {traj_id} has no steps")
            return []
        
        episodes = []
        previous_outcome = None
        
        for idx, step in enumerate(steps):
            episode = self._step_to_episode(
                step=step,
                trajectory=trajectory,
                traj_id=traj_id,
                step_index=idx,
                question=question,
                gold_sparql=gold_sparql,
                gold_answer=gold_answer,
                previous_outcome=previous_outcome
            )
            
            if episode:
                episodes.append(episode)
                previous_outcome = episode.get("outcome")
        
        logger.info(f"[TrajectoryCollector] Extracted {len(episodes)} episodes from trajectory {traj_id}")
        return episodes
    
    def extract_error_correction_pairs(
        self, 
        trajectory: Dict[str, Any]
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any], str]]:
        """
        Extract error-correction pairs from a trajectory.
        
        An error-correction pair consists of:
        - Error state: The state that caused the error
        - Correction action: The action that attempted to fix it
        - Result: Whether the correction succeeded
        
        Args:
            trajectory: A normalized trajectory dictionary
            
        Returns:
            List of (error_state, correction_action, outcome) tuples
        """
        episodes = self.extract_episodes(trajectory)
        pairs = []
        
        for i, episode in enumerate(episodes):
            state = episode.get("state", {})
            error_type = state.get("error_type")
            
            # Only extract pairs from episodes with errors or recovery actions
            if error_type and error_type != "none":
                action = episode.get("action", {})
                outcome = episode.get("outcome", "failure")
                
                error_state = {
                    "question": state.get("question", ""),
                    "linked_entities": state.get("linked_entities", []),
                    "current_sparql": state.get("current_sparql", ""),
                    "sparql_result_count": state.get("sparql_result_count", 0),
                    "error_type": error_type,
                    "error_message": state.get("error_message", "")
                }
                
                correction_action = {
                    "type": action.get("type", "revise"),
                    "reasoning": action.get("reasoning", ""),
                    "new_sparql": action.get("new_sparql", "")
                }
                
                pairs.append((error_state, correction_action, outcome))
        
        logger.info(f"[TrajectoryCollector] Extracted {len(pairs)} error-correction pairs")
        return pairs
    
    def collect_batch(self, input_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Collect episodes from multiple input sources.
        
        Args:
            input_paths: List of file or directory paths
            
        Returns:
            List of all episodes from all sources
        """
        all_episodes = []
        
        for path in input_paths:
            try:
                if os.path.isdir(path):
                    trajectories = self.load_trajectory_dir(path)
                elif path.endswith('.jsonl'):
                    trajectories = self.load_trajectory_jsonl(path)
                elif path.endswith('.json'):
                    trajectories = [self.load_trajectory(path)]
                else:
                    logger.warning(f"[TrajectoryCollector] Unsupported file type: {path}")
                    continue
                
                for traj in trajectories:
                    episodes = self.extract_episodes(traj)
                    all_episodes.extend(episodes)
                    
            except Exception as e:
                logger.error(f"[TrajectoryCollector] Error processing {path}: {e}")
                continue
        
        logger.info(f"[TrajectoryCollector] Collected {len(all_episodes)} total episodes from batch")
        return all_episodes
    
    def classify_episode(self, episode: Dict[str, Any]) -> str:
        """
        Classify an episode into a rule type category.
        
        Args:
            episode: An episode dictionary
            
        Returns:
            Rule type string (ERROR_RECOVERY, SUCCESS_SHORTCUT, etc.)
        """
        state = episode.get("state", {})
        action = episode.get("action", {})
        outcome = episode.get("outcome", "failure")
        is_recovery = episode.get("is_recovery", False)
        error_type = state.get("error_type")
        
        # Classification logic
        if is_recovery and outcome == "success":
            return EPISODE_TYPES["ERROR_RECOVERY"]
        elif outcome == "success" and not error_type:
            return EPISODE_TYPES["SUCCESS_SHORTCUT"]
        elif error_type == "type_mismatch":
            return EPISODE_TYPES["ERROR_RECOVERY"]
        elif outcome == "failure":
            return EPISODE_TYPES["FAILED_ATTEMPT"]
        elif outcome == "partial":
            return EPISODE_TYPES["PARTIAL_PROGRESS"]
        elif action.get("type") == "generate":
            return EPISODE_TYPES["INITIAL_GENERATION"]
        else:
            return EPISODE_TYPES["ERROR_RECOVERY"]
    
    def export_episodes(self, episodes: List[Dict[str, Any]], output_path: str) -> str:
        """
        Export episodes to a JSON file.
        
        Args:
            episodes: List of episode dictionaries
            output_path: Path to write the JSON output
            
        Returns:
            The output file path
        """
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        export_data = {
            "exported_at": datetime.utcnow().isoformat(),
            "episode_count": len(episodes),
            "episodes": episodes
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[TrajectoryCollector] Exported {len(episodes)} episodes to {output_path}")
        return output_path
    
    # ---- Private helper methods ----
    
    def _normalize_trajectory(
        self, 
        trajectory: Dict[str, Any], 
        source_file: str = ""
    ) -> Dict[str, Any]:
        """
        Normalize a trajectory to the expected format.
        
        Handles variations in input format and ensures all required fields exist.
        """
        # Ensure trajectory_id
        if "trajectory_id" not in trajectory:
            trajectory["trajectory_id"] = f"{self.trajectory_id_prefix}_{uuid.uuid4().hex[:8]}"
        
        # Ensure steps list
        if "steps" not in trajectory:
            trajectory["steps"] = []
        
        # Normalize each step
        for idx, step in enumerate(trajectory["steps"]):
            trajectory["steps"][idx] = self._normalize_step(step, idx)
        
        # Add metadata
        trajectory["_source_file"] = source_file
        trajectory["_loaded_at"] = datetime.utcnow().isoformat()
        
        return trajectory
    
    def _normalize_step(self, step: Dict[str, Any], step_index: int) -> Dict[str, Any]:
        """
        Normalize a step to the expected format.
        """
        # Ensure step_id
        if "step_id" not in step:
            step["step_id"] = step_index
        
        # Normalize state
        if "state" not in step:
            step["state"] = {}
        state = step["state"]
        
        # Handle alternative field names
        if "current_sparql_partial" in state and "current_sparql" not in state:
            state["current_sparql"] = state["current_sparql_partial"]
        
        if "sparql_result" in state and "execution_result" not in state:
            state["execution_result"] = state["sparql_result"]
        
        if "linked_entities" not in state:
            state["linked_entities"] = []
        
        # Normalize execution result
        exec_result = state.get("execution_result", {})
        if isinstance(exec_result, dict) and "count" in exec_result:
            state["sparql_result_count"] = exec_result.get("count", 0)
        else:
            state["sparql_result_count"] = state.get("sparql_result_count", 0)
        
        # Normalize action
        if "action" not in step:
            step["action"] = {}
        action = step["action"]
        
        # Map action type to normalized form
        raw_type = action.get("type", "generate")
        action["type"] = ACTION_TYPE_MAP.get(raw_type.lower(), raw_type)
        
        # Handle alternative field names
        if "content" in action and "reasoning" not in action:
            action["reasoning"] = action["content"]
        
        # Ensure outcome
        if "outcome" not in step:
            step["outcome"] = "failure"
        
        return step
    
    def _step_to_episode(
        self,
        step: Dict[str, Any],
        trajectory: Dict[str, Any],
        traj_id: str,
        step_index: int,
        question: str,
        gold_sparql: str,
        gold_answer: str,
        previous_outcome: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Convert a trajectory step into a structured episode.
        """
        try:
            state = step.get("state", {})
            action = step.get("action", {})
            outcome = step.get("outcome", "failure")
            
            # Determine error type from state
            error_type = self._classify_error_type(state)
            
            # Determine if this is a recovery step
            is_recovery = (
                previous_outcome == "failure" and outcome == "success"
            ) or (
                error_type is not None and action.get("type") in ["revise", "relink", "relax"]
            )
            
            episode = {
                "episode_id": str(uuid.uuid4()),
                "source_trajectory": traj_id,
                "step_index": step_index,
                "state": {
                    "question": state.get("question", question),
                    "linked_entities": state.get("linked_entities", []),
                    "current_sparql": state.get("current_sparql", ""),
                    "sparql_result_count": state.get("sparql_result_count", 0),
                    "error_type": error_type,
                    "error_message": state.get("error", state.get("error_message", ""))
                },
                "action": {
                    "type": action.get("type", "generate"),
                    "reasoning": action.get("reasoning", ""),
                    "new_sparql": action.get("new_sparql", "")
                },
                "outcome": outcome,
                "is_recovery": is_recovery,
                "gold_sparql": gold_sparql,
                "gold_answer": gold_answer,
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Classify the episode
            episode["episode_type"] = self.classify_episode(episode)
            
            return episode
            
        except Exception as e:
            logger.error(f"[TrajectoryCollector] Error converting step {step_index} to episode: {e}")
            return None
    
    def _classify_error_type(self, state: Dict[str, Any]) -> Optional[str]:
        """
        Classify the error type from a state dictionary.
        
        Returns:
            Error type string or None if no error detected
        """
        error_msg = (state.get("error") or state.get("error_message") or "").lower()
        exec_result = state.get("execution_result", {})
        result_count = state.get("sparql_result_count", 0)
        
        if not error_msg and result_count > 0:
            return None
        
        # Classify based on error message patterns
        if "empty" in error_msg or "no result" in error_msg or result_count == 0:
            return "empty_result"
        elif "type" in error_msg and "mismatch" in error_msg:
            return "type_mismatch"
        elif "syntax" in error_msg or "parse" in error_msg:
            return "syntax_error"
        elif "timeout" in error_msg:
            return "timeout"
        elif "constraint" in error_msg:
            return "constraint_error"
        elif "entity" in error_msg and ("not found" in error_msg or "link" in error_msg):
            return "entity_linking_error"
        elif error_msg:
            return "execution_error"
        elif result_count == 0:
            return "empty_result"
        
        return None


# ---- Standalone demo ----

if __name__ == "__main__":
    # Create a sample trajectory for demonstration
    sample_trajectory = {
        "trajectory_id": "demo_traj_001",
        "question": "What college did Obama attend?",
        "gold_sparql": "SELECT ?x WHERE { ns:Barack_Obama ns:education ?e . ?e ns:institution ?x }",
        "gold_answer": "Columbia University",
        "steps": [
            {
                "step_id": 0,
                "state": {
                    "question": "What college did Obama attend?",
                    "linked_entities": ["Barack_Obama"],
                    "current_sparql": "SELECT ?x WHERE { ns:m.02mjmr ns:education ?e . ?e ns:institution ?x }",
                    "execution_result": {"count": 0},
                    "error": "No results found"
                },
                "action": {
                    "type": "revise",
                    "reasoning": "The entity MID might be wrong, try using the named entity directly",
                    "new_sparql": "SELECT ?x WHERE { ns:Barack_Obama ns:education ?e . ?e ns:institution ?x }"
                },
                "outcome": "failure"
            },
            {
                "step_id": 1,
                "state": {
                    "question": "What college did Obama attend?",
                    "linked_entities": ["Barack_Obama"],
                    "current_sparql": "SELECT ?x WHERE { ns:Barack_Obama ns:education ?e . ?e ns:institution ?x }",
                    "execution_result": {"count": 0},
                    "error": "Empty result - predicate may not exist"
                },
                "action": {
                    "type": "relax",
                    "reasoning": "Try a broader predicate - use ns:education_institution instead of nested path",
                    "new_sparql": "SELECT ?x WHERE { ns:Barack_Obama ns:education_institution ?x }"
                },
                "outcome": "success"
            }
        ],
        "final_correct": True,
        "total_steps": 2,
        "recovery_steps": 1
    }
    
    # Initialize collector
    collector = TrajectoryCollector()
    
    # Extract episodes
    print("=" * 60)
    print("[Demo] Extracting episodes from sample trajectory...")
    episodes = collector.extract_episodes(sample_trajectory)
    
    print(f"\n[Demo] Extracted {len(episodes)} episodes:\n")
    for ep in episodes:
        print(f"  Episode ID: {ep['episode_id']}")
        print(f"  Step Index: {ep['step_index']}")
        print(f"  Episode Type: {ep['episode_type']}")
        print(f"  Error Type: {ep['state']['error_type']}")
        print(f"  Action Type: {ep['action']['type']}")
        print(f"  Outcome: {ep['outcome']}")
        print(f"  Is Recovery: {ep['is_recovery']}")
        print(f"  SPARQL Result Count: {ep['state']['sparql_result_count']}")
        print()
    
    # Extract error-correction pairs
    print("=" * 60)
    print("[Demo] Extracting error-correction pairs...")
    pairs = collector.extract_error_correction_pairs(sample_trajectory)
    
    print(f"\n[Demo] Extracted {len(pairs)} error-correction pairs:\n")
    for error_state, correction_action, outcome in pairs:
        print(f"  Error Type: {error_state['error_type']}")
        print(f"  Error Message: {error_state['error_message'][:50]}...")
        print(f"  Correction: {correction_action['type']} -> {correction_action['reasoning'][:50]}...")
        print(f"  Outcome: {outcome}")
        print()
    
    # Export demo
    print("=" * 60)
    output_path = "/tmp/demo_episodes.json"
    collector.export_episodes(episodes, output_path)
    print(f"[Demo] Episodes exported to {output_path}")
