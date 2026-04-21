"""
Experience Extractor for KBQA Experience Knowledge Base.

Uses LLM to analyze step-level episodes and extract structured error-correction rules.
Adapted from the vkbqa memory_worker.py extraction pattern.

Each episode is analyzed by the LLM to produce a generalized, reusable rule that captures:
- The error pattern or successful strategy
- The corrective action taken
- Classification into rule types (ERROR_RECOVERY, SUCCESS_SHORTCUT, etc.)
"""

import json
import os
import uuid
import time
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

try:
    import requests
except ImportError:
    requests = None
    logging.warning("[ExperienceExtractor] requests library not found. Install with: pip install requests")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger("ExperienceExtractor")


# Rule type taxonomy
RULE_TYPES = {
    "ERROR_RECOVERY": "ERROR_RECOVERY",
    "SUCCESS_SHORTCUT": "SUCCESS_SHORTCUT",
    "CONSTRAINT_GUIDE": "CONSTRAINT_GUIDE",
    "TYPE_MISMATCH": "TYPE_MISMATCH",
    "LOGICAL_STRUCTURE": "LOGICAL_STRUCTURE"
}

# Default extraction system prompt
DEFAULT_SYSTEM_PROMPT = """You are an expert KBQA (Knowledge Base Question Answering) error-correction analyst. 
Your task is to extract reusable, generalized error-correction rules from SPARQL generation trajectories.

Given a step-level episode from a KBQA trajectory, you must:
1. Analyze the error state and corrective action
2. Generate a GENERALIZED rule (not specific to this particular question)
3. Classify the rule type appropriately
4. Structure the output as JSON

Important guidelines:
- Rules must be reusable across different questions and domains
- Focus on the pattern, not the specific entities
- Include actionable steps in the action
- Provide a SPARQL hint/pattern when applicable
- Assign confidence based on how clear and reusable the rule is

Output ONLY valid JSON with these fields:
{
    "title": "short descriptive title (max 60 chars)",
    "description": "1-3 sentences explaining the error pattern and correction",
    "rule_type": "ERROR_RECOVERY|SUCCESS_SHORTCUT|CONSTRAINT_GUIDE|TYPE_MISMATCH|LOGICAL_STRUCTURE",
    "state_description": "description of when this rule applies (generalized)",
    "state_keywords": ["keyword1", "keyword2", "keyword3"],
    "action": {
        "description": "what corrective action to take",
        "steps": ["step 1", "step 2", "step 3"],
        "sparql_hint": "optional SPARQL pattern or template"
    },
    "embedding_text": "combined text for vector search embedding",
    "confidence": 0.0-1.0
}"""

# Few-shot examples for KBQA-specific patterns
FEW_SHOT_EXAMPLES = """
Example 1 - Empty Result Recovery:
Episode:
  Question: "What university did Einstein attend?"
  Error: SPARQL returns 0 results, entity linked as "Albert_Einstein_(physicist)"
  Action: Re-link entity to broader "Albert_Einstein" entity type
  Outcome: success

Rule:
{
    "title": "Verify entity type specificity for empty results",
    "description": "When a SPARQL query returns empty results with a specific entity type, try re-linking to a broader entity type. Entity linking may have selected a too-specific variant.",
    "rule_type": "ERROR_RECOVERY",
    "state_description": "SPARQL query with specific entity type returns empty results",
    "state_keywords": ["empty", "zero", "entity", "type", "specificity"],
    "action": {
        "description": "Re-link entity to broader type variant",
        "steps": ["Check current entity type specificity", "Search for broader entity variants", "Re-execute SPARQL with broader entity"],
        "sparql_hint": null
    },
    "embedding_text": "empty result entity type too specific: try broader entity type variant, re-link entity",
    "confidence": 0.9
}

Example 2 - Predicate Path Simplification:
Episode:
  Question: "Who is the president of France?"
  Error: Nested predicate path returns empty (ns:president -> ns:person -> ns:office)
  Action: Use direct predicate ns:president_of instead of nested path
  Outcome: success

Rule:
{
    "title": "Use direct predicates instead of nested paths",
    "description": "When a multi-hop predicate path returns empty, try using a direct predicate that captures the same relationship. Knowledge bases often have both granular and direct predicates.",
    "rule_type": "SUCCESS_SHORTCUT",
    "state_description": "Multi-hop predicate traversal returns empty or fails",
    "state_keywords": ["nested", "path", "predicate", "empty", "direct"],
    "action": {
        "description": "Replace nested predicate path with direct predicate",
        "steps": ["Identify the multi-hop path pattern", "Search for direct predicates covering the relationship", "Substitute the direct predicate in the SPARQL query"],
        "sparql_hint": "Replace {ns:A ns:p1 ?x . ?x ns:p2 ?y} with {ns:A ns:direct_p ?y}"
    },
    "embedding_text": "nested predicate path empty result: use direct predicate for relationship, simplify SPARQL path",
    "confidence": 0.85
}

Example 3 - Type Constraint Mismatch:
Episode:
  Question: "What books has Stephen King written?"
  Error: Predicate expects Person type but entity is linked as CreativeWork
  Action: Use reverse property path (book -> author -> person) 
  Outcome: success

Rule:
{
    "title": "Reverse property path for type mismatch",
    "description": "When a type mismatch occurs (expected type vs entity type), use a reverse property path to traverse from the correct type direction.",
    "rule_type": "TYPE_MISMATCH",
    "state_description": "Entity type does not match expected predicate type",
    "state_keywords": ["type", "mismatch", "reverse", "property", "direction"],
    "action": {
        "description": "Use reverse property path or inverse predicate",
        "steps": ["Identify the type mismatch (expected vs actual)", "Find inverse or reverse property", "Rewrite SPARQL with reversed path direction"],
        "sparql_hint": "Use ^predicate for reverse path: {ns:Book ^ns:author ns:Stephen_King}"
    },
    "embedding_text": "type mismatch predicate entity direction: use reverse property path, inverse predicate traversal",
    "confidence": 0.88
}
"""


class LLMClient:
    """
    LLM client for OpenAI-compatible APIs.
    Handles requests with retry logic and error handling.
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: int = 60
    ):
        """
        Initialize the LLM client.
        
        Args:
            base_url: Base URL for the LLM API (e.g., "https://api.openai.com/v1")
            api_key: API key for authentication
            model_name: Model name to use (e.g., "gpt-4", "qwen-72b")
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries in seconds (exponential backoff)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        
        # Construct the chat completions endpoint
        if '/chat/completions' not in self.base_url:
            self.endpoint = f"{self.base_url}/chat/completions"
        else:
            self.endpoint = self.base_url
        
        logger.info(f"[LLMClient] Initialized with model={model_name}, endpoint={self.endpoint}")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Send a chat completion request to the LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            response_format: Optional response format (e.g., {"type": "json_object"})
            
        Returns:
            The generated text response
            
        Raises:
            RuntimeError: If all retry attempts fail
        """
        if requests is None:
            raise RuntimeError("requests library is required. Install with: pip install requests")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if response_format:
            payload["response_format"] = response_format
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return content
                elif response.status_code == 429:
                    # Rate limited - wait longer
                    wait_time = self.retry_delay * (2 ** attempt) * 2
                    logger.warning(f"[LLMClient] Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                    logger.warning(f"[LLMClient] Attempt {attempt + 1} failed: {last_error}")
                    
            except requests.exceptions.Timeout:
                last_error = "Request timed out"
                logger.warning(f"[LLMClient] Attempt {attempt + 1} timed out")
            except requests.exceptions.ConnectionError:
                last_error = "Connection error"
                logger.warning(f"[LLMClient] Attempt {attempt + 1} connection error")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"[LLMClient] Attempt {attempt + 1} error: {e}")
            
            # Exponential backoff
            if attempt < self.max_retries - 1:
                wait_time = self.retry_delay * (2 ** attempt)
                time.sleep(wait_time)
        
        raise RuntimeError(f"LLM request failed after {self.max_retries} attempts: {last_error}")


class ExperienceExtractor:
    """
    Extracts structured error-correction rules from KBQA step-level episodes.
    
    Uses an LLM to analyze each episode and generate generalized, reusable rules
    that capture error patterns and their corrections.
    """
    
    def __init__(
        self,
        llm_base_url: str,
        llm_api_key: str,
        llm_model_name: str,
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
        temperature: float = 0.1
    ):
        """
        Initialize the ExperienceExtractor.
        
        Args:
            llm_base_url: Base URL for the LLM API
            llm_api_key: API key for authentication
            llm_model_name: Model name to use for extraction
            system_prompt: Optional custom system prompt (uses default if None)
            max_retries: Maximum retry attempts for LLM calls
            temperature: LLM sampling temperature
        """
        self.llm_client = LLMClient(
            base_url=llm_base_url,
            api_key=llm_api_key,
            model_name=llm_model_name,
            max_retries=max_retries
        )
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.temperature = temperature
        logger.info("[ExperienceExtractor] Initialized")
    
    def extract_rule(self, episode: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract a single rule from one episode.
        
        Args:
            episode: An episode dictionary from TrajectoryCollector
            
        Returns:
            A rule dictionary or None if extraction failed
        """
        logger.info(f"[ExperienceExtractor] Extracting rule from episode {episode.get('episode_id', 'unknown')}")
        
        try:
            # Build the extraction prompt
            prompt = self._build_extraction_prompt(episode)
            
            # Call the LLM
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response_text = self.llm_client.chat_completion(
                messages=messages,
                temperature=self.temperature,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            rule = self._parse_rule_response(response_text)
            
            if rule and self._validate_rule(rule):
                # Enrich the rule with metadata
                rule["rule_id"] = str(uuid.uuid4())
                rule["source_episode"] = episode.get("episode_id", "unknown")
                rule["source_trajectory"] = episode.get("source_trajectory", "unknown")
                rule["created_at"] = datetime.utcnow().isoformat()
                rule["success_count"] = 0
                rule["fail_count"] = 0
                rule["consolidated"] = False
                
                # Generate embedding text if not provided
                if not rule.get("embedding_text"):
                    rule["embedding_text"] = self._generate_embedding_text(rule)
                
                logger.info(f"[ExperienceExtractor] Extracted rule: {rule.get('title', 'untitled')}")
                return rule
            else:
                logger.warning("[ExperienceExtractor] Invalid rule extracted, skipping")
                return None
                
        except Exception as e:
            logger.error(f"[ExperienceExtractor] Error extracting rule: {e}")
            return None
    
    def extract_rules_batch(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract rules from a batch of episodes.
        
        Args:
            episodes: List of episode dictionaries
            
        Returns:
            List of extracted rule dictionaries
        """
        logger.info(f"[ExperienceExtractor] Extracting rules from {len(episodes)} episodes")
        
        rules = []
        for episode in episodes:
            rule = self.extract_rule(episode)
            if rule:
                rules.append(rule)
        
        logger.info(f"[ExperienceExtractor] Extracted {len(rules)} rules from {len(episodes)} episodes")
        return rules
    
    def extract_from_trajectory(self, trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract all rules from a single trajectory.
        
        First extracts episodes from the trajectory, then extracts rules from each episode.
        
        Args:
            trajectory: A trajectory dictionary
            
        Returns:
            List of extracted rule dictionaries
        """
        # Import here to avoid circular dependency
        from .trajectory_collector import TrajectoryCollector
        
        collector = TrajectoryCollector()
        episodes = collector.extract_episodes(trajectory)
        
        if not episodes:
            logger.warning("[ExperienceExtractor] No episodes extracted from trajectory")
            return []
        
        return self.extract_rules_batch(episodes)
    
    def _build_extraction_prompt(self, episode: Dict[str, Any]) -> str:
        """
        Build the LLM prompt for rule extraction from an episode.
        
        Args:
            episode: An episode dictionary
            
        Returns:
            Formatted prompt string
        """
        state = episode.get("state", {})
        action = episode.get("action", {})
        outcome = episode.get("outcome", "unknown")
        
        prompt = f"""Analyze this KBQA step-level episode and extract a generalized error-correction rule.

=== EPISODE DATA ===

Question: {state.get('question', 'N/A')}

Linked Entities: {json.dumps(state.get('linked_entities', []), ensure_ascii=False)}

Current SPARQL:
{state.get('current_sparql', 'N/A')}

SPARQL Result Count: {state.get('sparql_result_count', 0)}

Error Type: {state.get('error_type', 'none')}

Error Message: {state.get('error_message', 'N/A')}

--- Action Taken ---
Type: {action.get('type', 'N/A')}
Reasoning: {action.get('reasoning', 'N/A')}
New SPARQL: {action.get('new_sparql', 'N/A')}

Outcome: {outcome}

Is Recovery Step: {episode.get('is_recovery', False)}

=== FEW-SHOT EXAMPLES ===
{FEW_SHOT_EXAMPLES}

=== INSTRUCTIONS ===
1. Generalize the pattern - do NOT reference specific entities or question text
2. Focus on the error/corrective pattern that can be reused
3. Include actionable steps in the action
4. Provide a SPARQL hint/pattern if applicable
5. Output ONLY valid JSON matching the schema

Extract the rule now:"""
        
        return prompt
    
    def _parse_rule_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse the LLM JSON response into a rule dictionary.
        
        Args:
            response_text: Raw text response from LLM
            
        Returns:
            Parsed rule dictionary or None if parsing failed
        """
        try:
            # Try direct JSON parse first
            rule = json.loads(response_text)
            return rule
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            try:
                rule = json.loads(json_match.group(1))
                return rule
            except json.JSONDecodeError:
                pass
        
        # Try to find any JSON object in the response
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
        if json_match:
            try:
                rule = json.loads(json_match.group(0))
                return rule
            except json.JSONDecodeError:
                pass
        
        logger.error(f"[ExperienceExtractor] Failed to parse JSON from response: {response_text[:200]}...")
        return None
    
    def _validate_rule(self, rule: Dict[str, Any]) -> bool:
        """
        Validate that a rule has all required fields.
        
        Args:
            rule: A rule dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ["title", "description", "rule_type"]
        
        for field in required_fields:
            if field not in rule:
                logger.warning(f"[ExperienceExtractor] Rule missing required field: {field}")
                return False
        
        # Validate rule_type is known
        if rule.get("rule_type") not in RULE_TYPES:
            logger.warning(f"[ExperienceExtractor] Unknown rule_type: {rule.get('rule_type')}")
            # Don't fail, just log warning
        
        # Validate action structure
        action = rule.get("action", {})
        if not isinstance(action, dict):
            logger.warning("[ExperienceExtractor] Action is not a dict")
            return False
        
        # Validate confidence is a number
        confidence = rule.get("confidence", 0.5)
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            rule["confidence"] = 0.5
        
        # Ensure state_keywords is a list
        if "state_keywords" in rule and not isinstance(rule["state_keywords"], list):
            rule["state_keywords"] = []
        
        return True
    
    def _generate_embedding_text(self, rule: Dict[str, Any]) -> str:
        """
        Generate text for vector embedding from a rule.
        
        Combines the rule's state description, action, and keywords into
        a single text suitable for semantic embedding.
        
        Args:
            rule: A rule dictionary
            
        Returns:
            Combined text string for embedding
        """
        parts = []
        
        # Title and description
        if rule.get("title"):
            parts.append(rule["title"])
        if rule.get("description"):
            parts.append(rule["description"])
        
        # State description
        if rule.get("state_description"):
            parts.append(rule["state_description"])
        
        # Keywords
        keywords = rule.get("state_keywords", [])
        if keywords:
            parts.append("Keywords: " + ", ".join(keywords))
        
        # Action description and steps
        action = rule.get("action", {})
        if action.get("description"):
            parts.append("Action: " + action["description"])
        
        steps = action.get("steps", [])
        if steps:
            parts.append("Steps: " + "; ".join(str(s) for s in steps))
        
        # SPARQL hint
        if action.get("sparql_hint"):
            parts.append("SPARQL pattern: " + action["sparql_hint"])
        
        return " | ".join(parts)
    
    def export_rules(self, rules: List[Dict[str, Any]], output_path: str) -> str:
        """
        Export rules to a JSON file.
        
        Args:
            rules: List of rule dictionaries
            output_path: Path to write the JSON output
            
        Returns:
            The output file path
        """
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        export_data = {
            "exported_at": datetime.utcnow().isoformat(),
            "rule_count": len(rules),
            "rules": rules
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[ExperienceExtractor] Exported {len(rules)} rules to {output_path}")
        return output_path


# ---- Standalone demo ----

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("[Demo] Experience Extractor Demo")
    print("=" * 60)
    
    # Create a sample episode for demonstration
    sample_episode = {
        "episode_id": "demo_ep_001",
        "source_trajectory": "demo_traj_001",
        "step_index": 1,
        "state": {
            "question": "What college did the 44th US president attend?",
            "linked_entities": ["Barack_Obama"],
            "current_sparql": "SELECT ?x WHERE { ns:Barack_Obama ns:education ?e . ?e ns:institution ?x }",
            "sparql_result_count": 0,
            "error_type": "empty_result",
            "error_message": "No results found for the predicate path"
        },
        "action": {
            "type": "relax",
            "reasoning": "The nested predicate path might not exist; try using a direct education institution predicate",
            "new_sparql": "SELECT ?x WHERE { ns:Barack_Obama ns:education_institution ?x }"
        },
        "outcome": "success",
        "is_recovery": True,
        "episode_type": "error_recovery"
    }
    
    # Check if we have LLM configuration
    llm_base_url = os.environ.get("LLM_BASE_URL", "")
    llm_api_key = os.environ.get("LLM_API_KEY", "")
    llm_model = os.environ.get("LLM_MODEL_NAME", "")
    
    if not all([llm_base_url, llm_api_key, llm_model]):
        print("\n[Demo] No LLM configuration found.")
        print("[Demo] Set these environment variables to run the full extraction:")
        print("  LLM_BASE_URL  - Base URL for OpenAI-compatible API")
        print("  LLM_API_KEY   - API key")
        print("  LLM_MODEL_NAME - Model name")
        print("\n[Demo] Showing prompt construction only...\n")
        
        # Just show the prompt that would be sent
        extractor = ExperienceExtractor(
            llm_base_url="http://localhost:8000/v1",
            llm_api_key="dummy",
            llm_model_name="dummy"
        )
        
        prompt = extractor._build_extraction_prompt(sample_episode)
        print("--- Generated Prompt ---")
        print(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
        print("--- End Prompt ---")
        
        # Show what embedding text would look like
        print("\n--- Embedding Text Example ---")
        sample_rule = {
            "title": "Use direct predicates for empty nested paths",
            "description": "When a nested predicate traversal returns empty, try direct predicates",
            "state_description": "Multi-hop predicate path returns no results",
            "state_keywords": ["empty", "predicate", "path", "nested", "direct"],
            "action": {
                "description": "Replace nested path with direct predicate",
                "steps": ["Check for direct predicate", "Substitute in SPARQL", "Re-execute"],
                "sparql_hint": "Replace ?x ns:p1 ?y . ?y ns:p2 ?z with ?x ns:p_direct ?z"
            }
        }
        embedding_text = extractor._generate_embedding_text(sample_rule)
        print(embedding_text)
        
    else:
        # Run full extraction
        print(f"\n[Demo] Using LLM: {llm_model}")
        
        extractor = ExperienceExtractor(
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            llm_model_name=llm_model
        )
        
        print("\n[Demo] Extracting rule from sample episode...")
        rule = extractor.extract_rule(sample_episode)
        
        if rule:
            print("\n[Demo] Extracted Rule:")
            print(json.dumps(rule, indent=2, ensure_ascii=False))
            
            # Export to file
            output_path = "/tmp/demo_rules.json"
            extractor.export_rules([rule], output_path)
            print(f"\n[Demo] Rule exported to {output_path}")
        else:
            print("\n[Demo] Failed to extract rule")
