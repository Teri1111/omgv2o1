# KBQA Experience Knowledge Base - Architecture Design

## 1. Problem Statement

KBQA (Knowledge Base Question Answering) tasks require generating Logical Forms (e.g., SPARQL queries) from natural language questions. The generation process is multi-step and error-prone:
- Entity linking errors (wrong entity selected)
- Type mismatch errors (querying wrong predicate types)
- Constraint errors (too restrictive or too loose)
- Structural errors (incorrect SPARQL syntax/logic)

The KnowCoder-A1 paper uses RL curriculum to train models, producing "experience" in the form of successful and failed trajectories. This project converts that experience into a **structured, runtime-queryable Knowledge Base** that guides LLMs during Logical Form generation without further fine-tuning.

## 2. Key Insight: Error-Correction Library

Rather than training or fine-tuning, we build an **Error-Correction Library** — a collection of state→action rules extracted from RL trajectories:

```
STATE: "SPARQL query returns 0 results, entity 'college' linked but should be 'high school'"
ACTION: "Use SearchTypes to verify entity type; relax constraint on education level"
TYPE: ERROR_RECOVERY
```

At runtime, before each SPARQL generation step, the LLM queries this KB with its current state and receives targeted advice.

## 3. Architecture

### 3.1 Data Flow

```
[RL Trajectories / Teacher Demonstrations]
         |
         v
[TrajectoryCollector] -- parse & segment trajectories into step-level episodes
         |
         v
[Step-Level Episodes: {state, action, outcome, error_type}]
         |
         v
[ExperienceExtractor] -- LLM analyzes episodes, extracts structured rules
         |
         v
[Error-Correction Rules: {state_desc, action_desc, rule_type, confidence}]
         |
         v
[KnowledgeBase] -- embed rules, index with FAISS, store with metadata
         |
         v
[RuleRetriever] -- query with current state, return top-K relevant rules
         |
         v
[PipelineIntegration] -- inject rules into LLM prompt before SPARQL step
```

### 3.2 Module Specifications

#### Module 1: trajectory_collector.py
**Input**: Raw trajectory files (JSON) from RL training or teacher model
**Output**: Step-level episodes

Each trajectory contains:
```json
{
  "question": "What college did Obama attend?",
  "gold_sparql": "SELECT ?x WHERE { ... }",
  "steps": [
    {
      "step_id": 0,
      "state": {
        "question": "...",
        "linked_entities": ["Barack_Obama"],
        "current_sparql_partial": "SELECT ?x WHERE { ns:m.02mjmr ...",
        "sparql_result": {"count": 0, "error": "empty result set"},
        "error_message": "No results found"
      },
      "action": {
        "type": "entity_relink",
        "content": "Search for 'Obama' with broader entity search",
        "new_sparql": "..."
      },
      "outcome": "success|failure|partial"
    }
  ],
  "final_correct": true,
  "recovery_count": 2
}
```

#### Module 2: experience_extractor.py
Uses LLM to analyze each step and extract structured rules:

**Rule Schema**:
```json
{
  "rule_id": "uuid",
  "rule_type": "ERROR_RECOVERY|SUCCESS_SHORTCUT|CONSTRAINT_GUIDE|TYPE_MISMATCH",
  "state_pattern": {
    "error_type": "empty_result|type_mismatch|constraint_too_strict",
    "context_keywords": ["SPARQL", "empty", "relax"],
    "state_description": "Human-readable description of the triggering state"
  },
  "action": {
    "description": "What corrective action to take",
    "steps": ["Step 1: ...", "Step 2: ..."],
    "sparql_pattern": "Optional SPARQL template/pattern"
  },
  "embedding_text": "Combined text for vector embedding",
  "source_trajectories": ["traj_id_1", "traj_id_2"],
  "confidence": 0.85,
  "success_count": 0,
  "fail_count": 0,
  "created_at": "timestamp"
}
```

**Extraction Prompt Pattern** (adapted from existing memory_worker.py):
- Analyze the step-level trajectory
- Focus on the error state and the corrective action
- Generate a generalized, reusable rule (not specific to this question)
- Classify the rule type

#### Module 3: knowledge_base.py
Core storage and retrieval engine:

```python
class ExperienceKB:
    def __init__(self, kb_path, embedding_model="all-MiniLM-L6-v2"):
        # Load/create FAISS index
        # Load rule metadata from JSON
    
    def add_rule(self, rule: dict) -> str:
        # Generate embedding for rule.embedding_text
        # Add to FAISS index
        # Store metadata in JSON
        # Return rule_id
    
    def search(self, query_text: str, top_k: int = 5, 
               rule_types: List[str] = None, threshold: float = 0.6) -> List[dict]:
        # Embed query
        # Search FAISS
        # Filter by rule_type if specified
        # Filter by threshold
        # Return top-K rules with scores
    
    def update_stats(self, rule_id: str, success: bool):
        # Update success_count/fail_count
        # Recalculate confidence
    
    def consolidate(self):
        # Cluster similar episodic rules
        # Generate semantic (meta) rules via LLM
        # Similar to meta_consolidation.py pattern
```

**Embedding Strategy**:
- Use sentence-transformers (all-MiniLM-L6-v2 or bge-base-en)
- Embedding text = f"{state_description} {action_description}"
- FAISS IndexFlatIP for cosine similarity (normalized vectors)

#### Module 4: rule_retriever.py
Runtime retrieval interface:

```python
class RuleRetriever:
    def __init__(self, knowledge_base: ExperienceKB):
        self.kb = knowledge_base
    
    def retrieve_for_state(self, 
                           question: str,
                           current_sparql: str,
                           last_error: str = None,
                           entity_links: List[str] = None,
                           step_history: List[dict] = None) -> str:
        """
        Given the current SPARQL generation state, retrieve relevant 
        error-correction rules and format them as prompt guidance.
        
        Returns formatted string to inject into LLM prompt.
        """
        # Build state description from context
        # Query KB with state description
        # Filter and rank results
        # Format as prompt guidance text
```

**Retrieval Context Construction**:
```python
state_text = f"""
Question: {question}
Current SPARQL: {current_sparql}
Linked Entities: {entity_links}
Last Error: {last_error or 'none'}
Step: {len(step_history)} of generation
"""
```

**Prompt Injection Format**:
```
[Error-Correction Knowledge Base - Retrieved Rules]

Rule 1 (ERROR_RECOVERY, confidence: 0.92):
  State: When SPARQL returns empty results with a specific entity constraint
  Action: 1. Verify entity type with SearchTypes
          2. Try broader entity search
          3. Relax the constraint on the problematic triple pattern
  Source: Recovered from 5 similar cases

Rule 2 (CONSTRAINT_GUIDE, confidence: 0.87):
  ...
```

#### Module 5: pipeline_integration.py
Integration adapter for existing KBQA pipelines:

```python
class KBQAPipelineWithExperience:
    def __init__(self, sparql_generator, experience_kb_path):
        self.generator = sparql_generator
        self.kb = ExperienceKB(experience_kb_path)
        self.retriever = RuleRetriever(self.kb)
    
    def generate_sparql(self, question: str, entities: List[str]) -> str:
        """Generate SPARQL with experience-guided correction loop."""
        history = []
        current_sparql = ""
        
        for step in range(self.max_steps):
            # Retrieve relevant rules for current state
            guidance = self.retriever.retrieve_for_state(
                question=question,
                current_sparql=current_sparql,
                last_error=history[-1].get("error") if history else None,
                entity_links=entities,
                step_history=history
            )
            
            # Generate next SPARQL step with guidance injected
            result = self.generator.generate(
                question=question,
                current_sparql=current_sparql,
                guidance=guidance,
                step_history=history
            )
            
            # Execute SPARQL and check result
            sparql_result = self.executor.execute(result["sparql"])
            
            # Log step for future learning
            history.append({
                "step": step,
                "state": {...},
                "action": result,
                "outcome": sparql_result
            })
            
            if sparql_result.get("is_final"):
                break
        
        return current_sparql
```

### 3.3 Rule Types Taxonomy

| Rule Type | Trigger | Example |
|-----------|---------|---------|
| ERROR_RECOVERY | SPARQL execution fails/returns empty | "If empty result, check entity type with SearchTypes" |
| CONSTRAINT_GUIDE | Constraint too strict/loose | "For 'how many' questions, use COUNT with GROUP BY" |
| TYPE_MISMATCH | Entity/predicate type conflict | "If predicate expects Person but got Organization, re-link entity" |
| SUCCESS_SHORTCUT | Efficient path discovered | "For birth_date questions, direct property lookup is sufficient" |
| LOGICAL_STRUCTURE | Complex query patterns | "For negation questions, use FILTER NOT EXISTS pattern" |

### 3.4 Consolidation Strategy

Adapted from meta_consolidation.py:

1. **Episodic Rules**: Extracted from individual step-level episodes
2. **Semantic Rules**: Abstracted from clusters of similar episodic rules
3. **Consolidation Trigger**: When N unconsolidated episodic rules have cosine sim > threshold
4. **Abstraction**: LLM analyzes cluster → produces 1-2 generalized meta-rules

### 3.5 Dependencies

```
sentence-transformers  # Text embedding (lightweight, no CLIP needed for text-only)
faiss-cpu             # Vector similarity search
numpy                 # Numerical operations
json                  # Data storage
uuid                  # Rule ID generation
```

## 4. Relationship to Existing vkbqa Code

| vkbqa Component | experience_kb Adaptation |
|----------------|--------------------------|
| memory_worker.py | experience_extractor.py (same LLM extraction pattern) |
| memory_manager.py | knowledge_base.py (same FAISS + JSON storage) |
| meta_consolidation.py | knowledge_base.py.consolidate() (same clustering logic) |
| iterative_reasoner.py Step 0 | rule_retriever.py.retrieve_for_state() |
| CLIP embeddings | sentence-transformers (text-only) |
| Visual QA state | SPARQL generation state |

## 5. Data Format Specifications

### 5.1 Input Trajectory Format
```json
{
  "trajectory_id": "traj_xxx",
  "question": "Natural language question",
  "gold_sparql": "Ground truth SPARQL",
  "gold_answer": "Ground truth answer",
  "steps": [
    {
      "step_id": 0,
      "state": {
        "question": "...",
        "linked_entities": [...],
        "current_sparql": "...",
        "execution_result": {...},
        "error": "error message or null"
      },
      "action": {
        "type": "generate|revise|relink|relax",
        "reasoning": "LLM's reasoning for this step",
        "new_sparql": "..."
      },
      "outcome": "success|failure|partial"
    }
  ],
  "final_correct": true,
  "total_steps": 3,
  "recovery_steps": 1
}
```

### 5.2 Stored Rule Format
```json
{
  "rule_id": "rule_uuid8",
  "rule_type": "ERROR_RECOVERY",
  "state_pattern": {
    "error_type": "empty_result",
    "state_description": "SPARQL query with specific entity returns empty results",
    "context_keywords": ["empty", "zero", "no_results", "entity"]
  },
  "action": {
    "description": "Verify entity type and try broader search",
    "steps": [
      "1. Use SearchTypes to check the linked entity's type",
      "2. If type mismatch, search for alternative entities",
      "3. Relax the constraint that caused the empty result"
    ],
    "sparql_pattern": null
  },
  "embedding_text": "SPARQL empty result entity type mismatch: verify entity type with SearchTypes, try broader entity search, relax constraint",
  "source_trajectories": ["traj_abc", "traj_def"],
  "confidence": 0.85,
  "success_count": 3,
  "fail_count": 1,
  "created_at": "2026-04-15T00:00:00",
  "consolidated": false,
  "embedding": [0.123, -0.456, ...]
}
```
