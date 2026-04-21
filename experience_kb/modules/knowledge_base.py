"""
KBQA Experience Knowledge Base - Core KB Manager.

Manages error-correction rules with FAISS vector indexing and metadata storage.
Adapted from vkbqa/modules/memory_manager.py and meta_consolidation.py.

Architecture:
    - FAISS IndexFlatIP for cosine similarity search (L2-normalized vectors)
    - JSON metadata storage (rules.json) without embeddings for compactness
    - Numpy embedding storage (embeddings.npy) separate from metadata
    - Consolidation: episodic -> semantic rule abstraction via LLM clustering
"""

import os
import json
import uuid
import math
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np

# FAISS for vector similarity search
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("[ExperienceKB] WARNING: faiss not installed. Search will use brute-force cosine similarity.")

# Sentence-transformers for text embedding
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False
    print("[ExperienceKB] WARNING: sentence-transformers not installed. Embedding requires external input.")


# Valid rule types
RULE_TYPES = [
    "ERROR_RECOVERY",       # SPARQL execution fails/returns empty
    "SUCCESS_SHORTCUT",     # Efficient path discovered
    "CONSTRAINT_GUIDE",     # Constraint too strict/loose
    "TYPE_MISMATCH",        # Entity/predicate type conflict
    "LOGICAL_STRUCTURE",    # Complex query patterns
    "Semantic_Rule",        # Abstracted meta-rule from consolidation
]

# Required fields for a valid rule
REQUIRED_RULE_FIELDS = ["rule_type", "state_description", "action"]
# action must have these sub-fields
REQUIRED_ACTION_FIELDS = ["description", "steps"]


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2 normalize vectors for cosine similarity with IndexFlatIP."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return vectors / norms


class ExperienceKB:
    """
    Core Knowledge Base for KBQA error-correction rules.
    
    Storage layout (under kb_dir/):
        rules.json       - Rule metadata (no embeddings)
        embeddings.npy   - Numpy array of L2-normalized embeddings
        faiss_index.bin  - FAISS index (if faiss available)
        kb_stats.json    - Statistics and version info
    """

    def __init__(self, kb_dir: str, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Experience KB.
        
        Args:
            kb_dir: Directory for KB storage
            embedding_model_name: Sentence-transformers model name
        """
        self.kb_dir = kb_dir
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        self.rules_file = os.path.join(kb_dir, "rules.json")
        self.embeddings_file = os.path.join(kb_dir, "embeddings.npy")
        self.index_file = os.path.join(kb_dir, "faiss_index.bin")
        self.stats_file = os.path.join(kb_dir, "kb_stats.json")

        os.makedirs(kb_dir, exist_ok=True)

        # Initialize embedding model
        self.embed_model = None
        if HAS_ST:
            try:
                print(f"[ExperienceKB] Loading embedding model: {embedding_model_name}")
                self.embed_model = SentenceTransformer(embedding_model_name)
                self.embedding_dim = self.embed_model.get_sentence_embedding_dimension()
                print(f"[ExperienceKB] Embedding model loaded (dim={self.embedding_dim}).")
            except Exception as e:
                print(f"[ExperienceKB] Error loading embedding model: {e}")
                self.embed_model = None

        # Load or initialize
        self.rules: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index = None
        self._id_to_idx: Dict[str, int] = {}
        self.load()

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def get_embedding(self, texts: List[str]) -> np.ndarray:
        """Generate L2-normalized embeddings for a list of texts."""
        if self.embed_model is None:
            raise RuntimeError(
                "[ExperienceKB] No embedding model available. "
                "Install sentence-transformers or provide embeddings externally."
            )
        embeddings = self.embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(embeddings, dtype=np.float32)

    # ------------------------------------------------------------------
    # CRUD Operations
    # ------------------------------------------------------------------

    def add_rule(self, rule: dict, embedding: Optional[np.ndarray] = None) -> str:
        """
        Add a single rule to the KB.
        
        Args:
            rule: Rule dict with required fields
            embedding: Optional pre-computed embedding (skips model inference)
            
        Returns:
            rule_id of the added rule
        """
        # Validate
        if not self._validate_rule(rule):
            raise ValueError(f"[ExperienceKB] Invalid rule: missing required fields. Got keys: {list(rule.keys())}")

        # Generate rule_id if missing
        if "rule_id" not in rule or not rule["rule_id"]:
            rule["rule_id"] = f"rule_{uuid.uuid4().hex[:8]}"

        # Generate embedding_text if missing
        if "embedding_text" not in rule or not rule["embedding_text"]:
            rule["embedding_text"] = self._generate_embedding_text(rule)

        # Compute embedding
        if embedding is None:
            emb = self.get_embedding([rule["embedding_text"]])[0]
        else:
            emb = np.array(embedding, dtype=np.float32).flatten()
            # Normalize
            norm = np.linalg.norm(emb)
            if norm > 1e-12:
                emb = emb / norm

        # Initialize metadata fields
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rule.setdefault("confidence", 0.8)
        rule.setdefault("success_count", 0)
        rule.setdefault("fail_count", 0)
        rule.setdefault("created_at", now)
        rule.setdefault("last_updated", now)
        rule.setdefault("consolidated", rule.get("rule_type") == "Semantic_Rule")
        rule.setdefault("source_trajectories", [])

        # Store rule (without embedding in the JSON)
        rule_meta = {k: v for k, v in rule.items() if k != "embedding"}
        self.rules.append(rule_meta)
        self._id_to_idx[rule_meta["rule_id"]] = len(self.rules) - 1

        # Append embedding
        if self.embeddings is None:
            self.embeddings = emb.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, emb.reshape(1, -1)])

        # Update FAISS index
        self._add_to_index(emb)

        print(f"[ExperienceKB] Added rule {rule_meta['rule_id']} (type={rule_meta['rule_type']}). Total: {len(self.rules)}")
        return rule_meta["rule_id"]

    def add_rules_batch(self, rules: List[dict]) -> List[str]:
        """Add multiple rules in batch. Returns list of rule_ids."""
        if not rules:
            return []

        # Validate and prepare
        valid_rules = []
        texts = []
        for r in rules:
            if self._validate_rule(r):
                if "rule_id" not in r or not r["rule_id"]:
                    r["rule_id"] = f"rule_{uuid.uuid4().hex[:8]}"
                if "embedding_text" not in r or not r["embedding_text"]:
                    r["embedding_text"] = self._generate_embedding_text(r)
                valid_rules.append(r)
                texts.append(r["embedding_text"])
            else:
                print(f"[ExperienceKB] Skipping invalid rule: {list(r.keys())}")

        if not valid_rules:
            return []

        # Batch embed
        embeddings = self.get_embedding(texts)

        # Add each
        rule_ids = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for i, rule in enumerate(valid_rules):
            rule.setdefault("confidence", 0.8)
            rule.setdefault("success_count", 0)
            rule.setdefault("fail_count", 0)
            rule.setdefault("created_at", now)
            rule.setdefault("last_updated", now)
            rule.setdefault("consolidated", rule.get("rule_type") == "Semantic_Rule")
            rule.setdefault("source_trajectories", [])

            rule_meta = {k: v for k, v in rule.items() if k != "embedding"}
            self.rules.append(rule_meta)
            self._id_to_idx[rule_meta["rule_id"]] = len(self.rules) - 1
            rule_ids.append(rule_meta["rule_id"])

        # Stack embeddings
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

        # Rebuild index (faster than incremental for batch)
        self._rebuild_index()

        print(f"[ExperienceKB] Batch added {len(rule_ids)} rules. Total: {len(self.rules)}")
        return rule_ids

    def search(self, query_text: str, top_k: int = 5,
               rule_types: Optional[List[str]] = None,
               threshold: float = 0.6) -> List[Dict]:
        """
        Search for rules matching the query.
        
        Args:
            query_text: Natural language query describing the current state
            top_k: Number of results to return
            rule_types: Optional filter by rule types
            threshold: Minimum similarity score
            
        Returns:
            List of rule dicts with 'score' field added, sorted by score desc
        """
        if not self.rules or self.embeddings is None:
            print("[ExperienceKB] Search called but KB is empty.")
            return []

        # Embed query
        query_emb = self.get_embedding([query_text])[0]

        # Search
        if HAS_FAISS and self.index is not None:
            scores, indices = self._search_faiss(query_emb, top_k=max(top_k * 3, 10))
        else:
            scores, indices = self._search_brute(query_emb, top_k=max(top_k * 3, 10))

        # Collect and filter
        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.rules):
                continue
            if score < threshold:
                continue
            rule = self.rules[idx].copy()
            rule["score"] = float(score)

            # Type filter
            if rule_types and rule.get("rule_type") not in rule_types:
                continue

            # Boost Semantic_Rule by 15%
            if rule.get("rule_type") == "Semantic_Rule":
                rule["score"] *= 1.15

            results.append(rule)

        # T8: Add success_rate weighting (0.2 factor)
        for r in results:
            s = r.get("success_count", 0)
            f = r.get("fail_count", 0)
            total = s + f
            if total > 0:
                r["_success_rate"] = s / total
            else:
                r["_success_rate"] = 0.5  # neutral prior
            r["score"] = r.get("score", 0) + r["_success_rate"] * 0.2

        # Sort by boosted score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_rule(self, rule_id: str) -> Optional[Dict]:
        """Get a rule by its ID."""
        idx = self._id_to_idx.get(rule_id)
        if idx is not None and idx < len(self.rules):
            return self.rules[idx].copy()
        return None

    def update_stats(self, rule_id: str, success: bool):
        """Update success/fail counts for a rule."""
        idx = self._id_to_idx.get(rule_id)
        if idx is None:
            print(f"[ExperienceKB] update_stats: rule {rule_id} not found.")
            return
        if success:
            self.rules[idx]["success_count"] = self.rules[idx].get("success_count", 0) + 1
        else:
            self.rules[idx]["fail_count"] = self.rules[idx].get("fail_count", 0) + 1
        self.rules[idx]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Recalculate confidence
        sc = self.rules[idx].get("success_count", 0)
        fc = self.rules[idx].get("fail_count", 0)
        total = sc + fc
        if total > 0:
            self.rules[idx]["confidence"] = round(sc / total, 4)

    def update_success(self, rule_id: str) -> bool:
        """Increment success_count for a rule. Returns True if found."""
        idx = self._id_to_idx.get(rule_id)
        if idx is None:
            return False
        self.rules[idx]["success_count"] = self.rules[idx].get("success_count", 0) + 1
        self.rules[idx]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_rules()
        return True

    def update_failure(self, rule_id: str) -> bool:
        """Increment fail_count for a rule. Returns True if found."""
        idx = self._id_to_idx.get(rule_id)
        if idx is None:
            return False
        self.rules[idx]["fail_count"] = self.rules[idx].get("fail_count", 0) + 1
        self.rules[idx]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_rules()
        return True

    def get_success_rate(self, rule_id: str) -> float:
        """Return success rate for a rule. Returns 0.5 (neutral) if no data."""
        idx = self._id_to_idx.get(rule_id)
        if idx is None:
            return 0.0
        s = self.rules[idx].get("success_count", 0)
        f = self.rules[idx].get("fail_count", 0)
        total = s + f
        if total == 0:
            return 0.5  # neutral prior
        return s / total

    def _save_rules(self):
        """Save rules metadata to rules.json (no embeddings)."""
        with open(self.rules_file, 'w', encoding='utf-8') as f:
            json.dump(self.rules, f, ensure_ascii=False, indent=2)

    def delete_rule(self, rule_id: str):
        """Remove a rule from the KB."""
        idx = self._id_to_idx.get(rule_id)
        if idx is None:
            print(f"[ExperienceKB] delete_rule: rule {rule_id} not found.")
            return
        del self.rules[idx]
        if self.embeddings is not None and idx < len(self.embeddings):
            self.embeddings = np.delete(self.embeddings, idx, axis=0)
        # Rebuild index and id map
        self._rebuild_id_map()
        self._rebuild_index()
        print(f"[ExperienceKB] Deleted rule {rule_id}. Total: {len(self.rules)}")

    def list_rules(self, rule_type: Optional[str] = None,
                   consolidated: Optional[bool] = None) -> List[Dict]:
        """List rules with optional filters."""
        results = []
        for r in self.rules:
            if rule_type and r.get("rule_type") != rule_type:
                continue
            if consolidated is not None and r.get("consolidated") != consolidated:
                continue
            results.append(r.copy())
        return results

    def get_stats(self) -> Dict:
        """Return KB statistics."""
        type_counts = {}
        consolidated_count = 0
        total_success = 0
        total_fail = 0
        for r in self.rules:
            rt = r.get("rule_type", "unknown")
            type_counts[rt] = type_counts.get(rt, 0) + 1
            if r.get("consolidated"):
                consolidated_count += 1
            total_success += r.get("success_count", 0)
            total_fail += r.get("fail_count", 0)

        return {
            "total_rules": len(self.rules),
            "by_type": type_counts,
            "consolidated": consolidated_count,
            "unconsolidated": len(self.rules) - consolidated_count,
            "total_success": total_success,
            "total_fail": total_fail,
            "embedding_dim": self.embedding_dim,
        }

    # ------------------------------------------------------------------
    # Consolidation (adapted from meta_consolidation.py)
    # ------------------------------------------------------------------

    def consolidate(self, llm_client=None, cluster_sim_threshold: float = 0.75,
                    min_unconsolidated: int = 3, max_per_batch: int = 10):
        """
        Cluster unconsolidated episodic rules and abstract semantic meta-rules.
        
        Args:
            llm_client: LLM client with chat_completion method (from experience_extractor.LLMClient)
            cluster_sim_threshold: Cosine similarity threshold for clustering
            min_unconsolidated: Minimum unconsolidated rules to trigger consolidation
            max_per_batch: Max rules per abstraction batch
        """
        if llm_client is None:
            print("[ExperienceKB] Consolidation requires llm_client. Skipping.")
            return

        unconsolidated = [
            (i, r) for i, r in enumerate(self.rules)
            if not r.get("consolidated", False) and r.get("rule_type") != "Semantic_Rule"
        ]

        if len(unconsolidated) < min_unconsolidated:
            print(f"[ExperienceKB] Only {len(unconsolidated)} unconsolidated rules (need {min_unconsolidated}). Skipping.")
            return

        # Get embeddings for unconsolidated rules
        indices = [i for i, _ in unconsolidated]
        if self.embeddings is None or max(indices) >= len(self.embeddings):
            print("[ExperienceKB] Embedding index mismatch. Skipping consolidation.")
            return

        embs = self.embeddings[indices]
        embs_norm = _l2_normalize(embs.astype(np.float64))

        # Cluster by connected components
        sim_matrix = embs_norm @ embs_norm.T
        n = len(indices)
        visited = [False] * n
        clusters = []

        for i in range(n):
            if visited[i]:
                continue
            stack = [i]
            visited[i] = True
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in range(n):
                    if not visited[v] and sim_matrix[u, v] >= cluster_sim_threshold:
                        visited[v] = True
                        stack.append(v)
            clusters.append(comp)

        print(f"[ExperienceKB] Clustered {n} unconsolidated rules into {len(clusters)} groups.")

        for comp in clusters:
            if len(comp) < 2:
                # Mark singletons as consolidated
                for ci in comp:
                    self.rules[indices[ci]]["consolidated"] = True
                continue

            # Chunk if too large
            for chunk_start in range(0, len(comp), max_per_batch):
                chunk = comp[chunk_start:chunk_start + max_per_batch]
                self._do_consolidate_cluster(
                    [indices[ci] for ci in chunk], llm_client
                )

        self.save()
        print("[ExperienceKB] Consolidation complete.")

    def _do_consolidate_cluster(self, rule_indices: List[int], llm_client):
        """Abstract 1-2 semantic meta-rules from a cluster of episodic rules."""
        rules_text = ""
        for i, ri in enumerate(rule_indices):
            r = self.rules[ri]
            rules_text += (
                f"--- Rule {i+1} ---\n"
                f"Type: {r.get('rule_type')}\n"
                f"Title: {r.get('title', 'N/A')}\n"
                f"State: {r.get('state_description', 'N/A')}\n"
                f"Action: {r.get('action', {}).get('description', 'N/A')}\n"
                f"Steps: {r.get('action', {}).get('steps', [])}\n\n"
            )

        prompt = f"""You are the Cognitive Architect of a KBQA error-correction system.
Analyze the following episodic error-correction rules and abstract 1-2 high-level Semantic Rules.

{rules_text}
[Task]
Abstract these into 1-2 Meta-Rules (Semantic_Rule type).
- Generalize: rules must transfer to similar KBQA task families.
- Avoid tying the meta-rule to unique entity names or one-off details.
- Include ONE short anchor example in the content that illustrates the pattern.

For each rule, provide:
1. "title": A high-level rule title (e.g., "[Meta-Rule] Empty Result Recovery via Type Verification").
2. "description": Explain the overarching error pattern and strategic boundary.
3. "content": Actionable advice plus the anchor example.
4. "state_description": When this meta-rule applies.
5. "state_keywords": 3-5 keywords for retrieval.

Output valid JSON:
{{"meta_rules": [{{"title": "...", "description": "...", "content": "...", "state_description": "...", "state_keywords": [...]}}]}}
"""
        try:
            response = llm_client.chat_completion(
                [
                    {"role": "system", "content": "You are a KBQA error-correction analyst. Output raw JSON ONLY."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            content = response["choices"][0]["message"]["content"].strip()
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
            parsed = json.loads(content)
            meta_rules = parsed.get("meta_rules", [])
        except Exception as e:
            print(f"[ExperienceKB] Consolidation LLM call failed: {e}")
            # Mark as consolidated to avoid retrying
            for ri in rule_indices:
                self.rules[ri]["consolidated"] = True
            return

        for mr in meta_rules:
            if not isinstance(mr, dict):
                continue
            title = (mr.get("title") or "").strip()
            content = (mr.get("content") or "").strip()
            if not title or not content:
                continue

            new_rule = {
                "title": title,
                "description": mr.get("description", ""),
                "rule_type": "Semantic_Rule",
                "state_description": mr.get("state_description", ""),
                "state_keywords": mr.get("state_keywords", []),
                "action": {
                    "description": content,
                    "steps": [content],
                },
                "source_trajectories": [],
            }
            try:
                self.add_rule(new_rule)
                print(f"[ExperienceKB] Added meta-rule: {title}")
            except Exception as e:
                print(f"[ExperienceKB] Failed to add meta-rule: {e}")

        # Mark source rules as consolidated
        for ri in rule_indices:
            self.rules[ri]["consolidated"] = True

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        """Persist KB to disk."""
        # Save rules metadata
        tmp_rules = self.rules_file + ".tmp"
        with open(tmp_rules, "w", encoding="utf-8") as f:
            json.dump(self.rules, f, ensure_ascii=False, indent=2)
        os.replace(tmp_rules, self.rules_file)

        # Save embeddings
        if self.embeddings is not None:
            np.save(self.embeddings_file, self.embeddings.astype(np.float32))

        # Save FAISS index
        if HAS_FAISS and self.index is not None:
            try:
                faiss.write_index(self.index, self.index_file)
            except Exception as e:
                print(f"[ExperienceKB] Error saving FAISS index: {e}")

        # Save stats
        stats = self.get_stats()
        stats["last_saved"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        print(f"[ExperienceKB] Saved {len(self.rules)} rules to {self.kb_dir}")

    def load(self):
        """Load KB from disk."""
        # Load rules
        if os.path.exists(self.rules_file):
            try:
                with open(self.rules_file, "r", encoding="utf-8") as f:
                    self.rules = json.load(f)
                print(f"[ExperienceKB] Loaded {len(self.rules)} rules from {self.rules_file}")
            except Exception as e:
                print(f"[ExperienceKB] Error loading rules: {e}")
                self.rules = []

        # Rebuild ID map
        self._rebuild_id_map()

        # Load embeddings
        if os.path.exists(self.embeddings_file):
            try:
                self.embeddings = np.load(self.embeddings_file).astype(np.float32)
                print(f"[ExperienceKB] Loaded embeddings: shape={self.embeddings.shape}")
            except Exception as e:
                print(f"[ExperienceKB] Error loading embeddings: {e}")
                self.embeddings = None

        # Load FAISS index
        if HAS_FAISS and os.path.exists(self.index_file):
            try:
                self.index = faiss.read_index(self.index_file)
                print(f"[ExperienceKB] Loaded FAISS index: {self.index.ntotal} vectors")
            except Exception as e:
                print(f"[ExperienceKB] Error loading FAISS index: {e}")
                self.index = None
        elif HAS_FAISS and self.embeddings is not None and len(self.embeddings) > 0:
            self._rebuild_index()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_rule(self, rule: dict) -> bool:
        """Check that a rule has required fields."""
        if not isinstance(rule, dict):
            return False
        for field in REQUIRED_RULE_FIELDS:
            if field not in rule or not rule[field]:
                return False
        action = rule.get("action", {})
        if not isinstance(action, dict):
            return False
        for field in REQUIRED_ACTION_FIELDS:
            if field not in action or not action[field]:
                return False
        return True

    def _generate_embedding_text(self, rule: dict) -> str:
        """Combine rule fields into a single text for embedding."""
        parts = [
            rule.get("state_description", ""),
            rule.get("action", {}).get("description", ""),
            " ".join(rule.get("state_keywords", [])),
        ]
        if rule.get("title"):
            parts.insert(0, rule["title"])
        return " ".join(p for p in parts if p).strip()

    def _rebuild_id_map(self):
        """Rebuild the rule_id -> index mapping."""
        self._id_to_idx = {}
        for i, r in enumerate(self.rules):
            rid = r.get("rule_id")
            if rid:
                self._id_to_idx[rid] = i

    def _rebuild_index(self):
        """Rebuild the FAISS index from current embeddings."""
        if not HAS_FAISS or self.embeddings is None or len(self.embeddings) == 0:
            return
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        # Ensure L2 normalized
        embs = _l2_normalize(self.embeddings)
        self.index.add(embs)

    def _add_to_index(self, embedding: np.ndarray):
        """Add a single embedding to the FAISS index."""
        if not HAS_FAISS:
            return
        emb = embedding.reshape(1, -1).astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 1e-12:
            emb = emb / norm
        if self.index is None:
            self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    def _search_faiss(self, query_emb: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search using FAISS."""
        query = query_emb.reshape(1, -1).astype(np.float32)
        norm = np.linalg.norm(query)
        if norm > 1e-12:
            query = query / norm
        D, I = self.index.search(query, min(top_k, self.index.ntotal))
        return D[0], I[0]

    def _search_brute(self, query_emb: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Brute-force cosine similarity search (fallback when FAISS unavailable)."""
        if self.embeddings is None or len(self.embeddings) == 0:
            return np.array([]), np.array([], dtype=int)
        # Normalize
        q = query_emb / (np.linalg.norm(query_emb) + 1e-12)
        embs = _l2_normalize(self.embeddings)
        sims = embs @ q
        top_k = min(top_k, len(sims))
        idx = np.argsort(sims)[-top_k:][::-1]
        return sims[idx], idx


# ------------------------------------------------------------------
# Standalone demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("=== ExperienceKB Demo ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        kb = ExperienceKB(tmpdir)

        if kb.embed_model is None:
            print("No embedding model available. Install sentence-transformers to run full demo.")
            exit(0)

        # Add sample rules
        rules = [
            {
                "title": "Empty SPARQL Result Recovery",
                "description": "When a SPARQL query returns 0 results, the entity may be incorrectly linked.",
                "rule_type": "ERROR_RECOVERY",
                "state_description": "SPARQL query with specific entity constraint returns empty results",
                "state_keywords": ["empty", "zero_results", "entity", "sparql"],
                "action": {
                    "description": "Verify entity type and try broader entity search",
                    "steps": [
                        "Check entity type with SearchTypes",
                        "Try broader entity search with partial name match",
                        "Relax the constraint that caused the empty result",
                    ],
                },
            },
            {
                "title": "Birth Date Direct Lookup",
                "description": "For birth_date questions, direct property lookup is sufficient.",
                "rule_type": "SUCCESS_SHORTCUT",
                "state_description": "Question asks for birth date of a person entity",
                "state_keywords": ["birth_date", "born", "person", "date"],
                "action": {
                    "description": "Use direct property path without complex joins",
                    "steps": [
                        "Link to person entity",
                        "Use property.people.person.date_of_birth directly",
                        "No need for intermediate joins",
                    ],
                },
            },
            {
                "title": "Type Mismatch Person-Organization",
                "description": "When predicate expects Person but entity is Organization.",
                "rule_type": "TYPE_MISMATCH",
                "state_description": "SPARQL type error: predicate domain mismatch between Person and Organization",
                "state_keywords": ["type_error", "person", "organization", "domain"],
                "action": {
                    "description": "Re-link entity or find correct predicate",
                    "steps": [
                        "Check entity type with SearchTypes",
                        "If person linked as org, re-link with person filter",
                        "Find alternative predicate that matches entity type",
                    ],
                },
            },
        ]

        ids = kb.add_rules_batch(rules)
        print(f"\nAdded rule IDs: {ids}")
        print(f"KB Stats: {json.dumps(kb.get_stats(), indent=2)}")

        # Search
        print("\n--- Search: 'SPARQL returns empty set' ---")
        results = kb.search("SPARQL returns empty set, entity might be wrong", top_k=2)
        for r in results:
            print(f"  [{r['score']:.3f}] {r['rule_type']}: {r.get('title', 'N/A')}")

        print("\n--- Search: 'birth date question' ---")
        results = kb.search("What year was Einstein born?", top_k=2)
        for r in results:
            print(f"  [{r['score']:.3f}] {r['rule_type']}: {r.get('title', 'N/A')}")

        # Update stats
        kb.update_stats(ids[0], success=True)
        kb.update_stats(ids[0], success=True)
        kb.update_stats(ids[0], success=False)
        rule = kb.get_rule(ids[0])
        print(f"\nUpdated rule {ids[0]}: success={rule['success_count']}, fail={rule['fail_count']}, confidence={rule['confidence']}")

        # Save and reload
        kb.save()
        kb2 = ExperienceKB(tmpdir)
        print(f"\nReloaded KB: {len(kb2.rules)} rules")

        print("\n=== Demo Complete ===")
