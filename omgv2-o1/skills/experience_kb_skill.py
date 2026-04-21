"""Experience KB skill for omgv2-o1 — retrieves relevant error-correction rules
from the Experience Knowledge Base and formats them as LLM prompt guidance.

T6: Also loads SKILL.md documents from domain_skills/ and supports
retrieving them via keyword/category matching.
"""

import sys
import os
import re
import logging
from typing import Dict, List, Optional

logger = logging.getLogger("exp_kb")
# Uncomment for debug: logging.basicConfig(level=logging.DEBUG)

# Lazy-loaded singleton
_KB = None
_KB_LOAD_FAILED = False

# T8: Track last retrieved rule_ids for feedback loop
_LAST_RETRIEVED_RULE_IDS = []

def get_last_retrieved_rule_ids() -> list:
    """Return the rule IDs from the most recent search_experience_rules() call."""
    return list(_LAST_RETRIEVED_RULE_IDS)

# T6: SKILL.md documents — lazy-loaded
_SKILL_DOCS: Optional[Dict[str, dict]] = None
_SKILLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "domain_skills")

def _get_kb():
    global _KB, _KB_LOAD_FAILED
    if _KB_LOAD_FAILED:
        return None
    if _KB is None:
        try:
            sys.path.insert(0, "/data/gt/experience_kb")
            from modules.knowledge_base import ExperienceKB
            _KB = ExperienceKB("/data/gt/experience_kb/data/experience_kb")
        except Exception as e:
            logger.warning(f"KB load failed: {e}")
            _KB_LOAD_FAILED = True
            return None
    return _KB


def search_experience_rules(
    question: str,
    current_entity: str = "",
    current_expression: str = "",
    available_relations: list = None,
    last_failure: str = "",
    top_k: int = 3,
    threshold: float = 0.4,
) -> str:
    """Search the Experience KB for relevant rules. Returns formatted text for prompt injection.
    Returns empty string if no relevant rules found or KB unavailable.
    
    Strategy: prioritize error/relations over question text for better KB matching.
    """
    global _LAST_RETRIEVED_RULE_IDS
    kb = _get_kb()
    if kb is None:
        _LAST_RETRIEVED_RULE_IDS = []
        return ""

    # Build queries — try multiple strategies, take best results
    all_results = []
    seen_ids = set()

    queries = _build_queries(question, current_entity, available_relations, last_failure, current_expression)

    for query_text, q_threshold in queries:
        try:
            results = kb.search(query_text, top_k=top_k, threshold=q_threshold)
        except Exception:
            continue
        for r in results:
            rid = r.get("rule_id", r.get("title", ""))
            if rid not in seen_ids:
                seen_ids.add(rid)
                all_results.append(r)

    if not all_results:
        logger.debug(f"No KB match: q={question[:50]} fail={last_failure[:50]}")
        _LAST_RETRIEVED_RULE_IDS = []
        return ""

    # Sort by score, take top_k
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    top_results = all_results[:top_k]

    logger.debug(f"KB matched {len(top_results)} rules for: {question[:50]}")
    # T8: Record retrieved rule_ids for feedback loop
    _LAST_RETRIEVED_RULE_IDS = [r.get("rule_id", r.get("title", "")) for r in top_results]
    return format_rules_for_prompt(top_results)


def _build_queries(question, current_entity, available_relations, last_failure, current_expression=""):
    """Build prioritized query list. Returns [(query_text, threshold)]."""
    queries = []
    rels_str = ", ".join(available_relations[:8]) if available_relations else ""

    # Strategy 1: Error-focused (highest priority)
    if last_failure:
        err_q = f"error: {last_failure}"
        if current_expression:
            err_q += f" | current LF: {current_expression}"
        if rels_str:
            err_q += f" | relations: {rels_str}"
        queries.append((err_q, 0.3))

    # Strategy 2: Relation-focused (what tool to use next)
    if rels_str:
        rel_q = f"available relations for KBQA join: {rels_str}"
        queries.append((rel_q, 0.35))

    # Strategy 3: Question + relations (moderate threshold)
    if question:
        parts = [question]
        if rels_str:
            parts.append(f"candidate predicates: {rels_str}")
        queries.append((" | ".join(parts), 0.4))

    # Strategy 4: Rule-type fallback — inject structural guides unconditionally
    # when step-by-step is reached (path-guided already failed)
    queries.append(("KBQA step-by-step reasoning: choose correct relation, avoid wrong direction", 0.25))

    return queries


def consult_experience_active(
    state_description: str,
    last_error: str = "",
    current_expr: str = "",
    available_relations: list = None,
    top_k: int = 5,  # 主动查询时增加 top_k
    threshold: float = 0.3,  # 主动查询时降低阈值
) -> dict:
    """主动查询 Experience KB，返回结构化结果。"""
    kb = _get_kb()
    if kb is None:
        return {
            "matched_rules": [],
            "guidance_text": "",
            "rule_ids": [],
            "confidence": 0.0,
            "query_type": "active_consult"
        }

    # 构建查询 - 优先使用错误信息
    queries = []
    if last_error:
        queries.append((f"error: {last_error}", 0.3))
    if available_relations:
        queries.append((f"relations: {', '.join(available_relations[:5])}", 0.35))
    if state_description:
        queries.append((state_description, 0.4))

    all_results = []
    seen_ids = set()
    for query_text, q_threshold in queries:
        try:
            results = kb.search(query_text, top_k=top_k, threshold=q_threshold)
            for r in results:
                rid = r.get("rule_id", r.get("title", ""))
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    all_results.append(r)
        except Exception:
            continue

    if not all_results:
        return {
            "matched_rules": [],
            "guidance_text": "",
            "rule_ids": [],
            "confidence": 0.0,
            "query_type": "active_consult"
        }

    # 按分数排序
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    top_results = all_results[:top_k]

    # 计算置信度
    avg_score = sum(r.get("score", 0) for r in top_results) / len(top_results)

    return {
        "matched_rules": top_results,
        "guidance_text": format_rules_for_prompt(top_results),
        "rule_ids": [r.get("rule_id", r.get("title", "")) for r in top_results],
        "confidence": avg_score,
        "query_type": "active_consult"
    }


def _parse_frontmatter(content: str) -> tuple:
    """Parse YAML-like frontmatter from SKILL.md content.
    Returns (metadata_dict, body_text).
    """
    metadata = {}
    body = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1].strip()
            body = parts[2].strip()
            for line in fm_text.split("\n"):
                line = line.strip()
                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    # Try to parse JSON arrays
                    if val.startswith("["):
                        try:
                            import json
                            val = json.loads(val)
                        except Exception:
                            pass
                    # Try to parse numbers
                    elif val.replace(".", "").isdigit():
                        try:
                            val = float(val) if "." in val else int(val)
                        except Exception:
                            pass
                    metadata[key] = val
    return metadata, body


def _load_skill_docs() -> Dict[str, dict]:
    """Load all SKILL.md files from domain_skills/ directory."""
    global _SKILL_DOCS
    if _SKILL_DOCS is not None:
        return _SKILL_DOCS

    _SKILL_DOCS = {}
    if not os.path.isdir(_SKILLS_DIR):
        logger.debug(f"domain_skills dir not found: {_SKILLS_DIR}")
        return _SKILL_DOCS

    for fname in os.listdir(_SKILLS_DIR):
        if not fname.endswith(".md"):
            continue
        # Skip template file
        if fname == "template.md":
            continue
        fpath = os.path.join(_SKILLS_DIR, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            metadata, body = _parse_frontmatter(content)
            rule_id = metadata.get("rule_id", fname.replace(".md", ""))
            _SKILL_DOCS[rule_id] = {
                "metadata": metadata,
                "body": body,
                "full_content": content,
                "file_path": fpath,
            }
        except Exception as e:
            logger.warning(f"Failed to load {fname}: {e}")

    logger.debug(f"Loaded {len(_SKILL_DOCS)} SKILL.md docs from {_SKILLS_DIR}")
    return _SKILL_DOCS


def search_skill_docs(
    query: str,
    top_k: int = 3,
    category: str = None,
) -> List[dict]:
    """Search SKILL.md documents by keyword matching.
    Returns list of {rule_id, metadata, body, score, full_content}.
    """
    docs = _load_skill_docs()
    if not docs:
        return []

    query_lower = query.lower()
    query_words = set(re.findall(r'\w+', query_lower))

    scored = []
    for rule_id, doc in docs.items():
        meta = doc["metadata"]
        body = doc["body"]

        # Category filter
        if category and meta.get("rule_type", "").lower() != category.lower() and meta.get("category", "").lower() != category.lower():
            continue

        # Score by keyword overlap (keywords + tags + triggers for SKILL.md compat)
        keywords = meta.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [keywords]
        tags = meta.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        triggers = meta.get("triggers", [])
        if isinstance(triggers, str):
            triggers = [triggers]
        kw_set = set(k.lower() for k in list(keywords) + list(tags) + list(triggers))

        # Title + body text for matching
        title = meta.get("title", "").lower()
        text = (title + " " + body.lower())
        text_words = set(re.findall(r'\w+', text))

        # Compute score: keyword overlap + word overlap + title match
        kw_overlap = len(query_words & kw_set) * 3.0
        word_overlap = len(query_words & text_words) * 1.0
        title_match = sum(1 for w in query_words if w in title) * 2.0
        
        # Handle confidence - ensure it's a number
        conf_val = meta.get("confidence", 0.5)
        try:
            confidence = float(conf_val) * 1.0
        except (ValueError, TypeError):
            confidence = 0.5  # default if not a valid number

        score = kw_overlap + word_overlap + title_match + confidence

        if score > 0:
            scored.append({
                "rule_id": rule_id,
                "metadata": meta,
                "body": body,
                "full_content": doc["full_content"],
                "score": score,
                "title": meta.get("name", ""),
                "rule_type": meta.get("rule_type", ""),
                "confidence": confidence,
                "file_path": doc["file_path"],
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def get_skill_doc_content(rule_id: str) -> str:
    """Get the full SKILL.md content for a specific rule_id."""
    docs = _load_skill_docs()
    doc = docs.get(rule_id)
    if doc:
        return doc["full_content"]
    return ""


def list_skill_docs() -> List[dict]:
    """List all loaded SKILL.md documents with metadata."""
    docs = _load_skill_docs()
    return [
        {"rule_id": rid, "metadata": d["metadata"], "file_path": d["file_path"]}
        for rid, d in docs.items()
    ]


def format_rules_for_prompt(rules: list) -> str:
    """Format search results into a prompt-ready text block.
    
    P1-T3: Concise format — max 2 rules, truncated fields.
    """
    lines = [f"[Experience Guidance — {len(rules)} rules matched]"]
    for i, r in enumerate(rules[:2], 1):  # P1-T3: limit to top 2 rules
        rule_type = r.get("rule_type", "?")
        title = r.get("title", "N/A")
        score = r.get("score", 0)
        desc = r.get("action", {}).get("description", "")
        # Truncate long descriptions
        if len(desc) > 100:
            desc = desc[:97] + "..."
        avoid = r.get("avoid", "")
        lines.append(f"{i}. [{rule_type}] (score={score:.2f}) {title}")
        lines.append(f"   Do: {desc}")
        if avoid:
            if len(avoid) > 80:
                avoid = avoid[:77] + "..."
            lines.append(f"   Avoid: {avoid}")
    return "\n".join(lines)
