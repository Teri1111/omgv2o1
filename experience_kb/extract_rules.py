"""Extract experience rules from SFT episodes using LLM.

Groups episodes into trajectories, sends each to LLM for rule extraction,
outputs structured rules compatible with knowledge_base.py.
"""

import json
import os
import sys
import re
import time
import argparse
from collections import defaultdict
from datetime import datetime

import requests

# ── Config ──────────────────────────────────────────────────────────────────
LLM_API_URL = "http://127.0.0.1:8000/v1/completions"
LLM_MODEL = "qwen3.5-9b"
MAX_OBS_LEN = 300       # truncate observations
MAX_THINK_LEN = 400     # truncate think blocks
MAX_TRAJ_CHARS = 12000  # truncate entire trajectory text
REQUEST_TIMEOUT = 60
RATE_LIMIT_DELAY = 0.3  # seconds between LLM calls

# ── Paths ───────────────────────────────────────────────────────────────────
DEFAULT_EPISODES = "/data/gt/experience_kb/data/all_episodes.jsonl"
DEFAULT_OUTPUT_DIR = "/data/gt/experience_kb/data/extracted_rules"


# ── Data Loading ────────────────────────────────────────────────────────────

def load_episodes(path):
    """Load episodes from JSONL."""
    episodes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def group_by_trajectory(episodes):
    """Group episodes into complete trajectories by source_index."""
    trajs = defaultdict(list)
    for ep in episodes:
        trajs[ep["source_index"]].append(ep)
    result = {}
    for sid, eps in trajs.items():
        eps.sort(key=lambda e: e["step_index"])
        result[sid] = {
            "question": eps[0]["question"],
            "total_steps": eps[0].get("full_trajectory_length", len(eps)),
            "steps": eps,
        }
    return result


# ── Trajectory Formatting ──────────────────────────────────────────────────

def format_trajectory(traj):
    """Format a trajectory into readable text for LLM analysis."""
    lines = [f"Question: {traj['question']}", f"Total steps: {traj['total_steps']}", ""]

    for ep in traj["steps"]:
        step_n = ep["step_index"]
        atype = ep["action_type"]
        think = (ep.get("think") or "")[:MAX_THINK_LEN]
        action = ep.get("action", "")
        obs = (ep.get("observation") or "")[:MAX_OBS_LEN]
        obs_ok = ep.get("observation_success", True)
        is_final = ep.get("is_final_step", False)

        lines.append(f"=== Step {step_n} ({atype}) ===")
        if think:
            lines.append(f"Think: {think}")
        lines.append(f"Action: {action[:500]}")
        if obs:
            status = "OK" if obs_ok else "FAILED"
            lines.append(f"Observation [{status}]: {obs}")
        if is_final:
            lines.append("(FINAL STEP)")
        lines.append("")

    return "\n".join(lines)


# ── LLM Interaction ────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are an expert KBQA analyst. Analyze the following reasoning trajectory and extract 1-3 reusable error-correction or strategy rules.

Trajectory:
{trajectory_text}

Extract rules that are GENERALIZED (transfer to other KBQA questions, not specific to this one).

Focus on patterns like:
- Entity linking strategy (how entities are found and verified)
- Subgraph exploration patterns (SearchGraphPatterns semantic filtering, relation traversal)
- Error recovery (empty results, wrong entities, wrong relations)
- Aggregation/counting patterns (when to count, argmax, etc.)
- Multi-hop reasoning strategy (how to chain relations)
- SPARQL query construction patterns
- When to stop searching (finish condition)

Output a JSON array of 1-3 rule objects. Each rule MUST have:
{{
  "title": "short descriptive title (max 60 chars)",
  "rule_type": "ERROR_RECOVERY|SUCCESS_SHORTCUT|CONSTRAINT_GUIDE|TYPE_MISMATCH|LOGICAL_STRUCTURE",
  "state_description": "when this rule applies (generalized, not question-specific)",
  "state_keywords": ["keyword1", "keyword2", "keyword3"],
  "action": {{
    "description": "what to do (1-2 sentences)",
    "steps": ["step 1", "step 2", "step 3"]
  }},
  "avoid": "common pitfall to avoid",
  "confidence": 0.5-1.0
}}

Rules must be domain-general. Never mention specific entities from the trajectory.
Output ONLY the JSON array, no explanation."""


def call_llm(prompt):
    """Call vLLM completions API. Returns text or None."""
    try:
        payload = {
            "model": LLM_MODEL,
            "prompt": prompt,
            "max_tokens": 2048,
            "temperature": 0.1,
        }
        resp = requests.post(LLM_API_URL, json=payload, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("choices", [{}])[0].get("text", "").strip()
        else:
            print(f"  [WARN] HTTP {resp.status_code}", file=sys.stderr)
            return None
    except Exception as e:
        print(f"  [ERROR] LLM call failed: {e}", file=sys.stderr)
        return None


def parse_rules_from_response(raw_text):
    """Extract JSON array from LLM response text."""
    if not raw_text:
        return []

    # Try to find JSON array
    # First, look for ```json ... ``` blocks
    code_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw_text, re.DOTALL)
    if code_block:
        raw_text = code_block.group(1)

    # Find the outermost [ ... ]
    start = raw_text.find('[')
    end = raw_text.rfind(']')
    if start == -1 or end == -1 or end <= start:
        return []

    json_str = raw_text[start:end + 1]
    try:
        rules = json.loads(json_str)
        if isinstance(rules, list):
            return rules
    except json.JSONDecodeError:
        # Try fixing common issues
        # Remove trailing commas
        fixed = re.sub(r',\s*([}\]])', r'\1', json_str)
        try:
            rules = json.loads(fixed)
            if isinstance(rules, list):
                return rules
        except json.JSONDecodeError:
            pass

    return []


# ── Rule Validation ─────────────────────────────────────────────────────────

VALID_RULE_TYPES = {
    "ERROR_RECOVERY", "SUCCESS_SHORTCUT", "CONSTRAINT_GUIDE",
    "TYPE_MISMATCH", "LOGICAL_STRUCTURE"
}


def validate_rule(rule):
    """Validate a single rule dict. Returns (is_valid, error_msg)."""
    if not isinstance(rule, dict):
        return False, "not a dict"

    # Required top-level fields
    for field in ["rule_type", "state_description", "action"]:
        if field not in rule or not rule[field]:
            return False, f"missing field: {field}"

    # Valid rule_type
    if rule["rule_type"] not in VALID_RULE_TYPES:
        return False, f"invalid rule_type: {rule['rule_type']}"

    # Action must have description and steps
    action = rule["action"]
    if not isinstance(action, dict):
        return False, "action not a dict"
    for field in ["description", "steps"]:
        if field not in action or not action[field]:
            return False, f"action missing: {field}"
    if not isinstance(action["steps"], list) or len(action["steps"]) == 0:
        return False, "action.steps must be non-empty list"

    return True, None


def enrich_rule(rule, source_question, source_index):
    """Add missing fields to make rule compatible with knowledge_base.py."""
    import uuid

    # Generate rule_id
    if "rule_id" not in rule:
        rule["rule_id"] = f"rule_{uuid.uuid4().hex[:8]}"

    # Default title from state_description
    if "title" not in rule:
        rule["title"] = rule["state_description"][:60]

    # Default state_keywords
    if "state_keywords" not in rule:
        # Extract keywords from state_description
        words = re.findall(r'\b[a-z_]{3,}\b', rule["state_description"].lower())
        rule["state_keywords"] = list(set(words))[:5]

    # Default confidence
    if "confidence" not in rule:
        rule["confidence"] = 0.7

    # Ensure avoid field
    if "avoid" not in rule:
        rule["avoid"] = ""

    # Add source info
    rule["source_question"] = source_question
    rule["source_trajectory"] = source_index
    rule["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return rule


# ── Deduplication ───────────────────────────────────────────────────────────

def is_duplicate(rule, existing_rules, threshold=0.85):
    """Simple dedup by title similarity."""
    title = rule.get("title", "").lower()
    for er in existing_rules:
        et = er.get("title", "").lower()
        if title == et:
            return True
        # Jaccard on word sets
        s1 = set(title.split())
        s2 = set(et.split())
        if s1 and s2:
            jaccard = len(s1 & s2) / len(s1 | s2)
            if jaccard > threshold:
                return True
    return False


# ── Main Pipeline ───────────────────────────────────────────────────────────

def extract_rules_from_trajectory(traj, source_index):
    """Extract rules from one trajectory using LLM."""
    traj_text = format_trajectory(traj)
    if len(traj_text) > MAX_TRAJ_CHARS:
        traj_text = traj_text[:MAX_TRAJ_CHARS] + "\n... [truncated]"

    prompt = EXTRACTION_PROMPT.format(trajectory_text=traj_text)
    raw = call_llm(prompt)
    if not raw:
        return []

    raw_rules = parse_rules_from_response(raw)
    valid_rules = []
    for r in raw_rules:
        ok, err = validate_rule(r)
        if ok:
            r = enrich_rule(r, traj["question"], source_index)
            valid_rules.append(r)
        else:
            print(f"  [SKIP] Invalid rule: {err}", file=sys.stderr)

    return valid_rules


def run_extract(episodes_path, output_dir, sample_size=None, sample_only=False):
    """Main extraction pipeline."""
    print(f"Loading episodes from {episodes_path}")
    episodes = load_episodes(episodes_path)
    print(f"  Loaded {len(episodes)} episodes")

    trajs = group_by_trajectory(episodes)
    print(f"  Grouped into {len(trajs)} trajectories")

    # Length distribution
    from collections import Counter
    lens = Counter(len(t["steps"]) for t in trajs.values())
    print(f"  Length distribution: {dict(sorted(lens.items()))}")

    if sample_size:
        traj_items = sorted(trajs.items())[:sample_size]
    else:
        traj_items = sorted(trajs.items())

    print(f"\nProcessing {len(traj_items)} trajectories...")

    os.makedirs(output_dir, exist_ok=True)
    rules_path = os.path.join(output_dir, "rules.jsonl")

    all_rules = []
    success_count = 0
    error_count = 0
    total_rules = 0

    for idx, (sid, traj) in enumerate(traj_items):
        print(f"\n[{idx + 1}/{len(traj_items)}] Trajectory {sid}: "
              f"{traj['question'][:60]}... ({len(traj['steps'])} steps)")

        rules = extract_rules_from_trajectory(traj, sid)

        if rules:
            all_rules.extend(rules)
            total_rules += len(rules)
            success_count += 1
            print(f"  Extracted {len(rules)} rules")
            for r in rules:
                print(f"    [{r['rule_type']}] {r['title']}")
        else:
            print(f"  No rules extracted")
            error_count += 1

        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)

        # Incremental save every 50 trajectories
        if not sample_only and (idx + 1) % 50 == 0:
            _save_rules(all_rules, rules_path, do_dedup=True)
            print(f"\n  [SAVE] Checkpoint: {len(all_rules)} rules saved")

    # Final save
    if all_rules:
        _save_rules(all_rules, rules_path, do_dedup=True)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Extraction complete!")
    print(f"  Trajectories processed: {len(traj_items)}")
    print(f"  With rules: {success_count}")
    print(f"  No rules: {error_count}")
    print(f"  Total rules (before dedup): {total_rules}")
    print(f"  Final rules (after dedup): {len(all_rules)}")
    print(f"  Output: {rules_path}")

    # Rule type distribution
    type_counts = Counter(r["rule_type"] for r in all_rules)
    print(f"  Rule types: {dict(type_counts)}")

    return all_rules


def _save_rules(rules, path, do_dedup=True):
    """Save rules to JSONL, optionally deduplicating."""
    if do_dedup:
        seen_titles = set()
        unique = []
        for r in rules:
            t = r.get("title", "").lower()
            if t not in seen_titles:
                seen_titles.add(t)
                unique.append(r)
        rules = unique

    with open(path, "w", encoding="utf-8") as f:
        for r in rules:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(rules)} rules to {path}")


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract experience rules from SFT episodes")
    parser.add_argument("--episodes", default=DEFAULT_EPISODES, help="Path to episodes JSONL")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--sample", type=int, default=None,
                        help="Process only first N trajectories (for testing)")
    parser.add_argument("--sample-only", action="store_true",
                        help="Don't do final dedup/save (test mode)")

    args = parser.parse_args()
    run_extract(args.episodes, args.output_dir, args.sample, args.sample_only)
