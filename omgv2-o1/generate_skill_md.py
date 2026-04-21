#!/usr/bin/env python3
"""Generate new SKILL.md files with correct format and rule types."""

import json
import os

SKILLS_DIR = "/data/gt/omgv2-o1/skills/domain_skills"
RULES_FILE = "/data/gt/experience_kb/data/extracted_rules/all_rules_merged.jsonl"

# Categories to cover
CATEGORIES = {
    'RELATION_DIRECTION': ['direction', 'reverse', 'forward', 'incoming', 'outgoing', 'bidirectional'],
    'MULTI_HOP_PATTERN': ['multi-hop', 'multi hop', 'two-hop', 'two hop', 'chain', 'property path'],
    'CONSTRAINT_PATTERN': ['constraint', 'filter'],
    'CVT_NAVIGATION': ['cvt', 'compound value type'],
    'ERROR_RECOVERY': ['error', 'recovery', 'fix', 'repair']
}

def load_rules():
    """Load all rules from JSONL file."""
    rules = []
    with open(RULES_FILE, 'r') as f:
        for line in f:
            rule = json.loads(line.strip())
            rules.append(rule)
    return rules

def categorize_rule(rule):
    """Determine which category a rule belongs to."""
    title = rule.get('title', '').lower()
    desc = rule.get('state_description', '').lower()
    keywords = ' '.join(rule.get('state_keywords', [])).lower()
    text = f'{title} {desc} {keywords}'
    
    # Check categories in priority order
    for category, patterns in CATEGORIES.items():
        if any(pattern in text for pattern in patterns):
            return category
    return None

def select_rules(rules):
    """Select best rules for each category."""
    categorized = {cat: [] for cat in CATEGORIES}
    
    for rule in rules:
        category = categorize_rule(rule)
        if category:
            categorized[category].append(rule)
    
    # Sort each category by confidence and select top 3
    selected = []
    selected_ids = set()
    
    for category in CATEGORIES:
        sorted_rules = sorted(categorized[category], 
                            key=lambda x: x.get('confidence', 0), 
                            reverse=True)
        
        count = 0
        for rule in sorted_rules:
            if rule['rule_id'] not in selected_ids and count < 3:
                selected.append((category, rule))
                selected_ids.add(rule['rule_id'])
                count += 1
    
    return selected

def format_steps(steps):
    """Format steps as numbered list."""
    if not steps:
        return "1. Analyze the situation\n2. Apply the appropriate action\n3. Verify the result"
    
    formatted = []
    for i, step in enumerate(steps, 1):
        formatted.append(f"{i}. {step}")
    return "\n".join(formatted)

def generate_skill_md(category, rule):
    """Generate SKILL.md content for a rule."""
    title = rule.get('title', 'Unknown Rule')
    rule_id = rule.get('rule_id', 'unknown')
    confidence = rule.get('confidence', 0.5)
    state_desc = rule.get('state_description', '')
    action = rule.get('action', {})
    steps = action.get('steps', [])
    avoid = rule.get('avoid', '')
    keywords = rule.get('state_keywords', [])
    
    # Format keywords as YAML list
    keywords_yaml = json.dumps(keywords)
    
    # Format triggers (same as keywords for now)
    triggers_yaml = json.dumps(keywords)
    
    # Format steps
    steps_text = format_steps(steps)
    
    # Generate common cases based on state description
    common_cases = f"- When {state_desc.lower()}\n- Similar scenarios with the same pattern"
    
    content = f"""---
name: "{title}"
tags: {keywords_yaml}
triggers: {triggers_yaml}
confidence: {confidence}
success_count: 0
fail_count: 0
rule_type: "{category}"
rule_id: "{rule_id}"
version: "1.0"
source: "experience_kb"
---

# {title}

**Rule Type:** {category}  
**Confidence:** {confidence}  
**Rule ID:** {rule_id}

## When to Apply

{state_desc}

## Steps

{steps_text}

## Common Cases

{common_cases}

## Avoid

{avoid}
"""
    return content

def main():
    """Main function."""
    print("Loading rules...")
    rules = load_rules()
    print(f"Loaded {len(rules)} rules")
    
    print("\nSelecting rules for each category...")
    selected = select_rules(rules)
    
    # Count by category
    category_counts = {}
    for category, rule in selected:
        category_counts[category] = category_counts.get(category, 0) + 1
    
    print("\nSelected rules by category:")
    for cat in CATEGORIES:
        count = category_counts.get(cat, 0)
        print(f"  {cat}: {count} rules")
    
    print(f"\nTotal selected: {len(selected)} rules")
    
    # Remove old .md files (except template)
    print("\nRemoving old SKILL.md files...")
    for fname in os.listdir(SKILLS_DIR):
        if fname.endswith('.md') and fname != 'template.md':
            fpath = os.path.join(SKILLS_DIR, fname)
            os.remove(fpath)
            print(f"  Removed {fname}")
    
    # Generate new SKILL.md files
    print("\nGenerating new SKILL.md files...")
    for category, rule in selected:
        rule_id = rule['rule_id']
        fname = f"{rule_id}.md"
        fpath = os.path.join(SKILLS_DIR, fname)
        
        content = generate_skill_md(category, rule)
        
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  Created {fname} ({category})")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
