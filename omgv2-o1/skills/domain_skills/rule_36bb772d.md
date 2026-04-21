---
name: "Fix bracket syntax errors in SPARQL"
tags: ["syntax error", "bracket notation", "SPARQL compiler error", "complex pattern"]
triggers: ["syntax error", "bracket notation", "SPARQL compiler error", "complex pattern"]
confidence: 0.95
success_count: 0
fail_count: 0
rule_type: "ERROR_RECOVERY"
rule_id: "rule_36bb772d"
version: "1.0"
source: "experience_kb"
---

# Fix bracket syntax errors in SPARQL

**Rule Type:** ERROR_RECOVERY  
**Confidence:** 0.95  
**Rule ID:** rule_36bb772d

## When to Apply

SPARQL query fails with syntax error using bracket notation for paths

## Steps

1. Replace predicate [nested ?var] with explicit triple patterns
2. Use intermediate variables for each step
3. Example: ?actor predicate1 [predicate2 ?character] becomes ?actor predicate1 ?intermediate. ?intermediate predicate2 ?character

## Common Cases

- When sparql query fails with syntax error using bracket notation for paths
- Similar scenarios with the same pattern

## Avoid

Using non-standard SPARQL syntax for nested patterns
