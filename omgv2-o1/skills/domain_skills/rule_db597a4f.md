---
name: "Fix arrow notation errors in SPARQL"
tags: ["syntax error", "arrow notation", "->", "SPARQL compiler error"]
triggers: ["syntax error", "arrow notation", "->", "SPARQL compiler error"]
confidence: 0.95
success_count: 0
fail_count: 0
rule_type: "ERROR_RECOVERY"
rule_id: "rule_db597a4f"
version: "1.0"
source: "experience_kb"
---

# Fix arrow notation errors in SPARQL

**Rule Type:** ERROR_RECOVERY  
**Confidence:** 0.95  
**Rule ID:** rule_db597a4f

## When to Apply

SPARQL query fails with syntax error containing '>' or arrow notation

## Steps

1. Replace -> with / for property paths
2. Or split into separate triple patterns with intermediate variable
3. Example: ?x ns:predicate1 -> ns:predicate2 ?y becomes ?x ns:predicate1/ns:predicate2 ?y

## Common Cases

- When sparql query fails with syntax error containing '>' or arrow notation
- Similar scenarios with the same pattern

## Avoid

Copying arrow notation from SearchGraphPatterns directly into SPARQL
