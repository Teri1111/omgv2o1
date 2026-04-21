---
name: "Explore each constraint separately then combine in SPARQL"
tags: ["multi-constraint", "intersection", "AND condition", "combined filter"]
triggers: ["multi-constraint", "intersection", "AND condition", "combined filter"]
confidence: 0.95
success_count: 0
fail_count: 0
rule_type: "CONSTRAINT_PATTERN"
rule_id: "rule_1f61cd4a"
version: "1.0"
source: "experience_kb"
---

# Explore each constraint separately then combine in SPARQL

**Rule Type:** CONSTRAINT_PATTERN  
**Confidence:** 0.95  
**Rule ID:** rule_1f61cd4a

## When to Apply

Question has multiple constraints on different attributes of the answer entity (e.g., 'country that contains X AND uses currency Y').

## Steps

1. Use SearchGraphPatterns for first constraint to find relevant predicate
2. Use SearchGraphPatterns for second constraint to find relevant predicate
3. Combine both predicate paths into one SPARQL query sharing the answer variable
4. Execute and verify the combined query

## Common Cases

- When question has multiple constraints on different attributes of the answer entity (e.g., 'country that contains x and uses currency y').
- Similar scenarios with the same pattern

## Avoid

Trying to find a single predicate that satisfies both constraints simultaneously without first exploring each one.
